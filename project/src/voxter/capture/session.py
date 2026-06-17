"""Manual raw capture session orchestration."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol, TextIO

from voxter.capture.events import (
    InputEventReader,
    RawInputEvent,
    RawTerminalEvent,
    validate_input_events,
    validate_terminal_events,
)
from voxter.capture.frames import (
    FrameCaptureError,
    FrameCaptureRecord,
    GrimFrameCapture,
    parse_geometry,
    validate_frame_records,
)
from voxter.capture.pipewire import PipeWireGStreamerFrameCapture
from voxter.capture.preview import (
    PreviewGenerationError,
    PreviewGenerationResult,
    generate_capture_preview,
)
from voxter.contracts import ActionState, extract_action_transitions


class FrameCaptureBackend(Protocol):
    """Backend interface used by manual capture sessions."""

    @property
    def file_suffix(self) -> str:
        """Return the file suffix used for captured frames."""

    @property
    def backend_name(self) -> str:
        """Return a stable backend identifier."""

    def close(self) -> None:
        """Release backend resources."""

    def capture(
        self,
        frame_path: Path,
        *,
        run_id: str,
        attempt_id: str | None,
        frame_index: int,
        action: ActionState,
        action_sample_timestamp: float,
    ) -> FrameCaptureRecord:
        """Capture one frame."""


@dataclass(frozen=True, slots=True)
class CaptureSessionConfig:
    """Configuration for one manual raw capture session."""

    output_dir: Path
    run_id: str
    attempt_id: str | None
    geometry: str
    event_device: str
    duration_s: float
    target_hz: float
    backend: str = "grim"
    image_format: str = "jpeg"
    jpeg_quality: int = 70
    png_level: int = 0
    key_code: int = 17
    terminal_key_code: int | None = None
    terminal_type: str = "death"
    portal_source_types: int = 1
    portal_cursor_mode: int = 1
    portal_request_timeout_s: int = 20
    async_writes: bool = True
    write_queue_size: int = 8
    output_width: int | None = None
    output_height: int | None = None
    generate_preview: bool = True
    preview_name: str = "preview.mp4"


@dataclass(frozen=True, slots=True)
class CaptureSessionSummary:
    """Summary written after a manual raw capture session."""

    run_id: str
    attempt_id: str | None
    output_dir: str
    geometry: str
    event_device: str
    duration_s: float
    target_hz: float
    frame_count: int
    effective_hz: float
    input_event_count: int
    terminal_event_count: int
    press_release_count: int
    frame_transition_count: int
    held_frame_count: int
    released_frame_count: int
    min_capture_duration_ms: float | None
    mean_capture_duration_ms: float | None
    max_capture_duration_ms: float | None
    dropped_frame_count: int
    missed_deadline_estimate: int
    capture_backend: str
    image_format: str
    frame_width: int
    frame_height: int
    source_width: int
    source_height: int
    capture_resized: bool
    capture_side_preprocessing: tuple[str, ...]
    preview_path: str | None
    preview_generated: bool
    preview_error: str | None

    def to_json_dict(self) -> dict[str, object]:
        """Return a stable JSON-compatible representation."""

        return {
            "run_id": self.run_id,
            "attempt_id": self.attempt_id,
            "output_dir": self.output_dir,
            "geometry": self.geometry,
            "event_device": self.event_device,
            "duration_s": self.duration_s,
            "target_hz": self.target_hz,
            "frame_count": self.frame_count,
            "effective_hz": self.effective_hz,
            "input_event_count": self.input_event_count,
            "terminal_event_count": self.terminal_event_count,
            "press_release_count": self.press_release_count,
            "frame_transition_count": self.frame_transition_count,
            "held_frame_count": self.held_frame_count,
            "released_frame_count": self.released_frame_count,
            "min_capture_duration_ms": self.min_capture_duration_ms,
            "mean_capture_duration_ms": self.mean_capture_duration_ms,
            "max_capture_duration_ms": self.max_capture_duration_ms,
            "dropped_frame_count": self.dropped_frame_count,
            "missed_deadline_estimate": self.missed_deadline_estimate,
            "capture_backend": self.capture_backend,
            "image_format": self.image_format,
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
            "source_width": self.source_width,
            "source_height": self.source_height,
            "capture_resized": self.capture_resized,
            "capture_side_preprocessing": list(self.capture_side_preprocessing),
            "preview_path": self.preview_path,
            "preview_generated": self.preview_generated,
            "preview_error": self.preview_error,
        }


def run_capture_session(config: CaptureSessionConfig) -> CaptureSessionSummary:
    """Capture synchronized frame and input-event streams into `output_dir`."""

    if config.duration_s <= 0:
        raise ValueError("duration_s must be positive")
    if config.target_hz <= 0:
        raise ValueError("target_hz must be positive")
    if (config.output_width is None) != (config.output_height is None):
        raise ValueError("output_width and output_height must be set together")
    if config.output_width is not None and config.output_width <= 0:
        raise ValueError("output_width must be positive")
    if config.output_height is not None and config.output_height <= 0:
        raise ValueError("output_height must be positive")
    if config.backend != "pipewire" and config.output_width is not None:
        raise ValueError(
            "capture-side resizing is currently supported by pipewire only"
        )

    output_dir = config.output_dir
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    frame_manifest_path = output_dir / "frames.jsonl"
    input_events_path = output_dir / "input_events.jsonl"
    terminal_events_path = output_dir / "terminal_events.jsonl"
    summary_path = output_dir / "capture_summary.json"

    capture = _make_capture_backend(config)

    frame_records: list[FrameCaptureRecord] = []
    input_events: list[RawInputEvent] = []
    terminal_events: list[RawTerminalEvent] = []
    capture_durations: list[float] = []
    dropped_frame_count = 0
    missed_deadline_estimate = 0
    period_s = 1.0 / config.target_hz

    try:
        with (
            InputEventReader(
                config.event_device,
                run_id=config.run_id,
                attempt_id=config.attempt_id,
                key_code=config.key_code,
                terminal_key_code=config.terminal_key_code,
                terminal_type=config.terminal_type,
            ) as event_reader,
            frame_manifest_path.open("w", encoding="utf-8") as frame_file,
            input_events_path.open("w", encoding="utf-8") as event_file,
            terminal_events_path.open("w", encoding="utf-8") as terminal_file,
        ):
            started_at = time.monotonic()
            next_frame_at = started_at
            frame_index = 0

            while time.monotonic() - started_at < config.duration_s:
                new_events = event_reader.read_available()
                _write_input_events(event_file, new_events)
                input_events.extend(new_events)
                new_terminal_events = event_reader.pop_terminal_events()
                _write_terminal_events(terminal_file, new_terminal_events)
                terminal_events.extend(new_terminal_events)

                now = time.monotonic()
                if now < next_frame_at:
                    time.sleep(min(0.003, next_frame_at - now))
                    continue
                missed_periods = int((now - next_frame_at) // period_s)
                if missed_periods > 0:
                    missed_deadline_estimate += missed_periods
                    next_frame_at += missed_periods * period_s

                frame_path = frames_dir / f"{frame_index:06d}{capture.file_suffix}"
                action_sample_timestamp = time.time()
                try:
                    frame_record = capture.capture(
                        frame_path,
                        run_id=config.run_id,
                        attempt_id=config.attempt_id,
                        frame_index=frame_index,
                        action=event_reader.current_action,
                        action_sample_timestamp=action_sample_timestamp,
                    )
                except FrameCaptureError:
                    dropped_frame_count += 1
                    late_events = event_reader.read_available()
                    _write_input_events(event_file, late_events)
                    input_events.extend(late_events)
                    late_terminal_events = event_reader.pop_terminal_events()
                    _write_terminal_events(terminal_file, late_terminal_events)
                    terminal_events.extend(late_terminal_events)
                    next_frame_at += period_s
                    continue

                late_events = event_reader.read_available()
                _write_input_events(event_file, late_events)
                input_events.extend(late_events)
                late_terminal_events = event_reader.pop_terminal_events()
                _write_terminal_events(terminal_file, late_terminal_events)
                terminal_events.extend(late_terminal_events)

                corrected_action = _input_state_at(
                    input_events,
                    frame_record.action_sample_timestamp,
                )
                if corrected_action is not frame_record.action:
                    frame_record = replace(frame_record, action=corrected_action)

                frame_records.append(frame_record)
                frame_file.write(
                    json.dumps(frame_record.to_json_dict(), sort_keys=True)
                )
                frame_file.write("\n")
                capture_durations.append(frame_record.capture_duration_s)

                frame_index += 1
                next_frame_at += period_s
    finally:
        capture.close()

    validate_frame_records(frame_records)
    validate_input_events(input_events)
    validate_terminal_events(terminal_events)
    preview_result: PreviewGenerationResult | None = None
    preview_error: str | None = None
    if config.generate_preview:
        try:
            preview_result = generate_capture_preview(
                output_dir,
                frame_records,
                input_events,
                terminal_events,
                fps=config.target_hz,
                preview_name=config.preview_name,
                rejected_tail_s=0.35,
                rejected_skip_s=1.5,
            )
        except (OSError, ValueError, PreviewGenerationError) as exc:
            preview_error = str(exc)

    summary = _build_summary(
        config=config,
        capture_backend=capture.backend_name,
        frame_records=frame_records,
        input_events=input_events,
        terminal_events=terminal_events,
        capture_durations=capture_durations,
        dropped_frame_count=dropped_frame_count,
        missed_deadline_estimate=missed_deadline_estimate,
        preview_path=preview_result.preview_path if preview_result else None,
        preview_error=preview_error,
    )
    summary_path.write_text(
        json.dumps(summary.to_json_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if preview_error is not None:
        raise PreviewGenerationError(preview_error)
    return summary


def _make_capture_backend(config: CaptureSessionConfig) -> FrameCaptureBackend:
    if config.backend == "grim":
        return GrimFrameCapture(
            config.geometry,
            image_format=config.image_format,
            jpeg_quality=config.jpeg_quality,
            png_level=config.png_level,
        )
    if config.backend == "pipewire":
        return PipeWireGStreamerFrameCapture(
            config.geometry,
            source_types=config.portal_source_types,
            cursor_mode=config.portal_cursor_mode,
            portal_request_timeout_s=config.portal_request_timeout_s,
            image_format=config.image_format,
            jpeg_quality=config.jpeg_quality,
            async_writes=config.async_writes,
            write_queue_size=config.write_queue_size,
            output_width=config.output_width,
            output_height=config.output_height,
        )
    raise ValueError("backend must be 'grim' or 'pipewire'")


def _write_input_events(
    event_file: TextIO,
    events: list[RawInputEvent],
) -> None:
    for event in events:
        event_file.write(json.dumps(event.to_json_dict(), sort_keys=True))
        event_file.write("\n")


def _write_terminal_events(
    event_file: TextIO,
    events: list[RawTerminalEvent],
) -> None:
    for event in events:
        event_file.write(json.dumps(event.to_json_dict(), sort_keys=True))
        event_file.write("\n")


def _input_state_at(
    events: list[RawInputEvent],
    timestamp: float,
    *,
    initial_state: ActionState = ActionState.RELEASED,
) -> ActionState:
    state = initial_state
    for event in events:
        if event.timestamp > timestamp:
            break
        state = event.action
    return state


def _build_summary(
    *,
    config: CaptureSessionConfig,
    capture_backend: str,
    frame_records: list[FrameCaptureRecord],
    input_events: list[RawInputEvent],
    capture_durations: list[float],
    dropped_frame_count: int,
    missed_deadline_estimate: int,
    terminal_events: list[RawTerminalEvent] | None = None,
    preview_path: str | None = None,
    preview_error: str | None = None,
) -> CaptureSessionSummary:
    frame_transitions = extract_action_transitions(
        frame_record.action for frame_record in frame_records
    )
    press_release_count = sum(
        1 for event in input_events if event.kind.value in {"press", "release"}
    )
    terminal_event_count = len(terminal_events or [])
    held_frame_count = sum(
        1 for frame_record in frame_records if frame_record.action is ActionState.HELD
    )
    released_frame_count = len(frame_records) - held_frame_count
    effective_hz = len(frame_records) / config.duration_s
    capture_ms = [duration * 1000 for duration in capture_durations]
    geometry = parse_geometry(config.geometry)
    frame_width = config.output_width or geometry.width
    frame_height = config.output_height or geometry.height
    capture_resized = config.output_width is not None
    capture_side_preprocessing = (("resize",) if capture_resized else ()) + (
        ("grayscale",) if config.image_format == "gray8" else ()
    )

    return CaptureSessionSummary(
        run_id=config.run_id,
        attempt_id=config.attempt_id,
        output_dir=str(config.output_dir),
        geometry=config.geometry,
        event_device=config.event_device,
        duration_s=config.duration_s,
        target_hz=config.target_hz,
        frame_count=len(frame_records),
        effective_hz=effective_hz,
        input_event_count=len(input_events),
        terminal_event_count=terminal_event_count,
        press_release_count=press_release_count,
        frame_transition_count=len(frame_transitions),
        held_frame_count=held_frame_count,
        released_frame_count=released_frame_count,
        min_capture_duration_ms=min(capture_ms) if capture_ms else None,
        mean_capture_duration_ms=sum(capture_ms) / len(capture_ms)
        if capture_ms
        else None,
        max_capture_duration_ms=max(capture_ms) if capture_ms else None,
        dropped_frame_count=dropped_frame_count,
        missed_deadline_estimate=missed_deadline_estimate,
        capture_backend=capture_backend,
        image_format=config.image_format,
        frame_width=frame_width,
        frame_height=frame_height,
        source_width=geometry.width,
        source_height=geometry.height,
        capture_resized=capture_resized,
        capture_side_preprocessing=capture_side_preprocessing,
        preview_path=preview_path,
        preview_generated=preview_path is not None,
        preview_error=preview_error,
    )
