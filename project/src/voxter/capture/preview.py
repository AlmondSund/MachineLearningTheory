"""Preview video generation for completed raw capture sessions."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from voxter.capture.events import RawInputEvent, RawTerminalEvent
from voxter.capture.frames import FrameCaptureRecord
from voxter.contracts import ActionState, CaptureRecordError


class PreviewGenerationError(RuntimeError):
    """Raised when a capture preview video cannot be generated."""


@dataclass(frozen=True, slots=True)
class PreviewGenerationResult:
    """Result metadata for a generated capture preview."""

    preview_path: str
    subtitle_path: str
    frame_count: int
    fps: float
    rejected_window_count: int


def generate_capture_preview(
    output_dir: Path,
    frame_records: list[FrameCaptureRecord],
    input_events: list[RawInputEvent],
    terminal_events: list[RawTerminalEvent] | None = None,
    *,
    fps: float,
    preview_name: str = "preview.mp4",
    subtitle_name: str = "preview_actions.srt",
    rejected_tail_s: float = 0.35,
    rejected_skip_s: float = 1.5,
) -> PreviewGenerationResult:
    """Generate an MP4 preview with burned-in action-state subtitles."""

    if fps <= 0:
        raise CaptureRecordError("preview fps must be positive")
    if not frame_records:
        raise CaptureRecordError("at least one frame is required for preview")
    if not preview_name or Path(preview_name).name != preview_name:
        raise CaptureRecordError("preview_name must be a plain file name")
    if not subtitle_name or Path(subtitle_name).name != subtitle_name:
        raise CaptureRecordError("subtitle_name must be a plain file name")
    if rejected_tail_s < 0:
        raise CaptureRecordError("rejected_tail_s must be non-negative")
    if rejected_skip_s < 0:
        raise CaptureRecordError("rejected_skip_s must be non-negative")

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise PreviewGenerationError("ffmpeg is required to generate preview.mp4")

    sorted_frames = sorted(frame_records, key=lambda frame: frame.frame_index)
    _validate_preview_frame_sequence(sorted_frames)
    frame_pattern = _frame_pattern(sorted_frames[0])
    output_dir = output_dir.resolve()
    preview_path = output_dir / preview_name
    subtitle_path = output_dir / subtitle_name

    subtitle_path.write_text(
        _build_action_subtitles(
            sorted_frames,
            input_events,
            terminal_events or [],
            fps=fps,
        ),
        encoding="utf-8",
    )
    rejected_windows = _rejected_preview_windows(
        sorted_frames,
        terminal_events or [],
        fps=fps,
        rejected_tail_s=rejected_tail_s,
        rejected_skip_s=rejected_skip_s,
    )
    command = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-framerate",
        _format_fps(fps),
        "-i",
        str(frame_pattern),
        "-vf",
        _preview_filter(subtitle_path, rejected_windows),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(preview_path),
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        stderr = result.stderr.strip() or "ffmpeg exited without stderr"
        raise PreviewGenerationError(f"ffmpeg preview generation failed: {stderr}")
    if not preview_path.exists() or preview_path.stat().st_size <= 0:
        raise PreviewGenerationError("ffmpeg did not create a non-empty preview.mp4")
    return PreviewGenerationResult(
        preview_path=str(preview_path),
        subtitle_path=str(subtitle_path),
        frame_count=len(sorted_frames),
        fps=fps,
        rejected_window_count=len(rejected_windows),
    )


def _validate_preview_frame_sequence(frames: list[FrameCaptureRecord]) -> None:
    first_index = frames[0].frame_index
    for expected_index, frame in enumerate(frames, start=first_index):
        if frame.frame_index != expected_index:
            raise CaptureRecordError(
                "preview generation requires contiguous frame indexes"
            )
        if not Path(frame.frame_path).exists():
            raise CaptureRecordError(f"preview frame is missing: {frame.frame_path}")


def _frame_pattern(frame: FrameCaptureRecord) -> Path:
    frame_path = Path(frame.frame_path)
    suffix = frame_path.suffix
    stem = frame_path.stem
    if not stem.isdigit():
        raise CaptureRecordError("preview generation requires numeric frame file names")
    return frame_path.with_name(f"%0{len(stem)}d{suffix}")


def _build_action_subtitles(
    frames: list[FrameCaptureRecord],
    input_events: list[RawInputEvent],
    terminal_events: list[RawTerminalEvent],
    *,
    fps: float,
) -> str:
    events_by_frame = _events_by_frame(frames, input_events)
    terminal_events_by_frame = _terminal_events_by_frame(frames, terminal_events)
    entries: list[str] = []
    frame_duration = 1.0 / fps
    for index, frame in enumerate(frames, start=1):
        start_s = (index - 1) * frame_duration
        end_s = index * frame_duration
        action_text = "HELD" if frame.action is ActionState.HELD else "RELEASED"
        event_text = events_by_frame.get(frame.frame_index, "")
        terminal_text = terminal_events_by_frame.get(frame.frame_index, "")
        text = f"W ACTION: {action_text}"
        if event_text:
            text += f" | EVENT: {event_text}"
        if terminal_text:
            text += f" | TERMINAL: {terminal_text}"
        entries.append(
            "\n".join(
                [
                    str(index),
                    f"{_format_srt_time(start_s)} --> {_format_srt_time(end_s)}",
                    text,
                    "",
                ]
            )
        )
    return "\n".join(entries)


def _events_by_frame(
    frames: list[FrameCaptureRecord],
    input_events: list[RawInputEvent],
) -> dict[int, str]:
    event_text_by_frame: dict[int, str] = {}
    event_index = 0
    sorted_events = sorted(input_events, key=lambda event: event.timestamp)
    previous_frame_timestamp = float("-inf")
    for frame in frames:
        event_labels: list[str] = []
        while (
            event_index < len(sorted_events)
            and sorted_events[event_index].timestamp <= frame.timestamp
        ):
            event = sorted_events[event_index]
            if event.timestamp > previous_frame_timestamp:
                event_labels.append(event.kind.value.upper())
            event_index += 1
        if event_labels:
            event_text_by_frame[frame.frame_index] = ",".join(event_labels)
        previous_frame_timestamp = frame.timestamp
    return event_text_by_frame


def _terminal_events_by_frame(
    frames: list[FrameCaptureRecord],
    terminal_events: list[RawTerminalEvent],
) -> dict[int, str]:
    event_text_by_frame: dict[int, str] = {}
    event_index = 0
    sorted_events = sorted(terminal_events, key=lambda event: event.timestamp)
    previous_frame_timestamp = float("-inf")
    for frame in frames:
        event_labels: list[str] = []
        while (
            event_index < len(sorted_events)
            and sorted_events[event_index].timestamp <= frame.timestamp
        ):
            event = sorted_events[event_index]
            if event.timestamp > previous_frame_timestamp:
                event_labels.append(event.terminal_type.upper())
            event_index += 1
        if event_labels:
            event_text_by_frame[frame.frame_index] = ",".join(event_labels)
        previous_frame_timestamp = frame.timestamp
    return event_text_by_frame


def _rejected_preview_windows(
    frames: list[FrameCaptureRecord],
    terminal_events: list[RawTerminalEvent],
    *,
    fps: float,
    rejected_tail_s: float,
    rejected_skip_s: float,
) -> list[tuple[float, float]]:
    windows: list[tuple[float, float]] = []
    frame_duration = 1.0 / fps
    for terminal_event in terminal_events:
        if terminal_event.terminal_type == "active_start":
            continue
        first_index: int | None = None
        last_index: int | None = None
        start_timestamp = terminal_event.timestamp - rejected_tail_s
        end_timestamp = terminal_event.timestamp + rejected_skip_s
        for index, frame in enumerate(frames):
            if start_timestamp <= frame.timestamp <= end_timestamp:
                if first_index is None:
                    first_index = index
                last_index = index
        if first_index is None or last_index is None:
            continue
        windows.append(
            (
                first_index * frame_duration,
                (last_index + 1) * frame_duration,
            )
        )
    return _merge_windows(windows)


def _merge_windows(windows: list[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for start_s, end_s in sorted(windows):
        if start_s > end_s:
            continue
        if not merged or start_s > merged[-1][1]:
            merged.append((start_s, end_s))
        else:
            previous_start, previous_end = merged[-1]
            merged[-1] = (previous_start, max(previous_end, end_s))
    return merged


def _preview_filter(
    subtitle_path: Path,
    rejected_windows: list[tuple[float, float]],
) -> str:
    filters: list[str] = []
    if rejected_windows:
        filters.append(
            "drawbox="
            "x=0:y=0:w=iw:h=ih:"
            "color=red@0.32:t=fill:"
            f"enable='{_rejected_enable_expression(rejected_windows)}'"
        )
    filters.append(f"subtitles={_escape_filter_path(subtitle_path)}")
    return ",".join(filters)


def _rejected_enable_expression(windows: list[tuple[float, float]]) -> str:
    return "+".join(
        f"between(t,{_format_filter_float(start_s)},{_format_filter_float(end_s)})"
        for start_s, end_s in windows
    )


def _format_filter_float(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _format_srt_time(seconds: float) -> str:
    total_ms = max(0, round(seconds * 1000))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d},{milliseconds:03d}"


def _format_fps(fps: float) -> str:
    return f"{fps:.6f}".rstrip("0").rstrip(".")


def _escape_filter_path(path: Path) -> str:
    text = str(path)
    return text.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")
