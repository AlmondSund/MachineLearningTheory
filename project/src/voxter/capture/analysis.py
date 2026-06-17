"""Analysis helpers for completed raw capture runs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from math import ceil
from pathlib import Path

from voxter.capture.events import (
    InputEventKind,
    RawInputEvent,
    RawTerminalEvent,
    validate_input_events,
    validate_terminal_events,
)
from voxter.capture.frames import (
    FrameCaptureRecord,
    parse_geometry,
    validate_frame_records,
)
from voxter.contracts import ActionState, CaptureRecordError, coerce_action_state


@dataclass(frozen=True, slots=True)
class CaptureRunAnalysis:
    """Validation and timing summary for one completed capture directory."""

    output_dir: str
    frame_count: int
    event_count: int
    frame_actions: tuple[int, ...]
    event_actions: tuple[int, ...]
    event_kinds: tuple[str, ...]
    frame_timestamps_monotonic: bool
    action_sample_timestamps_monotonic: bool
    event_timestamps_monotonic: bool
    frame_timestamp_after_action_sample: bool
    sync_mismatch_count_at_action_sample: int
    missing_frame_file_count: int
    interval_mean_ms: float | None
    interval_min_ms: float | None
    interval_max_ms: float | None
    interval_p95_ms: float | None
    interval_p99_ms: float | None
    target_hz: float | None
    effective_hz: float | None
    dropped_frame_count: int | None
    missed_deadline_estimate: int | None
    failures: tuple[str, ...]

    @property
    def passed(self) -> bool:
        """Return whether the analyzed run satisfies the requested thresholds."""

        return not self.failures

    def to_json_dict(self) -> dict[str, object]:
        """Return a stable JSON-compatible representation."""

        return {
            "output_dir": self.output_dir,
            "frame_count": self.frame_count,
            "event_count": self.event_count,
            "frame_actions": list(self.frame_actions),
            "event_actions": list(self.event_actions),
            "event_kinds": list(self.event_kinds),
            "frame_timestamps_monotonic": self.frame_timestamps_monotonic,
            "action_sample_timestamps_monotonic": (
                self.action_sample_timestamps_monotonic
            ),
            "event_timestamps_monotonic": self.event_timestamps_monotonic,
            "frame_timestamp_after_action_sample": (
                self.frame_timestamp_after_action_sample
            ),
            "sync_mismatch_count_at_action_sample": (
                self.sync_mismatch_count_at_action_sample
            ),
            "missing_frame_file_count": self.missing_frame_file_count,
            "interval_mean_ms": self.interval_mean_ms,
            "interval_min_ms": self.interval_min_ms,
            "interval_max_ms": self.interval_max_ms,
            "interval_p95_ms": self.interval_p95_ms,
            "interval_p99_ms": self.interval_p99_ms,
            "target_hz": self.target_hz,
            "effective_hz": self.effective_hz,
            "dropped_frame_count": self.dropped_frame_count,
            "missed_deadline_estimate": self.missed_deadline_estimate,
            "passed": self.passed,
            "failures": list(self.failures),
        }


def analyze_capture_run(
    output_dir: Path,
    *,
    min_input_events: int = 0,
    require_both_actions: bool = False,
    max_sync_mismatches: int = 0,
    max_dropped_frames: int = 0,
    max_missing_frame_files: int = 0,
    max_missed_periods: int | None = None,
    max_p95_interval_ms: float | None = None,
    max_p99_interval_ms: float | None = None,
) -> CaptureRunAnalysis:
    """Analyze a completed capture directory and return validation metrics."""

    output_dir = output_dir.resolve()
    frame_records = load_frame_records(output_dir / "frames.jsonl")
    input_events = load_input_events(output_dir / "input_events.jsonl")
    summary = _load_summary(output_dir / "capture_summary.json")

    validate_frame_records(frame_records)
    validate_input_events(input_events)

    frame_timestamps = [record.timestamp for record in frame_records]
    action_sample_timestamps = [
        record.action_sample_timestamp for record in frame_records
    ]
    event_timestamps = [event.timestamp for event in input_events]
    intervals_ms = [
        (current - previous) * 1000
        for previous, current in zip(
            frame_timestamps,
            frame_timestamps[1:],
            strict=False,
        )
    ]
    sync_mismatch_count = sum(
        1
        for record in frame_records
        if _state_at(input_events, record.action_sample_timestamp) != record.action
    )
    missing_frame_file_count = sum(
        1 for record in frame_records if not Path(record.frame_path).exists()
    )

    frame_actions = tuple(sorted({int(record.action) for record in frame_records}))
    event_actions = tuple(sorted({int(event.action) for event in input_events}))
    event_kinds = tuple(sorted({event.kind.value for event in input_events}))

    target_hz = _optional_float(summary.get("target_hz"))
    effective_hz = _optional_float(summary.get("effective_hz"))
    dropped_frame_count = _optional_int(summary.get("dropped_frame_count"))
    missed_deadline_estimate = _optional_int(summary.get("missed_deadline_estimate"))

    frame_timestamps_monotonic = _monotonic(frame_timestamps)
    action_sample_timestamps_monotonic = _monotonic(action_sample_timestamps)
    event_timestamps_monotonic = _monotonic(event_timestamps)
    frame_timestamp_after_action_sample = all(
        record.action_sample_timestamp <= record.timestamp for record in frame_records
    )

    failures = _analysis_failures(
        frame_actions=frame_actions,
        event_actions=event_actions,
        frame_timestamps_monotonic=frame_timestamps_monotonic,
        action_sample_timestamps_monotonic=action_sample_timestamps_monotonic,
        event_timestamps_monotonic=event_timestamps_monotonic,
        frame_timestamp_after_action_sample=frame_timestamp_after_action_sample,
        event_count=len(input_events),
        min_input_events=min_input_events,
        require_both_actions=require_both_actions,
        sync_mismatch_count=sync_mismatch_count,
        max_sync_mismatches=max_sync_mismatches,
        dropped_frame_count=dropped_frame_count,
        max_dropped_frames=max_dropped_frames,
        missing_frame_file_count=missing_frame_file_count,
        max_missing_frame_files=max_missing_frame_files,
        missed_deadline_estimate=missed_deadline_estimate,
        max_missed_periods=max_missed_periods,
        interval_p95_ms=_percentile(intervals_ms, 95),
        max_p95_interval_ms=max_p95_interval_ms,
        interval_p99_ms=_percentile(intervals_ms, 99),
        max_p99_interval_ms=max_p99_interval_ms,
    )

    return CaptureRunAnalysis(
        output_dir=str(output_dir),
        frame_count=len(frame_records),
        event_count=len(input_events),
        frame_actions=frame_actions,
        event_actions=event_actions,
        event_kinds=event_kinds,
        frame_timestamps_monotonic=frame_timestamps_monotonic,
        action_sample_timestamps_monotonic=action_sample_timestamps_monotonic,
        event_timestamps_monotonic=event_timestamps_monotonic,
        frame_timestamp_after_action_sample=frame_timestamp_after_action_sample,
        sync_mismatch_count_at_action_sample=sync_mismatch_count,
        missing_frame_file_count=missing_frame_file_count,
        interval_mean_ms=sum(intervals_ms) / len(intervals_ms)
        if intervals_ms
        else None,
        interval_min_ms=min(intervals_ms) if intervals_ms else None,
        interval_max_ms=max(intervals_ms) if intervals_ms else None,
        interval_p95_ms=_percentile(intervals_ms, 95),
        interval_p99_ms=_percentile(intervals_ms, 99),
        target_hz=target_hz,
        effective_hz=effective_hz,
        dropped_frame_count=dropped_frame_count,
        missed_deadline_estimate=missed_deadline_estimate,
        failures=tuple(failures),
    )


def load_frame_records(path: Path) -> list[FrameCaptureRecord]:
    """Load raw frame records from a capture `frames.jsonl` file."""

    rows = _load_jsonl(path)
    records: list[FrameCaptureRecord] = []
    for row in rows:
        geometry = _required_str(row, "geometry")
        source_geometry = parse_geometry(geometry)
        source_width_value = _optional_int(row.get("source_width"))
        source_height_value = _optional_int(row.get("source_height"))
        source_width = (
            source_width_value
            if source_width_value is not None
            else source_geometry.width
        )
        source_height = (
            source_height_value
            if source_height_value is not None
            else source_geometry.height
        )
        frame_width_value = _optional_int(row.get("frame_width"))
        frame_height_value = _optional_int(row.get("frame_height"))
        frame_width = (
            frame_width_value if frame_width_value is not None else source_width
        )
        frame_height = (
            frame_height_value if frame_height_value is not None else source_height
        )
        image_format = _optional_str(row.get("image_format"), "image_format")
        if image_format is None:
            image_format = _image_format_from_path(_required_str(row, "frame_path"))
        records.append(
            FrameCaptureRecord(
                run_id=_required_str(row, "run_id"),
                attempt_id=_optional_str(row.get("attempt_id"), "attempt_id"),
                frame_index=_required_int(row, "frame_index"),
                timestamp=_required_float(row, "timestamp"),
                frame_path=_required_str(row, "frame_path"),
                action=coerce_action_state(_required_int(row, "action")),
                action_sample_timestamp=_required_float(
                    row,
                    "action_sample_timestamp",
                ),
                geometry=geometry,
                capture_duration_s=_required_float(row, "capture_duration_s"),
                capture_backend=_required_str(row, "capture_backend"),
                image_format=image_format,
                frame_width=frame_width,
                frame_height=frame_height,
                source_width=source_width,
                source_height=source_height,
                capture_resized=_optional_bool(
                    row.get("capture_resized"),
                    default=(frame_width, frame_height)
                    != (source_width, source_height),
                ),
            )
        )
    return records


def load_input_events(path: Path) -> list[RawInputEvent]:
    """Load raw input events from a capture `input_events.jsonl` file."""

    events: list[RawInputEvent] = []
    for row in _load_jsonl(path):
        events.append(
            RawInputEvent(
                run_id=_required_str(row, "run_id"),
                attempt_id=_optional_str(row.get("attempt_id"), "attempt_id"),
                timestamp=_required_float(row, "timestamp"),
                device=_required_str(row, "device"),
                key_code=_required_int(row, "key_code"),
                kind=InputEventKind(_required_str(row, "kind")),
                action=coerce_action_state(_required_int(row, "action")),
            )
        )
    return events


def load_terminal_events(path: Path) -> list[RawTerminalEvent]:
    """Load terminal marker events from a capture `terminal_events.jsonl` file."""

    if not path.exists():
        return []
    events: list[RawTerminalEvent] = []
    for row in _load_jsonl(path):
        events.append(
            RawTerminalEvent(
                run_id=_required_str(row, "run_id"),
                attempt_id=_optional_str(row.get("attempt_id"), "attempt_id"),
                timestamp=_required_float(row, "timestamp"),
                device=_required_str(row, "device"),
                key_code=_required_int(row, "key_code"),
                kind=InputEventKind(_required_str(row, "kind")),
                terminal_type=_required_str(row, "terminal_type"),
            )
        )
    validate_terminal_events(events)
    return events


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        raise CaptureRecordError(f"missing JSONL file: {path}")
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise CaptureRecordError(f"{path}:{line_number + 1} must be an object")
        rows.append(value)
    return rows


def _load_summary(path: Path) -> dict[str, object]:
    if not path.exists():
        raise CaptureRecordError(f"missing capture summary: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CaptureRecordError("capture summary must be an object")
    return value


def _analysis_failures(
    *,
    frame_actions: tuple[int, ...],
    event_actions: tuple[int, ...],
    frame_timestamps_monotonic: bool,
    action_sample_timestamps_monotonic: bool,
    event_timestamps_monotonic: bool,
    frame_timestamp_after_action_sample: bool,
    event_count: int,
    min_input_events: int,
    require_both_actions: bool,
    sync_mismatch_count: int,
    max_sync_mismatches: int,
    dropped_frame_count: int | None,
    max_dropped_frames: int,
    missing_frame_file_count: int,
    max_missing_frame_files: int,
    missed_deadline_estimate: int | None,
    max_missed_periods: int | None,
    interval_p95_ms: float | None,
    max_p95_interval_ms: float | None,
    interval_p99_ms: float | None,
    max_p99_interval_ms: float | None,
) -> list[str]:
    failures: list[str] = []
    if event_count < min_input_events:
        failures.append(
            f"input events below threshold: {event_count} < {min_input_events}"
        )
    if require_both_actions and frame_actions != (0, 1):
        failures.append("frame actions must include both 0 and 1")
    if any(action not in {0, 1} for action in frame_actions):
        failures.append("frame actions must be binary 0/1")
    if any(action not in {0, 1} for action in event_actions):
        failures.append("input-event actions must be binary 0/1")
    if not frame_timestamps_monotonic:
        failures.append("frame timestamps must be monotonic")
    if not action_sample_timestamps_monotonic:
        failures.append("action sample timestamps must be monotonic")
    if not event_timestamps_monotonic:
        failures.append("event timestamps must be monotonic")
    if not frame_timestamp_after_action_sample:
        failures.append("frame timestamps must be >= action sample timestamps")
    if sync_mismatch_count > max_sync_mismatches:
        failures.append(
            "frame action mismatches exceed threshold: "
            f"{sync_mismatch_count} > {max_sync_mismatches}"
        )
    if dropped_frame_count is not None and dropped_frame_count > max_dropped_frames:
        failures.append(
            "dropped frames exceed threshold: "
            f"{dropped_frame_count} > {max_dropped_frames}"
        )
    if missing_frame_file_count > max_missing_frame_files:
        failures.append(
            "missing frame files exceed threshold: "
            f"{missing_frame_file_count} > {max_missing_frame_files}"
        )
    if (
        max_missed_periods is not None
        and missed_deadline_estimate is not None
        and missed_deadline_estimate > max_missed_periods
    ):
        failures.append(
            "missed periods exceed threshold: "
            f"{missed_deadline_estimate} > {max_missed_periods}"
        )
    if (
        max_p95_interval_ms is not None
        and interval_p95_ms is not None
        and interval_p95_ms > max_p95_interval_ms
    ):
        failures.append(
            "p95 frame interval exceeds threshold: "
            f"{interval_p95_ms:.3f}ms > {max_p95_interval_ms:.3f}ms"
        )
    if (
        max_p99_interval_ms is not None
        and interval_p99_ms is not None
        and interval_p99_ms > max_p99_interval_ms
    ):
        failures.append(
            "p99 frame interval exceeds threshold: "
            f"{interval_p99_ms:.3f}ms > {max_p99_interval_ms:.3f}ms"
        )
    return failures


def _state_at(
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


def _monotonic(values: list[float]) -> bool:
    return all(
        previous <= current
        for previous, current in zip(values, values[1:], strict=False)
    )


def _percentile(values: list[float], percentile: int) -> float | None:
    if not values:
        return None
    sorted_values = sorted(values)
    index = ceil((percentile / 100) * len(sorted_values)) - 1
    return sorted_values[max(0, min(index, len(sorted_values) - 1))]


def _required_str(row: dict[str, object], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str):
        raise CaptureRecordError(f"{key} must be a string")
    return value


def _optional_str(value: object, key: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise CaptureRecordError(f"{key} must be null or a string")
    return value


def _required_int(row: dict[str, object], key: str) -> int:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise CaptureRecordError(f"{key} must be an integer")
    return value


def _required_float(row: dict[str, object], key: str) -> float:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise CaptureRecordError(f"{key} must be numeric")
    return float(value)


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise CaptureRecordError("optional integer summary value must be an integer")
    return value


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise CaptureRecordError("optional numeric summary value must be numeric")
    return float(value)


def _optional_bool(value: object, *, default: bool) -> bool:
    if value is None:
        return default
    if not isinstance(value, bool):
        raise CaptureRecordError("optional boolean summary value must be boolean")
    return value


def _image_format_from_path(path: str) -> str:
    suffix = Path(path).suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "jpeg"
    if suffix == ".png":
        return "png"
    if suffix == ".ppm":
        return "ppm"
    if suffix == ".pgm":
        return "pgm"
    raise CaptureRecordError("cannot infer image_format from frame_path suffix")
