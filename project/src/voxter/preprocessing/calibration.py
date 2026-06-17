"""System-delay calibration from raw capture logs."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from voxter.capture.analysis import load_frame_records, load_input_events
from voxter.capture.events import RawInputEvent, validate_input_events
from voxter.capture.frames import FrameCaptureRecord, validate_frame_records
from voxter.contracts import ActionState, CaptureRecordError, aligned_label_index

DELTA_SYS_CALIBRATION_SCHEMA_VERSION = "delta-sys-calibration-v1"


@dataclass(frozen=True, slots=True)
class DeltaSysCandidate:
    """Score for one candidate system-delay offset."""

    delta_sys: int
    compared_sample_count: int
    mismatch_count: int
    mismatch_rate: float | None
    dropped_sample_count: int

    def to_json_dict(self) -> dict[str, object]:
        """Return a stable JSON-compatible representation."""

        return {
            "delta_sys": self.delta_sys,
            "compared_sample_count": self.compared_sample_count,
            "mismatch_count": self.mismatch_count,
            "mismatch_rate": self.mismatch_rate,
            "dropped_sample_count": self.dropped_sample_count,
        }


@dataclass(frozen=True, slots=True)
class DeltaSysCalibrationReport:
    """Calibration report for candidate `delta_sys` offsets."""

    schema_version: str
    capture_dir: str
    min_delta_sys: int
    max_delta_sys: int
    best_delta_sys: int | None
    candidates: tuple[DeltaSysCandidate, ...]

    def to_json_dict(self) -> dict[str, object]:
        """Return a stable JSON-compatible representation."""

        return {
            "schema_version": self.schema_version,
            "capture_dir": self.capture_dir,
            "min_delta_sys": self.min_delta_sys,
            "max_delta_sys": self.max_delta_sys,
            "best_delta_sys": self.best_delta_sys,
            "candidates": [candidate.to_json_dict() for candidate in self.candidates],
        }


def calibrate_delta_sys(
    capture_dir: Path,
    *,
    min_delta_sys: int = -5,
    max_delta_sys: int = 5,
) -> DeltaSysCalibrationReport:
    """Score candidate `delta_sys` values against event-reconstructed labels.

    The raw frame row's sampled action is compared with the event-derived held
    state at `label_t = action_log_{t + delta_sys}`. This is a coarse
    log-level calibration report; visual calibration still requires inspecting
    frame contents or an explicit visual marker.
    """

    if min_delta_sys > max_delta_sys:
        raise CaptureRecordError("min_delta_sys must be <= max_delta_sys")

    capture_dir = capture_dir.resolve()
    frames = load_frame_records(capture_dir / "frames.jsonl")
    events = load_input_events(capture_dir / "input_events.jsonl")
    validate_frame_records(frames)
    validate_input_events(events)

    candidates = tuple(
        _score_delta_sys(frames, events, delta_sys=delta_sys)
        for delta_sys in range(min_delta_sys, max_delta_sys + 1)
    )
    best_candidate = _best_candidate(candidates)
    return DeltaSysCalibrationReport(
        schema_version=DELTA_SYS_CALIBRATION_SCHEMA_VERSION,
        capture_dir=str(capture_dir),
        min_delta_sys=min_delta_sys,
        max_delta_sys=max_delta_sys,
        best_delta_sys=best_candidate.delta_sys if best_candidate else None,
        candidates=candidates,
    )


def write_delta_sys_calibration_report(
    capture_dir: Path,
    output_path: Path,
    *,
    min_delta_sys: int = -5,
    max_delta_sys: int = 5,
) -> DeltaSysCalibrationReport:
    """Write a JSON calibration report and return the in-memory report."""

    report = calibrate_delta_sys(
        capture_dir,
        min_delta_sys=min_delta_sys,
        max_delta_sys=max_delta_sys,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report.to_json_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _score_delta_sys(
    frames: list[FrameCaptureRecord],
    events: list[RawInputEvent],
    *,
    delta_sys: int,
) -> DeltaSysCandidate:
    events_by_group = _events_by_group(events)
    mismatch_count = 0
    compared_sample_count = 0
    dropped_sample_count = 0

    for group, group_frames in _frames_by_group(frames).items():
        frames_by_index = {frame.frame_index: frame for frame in group_frames}
        group_events = events_by_group.get(group, [])

        for frame in group_frames:
            label_frame = frames_by_index.get(
                aligned_label_index(frame.frame_index, delta_sys)
            )
            if label_frame is None:
                dropped_sample_count += 1
                continue
            compared_sample_count += 1
            expected_action = _state_at(
                group_events,
                label_frame.action_sample_timestamp,
            )
            if frame.action != expected_action:
                mismatch_count += 1

    mismatch_rate = (
        mismatch_count / compared_sample_count if compared_sample_count > 0 else None
    )
    return DeltaSysCandidate(
        delta_sys=delta_sys,
        compared_sample_count=compared_sample_count,
        mismatch_count=mismatch_count,
        mismatch_rate=mismatch_rate,
        dropped_sample_count=dropped_sample_count,
    )


def _best_candidate(
    candidates: Iterable[DeltaSysCandidate],
) -> DeltaSysCandidate | None:
    comparable = [
        candidate for candidate in candidates if candidate.mismatch_rate is not None
    ]
    if not comparable:
        return None
    return min(
        comparable,
        key=lambda candidate: (
            candidate.mismatch_count,
            candidate.dropped_sample_count,
            abs(candidate.delta_sys),
            candidate.delta_sys,
        ),
    )


def _frames_by_group(
    frames: Iterable[FrameCaptureRecord],
) -> dict[tuple[str, str | None], list[FrameCaptureRecord]]:
    grouped: defaultdict[tuple[str, str | None], list[FrameCaptureRecord]] = (
        defaultdict(list)
    )
    for frame in frames:
        grouped[(frame.run_id, frame.attempt_id)].append(frame)
    return {
        group: sorted(group_frames, key=lambda item: item.frame_index)
        for group, group_frames in grouped.items()
    }


def _events_by_group(
    events: Iterable[RawInputEvent],
) -> dict[tuple[str, str | None], list[RawInputEvent]]:
    grouped: defaultdict[tuple[str, str | None], list[RawInputEvent]] = defaultdict(
        list
    )
    for event in events:
        grouped[(event.run_id, event.attempt_id)].append(event)
    return {
        group: sorted(group_events, key=lambda item: item.timestamp)
        for group, group_events in grouped.items()
    }


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
