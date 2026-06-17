"""Build causally aligned manifests from raw capture logs."""

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

ALIGNED_MANIFEST_SCHEMA_VERSION = "aligned-manifest-v1"


@dataclass(frozen=True, slots=True)
class AlignedManifestRow:
    """One processed manifest row with a causally aligned held-state label."""

    schema_version: str
    sample_id: str
    run_id: str
    attempt_id: str | None
    frame_index: int
    timestamp: float
    raw_frame_path: str
    action_held: ActionState
    label_source_timestamp: float
    delta_sys: int
    source_width: int
    source_height: int
    frame_width: int
    frame_height: int
    image_format: str
    capture_resized: bool
    split: str

    def to_json_dict(self) -> dict[str, object]:
        """Return a stable JSON-compatible representation."""

        return {
            "schema_version": self.schema_version,
            "sample_id": self.sample_id,
            "run_id": self.run_id,
            "attempt_id": self.attempt_id,
            "frame_index": self.frame_index,
            "timestamp": self.timestamp,
            "raw_frame_path": self.raw_frame_path,
            "action_held": int(self.action_held),
            "label_source_timestamp": self.label_source_timestamp,
            "delta_sys": self.delta_sys,
            "source_width": self.source_width,
            "source_height": self.source_height,
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
            "image_format": self.image_format,
            "capture_resized": self.capture_resized,
            "split": self.split,
        }


def build_aligned_manifest(
    capture_dir: Path,
    *,
    delta_sys: int = 0,
    split: str = "unsplit",
) -> list[AlignedManifestRow]:
    """Return aligned manifest rows for one raw capture directory.

    `delta_sys` uses the project sign convention: positive values take the label
    from a later frame index in the same run/attempt.
    """

    if not split:
        raise CaptureRecordError("split must be a non-empty string")

    capture_dir = capture_dir.resolve()
    frame_records = load_frame_records(capture_dir / "frames.jsonl")
    input_events = load_input_events(capture_dir / "input_events.jsonl")
    validate_frame_records(frame_records)
    validate_input_events(input_events)

    events_by_group = _events_by_group(input_events)
    rows: list[AlignedManifestRow] = []

    for group, group_frames in _frames_by_group(frame_records).items():
        frames_by_index = {frame.frame_index: frame for frame in group_frames}
        group_events = events_by_group.get(group, [])

        for frame in sorted(group_frames, key=lambda item: item.frame_index):
            label_index = aligned_label_index(frame.frame_index, delta_sys)
            label_frame = frames_by_index.get(label_index)
            if label_frame is None:
                continue
            rows.append(
                AlignedManifestRow(
                    schema_version=ALIGNED_MANIFEST_SCHEMA_VERSION,
                    sample_id=_sample_id(frame, delta_sys=delta_sys),
                    run_id=frame.run_id,
                    attempt_id=frame.attempt_id,
                    frame_index=frame.frame_index,
                    timestamp=frame.timestamp,
                    raw_frame_path=frame.frame_path,
                    action_held=_state_at(
                        group_events,
                        label_frame.action_sample_timestamp,
                    ),
                    label_source_timestamp=label_frame.action_sample_timestamp,
                    delta_sys=delta_sys,
                    source_width=frame.source_width,
                    source_height=frame.source_height,
                    frame_width=frame.frame_width,
                    frame_height=frame.frame_height,
                    image_format=frame.image_format,
                    capture_resized=frame.capture_resized,
                    split=split,
                )
            )

    return rows


def write_aligned_manifest(
    capture_dir: Path,
    output_dir: Path,
    *,
    delta_sys: int = 0,
    split: str = "unsplit",
    manifest_name: str = "aligned_manifest.jsonl",
) -> Path:
    """Build and write an aligned manifest, returning the output path."""

    if not manifest_name or Path(manifest_name).name != manifest_name:
        raise CaptureRecordError("manifest_name must be a plain file name")

    rows = build_aligned_manifest(capture_dir, delta_sys=delta_sys, split=split)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / manifest_name
    with manifest_path.open("w", encoding="utf-8") as manifest_file:
        for row in rows:
            manifest_file.write(json.dumps(row.to_json_dict(), sort_keys=True))
            manifest_file.write("\n")
    return manifest_path


def _frames_by_group(
    frames: Iterable[FrameCaptureRecord],
) -> dict[tuple[str, str | None], list[FrameCaptureRecord]]:
    grouped: defaultdict[tuple[str, str | None], list[FrameCaptureRecord]] = (
        defaultdict(list)
    )
    for frame in frames:
        grouped[(frame.run_id, frame.attempt_id)].append(frame)
    return dict(grouped)


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


def _sample_id(frame: FrameCaptureRecord, *, delta_sys: int) -> str:
    attempt = frame.attempt_id if frame.attempt_id is not None else "none"
    return f"{frame.run_id}:{attempt}:{frame.frame_index:06d}:delta{delta_sys:+d}"
