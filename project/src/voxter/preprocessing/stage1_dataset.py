"""Materialize Stage 1 behavior-cloning datasets from raw captures."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from voxter.capture.analysis import load_terminal_events
from voxter.capture.events import RawTerminalEvent
from voxter.contracts import ActionState, CaptureRecordError
from voxter.preprocessing.alignment import AlignedManifestRow, build_aligned_manifest
from voxter.preprocessing.observation import (
    OBSERVATION_SCHEMA_VERSION,
    ObservationConfig,
    preprocess_grayscale_observation,
)
from voxter.preprocessing.stack import (
    FRAME_STACK_SCHEMA_VERSION,
    FrameStackConfig,
    RollingFrameStack,
)

STAGE1_MANIFEST_SCHEMA_VERSION = "stage1-manifest-v1"
STAGE1_DATASET_SUMMARY_SCHEMA_VERSION = "stage1-dataset-summary-v1"


@dataclass(frozen=True, slots=True)
class PgmImage:
    """One binary PGM P5 grayscale image loaded from disk."""

    width: int
    height: int
    data: bytes


@dataclass(frozen=True, slots=True)
class Stage1DatasetConfig:
    """Configuration for one Stage 1 offline dataset materialization run."""

    capture_dir: Path
    output_dir: Path
    observation_width: int
    observation_height: int
    frame_stack_length: int = 4
    delta_sys: int = 0
    split: str = "unsplit"
    manifest_name: str = "stage1_manifest.jsonl"
    summary_name: str = "dataset_summary.json"
    death_tail_s: float = 0.35
    reset_skip_s: float = 1.5

    def __post_init__(self) -> None:
        if self.observation_width <= 0:
            raise CaptureRecordError("observation_width must be positive")
        if self.observation_height <= 0:
            raise CaptureRecordError("observation_height must be positive")
        if self.frame_stack_length <= 0:
            raise CaptureRecordError("frame_stack_length must be positive")
        if not self.split:
            raise CaptureRecordError("split must be a non-empty string")
        if (
            not self.manifest_name
            or Path(self.manifest_name).name != self.manifest_name
        ):
            raise CaptureRecordError("manifest_name must be a plain file name")
        if not self.summary_name or Path(self.summary_name).name != self.summary_name:
            raise CaptureRecordError("summary_name must be a plain file name")
        if self.death_tail_s < 0:
            raise CaptureRecordError("death_tail_s must be non-negative")
        if self.reset_skip_s < 0:
            raise CaptureRecordError("reset_skip_s must be non-negative")


@dataclass(frozen=True, slots=True)
class Stage1ManifestRow:
    """One materialized Stage 1 sample row."""

    schema_version: str
    sample_id: str
    run_id: str
    attempt_id: str | None
    frame_index: int
    timestamp: float
    raw_frame_path: str
    observation_path: str
    frame_stack_path: str
    action_held: ActionState
    label_source_timestamp: float
    delta_sys: int
    split: str
    observation_schema_version: str
    observation_width: int
    observation_height: int
    observation_dtype: str
    observation_layout: str
    frame_stack_schema_version: str
    frame_stack_length: int
    frame_stack_layout: str
    terminal_cleaning_applied: bool

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
            "observation_path": self.observation_path,
            "frame_stack_path": self.frame_stack_path,
            "action_held": int(self.action_held),
            "label_source_timestamp": self.label_source_timestamp,
            "delta_sys": self.delta_sys,
            "split": self.split,
            "observation_schema_version": self.observation_schema_version,
            "observation_width": self.observation_width,
            "observation_height": self.observation_height,
            "observation_dtype": self.observation_dtype,
            "observation_layout": self.observation_layout,
            "frame_stack_schema_version": self.frame_stack_schema_version,
            "frame_stack_length": self.frame_stack_length,
            "frame_stack_layout": self.frame_stack_layout,
            "terminal_cleaning_applied": self.terminal_cleaning_applied,
        }


@dataclass(frozen=True, slots=True)
class Stage1DatasetSummary:
    """Summary metadata for a materialized Stage 1 dataset."""

    schema_version: str
    source_capture_dir: str
    output_dir: str
    manifest_path: str
    sample_count: int
    released_count: int
    held_count: int
    class_weights: dict[str, float | None]
    delta_sys: int
    split: str
    observation_schema_version: str
    observation_width: int
    observation_height: int
    observation_dtype: str
    observation_layout: str
    frame_stack_schema_version: str
    frame_stack_length: int
    frame_stack_layout: str
    terminal_event_count: int
    discarded_terminal_window_count: int
    death_tail_s: float
    reset_skip_s: float

    def to_json_dict(self) -> dict[str, object]:
        """Return a stable JSON-compatible representation."""

        return {
            "schema_version": self.schema_version,
            "source_capture_dir": self.source_capture_dir,
            "output_dir": self.output_dir,
            "manifest_path": self.manifest_path,
            "sample_count": self.sample_count,
            "released_count": self.released_count,
            "held_count": self.held_count,
            "class_weights": self.class_weights,
            "delta_sys": self.delta_sys,
            "split": self.split,
            "observation_schema_version": self.observation_schema_version,
            "observation_width": self.observation_width,
            "observation_height": self.observation_height,
            "observation_dtype": self.observation_dtype,
            "observation_layout": self.observation_layout,
            "frame_stack_schema_version": self.frame_stack_schema_version,
            "frame_stack_length": self.frame_stack_length,
            "frame_stack_layout": self.frame_stack_layout,
            "terminal_event_count": self.terminal_event_count,
            "discarded_terminal_window_count": self.discarded_terminal_window_count,
            "death_tail_s": self.death_tail_s,
            "reset_skip_s": self.reset_skip_s,
        }


def build_stage1_dataset(config: Stage1DatasetConfig) -> Stage1DatasetSummary:
    """Write Stage 1 observations, frame stacks, manifest, and summary."""

    capture_dir = config.capture_dir.resolve()
    output_dir = config.output_dir.resolve()
    observation_dir = output_dir / "observations"
    stack_dir = output_dir / "stacks"
    output_dir.mkdir(parents=True, exist_ok=True)
    observation_dir.mkdir(parents=True, exist_ok=True)
    stack_dir.mkdir(parents=True, exist_ok=True)

    observation_config = ObservationConfig(
        width=config.observation_width,
        height=config.observation_height,
    )
    stack_config = FrameStackConfig(
        length=config.frame_stack_length,
        width=config.observation_width,
        height=config.observation_height,
    )

    aligned_rows = build_aligned_manifest(
        capture_dir,
        delta_sys=config.delta_sys,
        split=config.split,
    )
    terminal_events = load_terminal_events(capture_dir / "terminal_events.jsonl")
    terminal_windows = _terminal_discard_windows(
        terminal_events,
        death_tail_s=config.death_tail_s,
        reset_skip_s=config.reset_skip_s,
    )
    manifest_rows: list[Stage1ManifestRow] = []
    for group_rows in _rows_by_group(aligned_rows).values():
        stacker = RollingFrameStack(stack_config)
        for aligned_row in sorted(group_rows, key=lambda item: item.frame_index):
            if _is_discarded_by_terminal_window(aligned_row, terminal_windows):
                stacker = RollingFrameStack(stack_config)
                continue
            if aligned_row.image_format != "pgm":
                raise CaptureRecordError(
                    "Stage 1 materialization currently supports PGM/gray8 frames "
                    f"only; got {aligned_row.image_format!r} for "
                    f"{aligned_row.raw_frame_path}"
                )
            raw_frame_path = _resolve_raw_frame_path(
                capture_dir,
                aligned_row.raw_frame_path,
            )
            pgm = load_pgm_image(raw_frame_path)
            observation = preprocess_grayscale_observation(
                pgm.data,
                source_width=pgm.width,
                source_height=pgm.height,
                config=observation_config,
            )
            frame_stack = stacker.update(observation)

            file_stem = _safe_file_stem(aligned_row.sample_id)
            observation_path = observation_dir / f"{file_stem}.gray"
            stack_path = stack_dir / f"{file_stem}.bin"
            observation_path.write_bytes(observation.data)
            stack_path.write_bytes(frame_stack.data)

            manifest_rows.append(
                Stage1ManifestRow(
                    schema_version=STAGE1_MANIFEST_SCHEMA_VERSION,
                    sample_id=aligned_row.sample_id,
                    run_id=aligned_row.run_id,
                    attempt_id=aligned_row.attempt_id,
                    frame_index=aligned_row.frame_index,
                    timestamp=aligned_row.timestamp,
                    raw_frame_path=str(raw_frame_path),
                    observation_path=str(observation_path),
                    frame_stack_path=str(stack_path),
                    action_held=aligned_row.action_held,
                    label_source_timestamp=aligned_row.label_source_timestamp,
                    delta_sys=aligned_row.delta_sys,
                    split=aligned_row.split,
                    observation_schema_version=OBSERVATION_SCHEMA_VERSION,
                    observation_width=observation.width,
                    observation_height=observation.height,
                    observation_dtype=observation.dtype,
                    observation_layout=observation.layout,
                    frame_stack_schema_version=FRAME_STACK_SCHEMA_VERSION,
                    frame_stack_length=frame_stack.length,
                    frame_stack_layout=frame_stack.layout,
                    terminal_cleaning_applied=bool(terminal_events),
                )
            )

    manifest_path = output_dir / config.manifest_name
    _write_manifest(manifest_path, manifest_rows)
    summary = _build_summary(
        config,
        output_dir=output_dir,
        manifest_path=manifest_path,
        rows=manifest_rows,
        observation_config=observation_config,
        stack_config=stack_config,
        terminal_event_count=len(terminal_events),
        discarded_terminal_window_count=sum(
            len(windows) for windows in terminal_windows.values()
        ),
    )
    summary_path = output_dir / config.summary_name
    summary_path.write_text(
        json.dumps(summary.to_json_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def load_pgm_image(path: Path) -> PgmImage:
    """Load a binary PGM P5 image from disk."""

    data = path.read_bytes()
    tokens: list[bytes] = []
    index = 0
    for _ in range(4):
        token, index = _read_pgm_token(data, index)
        tokens.append(token)

    if tokens[0] != b"P5":
        raise CaptureRecordError("PGM image must use binary P5 format")
    try:
        width = int(tokens[1])
        height = int(tokens[2])
        max_value = int(tokens[3])
    except ValueError as exc:
        raise CaptureRecordError("PGM header dimensions must be integers") from exc

    if width <= 0 or height <= 0:
        raise CaptureRecordError("PGM dimensions must be positive")
    if max_value != 255:
        raise CaptureRecordError("PGM max value must be 255 for uint8 pixels")
    if index >= len(data) or data[index] not in b" \t\r\n":
        raise CaptureRecordError("PGM header must end with whitespace")

    payload_start = index + 1
    expected_size = width * height
    payload = data[payload_start : payload_start + expected_size]
    if len(payload) != expected_size:
        raise CaptureRecordError("PGM payload is smaller than width*height")
    return PgmImage(width=width, height=height, data=payload)


def _read_pgm_token(data: bytes, index: int) -> tuple[bytes, int]:
    while True:
        while index < len(data) and data[index] in b" \t\r\n":
            index += 1
        if index < len(data) and data[index] == ord("#"):
            while index < len(data) and data[index] not in b"\r\n":
                index += 1
            continue
        break

    if index >= len(data):
        raise CaptureRecordError("PGM header is incomplete")
    start = index
    while index < len(data) and data[index] not in b" \t\r\n":
        index += 1
    return data[start:index], index


def _rows_by_group(
    rows: Iterable[AlignedManifestRow],
) -> dict[tuple[str, str | None], list[AlignedManifestRow]]:
    grouped: defaultdict[tuple[str, str | None], list[AlignedManifestRow]] = (
        defaultdict(list)
    )
    for row in rows:
        grouped[(row.run_id, row.attempt_id)].append(row)
    return dict(grouped)


def _terminal_discard_windows(
    terminal_events: list[RawTerminalEvent],
    *,
    death_tail_s: float,
    reset_skip_s: float,
) -> dict[tuple[str, str | None], list[tuple[float, float]]]:
    grouped: defaultdict[tuple[str, str | None], list[tuple[float, float]]] = (
        defaultdict(list)
    )
    for event in terminal_events:
        grouped[(event.run_id, event.attempt_id)].append(
            (event.timestamp - death_tail_s, event.timestamp + reset_skip_s)
        )
    return {
        group: sorted(windows, key=lambda window: window[0])
        for group, windows in grouped.items()
    }


def _is_discarded_by_terminal_window(
    row: AlignedManifestRow,
    windows_by_group: dict[tuple[str, str | None], list[tuple[float, float]]],
) -> bool:
    for start_s, end_s in windows_by_group.get((row.run_id, row.attempt_id), []):
        if start_s <= row.timestamp <= end_s:
            return True
    return False


def _resolve_raw_frame_path(capture_dir: Path, raw_frame_path: str) -> Path:
    path = Path(raw_frame_path)
    if path.is_absolute():
        return path
    if path.exists():
        return path.resolve()
    return capture_dir / path


def _safe_file_stem(sample_id: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", sample_id).strip("._")
    if not stem:
        raise CaptureRecordError("sample_id cannot be converted to a file name")
    return stem


def _write_manifest(path: Path, rows: Iterable[Stage1ManifestRow]) -> None:
    with path.open("w", encoding="utf-8") as manifest_file:
        for row in rows:
            manifest_file.write(json.dumps(row.to_json_dict(), sort_keys=True))
            manifest_file.write("\n")


def _build_summary(
    config: Stage1DatasetConfig,
    *,
    output_dir: Path,
    manifest_path: Path,
    rows: list[Stage1ManifestRow],
    observation_config: ObservationConfig,
    stack_config: FrameStackConfig,
    terminal_event_count: int,
    discarded_terminal_window_count: int,
) -> Stage1DatasetSummary:
    released_count = sum(1 for row in rows if row.action_held == ActionState.RELEASED)
    held_count = sum(1 for row in rows if row.action_held == ActionState.HELD)
    return Stage1DatasetSummary(
        schema_version=STAGE1_DATASET_SUMMARY_SCHEMA_VERSION,
        source_capture_dir=str(config.capture_dir.resolve()),
        output_dir=str(output_dir),
        manifest_path=str(manifest_path),
        sample_count=len(rows),
        released_count=released_count,
        held_count=held_count,
        class_weights=_class_weights(
            released_count=released_count,
            held_count=held_count,
        ),
        delta_sys=config.delta_sys,
        split=config.split,
        observation_schema_version=OBSERVATION_SCHEMA_VERSION,
        observation_width=observation_config.width,
        observation_height=observation_config.height,
        observation_dtype=observation_config.dtype,
        observation_layout=observation_config.layout,
        frame_stack_schema_version=FRAME_STACK_SCHEMA_VERSION,
        frame_stack_length=stack_config.length,
        frame_stack_layout=stack_config.layout,
        terminal_event_count=terminal_event_count,
        discarded_terminal_window_count=discarded_terminal_window_count,
        death_tail_s=config.death_tail_s,
        reset_skip_s=config.reset_skip_s,
    )


def _class_weights(
    *,
    released_count: int,
    held_count: int,
) -> dict[str, float | None]:
    total = released_count + held_count
    return {
        "released": _binary_class_weight(total, released_count),
        "held": _binary_class_weight(total, held_count),
    }


def _binary_class_weight(total: int, class_count: int) -> float | None:
    if total == 0 or class_count == 0:
        return None
    return total / (2.0 * class_count)
