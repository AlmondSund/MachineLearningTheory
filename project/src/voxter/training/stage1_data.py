"""Dependency-free Stage 1 dataset loading smoke contracts."""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from voxter.contracts import CaptureRecordError
from voxter.preprocessing import (
    STAGE1_DATASET_SUMMARY_SCHEMA_VERSION,
    STAGE1_MANIFEST_SCHEMA_VERSION,
)

STAGE1_DATA_SMOKE_SCHEMA_VERSION = "stage1-data-smoke-v1"


@dataclass(frozen=True, slots=True)
class Stage1SampleRef:
    """Reference to one Stage 1 materialized sample payload."""

    sample_id: str
    run_id: str
    dataset_dir: str
    frame_index: int
    frame_stack_path: str
    observation_path: str
    action_held: int


@dataclass(frozen=True, slots=True)
class Stage1DatasetIndex:
    """In-memory index of Stage 1 sample references without payload bytes."""

    dataset_dirs: tuple[str, ...]
    samples: tuple[Stage1SampleRef, ...]
    sample_count: int
    held_count: int
    released_count: int
    observation_width: int
    observation_height: int
    observation_dtype: str
    frame_stack_length: int
    frame_stack_layout: str
    split: str
    delta_sys: int

    @property
    def expected_observation_bytes(self) -> int:
        """Return the expected byte count for one grayscale observation."""

        return self.observation_width * self.observation_height

    @property
    def expected_stack_bytes(self) -> int:
        """Return the expected byte count for one frame-stack payload."""

        return self.expected_observation_bytes * self.frame_stack_length

    @property
    def frame_stack_shape(self) -> tuple[int, int, int]:
        """Return the per-sample stack shape as `(K, H, W)`."""

        return (
            self.frame_stack_length,
            self.observation_height,
            self.observation_width,
        )


@dataclass(frozen=True, slots=True)
class _Stage1DatasetContract:
    observation_width: int
    observation_height: int
    observation_dtype: str
    observation_layout: str
    frame_stack_length: int
    frame_stack_layout: str
    split: str
    delta_sys: int


@dataclass(frozen=True, slots=True)
class Stage1Batch:
    """One dependency-free Stage 1 batch of frame-stack payload bytes."""

    samples: tuple[Stage1SampleRef, ...]
    frame_stacks: tuple[bytes, ...]
    labels: tuple[int, ...]
    shape: tuple[int, int, int, int]
    dtype: str
    layout: str

    @property
    def batch_size(self) -> int:
        """Return the number of samples in the batch."""

        return len(self.samples)


@dataclass(frozen=True, slots=True)
class Stage1DataSmokeReport:
    """Summary of a Stage 1 data-loading smoke run."""

    schema_version: str
    dataset_count: int
    sample_count: int
    held_count: int
    released_count: int
    checked_batch_count: int
    checked_sample_count: int
    batch_size: int
    frame_stack_shape: tuple[int, int, int]
    frame_stack_layout: str
    observation_dtype: str
    failures: tuple[str, ...]

    @property
    def passed(self) -> bool:
        """Return whether the smoke run found no failures."""

        return not self.failures

    def to_json_dict(self) -> dict[str, object]:
        """Return a stable JSON-compatible representation."""

        return {
            "schema_version": self.schema_version,
            "dataset_count": self.dataset_count,
            "sample_count": self.sample_count,
            "held_count": self.held_count,
            "released_count": self.released_count,
            "checked_batch_count": self.checked_batch_count,
            "checked_sample_count": self.checked_sample_count,
            "batch_size": self.batch_size,
            "frame_stack_shape": list(self.frame_stack_shape),
            "frame_stack_layout": self.frame_stack_layout,
            "observation_dtype": self.observation_dtype,
            "passed": self.passed,
            "failures": list(self.failures),
        }


def load_stage1_dataset_index(dataset_dirs: Sequence[Path]) -> Stage1DatasetIndex:
    """Load Stage 1 dataset manifests into a payload-light sample index."""

    if not dataset_dirs:
        raise CaptureRecordError("at least one Stage 1 dataset directory is required")

    resolved_dirs = tuple(path.resolve() for path in dataset_dirs)
    samples: list[Stage1SampleRef] = []
    held_count = 0
    released_count = 0
    expected_contract: _Stage1DatasetContract | None = None

    for dataset_dir in resolved_dirs:
        summary = _load_json_object(dataset_dir / "dataset_summary.json")
        _validate_summary_contract(summary, dataset_dir=dataset_dir)
        contract = _summary_contract(summary)
        if expected_contract is None:
            expected_contract = contract
        elif contract != expected_contract:
            raise CaptureRecordError(
                "Stage 1 dataset contracts must match across directories"
            )

        manifest_path = _manifest_path(summary, dataset_dir=dataset_dir)
        rows = _load_jsonl(manifest_path)
        if len(rows) != _required_int(summary, "sample_count"):
            raise CaptureRecordError(
                "manifest row count does not match summary sample_count"
            )

        for row in rows:
            _validate_manifest_row(row, dataset_dir=dataset_dir, summary=summary)
            action_held = _required_int(row, "action_held")
            held_count += 1 if action_held == 1 else 0
            released_count += 1 if action_held == 0 else 0
            samples.append(
                Stage1SampleRef(
                    sample_id=_required_str(row, "sample_id"),
                    run_id=_required_str(row, "run_id"),
                    dataset_dir=str(dataset_dir),
                    frame_index=_required_int(row, "frame_index"),
                    frame_stack_path=_required_str(row, "frame_stack_path"),
                    observation_path=_required_str(row, "observation_path"),
                    action_held=action_held,
                )
            )

    if expected_contract is None:
        raise CaptureRecordError("no Stage 1 dataset summaries were loaded")
    if held_count == 0 or released_count == 0:
        raise CaptureRecordError("Stage 1 index must include both action classes")

    return Stage1DatasetIndex(
        dataset_dirs=tuple(str(path) for path in resolved_dirs),
        samples=tuple(samples),
        sample_count=len(samples),
        held_count=held_count,
        released_count=released_count,
        observation_width=expected_contract.observation_width,
        observation_height=expected_contract.observation_height,
        observation_dtype=expected_contract.observation_dtype,
        frame_stack_length=expected_contract.frame_stack_length,
        frame_stack_layout=expected_contract.frame_stack_layout,
        split=expected_contract.split,
        delta_sys=expected_contract.delta_sys,
    )


def iter_stage1_batches(
    index: Stage1DatasetIndex,
    *,
    batch_size: int,
    max_batches: int | None = None,
) -> Iterator[Stage1Batch]:
    """Yield bounded Stage 1 batches by reading frame-stack payload bytes."""

    if batch_size <= 0:
        raise CaptureRecordError("batch_size must be positive")
    if max_batches is not None and max_batches < 0:
        raise CaptureRecordError("max_batches must be non-negative")

    for batch_index, start in enumerate(range(0, index.sample_count, batch_size)):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch_samples = index.samples[start : start + batch_size]
        frame_stacks = tuple(
            _read_payload(
                Path(sample.frame_stack_path),
                expected_bytes=index.expected_stack_bytes,
                description="frame stack",
            )
            for sample in batch_samples
        )
        labels = tuple(sample.action_held for sample in batch_samples)
        yield Stage1Batch(
            samples=batch_samples,
            frame_stacks=frame_stacks,
            labels=labels,
            shape=(
                len(batch_samples),
                index.frame_stack_length,
                index.observation_height,
                index.observation_width,
            ),
            dtype=index.observation_dtype,
            layout=index.frame_stack_layout,
        )


def smoke_stage1_batches(
    index: Stage1DatasetIndex,
    *,
    batch_size: int,
    max_batches: int = 2,
) -> Stage1DataSmokeReport:
    """Load a bounded number of Stage 1 batches and report contract failures."""

    failures: list[str] = []
    checked_batch_count = 0
    checked_sample_count = 0

    try:
        for batch in iter_stage1_batches(
            index,
            batch_size=batch_size,
            max_batches=max_batches,
        ):
            checked_batch_count += 1
            checked_sample_count += batch.batch_size
            failures.extend(_batch_failures(batch, index=index))
    except (OSError, ValueError, CaptureRecordError) as exc:
        failures.append(str(exc))

    if checked_batch_count == 0:
        failures.append("at least one batch must be checked")

    return Stage1DataSmokeReport(
        schema_version=STAGE1_DATA_SMOKE_SCHEMA_VERSION,
        dataset_count=len(index.dataset_dirs),
        sample_count=index.sample_count,
        held_count=index.held_count,
        released_count=index.released_count,
        checked_batch_count=checked_batch_count,
        checked_sample_count=checked_sample_count,
        batch_size=batch_size,
        frame_stack_shape=index.frame_stack_shape,
        frame_stack_layout=index.frame_stack_layout,
        observation_dtype=index.observation_dtype,
        failures=tuple(failures),
    )


def _batch_failures(
    batch: Stage1Batch,
    *,
    index: Stage1DatasetIndex,
) -> list[str]:
    failures: list[str] = []
    if batch.shape != (
        batch.batch_size,
        index.frame_stack_length,
        index.observation_height,
        index.observation_width,
    ):
        failures.append("batch shape does not match index contract")
    if batch.dtype != "uint8":
        failures.append("Stage 1 batches currently require uint8 payloads")
    if batch.layout != "khw":
        failures.append("Stage 1 batches currently require khw frame-stack layout")
    if len(batch.frame_stacks) != batch.batch_size:
        failures.append("frame-stack payload count does not match batch size")
    if len(batch.labels) != batch.batch_size:
        failures.append("label count does not match batch size")
    if not set(batch.labels) <= {0, 1}:
        failures.append("batch labels must be binary 0/1")
    return failures


def _validate_summary_contract(
    summary: dict[str, object],
    *,
    dataset_dir: Path,
) -> None:
    schema_version = _required_str(summary, "schema_version")
    if schema_version != STAGE1_DATASET_SUMMARY_SCHEMA_VERSION:
        raise CaptureRecordError("unsupported Stage 1 dataset summary schema")
    output_dir = Path(_required_str(summary, "output_dir")).resolve()
    if output_dir != dataset_dir:
        raise CaptureRecordError("dataset summary output_dir does not match directory")
    if _required_int(summary, "sample_count") <= 0:
        raise CaptureRecordError("Stage 1 dataset sample_count must be positive")
    if _required_int(summary, "held_count") <= 0:
        raise CaptureRecordError("Stage 1 dataset held_count must be positive")
    if _required_int(summary, "released_count") <= 0:
        raise CaptureRecordError("Stage 1 dataset released_count must be positive")


def _summary_contract(summary: dict[str, object]) -> _Stage1DatasetContract:
    return _Stage1DatasetContract(
        observation_width=_required_int(summary, "observation_width"),
        observation_height=_required_int(summary, "observation_height"),
        observation_dtype=_required_str(summary, "observation_dtype"),
        observation_layout=_required_str(summary, "observation_layout"),
        frame_stack_length=_required_int(summary, "frame_stack_length"),
        frame_stack_layout=_required_str(summary, "frame_stack_layout"),
        split=_required_str(summary, "split"),
        delta_sys=_required_int(summary, "delta_sys"),
    )


def _validate_manifest_row(
    row: dict[str, object],
    *,
    dataset_dir: Path,
    summary: dict[str, object],
) -> None:
    schema_version = _required_str(row, "schema_version")
    if schema_version != STAGE1_MANIFEST_SCHEMA_VERSION:
        raise CaptureRecordError("unsupported Stage 1 manifest schema")
    if _required_int(row, "action_held") not in {0, 1}:
        raise CaptureRecordError("manifest action_held must be binary 0/1")
    if _required_str(row, "observation_dtype") != "uint8":
        raise CaptureRecordError("Stage 1 observation_dtype must be uint8")
    if _required_str(row, "frame_stack_layout") != "khw":
        raise CaptureRecordError("Stage 1 frame_stack_layout must be khw")
    for row_key, summary_key in (
        ("observation_width", "observation_width"),
        ("observation_height", "observation_height"),
        ("frame_stack_length", "frame_stack_length"),
        ("split", "split"),
        ("delta_sys", "delta_sys"),
    ):
        if row[row_key] != summary[summary_key]:
            raise CaptureRecordError(f"manifest {row_key} does not match summary")
    _validate_payload_path(
        Path(_required_str(row, "frame_stack_path")),
        dataset_dir=dataset_dir,
        expected_root="stacks",
    )
    _validate_payload_path(
        Path(_required_str(row, "observation_path")),
        dataset_dir=dataset_dir,
        expected_root="observations",
    )


def _validate_payload_path(
    path: Path,
    *,
    dataset_dir: Path,
    expected_root: str,
) -> None:
    resolved = path.resolve()
    expected_dir = (dataset_dir / expected_root).resolve()
    if expected_dir not in (resolved.parent, *resolved.parents):
        raise CaptureRecordError(f"payload path must stay under {expected_root}/")


def _read_payload(
    path: Path,
    *,
    expected_bytes: int,
    description: str,
) -> bytes:
    payload = path.read_bytes()
    if len(payload) != expected_bytes:
        raise CaptureRecordError(
            f"{description} payload has wrong byte size: "
            f"{len(payload)} != {expected_bytes}"
        )
    return payload


def _manifest_path(summary: dict[str, object], *, dataset_dir: Path) -> Path:
    manifest_path = Path(_required_str(summary, "manifest_path"))
    if not manifest_path.is_absolute():
        manifest_path = dataset_dir / manifest_path
    resolved = manifest_path.resolve()
    if resolved.parent != dataset_dir:
        raise CaptureRecordError("manifest_path must resolve inside dataset_dir")
    return resolved


def _load_json_object(path: Path) -> dict[str, object]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise CaptureRecordError(f"{path} must contain a JSON object")
    return data


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as jsonl_file:
        for line_number, line in enumerate(jsonl_file, start=1):
            row = json.loads(line)
            if not isinstance(row, dict):
                raise CaptureRecordError(
                    f"{path}:{line_number} must contain a JSON object"
                )
            rows.append(row)
    return rows


def _required_str(row: dict[str, object], key: str) -> str:
    value = _required_value(row, key)
    if not isinstance(value, str) or not value:
        raise CaptureRecordError(f"{key} must be a non-empty string")
    return value


def _required_int(row: dict[str, object], key: str) -> int:
    value = _required_value(row, key)
    if not isinstance(value, int):
        raise CaptureRecordError(f"{key} must be an integer")
    return value


def _required_value(row: dict[str, object], key: str) -> Any:
    if key not in row:
        raise CaptureRecordError(f"missing required field: {key}")
    return row[key]
