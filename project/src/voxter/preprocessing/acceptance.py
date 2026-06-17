"""Acceptance records for materialized Stage 1 datasets."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from voxter.capture.analysis import CaptureRunAnalysis, analyze_capture_run
from voxter.contracts import CaptureRecordError

STAGE1_ACCEPTANCE_SCHEMA_VERSION = "stage1-acceptance-v1"


@dataclass(frozen=True, slots=True)
class Stage1DatasetAcceptance:
    """Validation result for one accepted Stage 1 dataset artifact."""

    schema_version: str
    capture_dir: str
    dataset_dir: str
    manifest_path: str
    capture_analysis: CaptureRunAnalysis
    sample_count: int
    released_count: int
    held_count: int
    observation_width: int
    observation_height: int
    frame_stack_length: int
    expected_observation_bytes: int
    expected_stack_bytes: int
    missing_payload_count: int
    bad_observation_size_count: int
    bad_stack_size_count: int
    first_stack_warmup_ok: bool
    failures: tuple[str, ...]

    @property
    def passed(self) -> bool:
        """Return whether the dataset satisfies the acceptance checks."""

        return self.capture_analysis.passed and not self.failures

    def to_json_dict(self) -> dict[str, object]:
        """Return a stable JSON-compatible representation."""

        return {
            "schema_version": self.schema_version,
            "capture_dir": self.capture_dir,
            "dataset_dir": self.dataset_dir,
            "manifest_path": self.manifest_path,
            "capture_analysis": self.capture_analysis.to_json_dict(),
            "sample_count": self.sample_count,
            "released_count": self.released_count,
            "held_count": self.held_count,
            "observation_width": self.observation_width,
            "observation_height": self.observation_height,
            "frame_stack_length": self.frame_stack_length,
            "expected_observation_bytes": self.expected_observation_bytes,
            "expected_stack_bytes": self.expected_stack_bytes,
            "missing_payload_count": self.missing_payload_count,
            "bad_observation_size_count": self.bad_observation_size_count,
            "bad_stack_size_count": self.bad_stack_size_count,
            "first_stack_warmup_ok": self.first_stack_warmup_ok,
            "passed": self.passed,
            "failures": list(self.failures),
        }


def accept_stage1_dataset(
    capture_dir: Path,
    dataset_dir: Path,
    *,
    min_input_events: int = 1,
    require_both_actions: bool = True,
    max_sync_mismatches: int = 0,
    max_dropped_frames: int = 0,
    max_missing_frame_files: int = 0,
    max_missed_periods: int | None = None,
    max_p95_interval_ms: float | None = None,
    max_p99_interval_ms: float | None = None,
) -> Stage1DatasetAcceptance:
    """Validate one raw capture plus its materialized Stage 1 dataset."""

    capture_dir = capture_dir.resolve()
    dataset_dir = dataset_dir.resolve()
    capture_analysis = analyze_capture_run(
        capture_dir,
        min_input_events=min_input_events,
        require_both_actions=require_both_actions,
        max_sync_mismatches=max_sync_mismatches,
        max_dropped_frames=max_dropped_frames,
        max_missing_frame_files=max_missing_frame_files,
        max_missed_periods=max_missed_periods,
        max_p95_interval_ms=max_p95_interval_ms,
        max_p99_interval_ms=max_p99_interval_ms,
    )

    summary = _load_json_object(dataset_dir / "dataset_summary.json")
    manifest_path = Path(_required_str(summary, "manifest_path"))
    if not manifest_path.is_absolute():
        manifest_path = dataset_dir / manifest_path
    manifest_rows = _load_jsonl(manifest_path)

    observation_width = _required_int(summary, "observation_width")
    observation_height = _required_int(summary, "observation_height")
    frame_stack_length = _required_int(summary, "frame_stack_length")
    expected_observation_bytes = observation_width * observation_height
    expected_stack_bytes = expected_observation_bytes * frame_stack_length

    failures = _dataset_failures(
        dataset_dir=dataset_dir,
        capture_dir=capture_dir,
        summary=summary,
        manifest_rows=manifest_rows,
        expected_observation_bytes=expected_observation_bytes,
        expected_stack_bytes=expected_stack_bytes,
    )
    missing_payload_count, bad_observation_size_count, bad_stack_size_count = (
        _payload_problem_counts(
            manifest_rows,
            expected_observation_bytes=expected_observation_bytes,
            expected_stack_bytes=expected_stack_bytes,
        )
    )
    first_stack_warmup_ok = _first_stack_warmup_ok(
        manifest_rows,
        frame_stack_length=frame_stack_length,
    )

    return Stage1DatasetAcceptance(
        schema_version=STAGE1_ACCEPTANCE_SCHEMA_VERSION,
        capture_dir=str(capture_dir),
        dataset_dir=str(dataset_dir),
        manifest_path=str(manifest_path),
        capture_analysis=capture_analysis,
        sample_count=len(manifest_rows),
        released_count=sum(1 for row in manifest_rows if row.get("action_held") == 0),
        held_count=sum(1 for row in manifest_rows if row.get("action_held") == 1),
        observation_width=observation_width,
        observation_height=observation_height,
        frame_stack_length=frame_stack_length,
        expected_observation_bytes=expected_observation_bytes,
        expected_stack_bytes=expected_stack_bytes,
        missing_payload_count=missing_payload_count,
        bad_observation_size_count=bad_observation_size_count,
        bad_stack_size_count=bad_stack_size_count,
        first_stack_warmup_ok=first_stack_warmup_ok,
        failures=tuple(failures),
    )


def write_stage1_dataset_acceptance(
    capture_dir: Path,
    dataset_dir: Path,
    output_path: Path,
    *,
    min_input_events: int = 1,
    require_both_actions: bool = True,
    max_sync_mismatches: int = 0,
    max_dropped_frames: int = 0,
    max_missing_frame_files: int = 0,
    max_missed_periods: int | None = None,
    max_p95_interval_ms: float | None = None,
    max_p99_interval_ms: float | None = None,
) -> Stage1DatasetAcceptance:
    """Validate and write one Stage 1 dataset acceptance JSON record."""

    acceptance = accept_stage1_dataset(
        capture_dir,
        dataset_dir,
        min_input_events=min_input_events,
        require_both_actions=require_both_actions,
        max_sync_mismatches=max_sync_mismatches,
        max_dropped_frames=max_dropped_frames,
        max_missing_frame_files=max_missing_frame_files,
        max_missed_periods=max_missed_periods,
        max_p95_interval_ms=max_p95_interval_ms,
        max_p99_interval_ms=max_p99_interval_ms,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(acceptance.to_json_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return acceptance


def _dataset_failures(
    *,
    dataset_dir: Path,
    capture_dir: Path,
    summary: dict[str, object],
    manifest_rows: list[dict[str, object]],
    expected_observation_bytes: int,
    expected_stack_bytes: int,
) -> list[str]:
    failures: list[str] = []
    summary_sample_count = _required_int(summary, "sample_count")
    if len(manifest_rows) != summary_sample_count:
        failures.append(
            "manifest row count does not match summary sample_count: "
            f"{len(manifest_rows)} != {summary_sample_count}"
        )

    source_capture_dir = Path(_required_str(summary, "source_capture_dir")).resolve()
    if source_capture_dir != capture_dir:
        failures.append(
            "dataset source_capture_dir does not match accepted capture_dir: "
            f"{source_capture_dir} != {capture_dir}"
        )

    summary_dataset_dir = Path(_required_str(summary, "output_dir")).resolve()
    if summary_dataset_dir != dataset_dir:
        failures.append(
            "dataset summary output_dir does not match dataset_dir: "
            f"{summary_dataset_dir} != {dataset_dir}"
        )

    action_values = {row.get("action_held") for row in manifest_rows}
    if not action_values <= {0, 1}:
        failures.append("manifest action_held values must be binary 0/1")
    if action_values != {0, 1}:
        failures.append("manifest must include both released and held samples")

    released_count = sum(1 for row in manifest_rows if row.get("action_held") == 0)
    held_count = sum(1 for row in manifest_rows if row.get("action_held") == 1)
    if released_count != _required_int(summary, "released_count"):
        failures.append("released_count does not match manifest")
    if held_count != _required_int(summary, "held_count"):
        failures.append("held_count does not match manifest")

    missing_payload_count, bad_observation_size_count, bad_stack_size_count = (
        _payload_problem_counts(
            manifest_rows,
            expected_observation_bytes=expected_observation_bytes,
            expected_stack_bytes=expected_stack_bytes,
        )
    )
    if missing_payload_count:
        failures.append(f"missing payload files: {missing_payload_count}")
    if bad_observation_size_count:
        failures.append(f"bad observation payload sizes: {bad_observation_size_count}")
    if bad_stack_size_count:
        failures.append(f"bad frame-stack payload sizes: {bad_stack_size_count}")
    if not _first_stack_warmup_ok(
        manifest_rows,
        frame_stack_length=_required_int(summary, "frame_stack_length"),
    ):
        failures.append("first frame stack in each run/attempt must warm up correctly")
    return failures


def _payload_problem_counts(
    rows: list[dict[str, object]],
    *,
    expected_observation_bytes: int,
    expected_stack_bytes: int,
) -> tuple[int, int, int]:
    missing_payload_count = 0
    bad_observation_size_count = 0
    bad_stack_size_count = 0
    for row in rows:
        observation_path = Path(_required_str(row, "observation_path"))
        stack_path = Path(_required_str(row, "frame_stack_path"))
        raw_frame_path = Path(_required_str(row, "raw_frame_path"))
        if (
            not observation_path.exists()
            or not stack_path.exists()
            or not raw_frame_path.exists()
        ):
            missing_payload_count += 1
            continue
        if observation_path.stat().st_size != expected_observation_bytes:
            bad_observation_size_count += 1
        if stack_path.stat().st_size != expected_stack_bytes:
            bad_stack_size_count += 1
    return missing_payload_count, bad_observation_size_count, bad_stack_size_count


def _first_stack_warmup_ok(
    rows: list[dict[str, object]],
    *,
    frame_stack_length: int,
) -> bool:
    grouped: defaultdict[tuple[str, str | None], list[dict[str, object]]] = defaultdict(
        list
    )
    for row in rows:
        attempt_id = row.get("attempt_id")
        if attempt_id is not None and not isinstance(attempt_id, str):
            raise CaptureRecordError("attempt_id must be null or a string")
        grouped[(_required_str(row, "run_id"), attempt_id)].append(row)

    for group_rows in grouped.values():
        first_row = min(group_rows, key=lambda row: _required_int(row, "frame_index"))
        observation_path = Path(_required_str(first_row, "observation_path"))
        frame_stack_path = Path(_required_str(first_row, "frame_stack_path"))
        if not observation_path.exists() or not frame_stack_path.exists():
            return False
        observation = observation_path.read_bytes()
        frame_stack = frame_stack_path.read_bytes()
        if frame_stack != observation * frame_stack_length:
            return False
    return True


def _load_json_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CaptureRecordError(f"{path} must contain a JSON object")
    return value


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        value = json.loads(line)
        if not isinstance(value, dict):
            raise CaptureRecordError(f"{path}:{line_number + 1} must be an object")
        rows.append(value)
    return rows


def _required_str(row: dict[str, object], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str):
        raise CaptureRecordError(f"{key} must be a string")
    return value


def _required_int(row: dict[str, object], key: str) -> int:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise CaptureRecordError(f"{key} must be an integer")
    return value
