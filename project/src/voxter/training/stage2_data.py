"""Stage 2 sequence dataset loading contracts built from Stage 1 manifests."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from voxter.contracts import CaptureRecordError
from voxter.training.stage1_data import (
    Stage1DatasetIndex,
    Stage1SampleRef,
    load_stage1_dataset_index,
)

STAGE2_DATA_SMOKE_SCHEMA_VERSION = "stage2-data-smoke-v1"


@dataclass(frozen=True, slots=True)
class Stage2SequenceRef:
    """Reference to one contiguous Stage 2 sequence window."""

    sequence_id: str
    dataset_dir: str
    run_id: str
    start_frame_index: int
    end_frame_index: int
    samples: tuple[Stage1SampleRef, ...]


@dataclass(frozen=True, slots=True)
class Stage2SequenceIndex:
    """Payload-light index of Stage 2 sequence windows."""

    stage1_index: Stage1DatasetIndex
    sequences: tuple[Stage2SequenceRef, ...]
    sequence_length: int
    stride: int
    sequence_count: int
    step_count: int
    held_count: int
    released_count: int

    @property
    def dataset_dirs(self) -> tuple[str, ...]:
        """Return source Stage 1 dataset directories."""

        return self.stage1_index.dataset_dirs

    @property
    def frame_stack_shape(self) -> tuple[int, int, int]:
        """Return one per-step frame-stack shape as `(K,H,W)`."""

        return self.stage1_index.frame_stack_shape

    @property
    def expected_stack_bytes(self) -> int:
        """Return expected byte count for one frame-stack payload."""

        return self.stage1_index.expected_stack_bytes


@dataclass(frozen=True, slots=True)
class Stage2SequenceBatch:
    """One dependency-light Stage 2 sequence batch."""

    sequences: tuple[Stage2SequenceRef, ...]
    frame_stacks: tuple[tuple[bytes, ...], ...]
    previous_actions: tuple[tuple[int, ...], ...]
    labels: tuple[tuple[int, ...], ...]
    shape: tuple[int, int, int, int, int]
    dtype: str
    layout: str

    @property
    def batch_size(self) -> int:
        """Return the number of sequences in the batch."""

        return len(self.sequences)


@dataclass(frozen=True, slots=True)
class Stage2DataSmokeReport:
    """Summary of a Stage 2 sequence data-loading smoke run."""

    schema_version: str
    dataset_count: int
    sequence_count: int
    step_count: int
    held_count: int
    released_count: int
    checked_batch_count: int
    checked_sequence_count: int
    sequence_length: int
    stride: int
    batch_size: int
    frame_stack_shape: tuple[int, int, int]
    frame_stack_layout: str
    observation_dtype: str
    failures: tuple[str, ...]

    @property
    def passed(self) -> bool:
        """Return whether the smoke run found no contract failures."""

        return not self.failures

    def to_json_dict(self) -> dict[str, object]:
        """Return a stable JSON-compatible representation."""

        return {
            "schema_version": self.schema_version,
            "dataset_count": self.dataset_count,
            "sequence_count": self.sequence_count,
            "step_count": self.step_count,
            "held_count": self.held_count,
            "released_count": self.released_count,
            "checked_batch_count": self.checked_batch_count,
            "checked_sequence_count": self.checked_sequence_count,
            "sequence_length": self.sequence_length,
            "stride": self.stride,
            "batch_size": self.batch_size,
            "frame_stack_shape": list(self.frame_stack_shape),
            "frame_stack_layout": self.frame_stack_layout,
            "observation_dtype": self.observation_dtype,
            "passed": self.passed,
            "failures": list(self.failures),
        }


def load_stage2_sequence_index(
    dataset_dirs: Sequence[Path],
    *,
    sequence_length: int,
    stride: int | None = None,
) -> Stage2SequenceIndex:
    """Build a Stage 2 contiguous sequence index from Stage 1 datasets."""

    if sequence_length <= 1:
        raise CaptureRecordError("sequence_length must be greater than 1")
    effective_stride = sequence_length if stride is None else stride
    if effective_stride <= 0:
        raise CaptureRecordError("stride must be positive")

    stage1_index = load_stage1_dataset_index(dataset_dirs)
    sequences = _build_sequences(
        stage1_index.samples,
        sequence_length=sequence_length,
        stride=effective_stride,
    )
    if not sequences:
        raise CaptureRecordError("Stage 2 index must contain at least one sequence")
    labels = [
        sample.action_held for sequence in sequences for sample in sequence.samples
    ]
    held_count = sum(1 for label in labels if label == 1)
    released_count = sum(1 for label in labels if label == 0)
    if held_count == 0 or released_count == 0:
        raise CaptureRecordError("Stage 2 index must include both action classes")
    return Stage2SequenceIndex(
        stage1_index=stage1_index,
        sequences=tuple(sequences),
        sequence_length=sequence_length,
        stride=effective_stride,
        sequence_count=len(sequences),
        step_count=len(labels),
        held_count=held_count,
        released_count=released_count,
    )


def iter_stage2_sequence_batches(
    index: Stage2SequenceIndex,
    *,
    batch_size: int,
    max_batches: int | None = None,
) -> Iterator[Stage2SequenceBatch]:
    """Yield bounded Stage 2 batches by reading frame-stack payload bytes."""

    if batch_size <= 0:
        raise CaptureRecordError("batch_size must be positive")
    if max_batches is not None and max_batches < 0:
        raise CaptureRecordError("max_batches must be non-negative")
    for batch_index, start in enumerate(range(0, index.sequence_count, batch_size)):
        if max_batches is not None and batch_index >= max_batches:
            break
        sequences = index.sequences[start : start + batch_size]
        frame_stacks = tuple(
            tuple(
                _read_payload(
                    Path(sample.frame_stack_path),
                    expected_bytes=index.expected_stack_bytes,
                    description="frame stack",
                )
                for sample in sequence.samples
            )
            for sequence in sequences
        )
        labels = tuple(
            tuple(sample.action_held for sample in sequence.samples)
            for sequence in sequences
        )
        previous_actions = tuple(_previous_actions(row) for row in labels)
        yield Stage2SequenceBatch(
            sequences=sequences,
            frame_stacks=frame_stacks,
            previous_actions=previous_actions,
            labels=labels,
            shape=(
                len(sequences),
                index.sequence_length,
                index.stage1_index.frame_stack_length,
                index.stage1_index.observation_height,
                index.stage1_index.observation_width,
            ),
            dtype=index.stage1_index.observation_dtype,
            layout=index.stage1_index.frame_stack_layout,
        )


def smoke_stage2_sequence_batches(
    index: Stage2SequenceIndex,
    *,
    batch_size: int,
    max_batches: int = 2,
) -> Stage2DataSmokeReport:
    """Load bounded Stage 2 batches and report contract failures."""

    failures: list[str] = []
    checked_batch_count = 0
    checked_sequence_count = 0
    try:
        for batch in iter_stage2_sequence_batches(
            index,
            batch_size=batch_size,
            max_batches=max_batches,
        ):
            checked_batch_count += 1
            checked_sequence_count += batch.batch_size
            failures.extend(_batch_failures(batch, index=index))
    except (OSError, ValueError, CaptureRecordError) as exc:
        failures.append(str(exc))
    if checked_batch_count == 0:
        failures.append("at least one batch must be checked")
    return Stage2DataSmokeReport(
        schema_version=STAGE2_DATA_SMOKE_SCHEMA_VERSION,
        dataset_count=len(index.dataset_dirs),
        sequence_count=index.sequence_count,
        step_count=index.step_count,
        held_count=index.held_count,
        released_count=index.released_count,
        checked_batch_count=checked_batch_count,
        checked_sequence_count=checked_sequence_count,
        sequence_length=index.sequence_length,
        stride=index.stride,
        batch_size=batch_size,
        frame_stack_shape=index.frame_stack_shape,
        frame_stack_layout=index.stage1_index.frame_stack_layout,
        observation_dtype=index.stage1_index.observation_dtype,
        failures=tuple(failures),
    )


def _build_sequences(
    samples: Sequence[Stage1SampleRef],
    *,
    sequence_length: int,
    stride: int,
) -> list[Stage2SequenceRef]:
    grouped: dict[tuple[str, str, str], list[Stage1SampleRef]] = {}
    for sample in samples:
        grouped.setdefault(
            (sample.dataset_dir, sample.run_id, _attempt_token(sample.sample_id)),
            [],
        ).append(sample)

    sequences: list[Stage2SequenceRef] = []
    for (dataset_dir, run_id, attempt), group_samples in sorted(grouped.items()):
        sorted_samples = sorted(group_samples, key=lambda sample: sample.frame_index)
        for contiguous in _contiguous_segments(sorted_samples):
            for start in range(0, len(contiguous) - sequence_length + 1, stride):
                window = tuple(contiguous[start : start + sequence_length])
                first = window[0]
                last = window[-1]
                sequences.append(
                    Stage2SequenceRef(
                        sequence_id=(
                            f"{run_id}:{attempt}:{first.frame_index:06d}-"
                            f"{last.frame_index:06d}:L{sequence_length}"
                        ),
                        dataset_dir=dataset_dir,
                        run_id=run_id,
                        start_frame_index=first.frame_index,
                        end_frame_index=last.frame_index,
                        samples=window,
                    )
                )
    return sequences


def _contiguous_segments(
    samples: Sequence[Stage1SampleRef],
) -> Iterator[list[Stage1SampleRef]]:
    segment: list[Stage1SampleRef] = []
    previous_frame_index: int | None = None
    for sample in samples:
        if (
            previous_frame_index is not None
            and sample.frame_index != previous_frame_index + 1
        ):
            if segment:
                yield segment
            segment = []
        segment.append(sample)
        previous_frame_index = sample.frame_index
    if segment:
        yield segment


def _previous_actions(labels: Sequence[int]) -> tuple[int, ...]:
    if not labels:
        return ()
    return (0, *labels[:-1])


def _attempt_token(sample_id: str) -> str:
    parts = sample_id.split(":")
    if len(parts) >= 3 and parts[1]:
        return parts[1]
    return "none"


def _read_payload(path: Path, *, expected_bytes: int, description: str) -> bytes:
    payload = path.read_bytes()
    if len(payload) != expected_bytes:
        raise CaptureRecordError(
            f"{description} payload has wrong byte size: "
            f"{len(payload)} != {expected_bytes}"
        )
    return payload


def _batch_failures(
    batch: Stage2SequenceBatch,
    *,
    index: Stage2SequenceIndex,
) -> list[str]:
    failures: list[str] = []
    if batch.shape != (
        batch.batch_size,
        index.sequence_length,
        index.stage1_index.frame_stack_length,
        index.stage1_index.observation_height,
        index.stage1_index.observation_width,
    ):
        failures.append("batch shape does not match index contract")
    if batch.dtype != "uint8":
        failures.append("Stage 2 batches currently require uint8 payloads")
    if batch.layout != "khw":
        failures.append("Stage 2 batches currently require khw frame-stack layout")
    for sequence, labels, previous_actions in zip(
        batch.sequences,
        batch.labels,
        batch.previous_actions,
        strict=True,
    ):
        if len(sequence.samples) != index.sequence_length:
            failures.append("sequence length does not match index contract")
        if len(labels) != index.sequence_length:
            failures.append("label length does not match index contract")
        if len(previous_actions) != index.sequence_length:
            failures.append("previous action length does not match index contract")
        if not set(labels) <= {0, 1}:
            failures.append("sequence labels must be binary 0/1")
        if not set(previous_actions) <= {0, 1}:
            failures.append("previous actions must be binary 0/1")
    return failures
