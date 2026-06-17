"""Core data contracts shared by Voxter capture and preprocessing.

The functions in this module are intentionally pure. They validate and align
manifest-like records without reading frames, touching the filesystem, or
calling platform capture/control APIs.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum, IntEnum
from math import isfinite


class VoxterContractError(ValueError):
    """Base error for invalid Voxter data-contract values."""


class CaptureRecordError(VoxterContractError):
    """Raised when raw capture records violate the manifest contract."""


class ActionState(IntEnum):
    """Binary held-state action used by Voxter policies and datasets."""

    RELEASED = 0
    HELD = 1


class TransitionKind(Enum):
    """Derived press/release transition kind."""

    PRESS = "press"
    RELEASE = "release"


@dataclass(frozen=True, slots=True, init=False)
class RawCaptureRecord:
    """A raw capture manifest row before preprocessing.

    `frame_path` is a manifest path string only. Existence checks belong in
    capture or preprocessing adapters that intentionally touch the filesystem.
    """

    run_id: str
    attempt_id: str | None
    frame_index: int
    timestamp: float
    frame_path: str
    action: ActionState

    def __init__(
        self,
        run_id: str,
        attempt_id: str | None,
        frame_index: int,
        timestamp: float,
        frame_path: str,
        action: ActionState | int,
    ) -> None:
        object.__setattr__(self, "run_id", run_id)
        object.__setattr__(self, "attempt_id", attempt_id)
        object.__setattr__(self, "frame_index", frame_index)
        object.__setattr__(self, "timestamp", timestamp)
        object.__setattr__(self, "frame_path", frame_path)
        object.__setattr__(self, "action", coerce_action_state(action))
        self.__post_init__()

    def __post_init__(self) -> None:
        if not self.run_id:
            raise CaptureRecordError("run_id must be a non-empty string")
        if self.attempt_id == "":
            raise CaptureRecordError("attempt_id must be None or a non-empty string")
        if self.frame_index < 0:
            raise CaptureRecordError("frame_index must be non-negative")
        if not isfinite(self.timestamp):
            raise CaptureRecordError("timestamp must be finite")
        if not self.frame_path:
            raise CaptureRecordError("frame_path must be a non-empty string")


@dataclass(frozen=True, slots=True)
class ActionTransition:
    """A derived transition between adjacent held-state actions."""

    index: int
    previous_action: ActionState
    current_action: ActionState
    kind: TransitionKind


@dataclass(frozen=True, slots=True)
class AlignedSample:
    """A capture record paired with the causally aligned held-state label."""

    record: RawCaptureRecord
    label_record: RawCaptureRecord
    label_index: int
    action_label: ActionState
    delta_sys: int


def coerce_action_state(value: ActionState | int) -> ActionState:
    """Convert a binary integer or existing action state into `ActionState`."""

    if isinstance(value, ActionState):
        return value
    if isinstance(value, bool):
        raise CaptureRecordError("action must be integer 0 or 1, not bool")
    if value == 0:
        return ActionState.RELEASED
    if value == 1:
        return ActionState.HELD
    raise CaptureRecordError(f"action must be 0 or 1, got {value!r}")


def validate_capture_records(records: Iterable[RawCaptureRecord]) -> None:
    """Validate raw capture records by run/attempt and frame order."""

    record_list = list(records)
    if not record_list:
        raise CaptureRecordError("at least one raw capture record is required")

    grouped = _records_by_run_attempt(record_list)
    for group_key, group_records in grouped.items():
        seen_frame_indexes: set[int] = set()
        previous_timestamp: float | None = None

        for record in sorted(group_records, key=lambda item: item.frame_index):
            if record.frame_index in seen_frame_indexes:
                raise CaptureRecordError(
                    "duplicate frame_index "
                    f"{record.frame_index} in run/attempt {group_key!r}"
                )
            seen_frame_indexes.add(record.frame_index)

            if previous_timestamp is not None and record.timestamp < previous_timestamp:
                raise CaptureRecordError(
                    "timestamps must be monotonic by frame_index in "
                    f"run/attempt {group_key!r}"
                )
            previous_timestamp = record.timestamp


def extract_action_transitions(
    actions: Iterable[ActionState | int],
) -> list[ActionTransition]:
    """Derive press/release transitions from a held-state action sequence."""

    action_list = [coerce_action_state(action) for action in actions]
    transitions: list[ActionTransition] = []

    for index, (previous_action, current_action) in enumerate(
        zip(action_list, action_list[1:], strict=False),
        start=1,
    ):
        if previous_action == current_action:
            continue
        kind = (
            TransitionKind.PRESS
            if current_action == ActionState.HELD
            else TransitionKind.RELEASE
        )
        transitions.append(
            ActionTransition(
                index=index,
                previous_action=previous_action,
                current_action=current_action,
                kind=kind,
            )
        )

    return transitions


def aligned_label_index(frame_index: int, delta_sys: int) -> int:
    """Return the label index under `label_t = action_log_{t + delta_sys}`."""

    if frame_index < 0:
        raise CaptureRecordError("frame_index must be non-negative")
    return frame_index + delta_sys


def align_records(
    records: Iterable[RawCaptureRecord],
    delta_sys: int,
) -> list[AlignedSample]:
    """Align records to held-state labels without crossing run/attempt groups.

    Samples whose aligned label index does not exist in the same run/attempt
    are discarded.
    """

    record_list = list(records)
    validate_capture_records(record_list)

    aligned_samples: list[AlignedSample] = []
    for group_records in _records_by_run_attempt(record_list).values():
        labels_by_index = {record.frame_index: record for record in group_records}

        for record in sorted(group_records, key=lambda item: item.frame_index):
            label_index = aligned_label_index(record.frame_index, delta_sys)
            label_record = labels_by_index.get(label_index)
            if label_record is None:
                continue
            aligned_samples.append(
                AlignedSample(
                    record=record,
                    label_record=label_record,
                    label_index=label_index,
                    action_label=coerce_action_state(label_record.action),
                    delta_sys=delta_sys,
                )
            )

    return aligned_samples


def _records_by_run_attempt(
    records: Iterable[RawCaptureRecord],
) -> dict[tuple[str, str | None], list[RawCaptureRecord]]:
    grouped: defaultdict[tuple[str, str | None], list[RawCaptureRecord]] = defaultdict(
        list
    )
    for record in records:
        grouped[(record.run_id, record.attempt_id)].append(record)
    return dict(grouped)
