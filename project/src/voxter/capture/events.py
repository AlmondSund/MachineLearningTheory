"""Linux input-event parsing for raw Voxter capture logs."""

from __future__ import annotations

import os
import selectors
import struct
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from math import isfinite

from voxter.contracts import ActionState, CaptureRecordError

EV_KEY = 1
KEY_W = 17
KEY_KPPLUS = 78
KEY_KPENTER = 96
LINUX_INPUT_EVENT = struct.Struct("llHHI")


class InputEventKind(Enum):
    """Semantic key event kind used in raw input logs."""

    PRESS = "press"
    RELEASE = "release"
    REPEAT = "repeat"


@dataclass(frozen=True, slots=True)
class KernelInputEvent:
    """Decoded Linux `struct input_event`."""

    timestamp: float
    event_type: int
    code: int
    value: int


@dataclass(frozen=True, slots=True)
class RawInputEvent:
    """Raw input event row written beside captured frames."""

    run_id: str
    attempt_id: str | None
    timestamp: float
    device: str
    key_code: int
    kind: InputEventKind
    action: ActionState

    def to_json_dict(self) -> dict[str, object]:
        """Return a stable JSON-compatible representation."""

        return {
            "run_id": self.run_id,
            "attempt_id": self.attempt_id,
            "timestamp": self.timestamp,
            "device": self.device,
            "key_code": self.key_code,
            "kind": self.kind.value,
            "action": int(self.action),
        }


@dataclass(frozen=True, slots=True)
class RawTerminalEvent:
    """Manual terminal marker row, such as a human death/reset marker."""

    run_id: str
    attempt_id: str | None
    timestamp: float
    device: str
    key_code: int
    kind: InputEventKind
    terminal_type: str

    def to_json_dict(self) -> dict[str, object]:
        """Return a stable JSON-compatible representation."""

        return {
            "run_id": self.run_id,
            "attempt_id": self.attempt_id,
            "timestamp": self.timestamp,
            "device": self.device,
            "key_code": self.key_code,
            "kind": self.kind.value,
            "terminal_type": self.terminal_type,
        }


def unpack_input_events(data: bytes) -> list[KernelInputEvent]:
    """Decode complete Linux input-event structs from a byte chunk."""

    events: list[KernelInputEvent] = []
    complete_size = len(data) - (len(data) % LINUX_INPUT_EVENT.size)
    for offset in range(0, complete_size, LINUX_INPUT_EVENT.size):
        seconds, microseconds, event_type, code, value = LINUX_INPUT_EVENT.unpack_from(
            data, offset
        )
        events.append(
            KernelInputEvent(
                timestamp=seconds + microseconds / 1_000_000,
                event_type=event_type,
                code=code,
                value=value,
            )
        )
    return events


def key_event_kind(value: int) -> InputEventKind | None:
    """Translate Linux key values to Voxter input-event kinds."""

    if value == 1:
        return InputEventKind.PRESS
    if value == 0:
        return InputEventKind.RELEASE
    if value == 2:
        return InputEventKind.REPEAT
    return None


def raw_input_event_from_kernel(
    event: KernelInputEvent,
    *,
    run_id: str,
    attempt_id: str | None,
    device: str,
    key_code: int = KEY_W,
) -> RawInputEvent | None:
    """Convert a kernel event into a Voxter input row when it matches `key_code`."""

    if event.event_type != EV_KEY or event.code != key_code:
        return None

    kind = key_event_kind(event.value)
    if kind is None:
        return None

    action = (
        ActionState.RELEASED if kind is InputEventKind.RELEASE else ActionState.HELD
    )
    return RawInputEvent(
        run_id=run_id,
        attempt_id=attempt_id,
        timestamp=event.timestamp,
        device=device,
        key_code=key_code,
        kind=kind,
        action=action,
    )


def raw_terminal_event_from_kernel(
    event: KernelInputEvent,
    *,
    run_id: str,
    attempt_id: str | None,
    device: str,
    key_code: int = KEY_KPPLUS,
    terminal_type: str = "death",
) -> RawTerminalEvent | None:
    """Convert a kernel event into a terminal marker row for `key_code` presses."""

    if event.event_type != EV_KEY or event.code != key_code:
        return None

    kind = key_event_kind(event.value)
    if kind is not InputEventKind.PRESS:
        return None

    return RawTerminalEvent(
        run_id=run_id,
        attempt_id=attempt_id,
        timestamp=event.timestamp,
        device=device,
        key_code=key_code,
        kind=kind,
        terminal_type=terminal_type,
    )


def validate_input_events(events: Iterable[RawInputEvent]) -> None:
    """Validate monotonic raw input events within each run/attempt/device/key."""

    event_list = list(events)
    previous_by_stream: dict[tuple[str, str | None, str, int], float] = {}

    for event in event_list:
        if not event.run_id:
            raise CaptureRecordError("input event run_id must be non-empty")
        if event.attempt_id == "":
            raise CaptureRecordError(
                "input event attempt_id must be None or a non-empty string"
            )
        if not isfinite(event.timestamp):
            raise CaptureRecordError("input event timestamp must be finite")
        if not event.device:
            raise CaptureRecordError("input event device must be non-empty")

        stream = (event.run_id, event.attempt_id, event.device, event.key_code)
        previous_timestamp = previous_by_stream.get(stream)
        if previous_timestamp is not None and event.timestamp < previous_timestamp:
            raise CaptureRecordError("input event timestamps must be monotonic")
        previous_by_stream[stream] = event.timestamp


def validate_terminal_events(events: Iterable[RawTerminalEvent]) -> None:
    """Validate monotonic terminal marker events within each run/attempt/key."""

    event_list = list(events)
    previous_by_stream: dict[tuple[str, str | None, str, int], float] = {}

    for event in event_list:
        if not event.run_id:
            raise CaptureRecordError("terminal event run_id must be non-empty")
        if event.attempt_id == "":
            raise CaptureRecordError(
                "terminal event attempt_id must be None or a non-empty string"
            )
        if not isfinite(event.timestamp):
            raise CaptureRecordError("terminal event timestamp must be finite")
        if not event.device:
            raise CaptureRecordError("terminal event device must be non-empty")
        if event.kind is not InputEventKind.PRESS:
            raise CaptureRecordError("terminal events must be press markers")
        if event.terminal_type not in {
            "death",
            "reset",
            "completion",
            "active_start",
        }:
            raise CaptureRecordError(
                "terminal_type must be death, reset, completion, or active_start"
            )

        stream = (event.run_id, event.attempt_id, event.device, event.key_code)
        previous_timestamp = previous_by_stream.get(stream)
        if previous_timestamp is not None and event.timestamp < previous_timestamp:
            raise CaptureRecordError("terminal event timestamps must be monotonic")
        previous_by_stream[stream] = event.timestamp


def reconstruct_held_state(
    events: Iterable[RawInputEvent],
    *,
    initial_state: ActionState = ActionState.RELEASED,
    include_repeats: bool = False,
) -> list[tuple[float, ActionState]]:
    """Reconstruct held state changes from press/release events."""

    state = initial_state
    timeline: list[tuple[float, ActionState]] = []

    for event in events:
        if event.kind is InputEventKind.PRESS:
            state = ActionState.HELD
        elif event.kind is InputEventKind.RELEASE:
            state = ActionState.RELEASED
        elif not include_repeats:
            continue
        timeline.append((event.timestamp, state))

    return timeline


class InputEventReader:
    """Read matching Linux key events from one event device."""

    def __init__(
        self,
        device: str,
        *,
        run_id: str,
        attempt_id: str | None,
        key_code: int = KEY_W,
        terminal_key_code: int | None = None,
        terminal_type: str = "death",
        terminal_markers: dict[int, str] | None = None,
        include_repeats: bool = False,
    ) -> None:
        self.device = device
        self.run_id = run_id
        self.attempt_id = attempt_id
        self.key_code = key_code
        self.terminal_key_code = terminal_key_code
        self.terminal_type = terminal_type
        self.terminal_markers = terminal_markers
        self.include_repeats = include_repeats
        self._fd: int | None = None
        self._selector = selectors.DefaultSelector()
        self.current_action = ActionState.RELEASED
        self._terminal_events: list[RawTerminalEvent] = []

    def __enter__(self) -> InputEventReader:
        self.open()
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()

    def open(self) -> None:
        """Open the input device in non-blocking read mode."""

        if self._fd is not None:
            return
        self._fd = os.open(self.device, os.O_RDONLY | os.O_NONBLOCK)
        self._selector.register(self._fd, selectors.EVENT_READ)

    def close(self) -> None:
        """Close the input device if it is open."""

        if self._fd is None:
            return
        self._selector.unregister(self._fd)
        os.close(self._fd)
        self._fd = None

    def read_available(self) -> list[RawInputEvent]:
        """Return all immediately available matching input events."""

        if self._fd is None:
            raise CaptureRecordError("input event reader is not open")

        rows: list[RawInputEvent] = []
        while self._selector.select(timeout=0):
            try:
                data = os.read(self._fd, LINUX_INPUT_EVENT.size * 64)
            except BlockingIOError:
                break
            if not data:
                break

            for kernel_event in unpack_input_events(data):
                row = raw_input_event_from_kernel(
                    kernel_event,
                    run_id=self.run_id,
                    attempt_id=self.attempt_id,
                    device=self.device,
                    key_code=self.key_code,
                )
                if row is None:
                    terminal_row = self._terminal_event_from_kernel(kernel_event)
                    if terminal_row is not None:
                        self._terminal_events.append(terminal_row)
                    continue
                self.current_action = row.action
                if row.kind is InputEventKind.REPEAT and not self.include_repeats:
                    continue
                rows.append(row)

        return rows

    def pop_terminal_events(self) -> list[RawTerminalEvent]:
        """Return terminal marker events collected by the last reads."""

        events = self._terminal_events
        self._terminal_events = []
        return events

    def _terminal_event_from_kernel(
        self,
        event: KernelInputEvent,
    ) -> RawTerminalEvent | None:
        if self.terminal_markers is not None:
            terminal_type = self.terminal_markers.get(event.code)
            if terminal_type is None:
                return None
            return raw_terminal_event_from_kernel(
                event,
                run_id=self.run_id,
                attempt_id=self.attempt_id,
                device=self.device,
                key_code=event.code,
                terminal_type=terminal_type,
            )
        if self.terminal_key_code is None:
            return None
        return raw_terminal_event_from_kernel(
            event,
            run_id=self.run_id,
            attempt_id=self.attempt_id,
            device=self.device,
            key_code=self.terminal_key_code,
            terminal_type=self.terminal_type,
        )
