"""Linux uinput keyboard control adapter."""

from __future__ import annotations

import fcntl
import os
import struct
import time
from dataclasses import dataclass

from voxter.contracts import ActionState, coerce_action_state

DEFAULT_ACTION_KEY_CODE = 17
DEFAULT_UINPUT_DEVICE = "/dev/uinput"

EV_SYN = 0x00
EV_KEY = 0x01
SYN_REPORT = 0
BUS_USB = 0x03

_UINPUT_IOCTL_BASE = ord("U")
_UINPUT_MAX_NAME_SIZE = 80
_ABS_CNT = 64
_INT_SIZE = struct.calcsize("i")
_UINPUT_SETUP_FORMAT = (
    f"{_UINPUT_MAX_NAME_SIZE}sHHHHI{_ABS_CNT}i{_ABS_CNT}i{_ABS_CNT}i{_ABS_CNT}i"
)
_UINPUT_SETUP_SIZE = struct.calcsize(_UINPUT_SETUP_FORMAT)
_INPUT_EVENT_FORMAT = "llHHI"


class ControlError(OSError):
    """Raised when a control adapter cannot apply an input state."""


@dataclass(frozen=True, slots=True)
class UInputKeyboardConfig:
    """Configuration for a one-key uinput keyboard device."""

    device_path: str = DEFAULT_UINPUT_DEVICE
    key_code: int = DEFAULT_ACTION_KEY_CODE
    device_name: str = "voxter-uinput-keyboard"
    settle_s: float = 0.1


class UInputKeyboardControl:
    """Apply Voxter binary held/released actions through Linux uinput.

    The adapter owns a virtual keyboard device and emits events only when the
    requested held-state changes. `close()` always attempts to release the key
    before destroying the virtual device.
    """

    def __init__(self, config: UInputKeyboardConfig | None = None) -> None:
        self.config = config or UInputKeyboardConfig()
        if self.config.key_code <= 0:
            raise ControlError("uinput key_code must be positive")
        if self.config.settle_s < 0:
            raise ControlError("uinput settle_s must be non-negative")
        self._fd = _open_uinput_device(self.config.device_path)
        self._closed = False
        self._current_action = ActionState.RELEASED
        try:
            _configure_uinput_device(
                self._fd,
                key_code=self.config.key_code,
                device_name=self.config.device_name,
            )
            if self.config.settle_s:
                time.sleep(self.config.settle_s)
        except BaseException:
            os.close(self._fd)
            self._closed = True
            raise

    @property
    def current_action(self) -> ActionState:
        """Return the last successfully applied action state."""

        return self._current_action

    def apply_action(self, action: ActionState | int) -> None:
        """Apply one binary held-state action."""

        self._raise_if_closed()
        desired = coerce_action_state(action)
        if desired == self._current_action:
            return
        event_value = 1 if desired is ActionState.HELD else 0
        _write_key_state(self._fd, key_code=self.config.key_code, value=event_value)
        self._current_action = desired

    def release(self) -> None:
        """Release the key if the adapter currently holds it."""

        if self._closed:
            return
        if self._current_action is ActionState.HELD:
            _write_key_state(self._fd, key_code=self.config.key_code, value=0)
            self._current_action = ActionState.RELEASED

    def close(self) -> None:
        """Release the key and destroy the virtual uinput device."""

        if self._closed:
            return
        try:
            self.release()
            fcntl.ioctl(self._fd, _ui_dev_destroy())
        finally:
            os.close(self._fd)
            self._closed = True

    def __enter__(self) -> UInputKeyboardControl:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _raise_if_closed(self) -> None:
        if self._closed:
            raise ControlError("uinput control adapter is closed")


def _open_uinput_device(device_path: str) -> int:
    try:
        return os.open(device_path, os.O_WRONLY | os.O_NONBLOCK)
    except FileNotFoundError as exc:
        raise ControlError(
            f"{device_path} does not exist; load the uinput kernel module and "
            "ensure the current user can write to the device"
        ) from exc
    except PermissionError as exc:
        raise ControlError(
            f"permission denied opening {device_path}; the current user needs "
            "write access to the uinput device"
        ) from exc


def _configure_uinput_device(fd: int, *, key_code: int, device_name: str) -> None:
    try:
        fcntl.ioctl(fd, _ui_set_evbit(), EV_KEY)
        fcntl.ioctl(fd, _ui_set_keybit(), key_code)
        _write_all(fd, _pack_uinput_user_device(device_name))
        fcntl.ioctl(fd, _ui_dev_create())
    except OSError as exc:
        raise ControlError(f"failed to configure uinput device: {exc}") from exc


def _write_key_state(fd: int, *, key_code: int, value: int) -> None:
    try:
        _write_input_event(fd, event_type=EV_KEY, code=key_code, value=value)
        _write_input_event(fd, event_type=EV_SYN, code=SYN_REPORT, value=0)
    except OSError as exc:
        raise ControlError(f"failed to write uinput key event: {exc}") from exc


def _write_input_event(fd: int, *, event_type: int, code: int, value: int) -> None:
    now = time.time()
    seconds = int(now)
    microseconds = int((now - seconds) * 1_000_000)
    payload = struct.pack(
        _INPUT_EVENT_FORMAT,
        seconds,
        microseconds,
        event_type,
        code,
        value,
    )
    _write_all(fd, payload)


def _pack_uinput_user_device(device_name: str) -> bytes:
    encoded_name = device_name.encode("utf-8")
    if len(encoded_name) >= _UINPUT_MAX_NAME_SIZE:
        raise ControlError("uinput device_name must fit in 79 UTF-8 bytes")
    return struct.pack(
        _UINPUT_SETUP_FORMAT,
        encoded_name,
        BUS_USB,
        0x1209,
        0x0001,
        0x0001,
        0,
        *([0] * _ABS_CNT),
        *([0] * _ABS_CNT),
        *([0] * _ABS_CNT),
        *([0] * _ABS_CNT),
    )


def _write_all(fd: int, payload: bytes) -> None:
    written_total = 0
    while written_total < len(payload):
        written = os.write(fd, payload[written_total:])
        if written <= 0:
            raise ControlError("short write while applying uinput payload")
        written_total += written


def _ui_dev_create() -> int:
    return _io(_UINPUT_IOCTL_BASE, 1)


def _ui_dev_destroy() -> int:
    return _io(_UINPUT_IOCTL_BASE, 2)


def _ui_set_evbit() -> int:
    return _iow(_UINPUT_IOCTL_BASE, 100, _INT_SIZE)


def _ui_set_keybit() -> int:
    return _iow(_UINPUT_IOCTL_BASE, 101, _INT_SIZE)


def _io(ioctl_type: int, number: int) -> int:
    return _ioc(0, ioctl_type, number, 0)


def _iow(ioctl_type: int, number: int, size: int) -> int:
    return _ioc(1, ioctl_type, number, size)


def _ioc(direction: int, ioctl_type: int, number: int, size: int) -> int:
    return (direction << 30) | (size << 16) | (ioctl_type << 8) | number
