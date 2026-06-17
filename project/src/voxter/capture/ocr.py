"""OCR helpers for unattended live-control terminal detection."""

from __future__ import annotations

import re
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from voxter.capture.events import InputEventKind, RawTerminalEvent
from voxter.capture.pipewire import GrayFrame, encode_gray_pgm
from voxter.contracts import CaptureRecordError

_ATTEMPT_RE = re.compile(r"\battempt\D{0,8}(\d+)\b", re.IGNORECASE)
_NUMBER_RE = re.compile(r"\b(\d{1,6})\b")


@dataclass(frozen=True, slots=True)
class AttemptOcrConfig:
    """Configuration for OCR-based Geometry Dash attempt detection."""

    roi: tuple[int, int, int, int]
    command: str = "tesseract"
    psm: int = 7
    scale: int = 3
    every_n_frames: int = 6
    min_change_interval_s: float = 0.75
    timeout_s: float = 1.0
    device: str = "voxter-ocr-attempt"
    key_code: int = 0
    emit_active_start_on_first_read: bool = True

    def __post_init__(self) -> None:
        x, y, width, height = self.roi
        if x < 0 or y < 0:
            raise CaptureRecordError("OCR ROI x and y must be non-negative")
        if width <= 0 or height <= 0:
            raise CaptureRecordError("OCR ROI width and height must be positive")
        if not self.command:
            raise CaptureRecordError("OCR command must be non-empty")
        if self.psm <= 0:
            raise CaptureRecordError("OCR psm must be positive")
        if self.scale <= 0:
            raise CaptureRecordError("OCR scale must be positive")
        if self.every_n_frames <= 0:
            raise CaptureRecordError("OCR every_n_frames must be positive")
        if self.min_change_interval_s < 0:
            raise CaptureRecordError("OCR min_change_interval_s must be non-negative")
        if self.timeout_s <= 0:
            raise CaptureRecordError("OCR timeout_s must be positive")
        if not self.device:
            raise CaptureRecordError("OCR terminal event device must be non-empty")


class AttemptOcrDetector:
    """Stateful detector that converts attempt-number changes into terminal events."""

    def __init__(
        self,
        config: AttemptOcrConfig,
        *,
        ocr_runner: Callable[[GrayFrame, AttemptOcrConfig], str] | None = None,
    ) -> None:
        self._config = config
        self._ocr_runner = ocr_runner or run_tesseract_attempt_ocr
        self._last_attempt: int | None = None
        self._last_terminal_timestamp: float | None = None

    def detect(
        self,
        frame: GrayFrame,
        *,
        frame_index: int,
        timestamp: float,
        run_id: str,
    ) -> list[RawTerminalEvent]:
        """Return terminal events detected for one frame."""

        if frame_index % self._config.every_n_frames != 0:
            return []

        text = self._ocr_runner(frame, self._config)
        attempt = parse_attempt_number(text)
        if attempt is None:
            return []

        if self._last_attempt is None:
            self._last_attempt = attempt
            if not self._config.emit_active_start_on_first_read:
                return []
            return [
                self._terminal_event(
                    run_id=run_id,
                    timestamp=timestamp,
                    terminal_type="active_start",
                )
            ]

        if attempt == self._last_attempt:
            return []

        elapsed = (
            None
            if self._last_terminal_timestamp is None
            else timestamp - self._last_terminal_timestamp
        )
        if elapsed is not None and elapsed < self._config.min_change_interval_s:
            return []

        previous_attempt = self._last_attempt
        self._last_attempt = attempt
        if attempt > previous_attempt:
            self._last_terminal_timestamp = timestamp
            return [
                self._terminal_event(
                    run_id=run_id,
                    timestamp=timestamp,
                    terminal_type="death",
                ),
                self._terminal_event(
                    run_id=run_id,
                    timestamp=timestamp,
                    terminal_type="active_start",
                ),
            ]

        self._last_terminal_timestamp = timestamp
        return [
            self._terminal_event(
                run_id=run_id,
                timestamp=timestamp,
                terminal_type="reset",
            ),
            self._terminal_event(
                run_id=run_id,
                timestamp=timestamp,
                terminal_type="active_start",
            ),
        ]

    def _terminal_event(
        self,
        *,
        run_id: str,
        timestamp: float,
        terminal_type: str,
    ) -> RawTerminalEvent:
        return RawTerminalEvent(
            run_id=run_id,
            attempt_id=None,
            timestamp=timestamp,
            device=self._config.device,
            key_code=self._config.key_code,
            kind=InputEventKind.PRESS,
            terminal_type=terminal_type,
        )


def parse_attempt_number(text: str) -> int | None:
    """Extract an attempt number from OCR text."""

    match = _ATTEMPT_RE.search(text)
    if match is not None:
        return int(match.group(1))
    number_match = _NUMBER_RE.search(text)
    if number_match is None:
        return None
    return int(number_match.group(1))


def parse_roi(text: str) -> tuple[int, int, int, int]:
    """Parse an OCR ROI as `x,y,width,height`."""

    parts = text.split(",")
    if len(parts) != 4:
        raise CaptureRecordError("OCR ROI must use x,y,width,height")
    try:
        x, y, width, height = (int(part) for part in parts)
    except ValueError as exc:
        raise CaptureRecordError("OCR ROI values must be integers") from exc
    roi = (x, y, width, height)
    config = AttemptOcrConfig(roi=roi)
    return config.roi


def run_tesseract_attempt_ocr(frame: GrayFrame, config: AttemptOcrConfig) -> str:
    """Run Tesseract on the configured ROI and return recognized text."""

    roi = crop_gray_frame(frame, config.roi)
    if config.scale > 1:
        roi = scale_gray_frame(roi, config.scale)

    with tempfile.TemporaryDirectory(prefix="voxter-ocr-") as temp_dir:
        image_path = Path(temp_dir) / "attempt.pgm"
        image_path.write_bytes(encode_gray_pgm(roi))
        command = [
            config.command,
            str(image_path),
            "stdout",
            "--psm",
            str(config.psm),
            "-c",
            "tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "abcdefghijklmnopqrstuvwxyz0123456789: ",
        ]
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=config.timeout_s,
            )
        except subprocess.TimeoutExpired as exc:
            raise CaptureRecordError(
                f"OCR command timed out after {config.timeout_s:.3f}s"
            ) from exc
    if completed.returncode != 0:
        message = completed.stderr.strip() or completed.stdout.strip()
        if not message:
            message = f"{config.command!r} exited with code {completed.returncode}"
        raise CaptureRecordError(f"OCR command failed: {message}")
    return completed.stdout


def read_gray_pgm(path: Path) -> GrayFrame:
    """Read a binary PGM image produced by the live-control preview path."""

    data = path.read_bytes()
    tokens: list[bytes] = []
    offset = 0
    while len(tokens) < 4:
        while offset < len(data) and data[offset] in b" \t\r\n":
            offset += 1
        if offset >= len(data):
            raise CaptureRecordError(f"incomplete PGM header: {path}")
        if data[offset : offset + 1] == b"#":
            while offset < len(data) and data[offset] not in b"\r\n":
                offset += 1
            continue
        start = offset
        while offset < len(data) and data[offset] not in b" \t\r\n":
            offset += 1
        tokens.append(data[start:offset])

    if tokens[0] != b"P5":
        raise CaptureRecordError(f"unsupported PGM magic in {path}")
    try:
        width = int(tokens[1])
        height = int(tokens[2])
        max_value = int(tokens[3])
    except ValueError as exc:
        raise CaptureRecordError(f"invalid PGM header in {path}") from exc
    if width <= 0 or height <= 0:
        raise CaptureRecordError(f"invalid PGM dimensions in {path}")
    if max_value != 255:
        raise CaptureRecordError(f"unsupported PGM max value in {path}")
    if offset >= len(data) or data[offset] not in b" \t\r\n":
        raise CaptureRecordError(f"missing PGM payload separator in {path}")
    if data[offset : offset + 2] == b"\r\n":
        offset += 2
    else:
        offset += 1
    expected_size = width * height
    pixels = data[offset : offset + expected_size]
    if len(pixels) != expected_size:
        raise CaptureRecordError(f"incomplete PGM payload in {path}")
    return GrayFrame(width=width, height=height, data=pixels)


def crop_gray_frame(frame: GrayFrame, roi: tuple[int, int, int, int]) -> GrayFrame:
    """Crop a grayscale frame to `x,y,width,height`."""

    x, y, width, height = roi
    if x < 0 or y < 0 or width <= 0 or height <= 0:
        raise CaptureRecordError("OCR ROI must be within positive dimensions")
    if x + width > frame.width or y + height > frame.height:
        raise CaptureRecordError("OCR ROI extends beyond the captured frame")
    expected_size = frame.width * frame.height
    if len(frame.data) < expected_size:
        raise CaptureRecordError("grayscale frame buffer is smaller than width*height")

    rows = []
    for row in range(y, y + height):
        start = row * frame.width + x
        rows.append(frame.data[start : start + width])
    return GrayFrame(width=width, height=height, data=b"".join(rows))


def scale_gray_frame(frame: GrayFrame, factor: int) -> GrayFrame:
    """Nearest-neighbor scale a grayscale frame by an integer factor."""

    if factor <= 0:
        raise CaptureRecordError("scale factor must be positive")
    if factor == 1:
        return frame

    scaled_rows = []
    for row in range(frame.height):
        start = row * frame.width
        source_row = frame.data[start : start + frame.width]
        scaled_row = b"".join(bytes([pixel]) * factor for pixel in source_row)
        scaled_rows.extend([scaled_row] * factor)
    return GrayFrame(
        width=frame.width * factor,
        height=frame.height * factor,
        data=b"".join(scaled_rows),
    )
