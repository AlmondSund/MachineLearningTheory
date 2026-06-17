"""Frame capture contracts and the offline `grim` adapter."""

from __future__ import annotations

import subprocess
import time
from collections.abc import Iterable
from dataclasses import dataclass
from math import isfinite
from pathlib import Path

from voxter.contracts import ActionState, CaptureRecordError, RawCaptureRecord


class FrameCaptureError(RuntimeError):
    """Raised when frame capture fails."""


@dataclass(frozen=True, slots=True)
class MonitorGeometry:
    """A rectangular capture region in compositor coordinates."""

    x: int
    y: int
    width: int
    height: int

    def __str__(self) -> str:
        return f"{self.x},{self.y} {self.width}x{self.height}"


@dataclass(frozen=True, slots=True)
class FrameCaptureRecord:
    """A raw captured frame row."""

    run_id: str
    attempt_id: str | None
    frame_index: int
    timestamp: float
    frame_path: str
    action: ActionState
    action_sample_timestamp: float
    geometry: str
    capture_duration_s: float
    capture_backend: str
    image_format: str
    frame_width: int
    frame_height: int
    source_width: int
    source_height: int
    capture_resized: bool

    def to_raw_capture_record(self) -> RawCaptureRecord:
        """Return the shared frame/action manifest record."""

        return RawCaptureRecord(
            run_id=self.run_id,
            attempt_id=self.attempt_id,
            frame_index=self.frame_index,
            timestamp=self.timestamp,
            frame_path=self.frame_path,
            action=self.action,
        )

    def to_json_dict(self) -> dict[str, object]:
        """Return a stable JSON-compatible representation."""

        return {
            "run_id": self.run_id,
            "attempt_id": self.attempt_id,
            "frame_index": self.frame_index,
            "timestamp": self.timestamp,
            "frame_path": self.frame_path,
            "action": int(self.action),
            "action_sample_timestamp": self.action_sample_timestamp,
            "geometry": self.geometry,
            "capture_duration_s": self.capture_duration_s,
            "capture_backend": self.capture_backend,
            "image_format": self.image_format,
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
            "source_width": self.source_width,
            "source_height": self.source_height,
            "capture_resized": self.capture_resized,
        }


def parse_geometry(value: str) -> MonitorGeometry:
    """Parse `x,y WIDTHxHEIGHT` geometry strings used by `grim -g`."""

    try:
        origin, size = value.split(" ", maxsplit=1)
        x_text, y_text = origin.split(",", maxsplit=1)
        width_text, height_text = size.split("x", maxsplit=1)
        geometry = MonitorGeometry(
            x=int(x_text),
            y=int(y_text),
            width=int(width_text),
            height=int(height_text),
        )
    except ValueError as exc:
        raise CaptureRecordError(
            "geometry must use the form 'x,y WIDTHxHEIGHT'"
        ) from exc

    if geometry.width <= 0 or geometry.height <= 0:
        raise CaptureRecordError("geometry width and height must be positive")
    return geometry


def validate_frame_records(records: Iterable[FrameCaptureRecord]) -> None:
    """Validate captured frame rows and their shared manifest contract."""

    record_list = list(records)
    raw_records = [record.to_raw_capture_record() for record in record_list]
    if raw_records:
        from voxter.contracts import validate_capture_records

        validate_capture_records(raw_records)
    else:
        raise CaptureRecordError("at least one frame record is required")

    for record in record_list:
        parse_geometry(record.geometry)
        if not isfinite(record.capture_duration_s) or record.capture_duration_s < 0:
            raise CaptureRecordError("capture_duration_s must be finite and >= 0")
        if not isfinite(record.action_sample_timestamp):
            raise CaptureRecordError("action_sample_timestamp must be finite")
        if record.action_sample_timestamp > record.timestamp:
            raise CaptureRecordError(
                "action_sample_timestamp must be <= frame timestamp"
            )
        if not record.capture_backend:
            raise CaptureRecordError("capture_backend must be non-empty")
        if record.image_format not in {"jpeg", "png", "ppm", "pgm"}:
            raise CaptureRecordError("image_format must be jpeg, png, ppm, or pgm")
        if record.frame_width <= 0 or record.frame_height <= 0:
            raise CaptureRecordError("frame dimensions must be positive")
        if record.source_width <= 0 or record.source_height <= 0:
            raise CaptureRecordError("source dimensions must be positive")
        if not record.capture_resized and (
            record.frame_width != record.source_width
            or record.frame_height != record.source_height
        ):
            raise CaptureRecordError(
                "unresized frame dimensions must match source dimensions"
            )


class GrimFrameCapture:
    """Offline/debug frame capture using the `grim` screenshot CLI."""

    def __init__(
        self,
        geometry: str,
        *,
        image_format: str = "jpeg",
        jpeg_quality: int = 70,
        png_level: int = 0,
    ) -> None:
        if image_format not in {"jpeg", "png", "ppm"}:
            raise CaptureRecordError("image_format must be jpeg, png, or ppm")
        if not 0 <= jpeg_quality <= 100:
            raise CaptureRecordError("jpeg_quality must be between 0 and 100")
        if not 0 <= png_level <= 9:
            raise CaptureRecordError("png_level must be between 0 and 9")

        self.geometry = str(parse_geometry(geometry))
        self._geometry = parse_geometry(geometry)
        self.image_format = image_format
        self.jpeg_quality = jpeg_quality
        self.png_level = png_level

    @property
    def file_suffix(self) -> str:
        """Return the conventional file suffix for the configured format."""

        if self.image_format == "jpeg":
            return ".jpg"
        if self.image_format == "png":
            return ".png"
        return ".ppm"

    @property
    def backend_name(self) -> str:
        """Return a stable backend identifier for logs."""

        return f"grim-{self.image_format}"

    def close(self) -> None:
        """Release backend resources."""

    def capture(
        self,
        frame_path: Path,
        *,
        run_id: str,
        attempt_id: str | None,
        frame_index: int,
        action: ActionState,
        action_sample_timestamp: float,
    ) -> FrameCaptureRecord:
        """Capture one frame and return its manifest row."""

        command = [
            "grim",
            "-g",
            self.geometry,
            "-t",
            self.image_format,
        ]
        if self.image_format == "jpeg":
            command.extend(["-q", str(self.jpeg_quality)])
        elif self.image_format == "png":
            command.extend(["-l", str(self.png_level)])
        command.append(str(frame_path))

        started_at = time.monotonic()
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
        finished_at = time.monotonic()
        frame_timestamp = time.time()
        if result.returncode != 0:
            raise FrameCaptureError(
                f"grim failed for frame {frame_index}: {result.stderr.strip()}"
            )

        return FrameCaptureRecord(
            run_id=run_id,
            attempt_id=attempt_id,
            frame_index=frame_index,
            timestamp=frame_timestamp,
            frame_path=str(frame_path),
            action=action,
            action_sample_timestamp=action_sample_timestamp,
            geometry=self.geometry,
            capture_duration_s=finished_at - started_at,
            capture_backend=self.backend_name,
            image_format=self.image_format,
            frame_width=self._geometry.width,
            frame_height=self._geometry.height,
            source_width=self._geometry.width,
            source_height=self._geometry.height,
            capture_resized=False,
        )
