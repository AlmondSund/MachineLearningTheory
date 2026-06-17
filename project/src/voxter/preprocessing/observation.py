"""Deterministic frame-to-observation preprocessing."""

from __future__ import annotations

from dataclasses import dataclass

from voxter.contracts import CaptureRecordError

OBSERVATION_SCHEMA_VERSION = "observation-v1"


@dataclass(frozen=True, slots=True)
class ObservationConfig:
    """Configuration for the first model observation contract."""

    width: int
    height: int
    color_mode: str = "grayscale"
    dtype: str = "uint8"
    layout: str = "hw"

    def __post_init__(self) -> None:
        if self.width <= 0:
            raise CaptureRecordError("observation width must be positive")
        if self.height <= 0:
            raise CaptureRecordError("observation height must be positive")
        if self.color_mode != "grayscale":
            raise CaptureRecordError("only grayscale observation mode is supported")
        if self.dtype != "uint8":
            raise CaptureRecordError("only uint8 observation dtype is supported")
        if self.layout != "hw":
            raise CaptureRecordError("only hw observation layout is supported")


@dataclass(frozen=True, slots=True)
class Observation:
    """One preprocessed model observation."""

    schema_version: str
    width: int
    height: int
    color_mode: str
    dtype: str
    layout: str
    data: bytes

    def to_json_metadata(self) -> dict[str, object]:
        """Return metadata for manifests or benchmark reports without payload."""

        return {
            "schema_version": self.schema_version,
            "width": self.width,
            "height": self.height,
            "color_mode": self.color_mode,
            "dtype": self.dtype,
            "layout": self.layout,
            "byte_count": len(self.data),
        }


def preprocess_rgb_observation(
    rgb_data: bytes,
    *,
    source_width: int,
    source_height: int,
    config: ObservationConfig,
) -> Observation:
    """Convert RGB bytes into a deterministic grayscale observation."""

    _validate_rgb_payload(rgb_data, width=source_width, height=source_height)
    if source_width == config.width and source_height == config.height:
        grayscale = _rgb_to_grayscale(rgb_data)
    else:
        grayscale = _resize_grayscale_nearest(
            _rgb_to_grayscale(rgb_data),
            source_width=source_width,
            source_height=source_height,
            target_width=config.width,
            target_height=config.height,
        )
    return Observation(
        schema_version=OBSERVATION_SCHEMA_VERSION,
        width=config.width,
        height=config.height,
        color_mode=config.color_mode,
        dtype=config.dtype,
        layout=config.layout,
        data=grayscale,
    )


def preprocess_grayscale_observation(
    grayscale_data: bytes,
    *,
    source_width: int,
    source_height: int,
    config: ObservationConfig,
) -> Observation:
    """Validate and optionally resize grayscale bytes into an observation."""

    _validate_grayscale_payload(
        grayscale_data,
        width=source_width,
        height=source_height,
    )
    if source_width == config.width and source_height == config.height:
        data = grayscale_data[: source_width * source_height]
    else:
        data = _resize_grayscale_nearest(
            grayscale_data,
            source_width=source_width,
            source_height=source_height,
            target_width=config.width,
            target_height=config.height,
        )
    return Observation(
        schema_version=OBSERVATION_SCHEMA_VERSION,
        width=config.width,
        height=config.height,
        color_mode=config.color_mode,
        dtype=config.dtype,
        layout=config.layout,
        data=data,
    )


def _validate_rgb_payload(data: bytes, *, width: int, height: int) -> None:
    if width <= 0 or height <= 0:
        raise CaptureRecordError("RGB source dimensions must be positive")
    expected_size = width * height * 3
    if len(data) < expected_size:
        raise CaptureRecordError("RGB payload is smaller than width*height*3")


def _validate_grayscale_payload(data: bytes, *, width: int, height: int) -> None:
    if width <= 0 or height <= 0:
        raise CaptureRecordError("grayscale source dimensions must be positive")
    expected_size = width * height
    if len(data) < expected_size:
        raise CaptureRecordError("grayscale payload is smaller than width*height")


def _rgb_to_grayscale(data: bytes) -> bytes:
    grayscale = bytearray(len(data) // 3)
    for output_index, input_index in enumerate(range(0, len(data) - 2, 3)):
        red = data[input_index]
        green = data[input_index + 1]
        blue = data[input_index + 2]
        # BT.601 luma approximation using integer arithmetic.
        grayscale[output_index] = (77 * red + 150 * green + 29 * blue) >> 8
    return bytes(grayscale)


def _resize_grayscale_nearest(
    data: bytes,
    *,
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
) -> bytes:
    expected_size = source_width * source_height
    if len(data) < expected_size:
        raise CaptureRecordError("grayscale payload is smaller than width*height")

    resized = bytearray(target_width * target_height)
    for target_y in range(target_height):
        source_y = (target_y * source_height) // target_height
        source_row = source_y * source_width
        target_row = target_y * target_width
        for target_x in range(target_width):
            source_x = (target_x * source_width) // target_width
            resized[target_row + target_x] = data[source_row + source_x]
    return bytes(resized)
