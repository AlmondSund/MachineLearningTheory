"""Rolling frame-stack construction for Stage 1 policy inputs."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from voxter.contracts import CaptureRecordError
from voxter.preprocessing.observation import Observation

FRAME_STACK_SCHEMA_VERSION = "frame-stack-v1"


@dataclass(frozen=True, slots=True)
class FrameStackConfig:
    """Configuration for stacked `observation-v1` policy inputs."""

    length: int
    width: int
    height: int
    dtype: str = "uint8"
    layout: str = "khw"

    def __post_init__(self) -> None:
        if self.length <= 0:
            raise CaptureRecordError("frame stack length must be positive")
        if self.width <= 0:
            raise CaptureRecordError("frame stack width must be positive")
        if self.height <= 0:
            raise CaptureRecordError("frame stack height must be positive")
        if self.dtype != "uint8":
            raise CaptureRecordError("only uint8 frame stacks are supported")
        if self.layout != "khw":
            raise CaptureRecordError("only khw frame stack layout is supported")


@dataclass(frozen=True, slots=True)
class FrameStack:
    """One fixed-size Stage 1 policy input stack."""

    schema_version: str
    length: int
    width: int
    height: int
    dtype: str
    layout: str
    data: bytes

    def to_json_metadata(self) -> dict[str, object]:
        """Return metadata for manifests or benchmark reports without payload."""

        return {
            "schema_version": self.schema_version,
            "length": self.length,
            "width": self.width,
            "height": self.height,
            "dtype": self.dtype,
            "layout": self.layout,
            "byte_count": len(self.data),
        }


class RollingFrameStack:
    """Maintain a deterministic rolling stack of recent observations."""

    def __init__(self, config: FrameStackConfig) -> None:
        self.config = config
        self._frames: deque[bytes] = deque(maxlen=config.length)

    def update(self, observation: Observation) -> FrameStack:
        """Append one observation and return the current fixed-size stack."""

        self._validate_observation(observation)
        if not self._frames:
            for _ in range(self.config.length):
                self._frames.append(observation.data)
        else:
            self._frames.append(observation.data)
        return FrameStack(
            schema_version=FRAME_STACK_SCHEMA_VERSION,
            length=self.config.length,
            width=self.config.width,
            height=self.config.height,
            dtype=self.config.dtype,
            layout=self.config.layout,
            data=b"".join(self._frames),
        )

    def _validate_observation(self, observation: Observation) -> None:
        if observation.width != self.config.width:
            raise CaptureRecordError("observation width does not match frame stack")
        if observation.height != self.config.height:
            raise CaptureRecordError("observation height does not match frame stack")
        if observation.dtype != self.config.dtype:
            raise CaptureRecordError("observation dtype does not match frame stack")
        if observation.color_mode != "grayscale":
            raise CaptureRecordError("frame stack requires grayscale observations")
        if observation.layout != "hw":
            raise CaptureRecordError("frame stack requires hw observations")
        expected_size = self.config.width * self.config.height
        if len(observation.data) != expected_size:
            raise CaptureRecordError("observation byte count does not match shape")
