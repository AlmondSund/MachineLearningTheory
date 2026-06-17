"""Stage 1 checkpoint loading and inference policy adapter."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from voxter.contracts import CaptureRecordError
from voxter.training.stage1_torch import (
    STAGE1_TRAINING_SCHEMA_VERSION,
    build_stage1_model,
)


@dataclass(frozen=True, slots=True)
class Stage1PolicyMetadata:
    """Shape and preprocessing contract loaded from a Stage 1 checkpoint."""

    checkpoint_path: str
    model_name: str
    observation_width: int
    observation_height: int
    observation_dtype: str
    frame_stack_length: int
    frame_stack_layout: str
    delta_sys: int
    threshold: float
    device: str

    @property
    def expected_stack_bytes(self) -> int:
        """Return the required byte count for one `K,H,W` frame stack."""

        return (
            self.frame_stack_length * self.observation_height * self.observation_width
        )

    @property
    def input_shape(self) -> tuple[int, int, int]:
        """Return the per-sample model input shape as `(K,H,W)`."""

        return (
            self.frame_stack_length,
            self.observation_height,
            self.observation_width,
        )

    def to_json_dict(self) -> dict[str, object]:
        """Return stable JSON-compatible metadata for reports."""

        return {
            "checkpoint_path": self.checkpoint_path,
            "model_name": self.model_name,
            "observation_width": self.observation_width,
            "observation_height": self.observation_height,
            "observation_dtype": self.observation_dtype,
            "frame_stack_length": self.frame_stack_length,
            "frame_stack_layout": self.frame_stack_layout,
            "delta_sys": self.delta_sys,
            "threshold": self.threshold,
            "device": self.device,
            "expected_stack_bytes": self.expected_stack_bytes,
        }


class Stage1Policy:
    """Runtime adapter for a Stage 1 reactive CNN checkpoint."""

    def __init__(
        self,
        *,
        torch: ModuleType,
        model: Any,
        metadata: Stage1PolicyMetadata,
        device: Any,
    ) -> None:
        self._torch = torch
        self._model = model
        self.metadata = metadata
        self._device = device

    def predict_probability(self, frame_stack: bytes) -> float:
        """Return `P(action_held=1)` for one serialized `K,H,W` frame stack."""

        if len(frame_stack) != self.metadata.expected_stack_bytes:
            raise CaptureRecordError(
                "frame stack byte count does not match checkpoint contract: "
                f"{len(frame_stack)} != {self.metadata.expected_stack_bytes}"
            )
        tensor = self._torch.frombuffer(bytearray(frame_stack), dtype=self._torch.uint8)
        tensor = tensor.reshape((1, *self.metadata.input_shape)).to(
            device=self._device,
            dtype=self._torch.float32,
        )
        with self._torch.no_grad():
            logits = self._model(tensor / 255.0).squeeze(0).squeeze(0)
            probability = self._torch.sigmoid(logits)
        return float(probability.item())

    def predict_action(
        self, frame_stack: bytes, *, threshold: float | None = None
    ) -> int:
        """Return binary held-state action using the configured threshold."""

        effective_threshold = (
            self.metadata.threshold if threshold is None else threshold
        )
        if not 0 < effective_threshold < 1:
            raise CaptureRecordError("threshold must be between 0 and 1")
        return int(self.predict_probability(frame_stack) >= effective_threshold)


def load_stage1_policy(
    checkpoint_path: Path,
    *,
    device: str = "auto",
    threshold: float | None = None,
) -> Stage1Policy:
    """Load a Stage 1 checkpoint for inference."""

    torch, nn = _require_torch()
    selected_device = _select_device(torch, device)
    checkpoint = torch.load(
        checkpoint_path,
        map_location=selected_device,
        weights_only=False,
    )
    if not isinstance(checkpoint, dict):
        raise CaptureRecordError("Stage 1 checkpoint must contain a dictionary")
    schema_version = checkpoint.get("schema_version")
    if schema_version != STAGE1_TRAINING_SCHEMA_VERSION:
        raise CaptureRecordError("unsupported Stage 1 checkpoint schema")
    model_name = _required_str(checkpoint, "model_name")
    if model_name != "stage1-reactive-cnn":
        raise CaptureRecordError("unsupported Stage 1 model name")
    metadata_row = checkpoint.get("metadata")
    if not isinstance(metadata_row, dict):
        raise CaptureRecordError("Stage 1 checkpoint metadata must be a dictionary")
    preprocessing = metadata_row.get("preprocessing")
    if not isinstance(preprocessing, dict):
        raise CaptureRecordError("Stage 1 checkpoint preprocessing must be recorded")
    training = metadata_row.get("training")
    if not isinstance(training, dict):
        raise CaptureRecordError("Stage 1 checkpoint training metadata is required")

    frame_stack_length = _required_int(preprocessing, "frame_stack_length")
    model = build_stage1_model(nn, in_channels=frame_stack_length)
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise CaptureRecordError("Stage 1 checkpoint model_state_dict is required")
    model.load_state_dict(state_dict)
    model.to(selected_device)
    model.eval()

    effective_threshold = (
        float(_required_number(training, "threshold"))
        if threshold is None
        else threshold
    )
    if not 0 < effective_threshold < 1:
        raise CaptureRecordError("threshold must be between 0 and 1")

    policy_metadata = Stage1PolicyMetadata(
        checkpoint_path=str(checkpoint_path.resolve()),
        model_name=model_name,
        observation_width=_required_int(preprocessing, "observation_width"),
        observation_height=_required_int(preprocessing, "observation_height"),
        observation_dtype=_required_str(preprocessing, "observation_dtype"),
        frame_stack_length=frame_stack_length,
        frame_stack_layout=_required_str(preprocessing, "frame_stack_layout"),
        delta_sys=_required_int(preprocessing, "delta_sys"),
        threshold=effective_threshold,
        device=str(selected_device),
    )
    if policy_metadata.observation_dtype != "uint8":
        raise CaptureRecordError("Stage 1 policy requires uint8 observations")
    if policy_metadata.frame_stack_layout != "khw":
        raise CaptureRecordError("Stage 1 policy requires khw frame stacks")
    return Stage1Policy(
        torch=torch,
        model=model,
        metadata=policy_metadata,
        device=selected_device,
    )


def _require_torch() -> tuple[ModuleType, Any]:
    try:
        torch = importlib.import_module("torch")
    except ModuleNotFoundError as exc:
        raise CaptureRecordError(
            "PyTorch is required for Stage 1 policy inference. "
            'Install the training extra with `python -m pip install -e ".[train]"`.'
        ) from exc
    return torch, torch.nn


def _select_device(torch: ModuleType, requested: str) -> Any:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise CaptureRecordError("CUDA was requested but torch.cuda is unavailable")
    if requested not in {"cpu", "cuda"}:
        raise CaptureRecordError("device must be auto, cpu, or cuda")
    return torch.device(requested)


def _required_str(row: dict[str, object], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value:
        raise CaptureRecordError(f"{key} must be a non-empty string")
    return value


def _required_int(row: dict[str, object], key: str) -> int:
    value = row.get(key)
    if not isinstance(value, int):
        raise CaptureRecordError(f"{key} must be an integer")
    return value


def _required_number(row: dict[str, object], key: str) -> int | float:
    value = row.get(key)
    if not isinstance(value, int | float):
        raise CaptureRecordError(f"{key} must be numeric")
    return value
