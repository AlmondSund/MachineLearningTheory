"""Stage 2 checkpoint loading and recurrent inference policy adapter."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from voxter.contracts import ActionState, CaptureRecordError, coerce_action_state
from voxter.training.stage2_torch import (
    STAGE2_CNN_GRU_MODEL_NAME,
    STAGE2_MOBILENET_GRU_MODEL_NAME,
    STAGE2_TRAINING_SCHEMA_VERSION,
    build_stage2_model,
)


@dataclass(frozen=True, slots=True)
class Stage2PolicyMetadata:
    """Shape, preprocessing, and recurrent contract loaded from a checkpoint."""

    checkpoint_path: str
    model_name: str
    observation_width: int
    observation_height: int
    observation_dtype: str
    frame_stack_length: int
    frame_stack_layout: str
    sequence_length: int
    stride: int
    delta_sys: int
    hidden_size: int
    pretrained_visual_encoder: bool
    freeze_visual_encoder: bool
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
        """Return the per-step model input shape as `(K,H,W)`."""

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
            "sequence_length": self.sequence_length,
            "stride": self.stride,
            "delta_sys": self.delta_sys,
            "hidden_size": self.hidden_size,
            "pretrained_visual_encoder": self.pretrained_visual_encoder,
            "freeze_visual_encoder": self.freeze_visual_encoder,
            "threshold": self.threshold,
            "device": self.device,
            "expected_stack_bytes": self.expected_stack_bytes,
        }


class Stage2Policy:
    """Runtime adapter for a Stage 2 recurrent checkpoint."""

    def __init__(
        self,
        *,
        torch: ModuleType,
        model: Any,
        metadata: Stage2PolicyMetadata,
        device: Any,
    ) -> None:
        self._torch = torch
        self._model = model
        self.metadata = metadata
        self._device = device
        self._hidden: Any | None = None
        self._previous_action = ActionState.RELEASED

    def reset_state(self) -> None:
        """Clear recurrent memory and restart from released previous action."""

        self._hidden = None
        self._previous_action = ActionState.RELEASED

    def observe_action(self, action: ActionState | int) -> None:
        """Record the action actually applied by the outer decision layer."""

        self._previous_action = coerce_action_state(action)

    def predict_probability(self, frame_stack: bytes) -> float:
        """Return `P(action_held=1)` for one serialized `K,H,W` frame stack."""

        tensor = self._frame_stack_tensor(frame_stack)
        previous_action = self._previous_action_tensor()
        with self._torch.no_grad():
            logit, next_hidden = self._model.step(
                tensor / 255.0,
                previous_action,
                self._hidden,
            )
            probability = self._torch.sigmoid(logit.squeeze(0))
        self._hidden = next_hidden.detach()
        return float(probability.item())

    def predict_head_probabilities(self, frame_stack: bytes) -> dict[str, float]:
        """Return held, press, and release probabilities for one recurrent step."""

        tensor = self._frame_stack_tensor(frame_stack)
        previous_action = self._previous_action_tensor()
        step_heads = getattr(self._model, "step_heads", None)
        if not callable(step_heads):
            raise CaptureRecordError("Stage 2 model does not expose transition heads")
        with self._torch.no_grad():
            held_logit, press_logit, release_logit, next_hidden = step_heads(
                tensor / 255.0,
                previous_action,
                self._hidden,
            )
            held_probability = self._torch.sigmoid(held_logit.squeeze(0))
            press_probability = self._torch.sigmoid(press_logit.squeeze(0))
            release_probability = self._torch.sigmoid(release_logit.squeeze(0))
        self._hidden = next_hidden.detach()
        return {
            "held_probability": float(held_probability.item()),
            "press_probability": float(press_probability.item()),
            "release_probability": float(release_probability.item()),
        }

    def predict_transition_head_action(
        self,
        frame_stack: bytes,
        *,
        press_threshold: float | None = None,
        release_threshold: float | None = None,
    ) -> int:
        """Return a binary action decoded from press/release transition heads."""

        effective_press_threshold = (
            self.metadata.threshold if press_threshold is None else press_threshold
        )
        effective_release_threshold = (
            self.metadata.threshold if release_threshold is None else release_threshold
        )
        _validate_probability_threshold(effective_press_threshold, "press_threshold")
        _validate_probability_threshold(
            effective_release_threshold,
            "release_threshold",
        )
        probabilities = self.predict_head_probabilities(frame_stack)
        if self._previous_action is ActionState.RELEASED:
            action = int(
                probabilities["press_probability"] >= effective_press_threshold
            )
        else:
            action = int(
                probabilities["release_probability"] < effective_release_threshold
            )
        self.observe_action(action)
        return action

    def _frame_stack_tensor(self, frame_stack: bytes) -> Any:
        if len(frame_stack) != self.metadata.expected_stack_bytes:
            raise CaptureRecordError(
                "frame stack byte count does not match checkpoint contract: "
                f"{len(frame_stack)} != {self.metadata.expected_stack_bytes}"
            )
        tensor = self._torch.frombuffer(bytearray(frame_stack), dtype=self._torch.uint8)
        return tensor.reshape((1, *self.metadata.input_shape)).to(
            device=self._device,
            dtype=self._torch.float32,
        )

    def _previous_action_tensor(self) -> Any:
        return self._torch.tensor(
            [int(self._previous_action)],
            dtype=self._torch.float32,
            device=self._device,
        )

    def predict_action(
        self, frame_stack: bytes, *, threshold: float | None = None
    ) -> int:
        """Return binary held-state action and feed it back as previous action."""

        effective_threshold = (
            self.metadata.threshold if threshold is None else threshold
        )
        if not 0 < effective_threshold < 1:
            raise CaptureRecordError("threshold must be between 0 and 1")
        action = int(self.predict_probability(frame_stack) >= effective_threshold)
        self.observe_action(action)
        return action


def load_stage2_policy(
    checkpoint_path: Path,
    *,
    device: str = "auto",
    threshold: float | None = None,
) -> Stage2Policy:
    """Load a Stage 2 checkpoint for recurrent inference."""

    torch, nn = _require_torch()
    selected_device = _select_device(torch, device)
    checkpoint = torch.load(
        checkpoint_path,
        map_location=selected_device,
        weights_only=False,
    )
    if not isinstance(checkpoint, dict):
        raise CaptureRecordError("Stage 2 checkpoint must contain a dictionary")
    schema_version = checkpoint.get("schema_version")
    if schema_version != STAGE2_TRAINING_SCHEMA_VERSION:
        raise CaptureRecordError("unsupported Stage 2 checkpoint schema")
    model_name = _required_str(checkpoint, "model_name")
    if model_name not in {STAGE2_CNN_GRU_MODEL_NAME, STAGE2_MOBILENET_GRU_MODEL_NAME}:
        raise CaptureRecordError("unsupported Stage 2 model name")
    metadata_row = checkpoint.get("metadata")
    if not isinstance(metadata_row, dict):
        raise CaptureRecordError("Stage 2 checkpoint metadata must be a dictionary")
    preprocessing = metadata_row.get("preprocessing")
    if not isinstance(preprocessing, dict):
        raise CaptureRecordError("Stage 2 checkpoint preprocessing must be recorded")
    training = metadata_row.get("training")
    if not isinstance(training, dict):
        raise CaptureRecordError("Stage 2 checkpoint training metadata is required")
    model_metadata = metadata_row.get("model")
    if not isinstance(model_metadata, dict):
        raise CaptureRecordError("Stage 2 checkpoint model metadata is required")

    frame_stack_length = _required_int(preprocessing, "frame_stack_length")
    hidden_size = _required_int(model_metadata, "hidden_size")
    model = build_stage2_model(
        torch,
        nn,
        in_channels=frame_stack_length,
        hidden_size=hidden_size,
        model_name=model_name,
        pretrained_visual_encoder=False,
        freeze_visual_encoder=_optional_bool(
            model_metadata,
            "freeze_visual_encoder",
            default=False,
        ),
    )
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise CaptureRecordError("Stage 2 checkpoint model_state_dict is required")
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

    policy_metadata = Stage2PolicyMetadata(
        checkpoint_path=str(checkpoint_path.resolve()),
        model_name=model_name,
        observation_width=_required_int(preprocessing, "observation_width"),
        observation_height=_required_int(preprocessing, "observation_height"),
        observation_dtype=_required_str(preprocessing, "observation_dtype"),
        frame_stack_length=frame_stack_length,
        frame_stack_layout=_required_str(preprocessing, "frame_stack_layout"),
        sequence_length=_required_int(preprocessing, "sequence_length"),
        stride=_required_int(preprocessing, "stride"),
        delta_sys=_required_int(preprocessing, "delta_sys"),
        hidden_size=hidden_size,
        pretrained_visual_encoder=_optional_bool(
            model_metadata,
            "pretrained_visual_encoder",
            default=False,
        ),
        freeze_visual_encoder=_optional_bool(
            model_metadata,
            "freeze_visual_encoder",
            default=False,
        ),
        threshold=effective_threshold,
        device=str(selected_device),
    )
    if policy_metadata.observation_dtype != "uint8":
        raise CaptureRecordError("Stage 2 policy requires uint8 observations")
    if policy_metadata.frame_stack_layout != "khw":
        raise CaptureRecordError("Stage 2 policy requires khw frame stacks")
    return Stage2Policy(
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
            "PyTorch is required for Stage 2 policy inference. "
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


def _required_number(row: dict[str, object], key: str) -> float:
    value = row.get(key)
    if not isinstance(value, int | float):
        raise CaptureRecordError(f"{key} must be numeric")
    return float(value)


def _validate_probability_threshold(value: float, name: str) -> None:
    if not 0 < value < 1:
        raise CaptureRecordError(f"{name} must be between 0 and 1")


def _optional_bool(row: dict[str, object], key: str, *, default: bool) -> bool:
    value = row.get(key, default)
    if not isinstance(value, bool):
        raise CaptureRecordError(f"{key} must be boolean")
    return value
