"""Stage 2 ONNX Runtime recurrent inference policy adapter."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from voxter.contracts import ActionState, CaptureRecordError, coerce_action_state
from voxter.policy.stage2_policy import Stage2PolicyMetadata


@dataclass(frozen=True, slots=True)
class Stage2OnnxPolicyConfig:
    """ONNX Runtime execution configuration for a Stage 2 policy."""

    provider: str = "CPUExecutionProvider"
    intra_op_num_threads: int = 1
    inter_op_num_threads: int = 1


class Stage2OnnxPolicy:
    """Runtime adapter for a Stage 2 single-step ONNX model."""

    def __init__(
        self,
        *,
        session: Any,
        metadata: Stage2PolicyMetadata,
        input_names: tuple[str, str, str],
        output_names: tuple[str, str, str, str],
        hidden_shape: tuple[int, int, int],
    ) -> None:
        self._session = session
        self.metadata = metadata
        self._input_names = input_names
        self._output_names = output_names
        self._hidden_shape = hidden_shape
        self._hidden = np.zeros(hidden_shape, dtype=np.float32)
        self._previous_action = ActionState.RELEASED

    def reset_state(self) -> None:
        """Clear recurrent memory and restart from released previous action."""

        self._hidden = np.zeros(self._hidden_shape, dtype=np.float32)
        self._previous_action = ActionState.RELEASED

    def observe_action(self, action: ActionState | int) -> None:
        """Record the action actually applied by the outer decision layer."""

        self._previous_action = coerce_action_state(action)

    def predict_probability(self, frame_stack: bytes) -> float:
        """Return `P(action_held=1)` for one serialized `K,H,W` frame stack."""

        return self.predict_head_probabilities(frame_stack)["held_probability"]

    def predict_head_probabilities(self, frame_stack: bytes) -> dict[str, float]:
        """Return held, press, and release probabilities for one recurrent step."""

        outputs = self._run_step(frame_stack)
        return {
            "held_probability": _sigmoid_scalar(outputs[0]),
            "press_probability": _sigmoid_scalar(outputs[1]),
            "release_probability": _sigmoid_scalar(outputs[2]),
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

    def _run_step(self, frame_stack: bytes) -> list[Any]:
        frame_stack_array = self._frame_stack_array(frame_stack)
        previous_action = np.array([int(self._previous_action)], dtype=np.float32)
        outputs = cast(
            list[Any],
            self._session.run(
                list(self._output_names),
                {
                    self._input_names[0]: frame_stack_array,
                    self._input_names[1]: previous_action,
                    self._input_names[2]: self._hidden,
                },
            ),
        )
        self._hidden = outputs[3]
        return outputs

    def _frame_stack_array(self, frame_stack: bytes) -> np.ndarray:
        if len(frame_stack) != self.metadata.expected_stack_bytes:
            raise CaptureRecordError(
                "frame stack byte count does not match ONNX policy contract: "
                f"{len(frame_stack)} != {self.metadata.expected_stack_bytes}"
            )
        tensor = np.frombuffer(frame_stack, dtype=np.uint8).reshape(
            (1, *self.metadata.input_shape)
        )
        return tensor.astype(np.float32) / 255.0


def load_stage2_onnx_policy(
    onnx_path: Path,
    *,
    metadata_path: Path | None = None,
    threshold: float | None = None,
    config: Stage2OnnxPolicyConfig | None = None,
) -> Stage2OnnxPolicy:
    """Load an exported Stage 2 ONNX single-step model for inference."""

    ort = _require_onnxruntime()
    config = config or Stage2OnnxPolicyConfig()
    if config.intra_op_num_threads <= 0:
        raise CaptureRecordError("intra_op_num_threads must be positive")
    if config.inter_op_num_threads <= 0:
        raise CaptureRecordError("inter_op_num_threads must be positive")
    available_providers = ort.get_available_providers()
    if config.provider not in available_providers:
        raise CaptureRecordError(
            f"ONNX provider {config.provider!r} is unavailable; available "
            f"providers: {', '.join(available_providers)}"
        )
    metadata_path = (
        metadata_path
        if metadata_path is not None
        else onnx_path.with_suffix(onnx_path.suffix + ".json")
    )
    export_metadata = _load_export_metadata(metadata_path)
    input_names = cast(
        tuple[str, str, str],
        _required_str_tuple(export_metadata, "input_names", length=3),
    )
    output_names = cast(
        tuple[str, str, str, str],
        _required_str_tuple(export_metadata, "output_names", length=4),
    )
    hidden_shape = cast(
        tuple[int, int, int],
        _required_int_tuple(export_metadata, "hidden_shape", length=3),
    )
    checkpoint_metadata = export_metadata.get("checkpoint_metadata")
    if not isinstance(checkpoint_metadata, dict):
        raise CaptureRecordError("ONNX metadata checkpoint_metadata is required")
    policy_metadata = _policy_metadata_from_export(
        onnx_path,
        checkpoint_metadata,
        threshold=threshold,
        device=f"onnx:{config.provider}",
    )

    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session_options.intra_op_num_threads = config.intra_op_num_threads
    session_options.inter_op_num_threads = config.inter_op_num_threads
    session = ort.InferenceSession(
        str(onnx_path.resolve()),
        sess_options=session_options,
        providers=[config.provider],
    )
    return Stage2OnnxPolicy(
        session=session,
        metadata=policy_metadata,
        input_names=input_names,
        output_names=output_names,
        hidden_shape=hidden_shape,
    )


def _policy_metadata_from_export(
    onnx_path: Path,
    checkpoint_metadata: dict[str, object],
    *,
    threshold: float | None,
    device: str,
) -> Stage2PolicyMetadata:
    preprocessing = checkpoint_metadata.get("preprocessing")
    if not isinstance(preprocessing, dict):
        raise CaptureRecordError("ONNX checkpoint preprocessing metadata is required")
    training = checkpoint_metadata.get("training")
    if not isinstance(training, dict):
        raise CaptureRecordError("ONNX checkpoint training metadata is required")
    model_metadata = checkpoint_metadata.get("model")
    if not isinstance(model_metadata, dict):
        raise CaptureRecordError("ONNX checkpoint model metadata is required")
    effective_threshold = (
        float(_required_number(training, "threshold"))
        if threshold is None
        else threshold
    )
    _validate_probability_threshold(effective_threshold, "threshold")
    return Stage2PolicyMetadata(
        checkpoint_path=str(onnx_path.resolve()),
        model_name=_required_str(model_metadata, "name"),
        observation_width=_required_int(preprocessing, "observation_width"),
        observation_height=_required_int(preprocessing, "observation_height"),
        observation_dtype=_required_str(preprocessing, "observation_dtype"),
        frame_stack_length=_required_int(preprocessing, "frame_stack_length"),
        frame_stack_layout=_required_str(preprocessing, "frame_stack_layout"),
        sequence_length=_required_int(preprocessing, "sequence_length"),
        stride=_required_int(preprocessing, "stride"),
        delta_sys=_required_int(preprocessing, "delta_sys"),
        hidden_size=_required_int(model_metadata, "hidden_size"),
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
        device=device,
    )


def _require_onnxruntime() -> Any:
    try:
        import onnxruntime as ort  # type: ignore[import-untyped]
    except ModuleNotFoundError as exc:
        raise CaptureRecordError(
            "ONNX Runtime is required for Stage 2 ONNX inference. "
            'Install with `python -m pip install -e ".[onnx]"`.'
        ) from exc
    return ort


def _load_export_metadata(metadata_path: Path) -> dict[str, object]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise CaptureRecordError("ONNX metadata must be a JSON object")
    return metadata


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


def _optional_bool(row: dict[str, object], key: str, *, default: bool) -> bool:
    value = row.get(key, default)
    if not isinstance(value, bool):
        raise CaptureRecordError(f"{key} must be boolean")
    return value


def _required_int_tuple(
    row: dict[str, object],
    key: str,
    *,
    length: int,
) -> tuple[int, ...]:
    value = row.get(key)
    if (
        not isinstance(value, list)
        or len(value) != length
        or any(not isinstance(item, int) for item in value)
    ):
        raise CaptureRecordError(f"{key} must be a list of {length} integers")
    return tuple(value)


def _required_str_tuple(
    row: dict[str, object],
    key: str,
    *,
    length: int,
) -> tuple[str, ...]:
    value = row.get(key)
    if (
        not isinstance(value, list)
        or len(value) != length
        or any(not isinstance(item, str) or not item for item in value)
    ):
        raise CaptureRecordError(f"{key} must be a list of {length} strings")
    return tuple(value)


def _validate_probability_threshold(value: float, name: str) -> None:
    if not 0 < value < 1:
        raise CaptureRecordError(f"{name} must be between 0 and 1")


def _sigmoid_scalar(value: np.ndarray) -> float:
    return float(1.0 / (1.0 + np.exp(-float(value.reshape(-1)[0]))))
