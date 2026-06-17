"""Optional PyTorch Stage 2 recurrent behavioral-cloning training."""

from __future__ import annotations

import importlib
import json
import random
import subprocess
import sys
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from voxter.contracts import CaptureRecordError
from voxter.training.stage1_data import Stage1SampleRef
from voxter.training.stage1_torch import compute_stage1_binary_metrics
from voxter.training.stage2_data import (
    Stage2SequenceBatch,
    Stage2SequenceIndex,
    Stage2SequenceRef,
    iter_stage2_sequence_batches,
    load_stage2_sequence_index,
)

STAGE2_TRAINING_SCHEMA_VERSION = "stage2-training-v1"
STAGE2_CNN_GRU_MODEL_NAME = "stage2-cnn-gru"
STAGE2_MOBILENET_GRU_MODEL_NAME = "stage2-mobilenetv3-small-gru"


@dataclass(frozen=True, slots=True)
class Stage2TrainingConfig:
    """Configuration for Stage 2 recurrent behavioral-cloning training."""

    dataset_dirs: tuple[Path, ...]
    output_dir: Path
    run_id: str = "stage2-local"
    epochs: int = 1
    batch_size: int = 4
    sequence_length: int = 32
    stride: int | None = None
    model_name: str = STAGE2_MOBILENET_GRU_MODEL_NAME
    hidden_size: int = 64
    pretrained_visual_encoder: bool = True
    freeze_visual_encoder: bool = True
    learning_rate: float = 1e-3
    validation_fraction: float = 0.2
    threshold: float = 0.5
    transition_weight_multiplier: float = 4.0
    transition_window_radius: int = 2
    transition_aux_loss_weight: float = 1.0
    device: str = "auto"
    seed: int = 0
    max_train_batches: int | None = None
    max_validation_batches: int | None = None
    log_every_batches: int | None = 50


@dataclass(frozen=True, slots=True)
class Stage2TrainingReport:
    """Machine-readable Stage 2 training result."""

    schema_version: str
    run_id: str
    output_dir: str
    checkpoint_path: str
    dataset_count: int
    sequence_count: int
    step_count: int
    train_sequence_count: int
    validation_sequence_count: int
    held_count: int
    released_count: int
    epochs: int
    batch_size: int
    sequence_length: int
    stride: int
    model_name: str
    hidden_size: int
    pretrained_visual_encoder: bool
    freeze_visual_encoder: bool
    learning_rate: float
    threshold: float
    transition_weight_multiplier: float
    transition_window_radius: int
    transition_aux_loss_weight: float
    device: str
    input_shape: tuple[int, int, int, int]
    train_loss: float
    validation_metrics: dict[str, object]
    split: dict[str, object]
    failures: tuple[str, ...]

    @property
    def passed(self) -> bool:
        """Return whether training produced a usable checkpoint and metrics."""

        return not self.failures

    def to_json_dict(self) -> dict[str, object]:
        """Return a stable JSON-compatible representation."""

        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "output_dir": self.output_dir,
            "checkpoint_path": self.checkpoint_path,
            "dataset_count": self.dataset_count,
            "sequence_count": self.sequence_count,
            "step_count": self.step_count,
            "train_sequence_count": self.train_sequence_count,
            "validation_sequence_count": self.validation_sequence_count,
            "held_count": self.held_count,
            "released_count": self.released_count,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "sequence_length": self.sequence_length,
            "stride": self.stride,
            "model_name": self.model_name,
            "hidden_size": self.hidden_size,
            "pretrained_visual_encoder": self.pretrained_visual_encoder,
            "freeze_visual_encoder": self.freeze_visual_encoder,
            "learning_rate": self.learning_rate,
            "threshold": self.threshold,
            "transition_weight_multiplier": self.transition_weight_multiplier,
            "transition_window_radius": self.transition_window_radius,
            "transition_aux_loss_weight": self.transition_aux_loss_weight,
            "device": self.device,
            "input_shape": list(self.input_shape),
            "train_loss": self.train_loss,
            "validation_metrics": self.validation_metrics,
            "split": self.split,
            "passed": self.passed,
            "failures": list(self.failures),
        }


def train_stage2_policy(config: Stage2TrainingConfig) -> Stage2TrainingReport:
    """Train and persist the Stage 2 recurrent behavior-cloning policy."""

    _validate_training_config(config)
    torch, nn = _require_torch()
    torch.manual_seed(config.seed)

    index = load_stage2_sequence_index(
        config.dataset_dirs,
        sequence_length=config.sequence_length,
        stride=config.stride,
    )
    split = _split_sequences_by_dataset(
        index,
        validation_fraction=config.validation_fraction,
        seed=config.seed,
    )
    selected_device = _select_device(torch, config.device)
    model = build_stage2_model(
        torch,
        nn,
        in_channels=index.stage1_index.frame_stack_length,
        hidden_size=config.hidden_size,
        model_name=config.model_name,
        pretrained_visual_encoder=config.pretrained_visual_encoder,
        freeze_visual_encoder=config.freeze_visual_encoder,
    )
    model.to(selected_device)
    model.train()

    pos_weight = index.released_count / index.held_count
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            [pos_weight],
            dtype=torch.float32,
            device=selected_device,
        ),
        reduction="none",
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    output_dir = config.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    training_log_path = output_dir / "training_log.jsonl"
    train_loss = 0.0
    with training_log_path.open("w", encoding="utf-8") as training_log:
        progress = _make_progress_logger(training_log.write)
        for epoch_index in range(config.epochs):
            epoch_sequences = list(split["train_sequences"])
            random.Random(config.seed + epoch_index).shuffle(epoch_sequences)
            train_loss = _train_one_epoch(
                torch,
                model,
                criterion,
                optimizer,
                epoch_sequences,
                index=index,
                batch_size=config.batch_size,
                device=selected_device,
                max_batches=config.max_train_batches,
                epoch=epoch_index + 1,
                run_id=config.run_id,
                transition_weight_multiplier=config.transition_weight_multiplier,
                transition_window_radius=config.transition_window_radius,
                transition_aux_loss_weight=config.transition_aux_loss_weight,
                log_every_batches=config.log_every_batches,
                progress=progress,
            )
            progress(
                {
                    "schema_version": STAGE2_TRAINING_SCHEMA_VERSION,
                    "run_id": config.run_id,
                    "phase": "train_epoch",
                    "epoch": epoch_index + 1,
                    "train_loss": train_loss,
                }
            )

    checkpoint_path = output_dir / "checkpoint.pt"
    metadata = _training_metadata(
        config,
        index=index,
        split=split,
        pos_weight=pos_weight,
        selected_device=str(selected_device),
    )
    torch.save(
        {
            "schema_version": STAGE2_TRAINING_SCHEMA_VERSION,
            "model_name": config.model_name,
            "model_state_dict": model.state_dict(),
            "metadata": metadata,
        },
        checkpoint_path,
    )

    validation_sequences = list(split["validation_sequences"])
    random.Random(config.seed).shuffle(validation_sequences)
    validation_metrics = _evaluate_stage2_sequences(
        torch,
        model,
        criterion,
        validation_sequences,
        index=index,
        batch_size=config.batch_size,
        device=selected_device,
        threshold=config.threshold,
        max_batches=config.max_validation_batches,
        run_id=config.run_id,
        log_every_batches=config.log_every_batches,
        progress=_make_progress_logger(lambda row: None),
    )

    dataset_summary = {
        "schema_version": STAGE2_TRAINING_SCHEMA_VERSION,
        "dataset_dirs": list(index.dataset_dirs),
        "dataset_count": len(index.dataset_dirs),
        "sequence_count": index.sequence_count,
        "step_count": index.step_count,
        "held_count": index.held_count,
        "released_count": index.released_count,
        "observation_width": index.stage1_index.observation_width,
        "observation_height": index.stage1_index.observation_height,
        "observation_dtype": index.stage1_index.observation_dtype,
        "frame_stack_length": index.stage1_index.frame_stack_length,
        "frame_stack_layout": index.stage1_index.frame_stack_layout,
        "sequence_length": index.sequence_length,
        "stride": index.stride,
        "delta_sys": index.stage1_index.delta_sys,
    }
    report = Stage2TrainingReport(
        schema_version=STAGE2_TRAINING_SCHEMA_VERSION,
        run_id=config.run_id,
        output_dir=str(output_dir),
        checkpoint_path=str(checkpoint_path),
        dataset_count=len(index.dataset_dirs),
        sequence_count=index.sequence_count,
        step_count=index.step_count,
        train_sequence_count=len(split["train_sequences"]),
        validation_sequence_count=len(split["validation_sequences"]),
        held_count=index.held_count,
        released_count=index.released_count,
        epochs=config.epochs,
        batch_size=config.batch_size,
        sequence_length=index.sequence_length,
        stride=index.stride,
        model_name=config.model_name,
        hidden_size=config.hidden_size,
        pretrained_visual_encoder=config.pretrained_visual_encoder,
        freeze_visual_encoder=config.freeze_visual_encoder,
        learning_rate=config.learning_rate,
        threshold=config.threshold,
        transition_weight_multiplier=config.transition_weight_multiplier,
        transition_window_radius=config.transition_window_radius,
        transition_aux_loss_weight=config.transition_aux_loss_weight,
        device=str(selected_device),
        input_shape=(
            index.sequence_length,
            index.stage1_index.frame_stack_length,
            index.stage1_index.observation_height,
            index.stage1_index.observation_width,
        ),
        train_loss=train_loss,
        validation_metrics=validation_metrics,
        split={
            "strategy": "dataset-directory",
            "train_dataset_dirs": split["train_dataset_dirs"],
            "validation_dataset_dirs": split["validation_dataset_dirs"],
        },
        failures=tuple(_training_failures(torch, train_loss, validation_metrics)),
    )
    (output_dir / "config.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "dataset_summary.json").write_text(
        json.dumps(dataset_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(report.to_json_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def write_stage2_training_report(
    config: Stage2TrainingConfig,
) -> Stage2TrainingReport:
    """Train Stage 2 and persist its standard artifact set."""

    return train_stage2_policy(config)


def build_stage2_model(
    torch: ModuleType,
    nn: Any,
    *,
    in_channels: int,
    hidden_size: int,
    model_name: str = STAGE2_MOBILENET_GRU_MODEL_NAME,
    pretrained_visual_encoder: bool = True,
    freeze_visual_encoder: bool = True,
) -> Any:
    """Build the Stage 2 recurrent architecture used by training/inference."""

    if model_name == STAGE2_MOBILENET_GRU_MODEL_NAME:
        return _build_mobilenet_stage2_model(
            torch,
            nn,
            in_channels=in_channels,
            hidden_size=hidden_size,
            pretrained_visual_encoder=pretrained_visual_encoder,
            freeze_visual_encoder=freeze_visual_encoder,
        )
    if model_name != STAGE2_CNN_GRU_MODEL_NAME:
        raise CaptureRecordError(f"unsupported Stage 2 model name: {model_name}")

    class Stage2CnnGru(nn.Module):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv2d(in_channels, 16, kernel_size=5, stride=2, padding=2),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),
                nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
            )
            self.gru = nn.GRU(
                input_size=65,
                hidden_size=hidden_size,
                batch_first=True,
            )
            self.head = nn.Linear(hidden_size, 1)
            self.press_head = nn.Linear(hidden_size, 1)
            self.release_head = nn.Linear(hidden_size, 1)

        def encode_sequence(
            self,
            frame_stacks: Any,
            previous_actions: Any,
            hidden: Any = None,
        ) -> Any:
            batch_size, sequence_length = frame_stacks.shape[:2]
            encoded = self.encoder(
                frame_stacks.reshape(
                    batch_size * sequence_length,
                    *frame_stacks.shape[2:],
                )
            )
            encoded = encoded.reshape(batch_size, sequence_length, 64)
            previous = previous_actions.to(dtype=encoded.dtype).unsqueeze(-1)
            recurrent_input = torch.cat((encoded, previous), dim=-1)
            output, _hidden = self.gru(recurrent_input, hidden)
            return output

        def forward_heads(
            self,
            frame_stacks: Any,
            previous_actions: Any,
            hidden: Any = None,
        ) -> tuple[Any, Any, Any]:
            output = self.encode_sequence(frame_stacks, previous_actions, hidden)
            return (
                self.head(output).squeeze(-1),
                self.press_head(output).squeeze(-1),
                self.release_head(output).squeeze(-1),
            )

        def forward(
            self,
            frame_stacks: Any,
            previous_actions: Any,
            hidden: Any = None,
        ) -> Any:
            output = self.encode_sequence(frame_stacks, previous_actions, hidden)
            return self.head(output).squeeze(-1)

        def step(
            self,
            frame_stack: Any,
            previous_action: Any,
            hidden: Any = None,
        ) -> tuple[Any, Any]:
            held_logit, _press_logit, _release_logit, next_hidden = self.step_heads(
                frame_stack,
                previous_action,
                hidden,
            )
            return held_logit, next_hidden

        def step_heads(
            self,
            frame_stack: Any,
            previous_action: Any,
            hidden: Any = None,
        ) -> tuple[Any, Any, Any, Any]:
            encoded = self.encoder(frame_stack).unsqueeze(1)
            previous = previous_action.to(dtype=encoded.dtype).reshape(1, 1, 1)
            recurrent_input = torch.cat((encoded, previous), dim=-1)
            output, next_hidden = self.gru(recurrent_input, hidden)
            step_output = output[:, -1, :]
            return (
                self.head(step_output).squeeze(-1),
                self.press_head(step_output).squeeze(-1),
                self.release_head(step_output).squeeze(-1),
                next_hidden,
            )

    return Stage2CnnGru()


def _build_mobilenet_stage2_model(
    torch: ModuleType,
    nn: Any,
    *,
    in_channels: int,
    hidden_size: int,
    pretrained_visual_encoder: bool,
    freeze_visual_encoder: bool,
) -> Any:
    torchvision_models = _require_torchvision_models()
    weights = (
        torchvision_models.MobileNet_V3_Small_Weights.DEFAULT
        if pretrained_visual_encoder
        else None
    )
    backbone = torchvision_models.mobilenet_v3_small(weights=weights)

    class Stage2MobileNetGru(nn.Module):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.stack_adapter = nn.Sequential(
                nn.Conv2d(in_channels, 3, kernel_size=1),
                nn.BatchNorm2d(3),
                nn.Hardswish(),
            )
            self.encoder = backbone.features
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            if freeze_visual_encoder:
                for parameter in self.encoder.parameters():
                    parameter.requires_grad = False
            self.gru = nn.GRU(
                input_size=577,
                hidden_size=hidden_size,
                batch_first=True,
            )
            self.head = nn.Linear(hidden_size, 1)
            self.press_head = nn.Linear(hidden_size, 1)
            self.release_head = nn.Linear(hidden_size, 1)

        def encode_frame_stacks(self, frame_stacks: Any) -> Any:
            adapted = self.stack_adapter(frame_stacks)
            encoded = self.pool(self.encoder(adapted))
            return torch.flatten(encoded, start_dim=1)

        def encode_sequence(
            self,
            frame_stacks: Any,
            previous_actions: Any,
            hidden: Any = None,
        ) -> Any:
            batch_size, sequence_length = frame_stacks.shape[:2]
            encoded = self.encode_frame_stacks(
                frame_stacks.reshape(
                    batch_size * sequence_length,
                    *frame_stacks.shape[2:],
                )
            )
            encoded = encoded.reshape(batch_size, sequence_length, 576)
            previous = previous_actions.to(dtype=encoded.dtype).unsqueeze(-1)
            recurrent_input = torch.cat((encoded, previous), dim=-1)
            output, _hidden = self.gru(recurrent_input, hidden)
            return output

        def forward_heads(
            self,
            frame_stacks: Any,
            previous_actions: Any,
            hidden: Any = None,
        ) -> tuple[Any, Any, Any]:
            output = self.encode_sequence(frame_stacks, previous_actions, hidden)
            return (
                self.head(output).squeeze(-1),
                self.press_head(output).squeeze(-1),
                self.release_head(output).squeeze(-1),
            )

        def forward(
            self,
            frame_stacks: Any,
            previous_actions: Any,
            hidden: Any = None,
        ) -> Any:
            output = self.encode_sequence(frame_stacks, previous_actions, hidden)
            return self.head(output).squeeze(-1)

        def step(
            self,
            frame_stack: Any,
            previous_action: Any,
            hidden: Any = None,
        ) -> tuple[Any, Any]:
            held_logit, _press_logit, _release_logit, next_hidden = self.step_heads(
                frame_stack,
                previous_action,
                hidden,
            )
            return held_logit, next_hidden

        def step_heads(
            self,
            frame_stack: Any,
            previous_action: Any,
            hidden: Any = None,
        ) -> tuple[Any, Any, Any, Any]:
            encoded = self.encode_frame_stacks(frame_stack).unsqueeze(1)
            previous = previous_action.to(dtype=encoded.dtype).reshape(1, 1, 1)
            recurrent_input = torch.cat((encoded, previous), dim=-1)
            output, next_hidden = self.gru(recurrent_input, hidden)
            step_output = output[:, -1, :]
            return (
                self.head(step_output).squeeze(-1),
                self.press_head(step_output).squeeze(-1),
                self.release_head(step_output).squeeze(-1),
                next_hidden,
            )

    return Stage2MobileNetGru()


def _batch_inputs(
    torch: ModuleType,
    batch: Stage2SequenceBatch,
    *,
    device: Any,
) -> Any:
    payload = bytearray()
    for sequence_payloads in batch.frame_stacks:
        for frame_stack in sequence_payloads:
            payload.extend(frame_stack)
    tensor = torch.frombuffer(payload, dtype=torch.uint8)
    tensor = tensor.reshape(batch.shape).to(device=device, dtype=torch.float32)
    return tensor / 255.0


def _batch_previous_actions(
    torch: ModuleType,
    batch: Stage2SequenceBatch,
    *,
    device: Any,
) -> Any:
    return torch.tensor(batch.previous_actions, dtype=torch.float32, device=device)


def _batch_targets(
    torch: ModuleType,
    batch: Stage2SequenceBatch,
    *,
    device: Any,
) -> Any:
    return torch.tensor(batch.labels, dtype=torch.float32, device=device)


def _batch_transition_targets(
    torch: ModuleType,
    batch: Stage2SequenceBatch,
    *,
    device: Any,
) -> tuple[Any, Any]:
    press_rows: list[list[float]] = []
    release_rows: list[list[float]] = []
    for labels in batch.labels:
        press_row = [0.0] * len(labels)
        release_row = [0.0] * len(labels)
        for step_index in range(1, len(labels)):
            previous_label = labels[step_index - 1]
            label = labels[step_index]
            press_row[step_index] = float(previous_label == 0 and label == 1)
            release_row[step_index] = float(previous_label == 1 and label == 0)
        press_rows.append(press_row)
        release_rows.append(release_row)
    return (
        torch.tensor(press_rows, dtype=torch.float32, device=device),
        torch.tensor(release_rows, dtype=torch.float32, device=device),
    )


def _validate_training_config(config: Stage2TrainingConfig) -> None:
    if config.epochs <= 0:
        raise CaptureRecordError("epochs must be positive")
    if config.batch_size <= 0:
        raise CaptureRecordError("batch_size must be positive")
    if config.sequence_length <= 1:
        raise CaptureRecordError("sequence_length must be greater than 1")
    if config.stride is not None and config.stride <= 0:
        raise CaptureRecordError("stride must be positive when provided")
    if config.hidden_size <= 0:
        raise CaptureRecordError("hidden_size must be positive")
    if config.model_name not in {
        STAGE2_CNN_GRU_MODEL_NAME,
        STAGE2_MOBILENET_GRU_MODEL_NAME,
    }:
        raise CaptureRecordError(f"unsupported Stage 2 model name: {config.model_name}")
    if config.learning_rate <= 0:
        raise CaptureRecordError("learning_rate must be positive")
    if not 0 < config.validation_fraction < 1:
        raise CaptureRecordError("validation_fraction must be between 0 and 1")
    if not 0 < config.threshold < 1:
        raise CaptureRecordError("threshold must be between 0 and 1")
    if config.transition_weight_multiplier < 1:
        raise CaptureRecordError("transition_weight_multiplier must be at least 1")
    if config.transition_window_radius < 0:
        raise CaptureRecordError("transition_window_radius must be non-negative")
    if config.transition_aux_loss_weight < 0:
        raise CaptureRecordError("transition_aux_loss_weight must be non-negative")
    if config.max_train_batches is not None and config.max_train_batches <= 0:
        raise CaptureRecordError("max_train_batches must be positive when provided")
    if config.max_validation_batches is not None and config.max_validation_batches <= 0:
        raise CaptureRecordError(
            "max_validation_batches must be positive when provided"
        )
    if config.log_every_batches is not None and config.log_every_batches <= 0:
        raise CaptureRecordError("log_every_batches must be positive when provided")


def _split_sequences_by_dataset(
    index: Stage2SequenceIndex,
    *,
    validation_fraction: float,
    seed: int,
) -> dict[str, Any]:
    dataset_dirs = sorted(index.dataset_dirs)
    if len(dataset_dirs) < 2:
        raise CaptureRecordError(
            "Stage 2 training requires at least two dataset directories for a "
            "non-frame-leaking train/validation split"
        )
    shuffled_dirs = list(dataset_dirs)
    random.Random(seed).shuffle(shuffled_dirs)
    validation_count = max(1, round(len(shuffled_dirs) * validation_fraction))
    validation_count = min(validation_count, len(shuffled_dirs) - 1)
    validation_dirs = set(shuffled_dirs[:validation_count])
    train_dirs = set(shuffled_dirs[validation_count:])
    train_sequences = tuple(
        sequence for sequence in index.sequences if sequence.dataset_dir in train_dirs
    )
    validation_sequences = tuple(
        sequence
        for sequence in index.sequences
        if sequence.dataset_dir in validation_dirs
    )
    if not train_sequences or not validation_sequences:
        raise CaptureRecordError("train and validation splits must both be non-empty")
    return {
        "train_dataset_dirs": sorted(train_dirs),
        "validation_dataset_dirs": sorted(validation_dirs),
        "train_sequences": train_sequences,
        "validation_sequences": validation_sequences,
    }


def _iter_sequence_batches(
    sequences: Sequence[Stage2SequenceRef],
    *,
    index: Stage2SequenceIndex,
    batch_size: int,
    max_batches: int | None = None,
) -> Iterable[Stage2SequenceBatch]:
    sliced_index = Stage2SequenceIndex(
        stage1_index=index.stage1_index,
        sequences=tuple(sequences),
        sequence_length=index.sequence_length,
        stride=index.stride,
        sequence_count=len(sequences),
        step_count=sum(len(sequence.samples) for sequence in sequences),
        held_count=index.held_count,
        released_count=index.released_count,
    )
    return iter_stage2_sequence_batches(
        sliced_index,
        batch_size=batch_size,
        max_batches=max_batches,
    )


def _train_one_epoch(
    torch: ModuleType,
    model: Any,
    criterion: Any,
    optimizer: Any,
    sequences: Sequence[Stage2SequenceRef],
    *,
    index: Stage2SequenceIndex,
    batch_size: int,
    device: Any,
    max_batches: int | None,
    epoch: int,
    run_id: str,
    transition_weight_multiplier: float,
    transition_window_radius: int,
    transition_aux_loss_weight: float,
    log_every_batches: int | None,
    progress: Callable[[dict[str, object]], None],
) -> float:
    model.train()
    loss_total = 0.0
    step_total = 0
    for batch_index, batch in enumerate(
        _iter_sequence_batches(
            sequences,
            index=index,
            batch_size=batch_size,
            max_batches=max_batches,
        ),
        start=1,
    ):
        inputs = _batch_inputs(torch, batch, device=device)
        previous_actions = _batch_previous_actions(torch, batch, device=device)
        targets = _batch_targets(torch, batch, device=device)
        press_targets, release_targets = _batch_transition_targets(
            torch,
            batch,
            device=device,
        )
        weights = _batch_transition_weights(
            torch,
            batch,
            multiplier=transition_weight_multiplier,
            radius=transition_window_radius,
            device=device,
        )
        optimizer.zero_grad(set_to_none=True)
        held_logits, press_logits, release_logits = model.forward_heads(
            inputs,
            previous_actions,
        )
        held_loss = _weighted_loss(
            criterion(held_logits, targets),
            weights,
        )
        transition_loss = _mean_loss(criterion(press_logits, press_targets)) + (
            _mean_loss(criterion(release_logits, release_targets))
        )
        loss = held_loss + transition_aux_loss_weight * transition_loss
        loss.backward()
        optimizer.step()
        step_count = batch.batch_size * index.sequence_length
        loss_total += float(loss.item()) * step_count
        step_total += step_count
        if log_every_batches is not None and batch_index % log_every_batches == 0:
            progress(
                {
                    "schema_version": STAGE2_TRAINING_SCHEMA_VERSION,
                    "run_id": run_id,
                    "phase": "train_batch",
                    "epoch": epoch,
                    "batch": batch_index,
                    "step_count": step_total,
                    "mean_loss": loss_total / step_total,
                }
            )
    if step_total == 0:
        raise CaptureRecordError("at least one training batch is required")
    return loss_total / step_total


def _batch_transition_weights(
    torch: ModuleType,
    batch: Stage2SequenceBatch,
    *,
    multiplier: float,
    radius: int,
    device: Any,
) -> Any:
    weights: list[list[float]] = []
    for labels in batch.labels:
        row = [1.0] * len(labels)
        for step_index in range(1, len(labels)):
            if labels[step_index] == labels[step_index - 1]:
                continue
            start = max(0, step_index - radius)
            end = min(len(labels), step_index + radius + 1)
            for weighted_index in range(start, end):
                row[weighted_index] = max(row[weighted_index], multiplier)
        weights.append(row)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _weighted_loss(losses: Any, weights: Any) -> Any:
    return (losses * weights).sum() / weights.sum()


def _mean_loss(losses: Any) -> Any:
    return losses.mean()


def _evaluate_stage2_sequences(
    torch: ModuleType,
    model: Any,
    criterion: Any,
    sequences: Sequence[Stage2SequenceRef],
    *,
    index: Stage2SequenceIndex,
    batch_size: int,
    device: Any,
    threshold: float,
    max_batches: int | None,
    run_id: str,
    log_every_batches: int | None,
    progress: Callable[[dict[str, object]], None],
) -> dict[str, object]:
    model.eval()
    loss_total = 0.0
    labels: list[int] = []
    predictions: list[int] = []
    probabilities: list[float] = []
    evaluated_samples: list[Stage1SampleRef] = []
    closed_loop_labels: list[int] = []
    closed_loop_predictions: list[int] = []
    closed_loop_probabilities: list[float] = []
    closed_loop_samples: list[Stage1SampleRef] = []
    transition_head_labels: list[int] = []
    transition_head_predictions: list[int] = []
    transition_head_probabilities: list[float] = []
    transition_head_samples: list[Stage1SampleRef] = []
    with torch.no_grad():
        for batch_index, batch in enumerate(
            _iter_sequence_batches(
                sequences,
                index=index,
                batch_size=batch_size,
                max_batches=max_batches,
            ),
            start=1,
        ):
            inputs = _batch_inputs(torch, batch, device=device)
            previous_actions = _batch_previous_actions(torch, batch, device=device)
            targets = _batch_targets(torch, batch, device=device)
            logits = model(inputs, previous_actions)
            loss = _mean_loss(criterion(logits, targets))
            probs = torch.sigmoid(logits).detach().cpu().reshape(-1).tolist()
            flat_labels = [label for row in batch.labels for label in row]
            flat_samples = [
                sample for sequence in batch.sequences for sample in sequence.samples
            ]
            loss_total += float(loss.item()) * len(flat_labels)
            labels.extend(flat_labels)
            probabilities.extend(float(prob) for prob in probs)
            predictions.extend(1 if float(prob) >= threshold else 0 for prob in probs)
            evaluated_samples.extend(flat_samples)
            closed_loop = _evaluate_closed_loop_batch(
                torch,
                model,
                inputs,
                batch,
                threshold=threshold,
                device=device,
            )
            closed_loop_labels.extend(closed_loop["labels"])
            closed_loop_predictions.extend(closed_loop["predictions"])
            closed_loop_probabilities.extend(closed_loop["probabilities"])
            closed_loop_samples.extend(closed_loop["samples"])
            transition_head = _evaluate_transition_head_closed_loop_batch(
                torch,
                model,
                inputs,
                batch,
                threshold=threshold,
                device=device,
            )
            transition_head_labels.extend(transition_head["labels"])
            transition_head_predictions.extend(transition_head["predictions"])
            transition_head_probabilities.extend(transition_head["probabilities"])
            transition_head_samples.extend(transition_head["samples"])
            if log_every_batches is not None and batch_index % log_every_batches == 0:
                progress(
                    {
                        "schema_version": STAGE2_TRAINING_SCHEMA_VERSION,
                        "run_id": run_id,
                        "phase": "validation_batch",
                        "batch": batch_index,
                        "step_count": len(labels),
                        "mean_loss": loss_total / len(labels),
                    }
                )
    if not labels:
        raise CaptureRecordError("at least one validation batch is required")
    metrics = compute_stage1_binary_metrics(
        labels=labels,
        predictions=predictions,
        samples=evaluated_samples,
    )
    metrics["loss"] = loss_total / len(labels)
    metrics["threshold"] = threshold
    metrics["evaluated_step_count"] = len(labels)
    metrics["mean_probability"] = sum(probabilities) / len(probabilities)
    closed_loop_metrics = compute_stage1_binary_metrics(
        labels=closed_loop_labels,
        predictions=closed_loop_predictions,
        samples=closed_loop_samples,
    )
    closed_loop_metrics["evaluated_step_count"] = len(closed_loop_labels)
    closed_loop_metrics["mean_probability"] = sum(closed_loop_probabilities) / len(
        closed_loop_probabilities
    )
    closed_loop_metrics["held_frame_count"] = sum(closed_loop_predictions)
    closed_loop_metrics["released_frame_count"] = len(closed_loop_predictions) - sum(
        closed_loop_predictions
    )
    metrics["closed_loop_metrics"] = closed_loop_metrics
    transition_head_metrics = compute_stage1_binary_metrics(
        labels=transition_head_labels,
        predictions=transition_head_predictions,
        samples=transition_head_samples,
    )
    transition_head_metrics["evaluated_step_count"] = len(transition_head_labels)
    transition_head_metrics["mean_probability"] = sum(
        transition_head_probabilities
    ) / len(transition_head_probabilities)
    transition_head_metrics["held_frame_count"] = sum(transition_head_predictions)
    transition_head_metrics["released_frame_count"] = len(
        transition_head_predictions
    ) - sum(transition_head_predictions)
    transition_head_metrics["decoder"] = "transition-heads"
    transition_head_metrics["press_threshold"] = threshold
    transition_head_metrics["release_threshold"] = threshold
    metrics["transition_head_closed_loop_metrics"] = transition_head_metrics
    return metrics


def _evaluate_closed_loop_batch(
    torch: ModuleType,
    model: Any,
    inputs: Any,
    batch: Stage2SequenceBatch,
    *,
    threshold: float,
    device: Any,
) -> dict[str, list[Any]]:
    labels: list[int] = []
    predictions: list[int] = []
    probabilities: list[float] = []
    samples: list[Stage1SampleRef] = []
    for sequence_index, sequence in enumerate(batch.sequences):
        hidden = None
        previous_action = torch.tensor([0.0], dtype=torch.float32, device=device)
        for step_index, sample in enumerate(sequence.samples):
            logit, hidden = model.step(
                inputs[sequence_index, step_index].unsqueeze(0),
                previous_action,
                hidden,
            )
            probability = float(torch.sigmoid(logit.squeeze(0)).item())
            prediction = int(probability >= threshold)
            labels.append(batch.labels[sequence_index][step_index])
            predictions.append(prediction)
            probabilities.append(probability)
            samples.append(sample)
            previous_action = torch.tensor(
                [float(prediction)],
                dtype=torch.float32,
                device=device,
            )
            hidden = hidden.detach()
    return {
        "labels": labels,
        "predictions": predictions,
        "probabilities": probabilities,
        "samples": samples,
    }


def _evaluate_transition_head_closed_loop_batch(
    torch: ModuleType,
    model: Any,
    inputs: Any,
    batch: Stage2SequenceBatch,
    *,
    threshold: float,
    device: Any,
) -> dict[str, list[Any]]:
    labels: list[int] = []
    predictions: list[int] = []
    probabilities: list[float] = []
    samples: list[Stage1SampleRef] = []
    for sequence_index, sequence in enumerate(batch.sequences):
        hidden = None
        previous_action_value = 0
        previous_action = torch.tensor([0.0], dtype=torch.float32, device=device)
        for step_index, sample in enumerate(sequence.samples):
            _held_logit, press_logit, release_logit, hidden = model.step_heads(
                inputs[sequence_index, step_index].unsqueeze(0),
                previous_action,
                hidden,
            )
            press_probability = float(torch.sigmoid(press_logit.squeeze(0)).item())
            release_probability = float(torch.sigmoid(release_logit.squeeze(0)).item())
            if previous_action_value == 0:
                prediction = int(press_probability >= threshold)
                decision_probability = press_probability
            else:
                prediction = int(release_probability < threshold)
                decision_probability = 1.0 - release_probability
            labels.append(batch.labels[sequence_index][step_index])
            predictions.append(prediction)
            probabilities.append(decision_probability)
            samples.append(sample)
            previous_action_value = prediction
            previous_action = torch.tensor(
                [float(prediction)],
                dtype=torch.float32,
                device=device,
            )
            hidden = hidden.detach()
    return {
        "labels": labels,
        "predictions": predictions,
        "probabilities": probabilities,
        "samples": samples,
    }


def _require_torch() -> tuple[ModuleType, Any]:
    try:
        torch = importlib.import_module("torch")
    except ModuleNotFoundError as exc:
        raise CaptureRecordError(
            "PyTorch is required for Stage 2 training. "
            'Install the training extra with `python -m pip install -e ".[train]"`.'
        ) from exc
    return torch, torch.nn


def _require_torchvision_models() -> Any:
    try:
        return importlib.import_module("torchvision.models")
    except ModuleNotFoundError as exc:
        raise CaptureRecordError(
            "torchvision is required for the MobileNetV3 Stage 2 model. "
            'Install the training extra with `python -m pip install -e ".[train]"`.'
        ) from exc


def _select_device(torch: ModuleType, requested: str) -> Any:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise CaptureRecordError("CUDA was requested but torch.cuda is unavailable")
    if requested not in {"cpu", "cuda"}:
        raise CaptureRecordError("device must be auto, cpu, or cuda")
    return torch.device(requested)


def _make_progress_logger(
    write_log: Callable[[str], object],
) -> Callable[[dict[str, object]], None]:
    def progress(row: dict[str, object]) -> None:
        line = json.dumps(row, sort_keys=True)
        write_log(line + "\n")
        print(line, file=sys.stderr, flush=True)

    return progress


def _training_metadata(
    config: Stage2TrainingConfig,
    *,
    index: Stage2SequenceIndex,
    split: dict[str, Any],
    pos_weight: float,
    selected_device: str,
) -> dict[str, object]:
    return {
        "schema_version": STAGE2_TRAINING_SCHEMA_VERSION,
        "run_id": config.run_id,
        "code_version": _git_commit_or_unknown(),
        "dataset_dirs": list(index.dataset_dirs),
        "train_dataset_dirs": split["train_dataset_dirs"],
        "validation_dataset_dirs": split["validation_dataset_dirs"],
        "dataset_manifest_version": "stage1-manifest-v1",
        "preprocessing": {
            "observation_width": index.stage1_index.observation_width,
            "observation_height": index.stage1_index.observation_height,
            "observation_dtype": index.stage1_index.observation_dtype,
            "frame_stack_length": index.stage1_index.frame_stack_length,
            "frame_stack_layout": index.stage1_index.frame_stack_layout,
            "sequence_length": index.sequence_length,
            "stride": index.stride,
            "delta_sys": index.stage1_index.delta_sys,
        },
        "model": {
            "name": config.model_name,
            "architecture": (
                "mobilenetv3-small-gru-binary-and-transition-heads"
                if config.model_name == STAGE2_MOBILENET_GRU_MODEL_NAME
                else "cnn-gru-binary-and-transition-heads"
            ),
            "hidden_size": config.hidden_size,
            "previous_action_input": True,
            "pretrained_visual_encoder": config.pretrained_visual_encoder,
            "freeze_visual_encoder": config.freeze_visual_encoder,
            "transition_auxiliary_heads": True,
        },
        "training": {
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "threshold": config.threshold,
            "transition_weight_multiplier": config.transition_weight_multiplier,
            "transition_window_radius": config.transition_window_radius,
            "transition_aux_loss_weight": config.transition_aux_loss_weight,
            "device": selected_device,
            "seed": config.seed,
            "pos_weight": pos_weight,
            "max_train_batches": config.max_train_batches,
            "max_validation_batches": config.max_validation_batches,
            "log_every_batches": config.log_every_batches,
        },
    }


def _git_commit_or_unknown() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def _training_failures(
    torch: ModuleType,
    train_loss: float,
    validation_metrics: dict[str, object],
) -> list[str]:
    failures: list[str] = []
    if not bool(torch.isfinite(torch.tensor(train_loss))):
        failures.append("train_loss must be finite")
    validation_loss = validation_metrics.get("loss")
    if not isinstance(validation_loss, float) or not bool(
        torch.isfinite(torch.tensor(validation_loss))
    ):
        failures.append("validation loss must be finite")
    if validation_metrics.get("evaluated_step_count", 0) == 0:
        failures.append("validation must evaluate at least one step")
    return failures
