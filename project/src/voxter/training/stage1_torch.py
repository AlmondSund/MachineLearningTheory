"""Optional PyTorch Stage 1 training and optimization smoke."""

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
from voxter.training.stage1_data import (
    Stage1Batch,
    Stage1DatasetIndex,
    Stage1SampleRef,
    load_stage1_dataset_index,
)

STAGE1_TORCH_SMOKE_SCHEMA_VERSION = "stage1-torch-smoke-v1"
STAGE1_TRAINING_SCHEMA_VERSION = "stage1-training-v1"


@dataclass(frozen=True, slots=True)
class Stage1TorchSmokeConfig:
    """Configuration for a tiny Stage 1 optimization smoke."""

    dataset_dirs: tuple[Path, ...]
    batch_size: int = 8
    train_steps: int = 3
    learning_rate: float = 1e-3
    device: str = "auto"
    seed: int = 0


@dataclass(frozen=True, slots=True)
class Stage1TorchSmokeReport:
    """Machine-readable result for the tiny Stage 1 optimization smoke."""

    schema_version: str
    dataset_count: int
    sample_count: int
    held_count: int
    released_count: int
    batch_size: int
    train_steps: int
    device: str
    model_name: str
    input_shape: tuple[int, int, int, int]
    initial_loss: float
    final_loss: float
    parameter_delta_l1: float
    failures: tuple[str, ...]

    @property
    def passed(self) -> bool:
        """Return whether the optimization smoke satisfied the contract."""

        return not self.failures

    def to_json_dict(self) -> dict[str, object]:
        """Return a stable JSON-compatible representation."""

        return {
            "schema_version": self.schema_version,
            "dataset_count": self.dataset_count,
            "sample_count": self.sample_count,
            "held_count": self.held_count,
            "released_count": self.released_count,
            "batch_size": self.batch_size,
            "train_steps": self.train_steps,
            "device": self.device,
            "model_name": self.model_name,
            "input_shape": list(self.input_shape),
            "initial_loss": self.initial_loss,
            "final_loss": self.final_loss,
            "parameter_delta_l1": self.parameter_delta_l1,
            "passed": self.passed,
            "failures": list(self.failures),
        }


@dataclass(frozen=True, slots=True)
class Stage1TrainingConfig:
    """Configuration for Stage 1 reactive behavioral-cloning training."""

    dataset_dirs: tuple[Path, ...]
    output_dir: Path
    run_id: str = "stage1-local"
    epochs: int = 1
    batch_size: int = 16
    learning_rate: float = 1e-3
    validation_fraction: float = 0.2
    threshold: float = 0.5
    device: str = "auto"
    seed: int = 0
    max_train_batches: int | None = None
    max_validation_batches: int | None = None
    log_every_batches: int | None = 100


@dataclass(frozen=True, slots=True)
class Stage1TrainingReport:
    """Machine-readable Stage 1 training result."""

    schema_version: str
    run_id: str
    output_dir: str
    checkpoint_path: str
    dataset_count: int
    sample_count: int
    train_sample_count: int
    validation_sample_count: int
    held_count: int
    released_count: int
    epochs: int
    batch_size: int
    learning_rate: float
    threshold: float
    device: str
    model_name: str
    input_shape: tuple[int, int, int]
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
            "sample_count": self.sample_count,
            "train_sample_count": self.train_sample_count,
            "validation_sample_count": self.validation_sample_count,
            "held_count": self.held_count,
            "released_count": self.released_count,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "threshold": self.threshold,
            "device": self.device,
            "model_name": self.model_name,
            "input_shape": list(self.input_shape),
            "train_loss": self.train_loss,
            "validation_metrics": self.validation_metrics,
            "split": self.split,
            "passed": self.passed,
            "failures": list(self.failures),
        }


def run_stage1_torch_smoke(config: Stage1TorchSmokeConfig) -> Stage1TorchSmokeReport:
    """Run a tiny Stage 1 CNN optimization smoke with optional PyTorch."""

    if config.batch_size <= 0:
        raise CaptureRecordError("batch_size must be positive")
    if config.train_steps <= 0:
        raise CaptureRecordError("train_steps must be positive")
    if config.learning_rate <= 0:
        raise CaptureRecordError("learning_rate must be positive")

    torch, nn = _require_torch()
    torch.manual_seed(config.seed)

    index = load_stage1_dataset_index(config.dataset_dirs)
    batch = next(_iter_one_stage1_batch(index, batch_size=config.batch_size), None)
    if batch is None:
        raise CaptureRecordError("at least one Stage 1 batch is required")

    selected_device = _select_device(torch, config.device)
    model = _build_stage1_smoke_model(nn, in_channels=index.frame_stack_length)
    model.to(selected_device)
    model.train()

    inputs = _batch_inputs(torch, batch, device=selected_device)
    targets = _batch_targets(torch, batch, device=selected_device)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            [index.released_count / index.held_count],
            dtype=torch.float32,
            device=selected_device,
        )
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    parameters_before = _flatten_parameters(torch, model).detach().clone()
    with torch.no_grad():
        initial_loss_tensor = criterion(model(inputs).squeeze(1), targets)
    initial_loss = float(initial_loss_tensor.item())

    final_loss = initial_loss
    for _ in range(config.train_steps):
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs).squeeze(1)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.item())

    parameters_after = _flatten_parameters(torch, model).detach()
    parameter_delta_l1 = float(
        torch.sum(torch.abs(parameters_after - parameters_before)).item()
    )
    failures = _smoke_failures(
        torch,
        initial_loss=initial_loss,
        final_loss=final_loss,
        parameter_delta_l1=parameter_delta_l1,
    )

    return Stage1TorchSmokeReport(
        schema_version=STAGE1_TORCH_SMOKE_SCHEMA_VERSION,
        dataset_count=len(index.dataset_dirs),
        sample_count=index.sample_count,
        held_count=index.held_count,
        released_count=index.released_count,
        batch_size=batch.batch_size,
        train_steps=config.train_steps,
        device=str(selected_device),
        model_name="stage1-smoke-cnn",
        input_shape=batch.shape,
        initial_loss=initial_loss,
        final_loss=final_loss,
        parameter_delta_l1=parameter_delta_l1,
        failures=tuple(failures),
    )


def train_stage1_policy(config: Stage1TrainingConfig) -> Stage1TrainingReport:
    """Train and persist the Stage 1 reactive CNN behavior-cloning baseline."""

    _validate_training_config(config)
    torch, nn = _require_torch()
    torch.manual_seed(config.seed)

    index = load_stage1_dataset_index(config.dataset_dirs)
    split = _split_samples_by_dataset(
        index,
        validation_fraction=config.validation_fraction,
        seed=config.seed,
    )
    selected_device = _select_device(torch, config.device)
    model = build_stage1_model(nn, in_channels=index.frame_stack_length)
    model.to(selected_device)
    model.train()

    pos_weight = index.released_count / index.held_count
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            [pos_weight],
            dtype=torch.float32,
            device=selected_device,
        )
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    output_dir = config.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    training_log_path = output_dir / "training_log.jsonl"
    train_loss = 0.0
    with training_log_path.open("w", encoding="utf-8") as training_log:
        progress = _make_progress_logger(training_log.write)
        for epoch_index in range(config.epochs):
            epoch_samples = list(split["train_samples"])
            random.Random(config.seed + epoch_index).shuffle(epoch_samples)
            train_loss = _train_one_epoch(
                torch,
                model,
                criterion,
                optimizer,
                epoch_samples,
                index=index,
                batch_size=config.batch_size,
                device=selected_device,
                max_batches=config.max_train_batches,
                epoch=epoch_index + 1,
                run_id=config.run_id,
                log_every_batches=config.log_every_batches,
                progress=progress,
            )
            log_row = {
                "schema_version": STAGE1_TRAINING_SCHEMA_VERSION,
                "run_id": config.run_id,
                "phase": "train_epoch",
                "epoch": epoch_index + 1,
                "train_loss": train_loss,
            }
            progress(log_row)

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
            "schema_version": STAGE1_TRAINING_SCHEMA_VERSION,
            "model_name": "stage1-reactive-cnn",
            "model_state_dict": model.state_dict(),
            "metadata": metadata,
        },
        checkpoint_path,
    )
    validation_samples = list(split["validation_samples"])
    random.Random(config.seed).shuffle(validation_samples)
    validation_metrics = _evaluate_stage1_samples(
        torch,
        model,
        criterion,
        validation_samples,
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
        "schema_version": STAGE1_TRAINING_SCHEMA_VERSION,
        "dataset_dirs": list(index.dataset_dirs),
        "dataset_count": len(index.dataset_dirs),
        "sample_count": index.sample_count,
        "held_count": index.held_count,
        "released_count": index.released_count,
        "observation_width": index.observation_width,
        "observation_height": index.observation_height,
        "observation_dtype": index.observation_dtype,
        "frame_stack_length": index.frame_stack_length,
        "frame_stack_layout": index.frame_stack_layout,
        "delta_sys": index.delta_sys,
    }
    report = Stage1TrainingReport(
        schema_version=STAGE1_TRAINING_SCHEMA_VERSION,
        run_id=config.run_id,
        output_dir=str(output_dir),
        checkpoint_path=str(checkpoint_path),
        dataset_count=len(index.dataset_dirs),
        sample_count=index.sample_count,
        train_sample_count=len(split["train_samples"]),
        validation_sample_count=len(split["validation_samples"]),
        held_count=index.held_count,
        released_count=index.released_count,
        epochs=config.epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        threshold=config.threshold,
        device=str(selected_device),
        model_name="stage1-reactive-cnn",
        input_shape=index.frame_stack_shape,
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


def write_stage1_torch_smoke_report(
    config: Stage1TorchSmokeConfig,
    output_path: Path,
) -> Stage1TorchSmokeReport:
    """Run a Stage 1 torch smoke and write the report JSON."""

    report = run_stage1_torch_smoke(config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report.to_json_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def write_stage1_training_report(
    config: Stage1TrainingConfig,
) -> Stage1TrainingReport:
    """Train Stage 1 and persist its standard artifact set."""

    return train_stage1_policy(config)


def _iter_one_stage1_batch(index: Any, *, batch_size: int) -> Any:
    from voxter.training.stage1_data import iter_stage1_batches

    return iter_stage1_batches(index, batch_size=batch_size, max_batches=1)


def _require_torch() -> tuple[ModuleType, Any]:
    try:
        torch = importlib.import_module("torch")
    except ModuleNotFoundError as exc:
        raise CaptureRecordError(
            "PyTorch is required for Stage 1 optimization smoke. "
            'Install the training extra with `python -m pip install -e ".[train]"` '
            "on a Python version supported by PyTorch."
        ) from exc
    nn = torch.nn
    return torch, nn


def _select_device(torch: ModuleType, requested: str) -> Any:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise CaptureRecordError("CUDA was requested but torch.cuda is unavailable")
    if requested not in {"cpu", "cuda"}:
        raise CaptureRecordError("device must be auto, cpu, or cuda")
    return torch.device(requested)


def _build_stage1_smoke_model(nn: Any, *, in_channels: int) -> Any:
    return nn.Sequential(
        nn.Conv2d(in_channels, 8, kernel_size=5, stride=4, padding=2),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(8, 1),
    )


def build_stage1_model(nn: Any, *, in_channels: int) -> Any:
    """Build the Stage 1 reactive CNN architecture used by training/inference."""

    return nn.Sequential(
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
        nn.Linear(64, 1),
    )


def _batch_inputs(torch: ModuleType, batch: Stage1Batch, *, device: Any) -> Any:
    payload = bytearray().join(batch.frame_stacks)
    tensor = torch.frombuffer(payload, dtype=torch.uint8)
    tensor = tensor.reshape(batch.shape).to(device=device, dtype=torch.float32)
    return tensor / 255.0


def _batch_targets(torch: ModuleType, batch: Stage1Batch, *, device: Any) -> Any:
    return torch.tensor(batch.labels, dtype=torch.float32, device=device)


def _flatten_parameters(torch: ModuleType, model: Any) -> Any:
    return torch.cat([parameter.detach().flatten() for parameter in model.parameters()])


def _validate_training_config(config: Stage1TrainingConfig) -> None:
    if config.epochs <= 0:
        raise CaptureRecordError("epochs must be positive")
    if config.batch_size <= 0:
        raise CaptureRecordError("batch_size must be positive")
    if config.learning_rate <= 0:
        raise CaptureRecordError("learning_rate must be positive")
    if not 0 < config.validation_fraction < 1:
        raise CaptureRecordError("validation_fraction must be between 0 and 1")
    if not 0 < config.threshold < 1:
        raise CaptureRecordError("threshold must be between 0 and 1")
    if config.max_train_batches is not None and config.max_train_batches <= 0:
        raise CaptureRecordError("max_train_batches must be positive when provided")
    if config.max_validation_batches is not None and config.max_validation_batches <= 0:
        raise CaptureRecordError(
            "max_validation_batches must be positive when provided"
        )
    if config.log_every_batches is not None and config.log_every_batches <= 0:
        raise CaptureRecordError("log_every_batches must be positive when provided")


def _split_samples_by_dataset(
    index: Stage1DatasetIndex,
    *,
    validation_fraction: float,
    seed: int,
) -> dict[str, Any]:
    dataset_dirs = sorted(index.dataset_dirs)
    if len(dataset_dirs) < 2:
        raise CaptureRecordError(
            "Stage 1 training requires at least two dataset directories for a "
            "non-frame-leaking train/validation split"
        )
    shuffled_dirs = list(dataset_dirs)
    random.Random(seed).shuffle(shuffled_dirs)
    validation_count = max(1, round(len(shuffled_dirs) * validation_fraction))
    validation_count = min(validation_count, len(shuffled_dirs) - 1)
    validation_dirs = set(shuffled_dirs[:validation_count])
    train_dirs = set(shuffled_dirs[validation_count:])
    train_samples = tuple(
        sample for sample in index.samples if sample.dataset_dir in train_dirs
    )
    validation_samples = tuple(
        sample for sample in index.samples if sample.dataset_dir in validation_dirs
    )
    if not train_samples or not validation_samples:
        raise CaptureRecordError("train and validation splits must both be non-empty")
    return {
        "train_dataset_dirs": sorted(train_dirs),
        "validation_dataset_dirs": sorted(validation_dirs),
        "train_samples": train_samples,
        "validation_samples": validation_samples,
    }


def _iter_sample_batches(
    samples: Sequence[Stage1SampleRef],
    *,
    index: Stage1DatasetIndex,
    batch_size: int,
    max_batches: int | None = None,
) -> Iterable[Stage1Batch]:
    for batch_index, start in enumerate(range(0, len(samples), batch_size)):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch_samples = tuple(samples[start : start + batch_size])
        frame_stacks = tuple(
            Path(sample.frame_stack_path).read_bytes() for sample in batch_samples
        )
        labels = tuple(sample.action_held for sample in batch_samples)
        yield Stage1Batch(
            samples=batch_samples,
            frame_stacks=frame_stacks,
            labels=labels,
            shape=(
                len(batch_samples),
                index.frame_stack_length,
                index.observation_height,
                index.observation_width,
            ),
            dtype=index.observation_dtype,
            layout=index.frame_stack_layout,
        )


def _train_one_epoch(
    torch: ModuleType,
    model: Any,
    criterion: Any,
    optimizer: Any,
    samples: Sequence[Stage1SampleRef],
    *,
    index: Stage1DatasetIndex,
    batch_size: int,
    device: Any,
    max_batches: int | None,
    epoch: int,
    run_id: str,
    log_every_batches: int | None,
    progress: Callable[[dict[str, object]], None],
) -> float:
    model.train()
    loss_total = 0.0
    sample_total = 0
    for batch_index, batch in enumerate(
        _iter_sample_batches(
            samples,
            index=index,
            batch_size=batch_size,
            max_batches=max_batches,
        ),
        start=1,
    ):
        inputs = _batch_inputs(torch, batch, device=device)
        targets = _batch_targets(torch, batch, device=device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(inputs).squeeze(1), targets)
        loss.backward()
        optimizer.step()
        loss_total += float(loss.item()) * batch.batch_size
        sample_total += batch.batch_size
        if log_every_batches is not None and batch_index % log_every_batches == 0:
            progress(
                {
                    "schema_version": STAGE1_TRAINING_SCHEMA_VERSION,
                    "run_id": run_id,
                    "phase": "train_batch",
                    "epoch": epoch,
                    "batch": batch_index,
                    "sample_count": sample_total,
                    "mean_loss": loss_total / sample_total,
                }
            )
    if sample_total == 0:
        raise CaptureRecordError("at least one training batch is required")
    return loss_total / sample_total


def _evaluate_stage1_samples(
    torch: ModuleType,
    model: Any,
    criterion: Any,
    samples: Sequence[Stage1SampleRef],
    *,
    index: Stage1DatasetIndex,
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
    with torch.no_grad():
        for batch_index, batch in enumerate(
            _iter_sample_batches(
                samples,
                index=index,
                batch_size=batch_size,
                max_batches=max_batches,
            ),
            start=1,
        ):
            inputs = _batch_inputs(torch, batch, device=device)
            targets = _batch_targets(torch, batch, device=device)
            logits = model(inputs).squeeze(1)
            loss = criterion(logits, targets)
            probs = torch.sigmoid(logits).detach().cpu().tolist()
            loss_total += float(loss.item()) * batch.batch_size
            evaluated_samples.extend(batch.samples)
            labels.extend(batch.labels)
            probabilities.extend(float(prob) for prob in probs)
            predictions.extend(1 if float(prob) >= threshold else 0 for prob in probs)
            if log_every_batches is not None and batch_index % log_every_batches == 0:
                progress(
                    {
                        "schema_version": STAGE1_TRAINING_SCHEMA_VERSION,
                        "run_id": run_id,
                        "phase": "validation_batch",
                        "batch": batch_index,
                        "sample_count": len(labels),
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
    metrics["evaluated_sample_count"] = len(labels)
    metrics["mean_probability"] = sum(probabilities) / len(probabilities)
    return metrics


def compute_stage1_binary_metrics(
    *,
    labels: Sequence[int],
    predictions: Sequence[int],
    samples: Sequence[Stage1SampleRef],
) -> dict[str, object]:
    """Compute Stage 1 held-state and transition metrics."""

    if len(labels) != len(predictions) or len(labels) != len(samples):
        raise CaptureRecordError("labels, predictions, and samples must align")
    if not labels:
        raise CaptureRecordError("at least one prediction is required")
    pairs = tuple(zip(labels, predictions, strict=True))
    true_positive = sum(1 for label, pred in pairs if label == pred == 1)
    true_negative = sum(1 for label, pred in pairs if label == pred == 0)
    false_positive = sum(1 for label, pred in pairs if label == 0 and pred == 1)
    false_negative = sum(1 for label, pred in pairs if label == 1 and pred == 0)
    held_precision = _safe_divide(true_positive, true_positive + false_positive)
    held_recall = _safe_divide(true_positive, true_positive + false_negative)
    held_f1 = _safe_divide(
        2 * held_precision * held_recall,
        held_precision + held_recall,
    )
    release_recall = _safe_divide(true_negative, true_negative + false_positive)
    transition_metrics = _transition_metrics(labels, predictions, samples)
    return {
        "accuracy": _safe_divide(true_positive + true_negative, len(labels)),
        "balanced_accuracy": (held_recall + release_recall) / 2,
        "held_precision": held_precision,
        "held_recall": held_recall,
        "held_f1": held_f1,
        "released_recall": release_recall,
        "confusion_matrix": {
            "true_positive": true_positive,
            "true_negative": true_negative,
            "false_positive": false_positive,
            "false_negative": false_negative,
        },
        **transition_metrics,
    }


def _transition_metrics(
    labels: Sequence[int],
    predictions: Sequence[int],
    samples: Sequence[Stage1SampleRef],
) -> dict[str, object]:
    rows = sorted(
        zip(samples, labels, predictions, strict=True),
        key=lambda row: (row[0].run_id, row[0].frame_index),
    )
    label_presses = 0
    predicted_presses = 0
    matched_presses = 0
    label_releases = 0
    predicted_releases = 0
    matched_releases = 0
    previous_run: str | None = None
    previous_label: int | None = None
    previous_prediction: int | None = None
    for sample, label, prediction in rows:
        if sample.run_id != previous_run:
            previous_run = sample.run_id
            previous_label = label
            previous_prediction = prediction
            continue
        label_press = previous_label == 0 and label == 1
        predicted_press = previous_prediction == 0 and prediction == 1
        label_release = previous_label == 1 and label == 0
        predicted_release = previous_prediction == 1 and prediction == 0
        label_presses += int(label_press)
        predicted_presses += int(predicted_press)
        matched_presses += int(label_press and predicted_press)
        label_releases += int(label_release)
        predicted_releases += int(predicted_release)
        matched_releases += int(label_release and predicted_release)
        previous_label = label
        previous_prediction = prediction
    return {
        "press_precision": _safe_divide(matched_presses, predicted_presses),
        "press_recall": _safe_divide(matched_presses, label_presses),
        "release_precision": _safe_divide(matched_releases, predicted_releases),
        "release_recall": _safe_divide(matched_releases, label_releases),
        "transition_counts": {
            "label_presses": label_presses,
            "predicted_presses": predicted_presses,
            "matched_presses": matched_presses,
            "label_releases": label_releases,
            "predicted_releases": predicted_releases,
            "matched_releases": matched_releases,
        },
    }


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _make_progress_logger(
    write_log: Callable[[str], object],
) -> Callable[[dict[str, object]], None]:
    def progress(row: dict[str, object]) -> None:
        line = json.dumps(row, sort_keys=True)
        write_log(line + "\n")
        print(line, file=sys.stderr, flush=True)

    return progress


def _training_metadata(
    config: Stage1TrainingConfig,
    *,
    index: Stage1DatasetIndex,
    split: dict[str, Any],
    pos_weight: float,
    selected_device: str,
) -> dict[str, object]:
    return {
        "schema_version": STAGE1_TRAINING_SCHEMA_VERSION,
        "run_id": config.run_id,
        "code_version": _git_commit_or_unknown(),
        "dataset_dirs": list(index.dataset_dirs),
        "train_dataset_dirs": split["train_dataset_dirs"],
        "validation_dataset_dirs": split["validation_dataset_dirs"],
        "dataset_manifest_version": "stage1-manifest-v1",
        "preprocessing": {
            "observation_width": index.observation_width,
            "observation_height": index.observation_height,
            "observation_dtype": index.observation_dtype,
            "frame_stack_length": index.frame_stack_length,
            "frame_stack_layout": index.frame_stack_layout,
            "delta_sys": index.delta_sys,
        },
        "model": {
            "name": "stage1-reactive-cnn",
            "architecture": "cnn-binary-head",
        },
        "training": {
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "threshold": config.threshold,
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
    if validation_metrics.get("evaluated_sample_count", 0) == 0:
        failures.append("validation must evaluate at least one sample")
    return failures


def _smoke_failures(
    torch: ModuleType,
    *,
    initial_loss: float,
    final_loss: float,
    parameter_delta_l1: float,
) -> list[str]:
    failures: list[str] = []
    if not bool(torch.isfinite(torch.tensor(initial_loss))):
        failures.append("initial_loss must be finite")
    if not bool(torch.isfinite(torch.tensor(final_loss))):
        failures.append("final_loss must be finite")
    if parameter_delta_l1 <= 0:
        failures.append("optimization step must change at least one parameter")
    return failures
