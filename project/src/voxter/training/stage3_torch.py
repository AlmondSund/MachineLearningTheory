"""Optional PyTorch Stage 3 offline policy-gradient fine-tuning."""

from __future__ import annotations

import importlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from voxter.contracts import CaptureRecordError, coerce_action_state
from voxter.training.stage2_torch import (
    STAGE2_CNN_GRU_MODEL_NAME,
    STAGE2_MOBILENET_GRU_MODEL_NAME,
    STAGE2_TRAINING_SCHEMA_VERSION,
    build_stage2_model,
)
from voxter.training.stage3_rollout import (
    STAGE3_ROLLOUT_SCHEMA_VERSION,
    Stage3RolloutStep,
)

STAGE3_TRAINING_SCHEMA_VERSION = "stage3-training-v1"


@dataclass(frozen=True, slots=True)
class Stage3TrainingConfig:
    """Configuration for offline Stage 3 policy-gradient fine-tuning."""

    rollout_paths: tuple[Path, ...]
    checkpoint_path: Path
    output_dir: Path
    run_id: str = "stage3-local"
    epochs: int = 1
    learning_rate: float = 1e-5
    gamma: float = 0.99
    kl_weight: float = 0.05
    entropy_weight: float = 0.001
    normalize_advantages: bool = True
    train_visual_encoder: bool = False
    device: str = "auto"
    max_episodes: int | None = None
    max_episode_steps: int | None = None


@dataclass(frozen=True, slots=True)
class Stage3TrainingReport:
    """Machine-readable Stage 3 fine-tuning result."""

    schema_version: str
    run_id: str
    output_dir: str
    checkpoint_path: str
    source_checkpoint_path: str
    rollout_count: int
    episode_count: int
    step_count: int
    terminal_count: int
    total_reward: float
    mean_reward: float
    epochs: int
    learning_rate: float
    gamma: float
    kl_weight: float
    entropy_weight: float
    normalize_advantages: bool
    train_visual_encoder: bool
    device: str
    final_loss: float
    policy_loss: float
    kl_loss: float
    entropy: float
    failures: tuple[str, ...]

    @property
    def passed(self) -> bool:
        """Return whether training produced a usable checkpoint."""

        return not self.failures

    def to_json_dict(self) -> dict[str, object]:
        """Return a stable JSON-compatible training report."""

        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "output_dir": self.output_dir,
            "checkpoint_path": self.checkpoint_path,
            "source_checkpoint_path": self.source_checkpoint_path,
            "rollout_count": self.rollout_count,
            "episode_count": self.episode_count,
            "step_count": self.step_count,
            "terminal_count": self.terminal_count,
            "total_reward": self.total_reward,
            "mean_reward": self.mean_reward,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "kl_weight": self.kl_weight,
            "entropy_weight": self.entropy_weight,
            "normalize_advantages": self.normalize_advantages,
            "train_visual_encoder": self.train_visual_encoder,
            "device": self.device,
            "final_loss": self.final_loss,
            "policy_loss": self.policy_loss,
            "kl_loss": self.kl_loss,
            "entropy": self.entropy,
            "passed": self.passed,
            "failures": list(self.failures),
        }


@dataclass(frozen=True, slots=True)
class Stage3RolloutEpisode:
    """One ordered Stage 3 rollout episode."""

    episode_id: str
    steps: tuple[Stage3RolloutStep, ...]


def load_stage3_rollout_steps(paths: tuple[Path, ...]) -> list[Stage3RolloutStep]:
    """Load and validate Stage 3 rollout JSONL rows."""

    if not paths:
        raise CaptureRecordError("at least one rollout path is required")
    steps: list[Stage3RolloutStep] = []
    for path in paths:
        if not path.exists():
            raise CaptureRecordError(f"missing Stage 3 rollout path: {path}")
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise CaptureRecordError("Stage 3 rollout row must be a dictionary")
            steps.append(_rollout_step_from_json(row))
    _validate_rollout_steps(steps)
    return steps


def group_stage3_rollout_episodes(
    steps: list[Stage3RolloutStep],
) -> list[Stage3RolloutEpisode]:
    """Group ordered rollout steps into episodes."""

    grouped: dict[str, list[Stage3RolloutStep]] = defaultdict(list)
    for step in steps:
        grouped[step.episode_id].append(step)
    episodes = [
        Stage3RolloutEpisode(
            episode_id=episode_id,
            steps=tuple(sorted(items, key=lambda step: step.step_index)),
        )
        for episode_id, items in grouped.items()
    ]
    return sorted(episodes, key=lambda episode: episode.steps[0].timestamp)


def train_stage3_policy(config: Stage3TrainingConfig) -> Stage3TrainingReport:
    """Fine-tune a Stage 2 policy from offline Stage 3 rollout rows."""

    _validate_training_config(config)
    torch, nn = _require_torch()
    selected_device = _select_device(torch, config.device)
    checkpoint = _load_stage2_checkpoint(torch, config.checkpoint_path, selected_device)
    metadata = checkpoint["metadata"]
    preprocessing = _required_dict(metadata, "preprocessing")
    model_name = _required_str(checkpoint, "model_name")
    model = _model_from_checkpoint(
        torch,
        nn,
        checkpoint,
        selected_device=selected_device,
    )
    reference_model = _model_from_checkpoint(
        torch,
        nn,
        checkpoint,
        selected_device=selected_device,
    )
    reference_model.eval()
    for parameter in reference_model.parameters():
        parameter.requires_grad = False
    if not config.train_visual_encoder:
        _freeze_visual_layers(model)
    model.train()

    steps = load_stage3_rollout_steps(config.rollout_paths)
    episodes = group_stage3_rollout_episodes(steps)
    if config.max_episodes is not None:
        episodes = episodes[: config.max_episodes]
    if config.max_episode_steps is not None:
        episodes = _truncate_stage3_rollout_episodes(
            episodes,
            max_episode_steps=config.max_episode_steps,
        )
    if not episodes:
        raise CaptureRecordError("at least one Stage 3 episode is required")

    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config.learning_rate,
    )
    output_dir = config.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    training_log_path = output_dir / "training_log.jsonl"
    final_metrics: dict[str, float] = {
        "loss": 0.0,
        "policy_loss": 0.0,
        "kl_loss": 0.0,
        "entropy": 0.0,
    }
    with training_log_path.open("w", encoding="utf-8") as training_log:
        for epoch in range(1, config.epochs + 1):
            final_metrics = _train_stage3_epoch(
                torch,
                model,
                reference_model,
                optimizer,
                episodes,
                preprocessing=preprocessing,
                gamma=config.gamma,
                kl_weight=config.kl_weight,
                entropy_weight=config.entropy_weight,
                normalize_advantages=config.normalize_advantages,
                device=selected_device,
            )
            training_log.write(
                json.dumps(
                    {
                        "schema_version": STAGE3_TRAINING_SCHEMA_VERSION,
                        "run_id": config.run_id,
                        "epoch": epoch,
                        **final_metrics,
                    },
                    sort_keys=True,
                )
                + "\n"
            )

    checkpoint_path = output_dir / "checkpoint.pt"
    stage3_metadata = dict(metadata)
    stage3_metadata["stage3"] = {
        "schema_version": STAGE3_TRAINING_SCHEMA_VERSION,
        "source_checkpoint_path": str(config.checkpoint_path.resolve()),
        "rollout_paths": [str(path.resolve()) for path in config.rollout_paths],
        "training": {
            "epochs": config.epochs,
            "learning_rate": config.learning_rate,
            "gamma": config.gamma,
            "kl_weight": config.kl_weight,
            "entropy_weight": config.entropy_weight,
            "normalize_advantages": config.normalize_advantages,
            "train_visual_encoder": config.train_visual_encoder,
            "max_episodes": config.max_episodes,
            "max_episode_steps": config.max_episode_steps,
        },
    }
    torch.save(
        {
            "schema_version": STAGE2_TRAINING_SCHEMA_VERSION,
            "model_name": model_name,
            "model_state_dict": model.state_dict(),
            "metadata": stage3_metadata,
        },
        checkpoint_path,
    )

    total_reward = sum(step.reward for episode in episodes for step in episode.steps)
    step_count = sum(len(episode.steps) for episode in episodes)
    terminal_count = sum(
        1 for episode in episodes for step in episode.steps if step.done
    )
    report = Stage3TrainingReport(
        schema_version=STAGE3_TRAINING_SCHEMA_VERSION,
        run_id=config.run_id,
        output_dir=str(output_dir),
        checkpoint_path=str(checkpoint_path),
        source_checkpoint_path=str(config.checkpoint_path.resolve()),
        rollout_count=len(config.rollout_paths),
        episode_count=len(episodes),
        step_count=step_count,
        terminal_count=terminal_count,
        total_reward=total_reward,
        mean_reward=total_reward / step_count,
        epochs=config.epochs,
        learning_rate=config.learning_rate,
        gamma=config.gamma,
        kl_weight=config.kl_weight,
        entropy_weight=config.entropy_weight,
        normalize_advantages=config.normalize_advantages,
        train_visual_encoder=config.train_visual_encoder,
        device=str(selected_device),
        final_loss=final_metrics["loss"],
        policy_loss=final_metrics["policy_loss"],
        kl_loss=final_metrics["kl_loss"],
        entropy=final_metrics["entropy"],
        failures=tuple(_training_failures(final_metrics, terminal_count)),
    )
    (output_dir / "config.json").write_text(
        json.dumps(stage3_metadata["stage3"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(report.to_json_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def write_stage3_training_report(
    config: Stage3TrainingConfig,
) -> Stage3TrainingReport:
    """Fine-tune Stage 3 and persist its standard artifact set."""

    return train_stage3_policy(config)


def _truncate_stage3_rollout_episodes(
    episodes: list[Stage3RolloutEpisode],
    *,
    max_episode_steps: int,
) -> list[Stage3RolloutEpisode]:
    """Return episodes truncated to bound per-episode training memory."""

    return [
        Stage3RolloutEpisode(
            episode_id=episode.episode_id,
            steps=episode.steps[:max_episode_steps],
        )
        for episode in episodes
        if episode.steps
    ]


def discounted_returns(
    rewards: tuple[float, ...],
    *,
    gamma: float,
) -> tuple[float, ...]:
    """Compute discounted returns for one episode."""

    if not rewards:
        raise CaptureRecordError("at least one reward is required")
    running = 0.0
    returns: list[float] = []
    for reward in reversed(rewards):
        running = reward + gamma * running
        returns.append(running)
    return tuple(reversed(returns))


def _train_stage3_epoch(
    torch: ModuleType,
    model: Any,
    reference_model: Any,
    optimizer: Any,
    episodes: list[Stage3RolloutEpisode],
    *,
    preprocessing: dict[str, object],
    gamma: float,
    kl_weight: float,
    entropy_weight: float,
    normalize_advantages: bool,
    device: Any,
) -> dict[str, float]:
    loss_total = 0.0
    policy_loss_total = 0.0
    kl_loss_total = 0.0
    entropy_total = 0.0
    step_total = 0
    for episode in episodes:
        inputs = _episode_inputs(torch, episode, preprocessing, device=device)
        previous_actions = _episode_previous_actions(torch, episode, device=device)
        actions = _episode_actions(torch, episode, device=device)
        rewards = tuple(step.reward for step in episode.steps)
        returns = torch.tensor(
            discounted_returns(rewards, gamma=gamma),
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)
        advantages = _advantages(torch, returns, normalize=normalize_advantages)

        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs, previous_actions)
        with torch.no_grad():
            reference_logits = reference_model(inputs, previous_actions)
        distribution = torch.distributions.Bernoulli(logits=logits)
        reference_distribution = torch.distributions.Bernoulli(logits=reference_logits)
        log_probs = distribution.log_prob(actions)
        policy_loss = -(log_probs * advantages).mean()
        kl_loss = torch.distributions.kl_divergence(
            distribution,
            reference_distribution,
        ).mean()
        entropy = distribution.entropy().mean()
        loss = policy_loss + kl_weight * kl_loss - entropy_weight * entropy
        loss.backward()
        optimizer.step()

        step_count = len(episode.steps)
        loss_total += float(loss.item()) * step_count
        policy_loss_total += float(policy_loss.item()) * step_count
        kl_loss_total += float(kl_loss.item()) * step_count
        entropy_total += float(entropy.item()) * step_count
        step_total += step_count
    if step_total == 0:
        raise CaptureRecordError("at least one Stage 3 training step is required")
    return {
        "loss": loss_total / step_total,
        "policy_loss": policy_loss_total / step_total,
        "kl_loss": kl_loss_total / step_total,
        "entropy": entropy_total / step_total,
    }


def _episode_inputs(
    torch: ModuleType,
    episode: Stage3RolloutEpisode,
    preprocessing: dict[str, object],
    *,
    device: Any,
) -> Any:
    frame_stack_length = _required_int(preprocessing, "frame_stack_length")
    width = _required_int(preprocessing, "observation_width")
    height = _required_int(preprocessing, "observation_height")
    frames = [
        _read_step_observation(step, width=width, height=height)
        for step in episode.steps
    ]
    stacks: list[bytes] = []
    rolling: list[bytes] = []
    for frame in frames:
        if not rolling:
            rolling = [frame] * frame_stack_length
        else:
            rolling.append(frame)
            rolling = rolling[-frame_stack_length:]
        stacks.append(b"".join(rolling))
    payload = b"".join(stacks)
    tensor = torch.frombuffer(bytearray(payload), dtype=torch.uint8)
    tensor = tensor.reshape((1, len(stacks), frame_stack_length, height, width))
    return tensor.to(device=device, dtype=torch.float32) / 255.0


def _episode_previous_actions(
    torch: ModuleType,
    episode: Stage3RolloutEpisode,
    *,
    device: Any,
) -> Any:
    return torch.tensor(
        [[int(step.previous_action) for step in episode.steps]],
        dtype=torch.float32,
        device=device,
    )


def _episode_actions(
    torch: ModuleType,
    episode: Stage3RolloutEpisode,
    *,
    device: Any,
) -> Any:
    return torch.tensor(
        [[int(step.action) for step in episode.steps]],
        dtype=torch.float32,
        device=device,
    )


def _advantages(torch: ModuleType, returns: Any, *, normalize: bool) -> Any:
    if not normalize or returns.numel() < 2:
        return returns
    std = returns.std(unbiased=False)
    if float(std.item()) <= 1e-8:
        return returns - returns.mean()
    return (returns - returns.mean()) / (std + 1e-8)


def _read_step_observation(
    step: Stage3RolloutStep,
    *,
    width: int,
    height: int,
) -> bytes:
    if step.observation_path is None:
        raise CaptureRecordError("Stage 3 training requires observation_path rows")
    return _read_binary_pgm(Path(step.observation_path), width=width, height=height)


def _read_binary_pgm(path: Path, *, width: int, height: int) -> bytes:
    data = path.read_bytes()
    tokens: list[bytes] = []
    index = 0
    while len(tokens) < 4:
        while index < len(data) and data[index] in b" \t\r\n":
            index += 1
        if index >= len(data):
            raise CaptureRecordError("PGM header is incomplete")
        if data[index] == ord("#"):
            while index < len(data) and data[index] not in b"\r\n":
                index += 1
            continue
        start = index
        while index < len(data) and data[index] not in b" \t\r\n":
            index += 1
        tokens.append(data[start:index])
    if index < len(data) and data[index] in b" \t\r\n":
        index += 1
    if tokens[0] != b"P5":
        raise CaptureRecordError("Stage 3 observations must be binary PGM/P5")
    parsed_width = int(tokens[1])
    parsed_height = int(tokens[2])
    max_value = int(tokens[3])
    if parsed_width != width or parsed_height != height:
        raise CaptureRecordError("PGM shape does not match checkpoint preprocessing")
    if max_value != 255:
        raise CaptureRecordError("PGM max value must be 255")
    payload = data[index:]
    expected_size = width * height
    if len(payload) != expected_size:
        raise CaptureRecordError("PGM payload byte count does not match shape")
    return payload


def _rollout_step_from_json(row: dict[str, object]) -> Stage3RolloutStep:
    schema_version = row.get("schema_version")
    if schema_version != STAGE3_ROLLOUT_SCHEMA_VERSION:
        raise CaptureRecordError("unsupported Stage 3 rollout schema")
    return Stage3RolloutStep(
        run_id=_required_str(row, "run_id"),
        episode_id=_required_str(row, "episode_id"),
        step_index=_required_int(row, "step_index"),
        frame_index=_required_int(row, "frame_index"),
        timestamp=_required_float(row, "timestamp"),
        previous_action=coerce_action_state(_required_int(row, "previous_action")),
        action=coerce_action_state(_required_int(row, "action")),
        probability=_probability(row, "probability"),
        reward=_required_float(row, "reward"),
        done=_required_bool(row, "done"),
        terminal_type=_optional_str(row, "terminal_type"),
        active=_required_bool(row, "active"),
        deadline_missed=_required_bool(row, "deadline_missed"),
        observation_path=_optional_str(row, "observation_path"),
        held_probability=_optional_probability(row, "held_probability"),
        press_probability=_optional_probability(row, "press_probability"),
        release_probability=_optional_probability(row, "release_probability"),
        progress=_optional_float(row, "progress"),
        progress_delta=_optional_float(row, "progress_delta"),
    )


def _validate_rollout_steps(steps: list[Stage3RolloutStep]) -> None:
    if not steps:
        raise CaptureRecordError("at least one Stage 3 rollout step is required")
    previous_by_episode: dict[str, int] = {}
    for step in steps:
        previous_index = previous_by_episode.get(step.episode_id)
        if previous_index is not None and step.step_index <= previous_index:
            raise CaptureRecordError("Stage 3 rollout step_index values must increase")
        previous_by_episode[step.episode_id] = step.step_index
        if step.terminal_type is not None and step.terminal_type not in {
            "death",
            "reset",
            "completion",
        }:
            raise CaptureRecordError("unsupported Stage 3 terminal_type")


def _model_from_checkpoint(
    torch: ModuleType,
    nn: Any,
    checkpoint: dict[str, Any],
    *,
    selected_device: Any,
) -> Any:
    metadata = _required_dict(checkpoint, "metadata")
    preprocessing = _required_dict(metadata, "preprocessing")
    model_metadata = _required_dict(metadata, "model")
    model_name = _required_str(checkpoint, "model_name")
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
    return model


def _load_stage2_checkpoint(
    torch: ModuleType,
    checkpoint_path: Path,
    selected_device: Any,
) -> dict[str, Any]:
    checkpoint = torch.load(
        checkpoint_path,
        map_location=selected_device,
        weights_only=False,
    )
    if not isinstance(checkpoint, dict):
        raise CaptureRecordError("Stage 2 checkpoint must contain a dictionary")
    schema_version = checkpoint.get("schema_version")
    if schema_version != STAGE2_TRAINING_SCHEMA_VERSION:
        raise CaptureRecordError("unsupported source checkpoint schema")
    model_name = _required_str(checkpoint, "model_name")
    if model_name not in {STAGE2_CNN_GRU_MODEL_NAME, STAGE2_MOBILENET_GRU_MODEL_NAME}:
        raise CaptureRecordError("unsupported Stage 2 model name")
    return checkpoint


def _freeze_visual_layers(model: Any) -> None:
    for name, parameter in model.named_parameters():
        if not (
            name.startswith("head.")
            or name.startswith("press_head.")
            or name.startswith("release_head.")
            or name.startswith("gru.")
        ):
            parameter.requires_grad = False
    for module_name in ("stack_adapter", "encoder", "pool"):
        module = getattr(model, module_name, None)
        if module is not None:
            module.eval()


def _validate_training_config(config: Stage3TrainingConfig) -> None:
    if not config.rollout_paths:
        raise CaptureRecordError("at least one rollout path is required")
    if config.epochs <= 0:
        raise CaptureRecordError("epochs must be positive")
    if config.learning_rate <= 0:
        raise CaptureRecordError("learning_rate must be positive")
    if not 0.0 <= config.gamma <= 1.0:
        raise CaptureRecordError("gamma must be between 0 and 1")
    if config.kl_weight < 0:
        raise CaptureRecordError("kl_weight must be non-negative")
    if config.entropy_weight < 0:
        raise CaptureRecordError("entropy_weight must be non-negative")
    if config.max_episodes is not None and config.max_episodes <= 0:
        raise CaptureRecordError("max_episodes must be positive when provided")
    if config.max_episode_steps is not None and config.max_episode_steps <= 0:
        raise CaptureRecordError("max_episode_steps must be positive when provided")


def _training_failures(
    final_metrics: dict[str, float],
    terminal_count: int,
) -> list[str]:
    failures: list[str] = []
    for name, value in final_metrics.items():
        if not isinstance(value, int | float) or value != value:
            failures.append(f"{name} is not finite")
    if terminal_count == 0:
        failures.append("rollout has no terminal steps")
    return failures


def _require_torch() -> tuple[ModuleType, Any]:
    try:
        torch = importlib.import_module("torch")
    except ModuleNotFoundError as exc:
        raise CaptureRecordError(
            "PyTorch is required for Stage 3 training. "
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


def _required_dict(row: dict[str, Any], key: str) -> dict[str, Any]:
    value = row.get(key)
    if not isinstance(value, dict):
        raise CaptureRecordError(f"{key} must be a dictionary")
    return value


def _required_str(row: dict[str, object], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value:
        raise CaptureRecordError(f"{key} must be a non-empty string")
    return value


def _optional_str(row: dict[str, object], key: str) -> str | None:
    value = row.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise CaptureRecordError(f"{key} must be None or a non-empty string")
    return value


def _required_int(row: dict[str, object], key: str) -> int:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise CaptureRecordError(f"{key} must be an integer")
    return value


def _required_float(row: dict[str, object], key: str) -> float:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise CaptureRecordError(f"{key} must be a finite number")
    float_value = float(value)
    if float_value != float_value:
        raise CaptureRecordError(f"{key} must be finite")
    return float_value


def _optional_float(row: dict[str, object], key: str) -> float | None:
    value = row.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise CaptureRecordError(f"{key} must be a finite number when provided")
    float_value = float(value)
    if float_value != float_value:
        raise CaptureRecordError(f"{key} must be finite when provided")
    return float_value


def _required_bool(row: dict[str, object], key: str) -> bool:
    value = row.get(key)
    if not isinstance(value, bool):
        raise CaptureRecordError(f"{key} must be a boolean")
    return value


def _probability(row: dict[str, object], key: str) -> float:
    value = _required_float(row, key)
    _validate_probability(value, key)
    return value


def _optional_probability(row: dict[str, object], key: str) -> float | None:
    value = _optional_float(row, key)
    if value is not None:
        _validate_probability(value, key)
    return value


def _validate_probability(value: float, name: str) -> None:
    if not 0.0 <= value <= 1.0:
        raise CaptureRecordError(f"{name} must be between 0 and 1")


def _optional_bool(row: dict[str, object], key: str, *, default: bool) -> bool:
    value = row.get(key, default)
    if not isinstance(value, bool):
        raise CaptureRecordError(f"{key} must be boolean")
    return value
