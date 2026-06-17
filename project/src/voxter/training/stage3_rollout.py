"""Stage 3 rollout and reward contracts.

This module converts live-control logs into reward-bearing rollout rows for
reinforcement fine-tuning. It is intentionally dependency-light and pure apart
from the explicit file read/write helpers at the boundary.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from math import isfinite
from pathlib import Path

from voxter.capture.events import InputEventKind, RawTerminalEvent
from voxter.contracts import ActionState, CaptureRecordError, coerce_action_state

STAGE3_ROLLOUT_SCHEMA_VERSION = "stage3-rollout-v1"

_TERMINAL_TYPES = {"death", "reset", "completion"}


@dataclass(frozen=True, slots=True)
class Stage3RewardConfig:
    """Reward weights for fixed-policy Stage 3 rollout construction."""

    alive_reward: float = 0.001
    progress_reward_scale: float = 1.0
    death_penalty: float = 1.0
    reset_penalty: float = 0.25
    completion_reward: float = 2.0
    instability_penalty: float = 0.01
    reward_active_only: bool = False
    allow_negative_progress: bool = False

    def __post_init__(self) -> None:
        for name, value in self.to_json_dict().items():
            if isinstance(value, float) and not isfinite(value):
                raise CaptureRecordError(f"{name} must be finite")

    def to_json_dict(self) -> dict[str, object]:
        """Return stable JSON-compatible reward configuration."""

        return {
            "alive_reward": self.alive_reward,
            "progress_reward_scale": self.progress_reward_scale,
            "death_penalty": self.death_penalty,
            "reset_penalty": self.reset_penalty,
            "completion_reward": self.completion_reward,
            "instability_penalty": self.instability_penalty,
            "reward_active_only": self.reward_active_only,
            "allow_negative_progress": self.allow_negative_progress,
        }


@dataclass(frozen=True, slots=True)
class Stage3LiveFrame:
    """A live-control frame row used as the source for Stage 3 rollouts."""

    frame_index: int
    timestamp: float
    previous_action: ActionState
    action: ActionState
    probability: float
    deadline_missed: bool
    held_probability: float | None = None
    press_probability: float | None = None
    release_probability: float | None = None
    progress: float | None = None
    observation_path: str | None = None


@dataclass(frozen=True, slots=True)
class Stage3RolloutStep:
    """One reward-bearing Stage 3 rollout step."""

    run_id: str
    episode_id: str
    step_index: int
    frame_index: int
    timestamp: float
    previous_action: ActionState
    action: ActionState
    probability: float
    reward: float
    done: bool
    terminal_type: str | None
    active: bool
    deadline_missed: bool
    observation_path: str | None = None
    held_probability: float | None = None
    press_probability: float | None = None
    release_probability: float | None = None
    progress: float | None = None
    progress_delta: float | None = None

    def to_json_dict(self) -> dict[str, object]:
        """Return a stable JSON-compatible rollout row."""

        return {
            "schema_version": STAGE3_ROLLOUT_SCHEMA_VERSION,
            "run_id": self.run_id,
            "episode_id": self.episode_id,
            "step_index": self.step_index,
            "frame_index": self.frame_index,
            "timestamp": self.timestamp,
            "previous_action": int(self.previous_action),
            "action": int(self.action),
            "probability": self.probability,
            "held_probability": self.held_probability,
            "press_probability": self.press_probability,
            "release_probability": self.release_probability,
            "reward": self.reward,
            "done": self.done,
            "terminal_type": self.terminal_type,
            "active": self.active,
            "deadline_missed": self.deadline_missed,
            "observation_path": self.observation_path,
            "progress": self.progress,
            "progress_delta": self.progress_delta,
        }


@dataclass(frozen=True, slots=True)
class Stage3RolloutReport:
    """Summary of a Stage 3 rollout artifact."""

    run_id: str
    step_count: int
    episode_count: int
    terminal_count: int
    death_count: int
    reset_count: int
    completion_count: int
    total_reward: float
    mean_reward: float
    action_transition_count: int
    missed_deadline_count: int
    active_step_count: int
    reward_config: Stage3RewardConfig

    @property
    def passed(self) -> bool:
        """Return whether the rollout artifact is structurally usable."""

        return self.step_count > 0 and self.episode_count > 0

    def to_json_dict(self) -> dict[str, object]:
        """Return a stable JSON-compatible report."""

        return {
            "schema_version": STAGE3_ROLLOUT_SCHEMA_VERSION,
            "run_id": self.run_id,
            "step_count": self.step_count,
            "episode_count": self.episode_count,
            "terminal_count": self.terminal_count,
            "death_count": self.death_count,
            "reset_count": self.reset_count,
            "completion_count": self.completion_count,
            "total_reward": self.total_reward,
            "mean_reward": self.mean_reward,
            "action_transition_count": self.action_transition_count,
            "missed_deadline_count": self.missed_deadline_count,
            "active_step_count": self.active_step_count,
            "reward_config": self.reward_config.to_json_dict(),
            "passed": self.passed,
        }


def build_stage3_rollout(
    frames: Iterable[Stage3LiveFrame],
    *,
    terminal_events: Iterable[RawTerminalEvent] = (),
    run_id: str,
    reward_config: Stage3RewardConfig | None = None,
) -> list[Stage3RolloutStep]:
    """Build reward-bearing rollout steps from live-control frames."""

    frame_list = sorted(list(frames), key=lambda frame: frame.frame_index)
    if not frame_list:
        raise CaptureRecordError("at least one live-control frame is required")
    _validate_live_frames(frame_list)
    terminal_list = sorted(list(terminal_events), key=lambda event: event.timestamp)
    _validate_stage3_terminal_events(terminal_list)

    config = reward_config or Stage3RewardConfig()
    steps: list[Stage3RolloutStep] = []
    terminal_index = 0
    active = not config.reward_active_only
    previous_progress: float | None = None
    episode_number = 0
    episode_step_index = 0

    for frame in frame_list:
        terminal_type: str | None = None
        done = False
        while (
            terminal_index < len(terminal_list)
            and terminal_list[terminal_index].timestamp <= frame.timestamp
        ):
            event = terminal_list[terminal_index]
            terminal_index += 1
            if event.terminal_type == "active_start":
                active = True
                continue
            if event.terminal_type in _TERMINAL_TYPES:
                terminal_type = event.terminal_type
                done = True
                active = False if config.reward_active_only else active
                break

        progress_delta = _progress_delta(
            frame.progress,
            previous_progress=previous_progress,
            allow_negative=config.allow_negative_progress,
        )
        if frame.progress is not None:
            previous_progress = frame.progress

        reward = compute_stage3_reward(
            frame,
            progress_delta=progress_delta,
            terminal_type=terminal_type,
            active=active,
            config=config,
        )
        steps.append(
            Stage3RolloutStep(
                run_id=run_id,
                episode_id=f"{run_id}:episode-{episode_number:04d}",
                step_index=episode_step_index,
                frame_index=frame.frame_index,
                timestamp=frame.timestamp,
                previous_action=frame.previous_action,
                action=frame.action,
                probability=frame.probability,
                reward=reward,
                done=done,
                terminal_type=terminal_type,
                active=active,
                deadline_missed=frame.deadline_missed,
                observation_path=frame.observation_path,
                held_probability=frame.held_probability,
                press_probability=frame.press_probability,
                release_probability=frame.release_probability,
                progress=frame.progress,
                progress_delta=progress_delta,
            )
        )
        if done:
            episode_number += 1
            episode_step_index = 0
            previous_progress = None
        else:
            episode_step_index += 1

    return steps


def compute_stage3_reward(
    frame: Stage3LiveFrame,
    *,
    progress_delta: float | None,
    terminal_type: str | None,
    active: bool,
    config: Stage3RewardConfig,
) -> float:
    """Compute the scalar reward for one Stage 3 rollout step."""

    reward = 0.0
    if active:
        reward += config.alive_reward
    if progress_delta is not None:
        reward += config.progress_reward_scale * progress_delta
    if frame.action != frame.previous_action:
        reward -= config.instability_penalty
    if terminal_type == "death":
        reward -= config.death_penalty
    elif terminal_type == "reset":
        reward -= config.reset_penalty
    elif terminal_type == "completion":
        reward += config.completion_reward
    return reward


def summarize_stage3_rollout(
    steps: Iterable[Stage3RolloutStep],
    *,
    reward_config: Stage3RewardConfig,
) -> Stage3RolloutReport:
    """Summarize Stage 3 rollout steps."""

    step_list = list(steps)
    if not step_list:
        raise CaptureRecordError("at least one rollout step is required")
    run_ids = {step.run_id for step in step_list}
    if len(run_ids) != 1:
        raise CaptureRecordError("rollout steps must belong to one run_id")
    terminal_types = [step.terminal_type for step in step_list if step.done]
    total_reward = sum(step.reward for step in step_list)
    return Stage3RolloutReport(
        run_id=step_list[0].run_id,
        step_count=len(step_list),
        episode_count=len({step.episode_id for step in step_list}),
        terminal_count=len(terminal_types),
        death_count=sum(1 for value in terminal_types if value == "death"),
        reset_count=sum(1 for value in terminal_types if value == "reset"),
        completion_count=sum(1 for value in terminal_types if value == "completion"),
        total_reward=total_reward,
        mean_reward=total_reward / len(step_list),
        action_transition_count=sum(
            1 for step in step_list if step.action != step.previous_action
        ),
        missed_deadline_count=sum(1 for step in step_list if step.deadline_missed),
        active_step_count=sum(1 for step in step_list if step.active),
        reward_config=reward_config,
    )


def write_stage3_rollout_artifacts(
    live_control_dir: Path,
    *,
    output_dir: Path | None = None,
    reward_config: Stage3RewardConfig | None = None,
    run_id: str | None = None,
) -> Stage3RolloutReport:
    """Read a live-control directory and write Stage 3 rollout artifacts."""

    source_dir = live_control_dir.resolve()
    destination = output_dir.resolve() if output_dir is not None else source_dir
    destination.mkdir(parents=True, exist_ok=True)
    config = reward_config or Stage3RewardConfig()
    effective_run_id = run_id or source_dir.name
    frames = load_stage3_live_frames(source_dir)
    terminal_events = load_stage3_terminal_events(source_dir)
    steps = build_stage3_rollout(
        frames,
        terminal_events=terminal_events,
        run_id=effective_run_id,
        reward_config=config,
    )
    report = summarize_stage3_rollout(steps, reward_config=config)

    rollout_path = destination / "stage3_rollout.jsonl"
    summary_path = destination / "stage3_rollout_summary.json"
    rollout_path.write_text(
        "".join(
            json.dumps(step.to_json_dict(), sort_keys=True) + "\n" for step in steps
        ),
        encoding="utf-8",
    )
    summary_payload = report.to_json_dict()
    summary_payload["source_live_control_dir"] = str(source_dir)
    summary_payload["rollout_path"] = str(rollout_path)
    summary_path.write_text(
        json.dumps(summary_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def load_stage3_live_frames(live_control_dir: Path) -> list[Stage3LiveFrame]:
    """Load Stage 3 source frames from a live-control output directory."""

    log_path = live_control_dir / "control_log.jsonl"
    if not log_path.exists():
        raise CaptureRecordError(f"missing live-control log: {log_path}")
    rows = [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    frames = [
        _live_frame_from_json(row, live_control_dir=live_control_dir) for row in rows
    ]
    _validate_live_frames(frames)
    return frames


def load_stage3_terminal_events(live_control_dir: Path) -> list[RawTerminalEvent]:
    """Load terminal marker events from a live-control output directory if present."""

    terminal_path = live_control_dir / "terminal_events.jsonl"
    if not terminal_path.exists():
        return []
    events = [
        _terminal_event_from_json(json.loads(line))
        for line in terminal_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    _validate_stage3_terminal_events(events)
    return events


def _live_frame_from_json(
    row: dict[str, object],
    *,
    live_control_dir: Path,
) -> Stage3LiveFrame:
    frame_index = _required_int(row, "frame_index")
    preview_path = live_control_dir / "preview_frames" / f"{frame_index:06d}.pgm"
    observation_path = str(preview_path) if preview_path.exists() else None
    return Stage3LiveFrame(
        frame_index=frame_index,
        timestamp=_required_float(row, "timestamp"),
        previous_action=coerce_action_state(_required_int(row, "previous_action")),
        action=coerce_action_state(_required_int(row, "action")),
        probability=_probability(row, "probability"),
        deadline_missed=_required_bool(row, "deadline_missed"),
        held_probability=_optional_probability(row, "held_probability"),
        press_probability=_optional_probability(row, "press_probability"),
        release_probability=_optional_probability(row, "release_probability"),
        progress=_optional_float(row, "progress"),
        observation_path=observation_path,
    )


def _terminal_event_from_json(row: dict[str, object]) -> RawTerminalEvent:
    kind = row.get("kind")
    if kind != InputEventKind.PRESS.value:
        raise CaptureRecordError("terminal event kind must be press")
    terminal_type = row.get("terminal_type")
    if not isinstance(terminal_type, str):
        raise CaptureRecordError("terminal event terminal_type must be a string")
    return RawTerminalEvent(
        run_id=_required_string(row, "run_id"),
        attempt_id=_optional_string(row, "attempt_id"),
        timestamp=_required_float(row, "timestamp"),
        device=_required_string(row, "device"),
        key_code=_required_int(row, "key_code"),
        kind=InputEventKind.PRESS,
        terminal_type=terminal_type,
    )


def _validate_live_frames(frames: list[Stage3LiveFrame]) -> None:
    if not frames:
        raise CaptureRecordError("at least one live-control frame is required")
    seen: set[int] = set()
    previous_frame_index: int | None = None
    previous_timestamp: float | None = None
    for frame in frames:
        if frame.frame_index in seen:
            raise CaptureRecordError("duplicate live-control frame_index")
        seen.add(frame.frame_index)
        if (
            previous_frame_index is not None
            and frame.frame_index <= previous_frame_index
        ):
            raise CaptureRecordError("live-control frame_index values must increase")
        if previous_timestamp is not None and frame.timestamp < previous_timestamp:
            raise CaptureRecordError("live-control timestamps must be monotonic")
        _validate_probability(frame.probability, "probability")
        for name, value in {
            "held_probability": frame.held_probability,
            "press_probability": frame.press_probability,
            "release_probability": frame.release_probability,
        }.items():
            if value is not None:
                _validate_probability(value, name)
        if frame.progress is not None and not isfinite(frame.progress):
            raise CaptureRecordError("progress must be finite when provided")
        previous_frame_index = frame.frame_index
        previous_timestamp = frame.timestamp


def _validate_stage3_terminal_events(events: list[RawTerminalEvent]) -> None:
    previous_timestamp: float | None = None
    for event in events:
        if event.terminal_type not in {*_TERMINAL_TYPES, "active_start"}:
            raise CaptureRecordError(
                "terminal_type must be death, reset, completion, or active_start"
            )
        if previous_timestamp is not None and event.timestamp < previous_timestamp:
            raise CaptureRecordError("terminal events must be monotonic")
        previous_timestamp = event.timestamp


def _progress_delta(
    progress: float | None,
    *,
    previous_progress: float | None,
    allow_negative: bool,
) -> float | None:
    if progress is None or previous_progress is None:
        return None
    delta = progress - previous_progress
    return delta if allow_negative else max(0.0, delta)


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
    if not isfinite(float_value):
        raise CaptureRecordError(f"{key} must be finite")
    return float_value


def _optional_float(row: dict[str, object], key: str) -> float | None:
    value = row.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise CaptureRecordError(f"{key} must be a finite number when provided")
    float_value = float(value)
    if not isfinite(float_value):
        raise CaptureRecordError(f"{key} must be finite when provided")
    return float_value


def _required_bool(row: dict[str, object], key: str) -> bool:
    value = row.get(key)
    if not isinstance(value, bool):
        raise CaptureRecordError(f"{key} must be a boolean")
    return value


def _required_string(row: dict[str, object], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value:
        raise CaptureRecordError(f"{key} must be a non-empty string")
    return value


def _optional_string(row: dict[str, object], key: str) -> str | None:
    value = row.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise CaptureRecordError(f"{key} must be None or a non-empty string")
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
