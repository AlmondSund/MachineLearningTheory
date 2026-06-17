"""Runtime-shaped latency benchmark loop."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from math import ceil, isfinite
from typing import TypeVar

from voxter.contracts import ActionState, CaptureRecordError

FrameT = TypeVar("FrameT")
ObservationT = TypeVar("ObservationT")

RUNTIME_BENCHMARK_SCHEMA_VERSION = "runtime-benchmark-v1"
FRAME_DRIVEN_RUNTIME_BENCHMARK_SCHEMA_VERSION = "frame-driven-runtime-benchmark-v1"


@dataclass(frozen=True, slots=True)
class RuntimeBenchmarkConfig:
    """Configuration for a runtime-shaped benchmark run."""

    duration_s: float
    target_hz: float
    threshold: float = 0.5
    max_cycles: int | None = None
    warmup_cycles: int = 0


@dataclass(frozen=True, slots=True)
class RuntimeCycleTiming:
    """Per-cycle timing for the runtime-shaped loop."""

    cycle_index: int
    scheduled_lag_ms: float
    capture_ms: float
    preprocess_ms: float
    inference_ms: float
    control_ms: float
    total_ms: float
    probability: float
    action: ActionState
    deadline_missed: bool

    def to_json_dict(self) -> dict[str, object]:
        """Return a stable JSON-compatible representation."""

        return {
            "cycle_index": self.cycle_index,
            "scheduled_lag_ms": self.scheduled_lag_ms,
            "capture_ms": self.capture_ms,
            "preprocess_ms": self.preprocess_ms,
            "inference_ms": self.inference_ms,
            "control_ms": self.control_ms,
            "total_ms": self.total_ms,
            "probability": self.probability,
            "action": int(self.action),
            "deadline_missed": self.deadline_missed,
        }


@dataclass(frozen=True, slots=True)
class FrameDrivenCycleTiming:
    """Per-frame timing for a frame-driven runtime loop."""

    frame_index: int
    frame_wait_ms: float
    preprocess_ms: float
    inference_ms: float
    control_ms: float
    decision_ms: float
    probability: float
    action: ActionState
    deadline_missed: bool

    def to_json_dict(self) -> dict[str, object]:
        """Return a stable JSON-compatible representation."""

        return {
            "frame_index": self.frame_index,
            "frame_wait_ms": self.frame_wait_ms,
            "preprocess_ms": self.preprocess_ms,
            "inference_ms": self.inference_ms,
            "control_ms": self.control_ms,
            "decision_ms": self.decision_ms,
            "probability": self.probability,
            "action": int(self.action),
            "deadline_missed": self.deadline_missed,
        }


@dataclass(frozen=True, slots=True)
class StageTimingSummary:
    """Aggregate timing statistics for one benchmark stage."""

    count: int
    min_ms: float | None
    mean_ms: float | None
    p50_ms: float | None
    p95_ms: float | None
    p99_ms: float | None
    max_ms: float | None

    def to_json_dict(self) -> dict[str, object]:
        """Return a stable JSON-compatible representation."""

        return {
            "count": self.count,
            "min_ms": self.min_ms,
            "mean_ms": self.mean_ms,
            "p50_ms": self.p50_ms,
            "p95_ms": self.p95_ms,
            "p99_ms": self.p99_ms,
            "max_ms": self.max_ms,
        }


@dataclass(frozen=True, slots=True)
class RuntimeBenchmarkReport:
    """Aggregate report for the runtime-shaped benchmark."""

    schema_version: str
    target_hz: float
    tick_budget_ms: float
    measured_cycle_count: int
    warmup_cycle_count: int
    skipped_period_count: int
    missed_deadline_count: int
    capture: StageTimingSummary
    preprocess: StageTimingSummary
    inference: StageTimingSummary
    control: StageTimingSummary
    total: StageTimingSummary
    cycles: tuple[RuntimeCycleTiming, ...]

    @property
    def passed_tick_budget(self) -> bool:
        """Return whether every measured cycle stayed within one tick."""

        return self.missed_deadline_count == 0

    def to_json_dict(self, *, include_cycles: bool = True) -> dict[str, object]:
        """Return a stable JSON-compatible representation."""

        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "target_hz": self.target_hz,
            "tick_budget_ms": self.tick_budget_ms,
            "measured_cycle_count": self.measured_cycle_count,
            "warmup_cycle_count": self.warmup_cycle_count,
            "skipped_period_count": self.skipped_period_count,
            "missed_deadline_count": self.missed_deadline_count,
            "passed_tick_budget": self.passed_tick_budget,
            "capture": self.capture.to_json_dict(),
            "preprocess": self.preprocess.to_json_dict(),
            "inference": self.inference.to_json_dict(),
            "control": self.control.to_json_dict(),
            "total": self.total.to_json_dict(),
        }
        if include_cycles:
            payload["cycles"] = [cycle.to_json_dict() for cycle in self.cycles]
        return payload


@dataclass(frozen=True, slots=True)
class FrameDrivenRuntimeBenchmarkReport:
    """Aggregate report for a frame-arrival-driven runtime benchmark."""

    schema_version: str
    target_hz: float
    tick_budget_ms: float
    measured_frame_count: int
    warmup_frame_count: int
    missed_deadline_count: int
    frame_wait: StageTimingSummary
    preprocess: StageTimingSummary
    inference: StageTimingSummary
    control: StageTimingSummary
    decision: StageTimingSummary
    frames: tuple[FrameDrivenCycleTiming, ...]

    @property
    def passed_decision_budget(self) -> bool:
        """Return whether every measured frame decision stayed within one tick."""

        return self.missed_deadline_count == 0

    def to_json_dict(self, *, include_cycles: bool = True) -> dict[str, object]:
        """Return a stable JSON-compatible representation."""

        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "target_hz": self.target_hz,
            "tick_budget_ms": self.tick_budget_ms,
            "measured_frame_count": self.measured_frame_count,
            "warmup_frame_count": self.warmup_frame_count,
            "missed_deadline_count": self.missed_deadline_count,
            "passed_decision_budget": self.passed_decision_budget,
            "frame_wait": self.frame_wait.to_json_dict(),
            "preprocess": self.preprocess.to_json_dict(),
            "inference": self.inference.to_json_dict(),
            "control": self.control.to_json_dict(),
            "decision": self.decision.to_json_dict(),
        }
        if include_cycles:
            payload["frames"] = [frame.to_json_dict() for frame in self.frames]
        return payload


def run_runtime_benchmark(
    config: RuntimeBenchmarkConfig,
    *,
    capture_frame: Callable[[int], FrameT],
    preprocess_frame: Callable[[FrameT], ObservationT],
    run_policy: Callable[[ObservationT], float],
    apply_action: Callable[[ActionState], None],
) -> RuntimeBenchmarkReport:
    """Run a capture-to-control latency benchmark using injected stages."""

    _validate_config(config)
    period_s = 1.0 / config.target_hz
    tick_budget_ms = period_s * 1000
    started_at = time.perf_counter()
    next_cycle_at = started_at
    cycle_index = 0
    skipped_period_count = 0
    measured_cycles: list[RuntimeCycleTiming] = []

    while True:
        if config.max_cycles is not None and cycle_index >= config.max_cycles:
            break
        now = time.perf_counter()
        if config.max_cycles is None and now - started_at >= config.duration_s:
            break
        missed_periods = int((now - next_cycle_at) // period_s)
        if missed_periods > 0:
            skipped_period_count += missed_periods
            next_cycle_at += missed_periods * period_s
        while now < next_cycle_at:
            time.sleep(min(0.003, next_cycle_at - now))
            now = time.perf_counter()

        scheduled_at = next_cycle_at
        cycle_started_at = time.perf_counter()
        capture_started_at = time.perf_counter()
        frame = capture_frame(cycle_index)
        capture_ms = (time.perf_counter() - capture_started_at) * 1000

        preprocess_started_at = time.perf_counter()
        observation = preprocess_frame(frame)
        preprocess_ms = (time.perf_counter() - preprocess_started_at) * 1000

        inference_started_at = time.perf_counter()
        probability = run_policy(observation)
        inference_ms = (time.perf_counter() - inference_started_at) * 1000

        action = _threshold_action(probability, config.threshold)

        control_started_at = time.perf_counter()
        apply_action(action)
        control_ms = (time.perf_counter() - control_started_at) * 1000
        cycle_finished_at = time.perf_counter()

        total_ms = (cycle_finished_at - cycle_started_at) * 1000
        if cycle_index >= config.warmup_cycles:
            scheduled_lag_ms = (cycle_started_at - scheduled_at) * 1000
            measured_cycles.append(
                RuntimeCycleTiming(
                    cycle_index=cycle_index,
                    scheduled_lag_ms=scheduled_lag_ms,
                    capture_ms=capture_ms,
                    preprocess_ms=preprocess_ms,
                    inference_ms=inference_ms,
                    control_ms=control_ms,
                    total_ms=total_ms,
                    probability=probability,
                    action=action,
                    deadline_missed=(
                        total_ms > tick_budget_ms or scheduled_lag_ms > tick_budget_ms
                    ),
                )
            )

        cycle_index += 1
        next_cycle_at += period_s

    return RuntimeBenchmarkReport(
        schema_version=RUNTIME_BENCHMARK_SCHEMA_VERSION,
        target_hz=config.target_hz,
        tick_budget_ms=tick_budget_ms,
        measured_cycle_count=len(measured_cycles),
        warmup_cycle_count=config.warmup_cycles,
        skipped_period_count=skipped_period_count,
        missed_deadline_count=sum(
            1 for cycle in measured_cycles if cycle.deadline_missed
        ),
        capture=_summarize([cycle.capture_ms for cycle in measured_cycles]),
        preprocess=_summarize([cycle.preprocess_ms for cycle in measured_cycles]),
        inference=_summarize([cycle.inference_ms for cycle in measured_cycles]),
        control=_summarize([cycle.control_ms for cycle in measured_cycles]),
        total=_summarize([cycle.total_ms for cycle in measured_cycles]),
        cycles=tuple(measured_cycles),
    )


def run_frame_driven_runtime_benchmark(
    config: RuntimeBenchmarkConfig,
    *,
    receive_frame: Callable[[int], FrameT],
    preprocess_frame: Callable[[FrameT], ObservationT],
    run_policy: Callable[[ObservationT], float],
    apply_action: Callable[[ActionState], None],
) -> FrameDrivenRuntimeBenchmarkReport:
    """Run a benchmark where each cycle starts when a frame is received."""

    _validate_config(config)
    tick_budget_ms = (1.0 / config.target_hz) * 1000
    started_at = time.perf_counter()
    frame_index = 0
    measured_frames: list[FrameDrivenCycleTiming] = []

    while True:
        if config.max_cycles is not None and frame_index >= config.max_cycles:
            break
        if (
            config.max_cycles is None
            and time.perf_counter() - started_at >= config.duration_s
        ):
            break

        wait_started_at = time.perf_counter()
        frame = receive_frame(frame_index)
        frame_wait_ms = (time.perf_counter() - wait_started_at) * 1000

        decision_started_at = time.perf_counter()

        preprocess_started_at = time.perf_counter()
        observation = preprocess_frame(frame)
        preprocess_ms = (time.perf_counter() - preprocess_started_at) * 1000

        inference_started_at = time.perf_counter()
        probability = run_policy(observation)
        inference_ms = (time.perf_counter() - inference_started_at) * 1000

        action = _threshold_action(probability, config.threshold)

        control_started_at = time.perf_counter()
        apply_action(action)
        control_ms = (time.perf_counter() - control_started_at) * 1000

        decision_ms = (time.perf_counter() - decision_started_at) * 1000
        if frame_index >= config.warmup_cycles:
            measured_frames.append(
                FrameDrivenCycleTiming(
                    frame_index=frame_index,
                    frame_wait_ms=frame_wait_ms,
                    preprocess_ms=preprocess_ms,
                    inference_ms=inference_ms,
                    control_ms=control_ms,
                    decision_ms=decision_ms,
                    probability=probability,
                    action=action,
                    deadline_missed=decision_ms > tick_budget_ms,
                )
            )

        frame_index += 1

    return FrameDrivenRuntimeBenchmarkReport(
        schema_version=FRAME_DRIVEN_RUNTIME_BENCHMARK_SCHEMA_VERSION,
        target_hz=config.target_hz,
        tick_budget_ms=tick_budget_ms,
        measured_frame_count=len(measured_frames),
        warmup_frame_count=config.warmup_cycles,
        missed_deadline_count=sum(
            1 for frame in measured_frames if frame.deadline_missed
        ),
        frame_wait=_summarize([frame.frame_wait_ms for frame in measured_frames]),
        preprocess=_summarize([frame.preprocess_ms for frame in measured_frames]),
        inference=_summarize([frame.inference_ms for frame in measured_frames]),
        control=_summarize([frame.control_ms for frame in measured_frames]),
        decision=_summarize([frame.decision_ms for frame in measured_frames]),
        frames=tuple(measured_frames),
    )


def _validate_config(config: RuntimeBenchmarkConfig) -> None:
    if not isfinite(config.duration_s) or config.duration_s <= 0:
        raise CaptureRecordError("duration_s must be positive and finite")
    if not isfinite(config.target_hz) or config.target_hz <= 0:
        raise CaptureRecordError("target_hz must be positive and finite")
    if not 0 <= config.threshold <= 1:
        raise CaptureRecordError("threshold must be between 0 and 1")
    if config.max_cycles is not None and config.max_cycles <= 0:
        raise CaptureRecordError("max_cycles must be positive when set")
    if config.warmup_cycles < 0:
        raise CaptureRecordError("warmup_cycles must be non-negative")
    if config.max_cycles is not None and config.warmup_cycles >= config.max_cycles:
        raise CaptureRecordError("warmup_cycles must be less than max_cycles")


def _threshold_action(probability: float, threshold: float) -> ActionState:
    if not isfinite(probability) or not 0 <= probability <= 1:
        raise CaptureRecordError("policy probability must be finite and in [0, 1]")
    if probability > threshold:
        return ActionState.HELD
    return ActionState.RELEASED


def _summarize(values: list[float]) -> StageTimingSummary:
    if not values:
        return StageTimingSummary(
            count=0,
            min_ms=None,
            mean_ms=None,
            p50_ms=None,
            p95_ms=None,
            p99_ms=None,
            max_ms=None,
        )
    return StageTimingSummary(
        count=len(values),
        min_ms=min(values),
        mean_ms=sum(values) / len(values),
        p50_ms=_percentile(values, 50),
        p95_ms=_percentile(values, 95),
        p99_ms=_percentile(values, 99),
        max_ms=max(values),
    )


def _percentile(values: list[float], percentile: int) -> float:
    sorted_values = sorted(values)
    index = ceil((percentile / 100) * len(sorted_values)) - 1
    return sorted_values[max(0, min(index, len(sorted_values) - 1))]
