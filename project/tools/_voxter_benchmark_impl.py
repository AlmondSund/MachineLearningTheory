#!/usr/bin/env python
"""Benchmark a single-step Stage 2 ONNX model."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class Stage2OnnxBenchmarkReport:
    """Machine-readable ONNX Runtime benchmark result."""

    onnx_path: str
    metadata_path: str
    provider: str
    providers: tuple[str, ...]
    iterations: int
    warmup_iterations: int
    target_hz: float
    tick_budget_ms: float
    latency_ms: dict[str, float]
    passed_tick_budget: bool
    output_shapes: dict[str, list[int]]

    def to_json_dict(self) -> dict[str, object]:
        """Return a stable JSON-compatible report."""

        return {
            "onnx_path": self.onnx_path,
            "metadata_path": self.metadata_path,
            "provider": self.provider,
            "providers": list(self.providers),
            "iterations": self.iterations,
            "warmup_iterations": self.warmup_iterations,
            "target_hz": self.target_hz,
            "tick_budget_ms": self.tick_budget_ms,
            "latency_ms": self.latency_ms,
            "passed_tick_budget": self.passed_tick_budget,
            "output_shapes": self.output_shapes,
        }


def main() -> int:
    parser = _make_parser()
    args = parser.parse_args()
    try:
        report = benchmark_stage2_onnx(
            args.onnx_path,
            metadata_path=args.metadata,
            provider=args.provider,
            iterations=args.iterations,
            warmup_iterations=args.warmup_iterations,
            target_hz=args.target_hz,
            intra_op_num_threads=args.intra_op_threads,
            inter_op_num_threads=args.inter_op_threads,
        )
    except (OSError, ValueError) as exc:
        print(f"Stage 2 ONNX benchmark failed: {exc}", file=sys.stderr)
        return 2

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report.to_json_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report.to_json_dict(), indent=2, sort_keys=True))
    return 0


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("onnx_path", type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--provider", default="CPUExecutionProvider")
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--warmup-iterations", type=int, default=50)
    parser.add_argument("--target-hz", type=float, default=60.0)
    parser.add_argument("--intra-op-threads", type=int, default=1)
    parser.add_argument("--inter-op-threads", type=int, default=1)
    parser.add_argument("--output", type=Path)
    return parser


def benchmark_stage2_onnx(
    onnx_path: Path,
    *,
    metadata_path: Path | None = None,
    provider: str = "CPUExecutionProvider",
    iterations: int = 500,
    warmup_iterations: int = 50,
    target_hz: float = 60.0,
    intra_op_num_threads: int = 1,
    inter_op_num_threads: int = 1,
) -> Stage2OnnxBenchmarkReport:
    """Benchmark a Stage 2 ONNX single-step graph."""

    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if warmup_iterations < 0:
        raise ValueError("warmup-iterations must be non-negative")
    if target_hz <= 0:
        raise ValueError("target-hz must be positive")
    if intra_op_num_threads <= 0:
        raise ValueError("intra-op-threads must be positive")
    if inter_op_num_threads <= 0:
        raise ValueError("inter-op-threads must be positive")

    ort = _require_onnxruntime()
    available_providers = tuple(ort.get_available_providers())
    if provider not in available_providers:
        raise ValueError(
            f"provider {provider!r} is unavailable; available providers: "
            f"{', '.join(available_providers)}"
        )
    metadata_path = (
        metadata_path
        if metadata_path is not None
        else onnx_path.with_suffix(onnx_path.suffix + ".json")
    )
    metadata = _load_metadata(metadata_path)
    input_shape = _required_int_tuple(metadata, "input_shape", length=4)
    hidden_shape = _required_int_tuple(metadata, "hidden_shape", length=3)
    input_names = _required_str_tuple(metadata, "input_names", length=3)
    output_names = _required_str_tuple(metadata, "output_names", length=4)

    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session_options.intra_op_num_threads = intra_op_num_threads
    session_options.inter_op_num_threads = inter_op_num_threads
    session = ort.InferenceSession(
        str(onnx_path.resolve()),
        sess_options=session_options,
        providers=[provider],
    )

    frame_stack = np.zeros(input_shape, dtype=np.float32)
    previous_action = np.zeros((input_shape[0],), dtype=np.float32)
    hidden = np.zeros(hidden_shape, dtype=np.float32)
    feeds = {
        input_names[0]: frame_stack,
        input_names[1]: previous_action,
        input_names[2]: hidden,
    }

    for _ in range(warmup_iterations):
        outputs = session.run(list(output_names), feeds)
        hidden = outputs[3]
        feeds[input_names[2]] = hidden

    latencies_ms: list[float] = []
    output_shapes: dict[str, list[int]] = {}
    for _ in range(iterations):
        started_at = time.perf_counter()
        outputs = session.run(list(output_names), feeds)
        latencies_ms.append((time.perf_counter() - started_at) * 1000)
        hidden = outputs[3]
        feeds[input_names[2]] = hidden
        output_shapes = {
            name: list(output.shape)
            for name, output in zip(output_names, outputs, strict=True)
        }

    tick_budget_ms = 1000.0 / target_hz
    latency = _latency_summary(latencies_ms)
    return Stage2OnnxBenchmarkReport(
        onnx_path=str(onnx_path.resolve()),
        metadata_path=str(metadata_path.resolve()),
        provider=provider,
        providers=available_providers,
        iterations=iterations,
        warmup_iterations=warmup_iterations,
        target_hz=target_hz,
        tick_budget_ms=tick_budget_ms,
        latency_ms=latency,
        passed_tick_budget=latency["p95"] <= tick_budget_ms,
        output_shapes=output_shapes,
    )


def _require_onnxruntime() -> Any:
    try:
        import onnxruntime as ort
    except ModuleNotFoundError as exc:
        raise ValueError(
            "ONNX Runtime is required. Install with "
            '`python -m pip install -e ".[onnx]"`.'
        ) from exc
    return ort


def _load_metadata(metadata_path: Path) -> dict[str, object]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("ONNX metadata must be a JSON object")
    return metadata


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
        raise ValueError(f"{key} must be a list of {length} integers")
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
        raise ValueError(f"{key} must be a list of {length} strings")
    return tuple(value)


def _latency_summary(latencies_ms: list[float]) -> dict[str, float]:
    ordered = sorted(latencies_ms)
    return {
        "min": ordered[0],
        "mean": statistics.fmean(ordered),
        "p50": _percentile(ordered, 50),
        "p95": _percentile(ordered, 95),
        "p99": _percentile(ordered, 99),
        "max": ordered[-1],
    }


def _percentile(ordered_values: list[float], percentile: float) -> float:
    if len(ordered_values) == 1:
        return ordered_values[0]
    index = round((percentile / 100) * (len(ordered_values) - 1))
    return ordered_values[index]


if __name__ == "__main__":
    raise SystemExit(main())
