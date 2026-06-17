#!/usr/bin/env python
"""Run guarded live Stage 1 inference with optional input control."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from voxter.capture.events import (
    InputEventKind,
    InputEventReader,
    RawInputEvent,
    RawTerminalEvent,
)
from voxter.capture.frames import FrameCaptureError, FrameCaptureRecord
from voxter.capture.ocr import AttemptOcrConfig, AttemptOcrDetector, parse_roi
from voxter.capture.pipewire import (
    GrayFrame,
    PipeWireFramePayload,
    PipeWireGStreamerFrameCapture,
    encode_gray_pgm,
)
from voxter.capture.preview import PreviewGenerationResult, generate_capture_preview
from voxter.contracts import ActionState, VoxterContractError
from voxter.control import UInputKeyboardControl
from voxter.control.uinput import UInputKeyboardConfig
from voxter.policy import (
    Stage2OnnxPolicyConfig,
    load_stage1_policy,
    load_stage2_onnx_policy,
    load_stage2_policy,
)
from voxter.preprocessing import (
    FrameStackConfig,
    Observation,
    ObservationConfig,
    RollingFrameStack,
    preprocess_grayscale_observation,
    preprocess_rgb_observation,
)


class LivePolicyMetadata(Protocol):
    """Checkpoint metadata needed by live capture and reporting."""

    observation_width: int
    observation_height: int
    frame_stack_length: int
    threshold: float

    def to_json_dict(self) -> dict[str, object]:
        """Return stable JSON-compatible metadata."""


class Stage1ProbabilityPolicy(Protocol):
    """Runtime subset needed from a live probability policy adapter."""

    metadata: LivePolicyMetadata

    def predict_probability(self, frame_stack: bytes) -> float:
        """Return `P(action_held=1)` for a serialized frame stack."""


@dataclass(frozen=True, slots=True)
class LiveControlFrame:
    """One live runtime frame decision record."""

    frame_index: int
    timestamp: float
    capture_ms: float
    preprocess_ms: float
    inference_ms: float
    control_ms: float
    decision_ms: float
    probability: float
    previous_action: ActionState
    action: ActionState
    control_applied: bool
    deadline_missed: bool
    held_probability: float | None = None
    press_probability: float | None = None
    release_probability: float | None = None

    def to_json_dict(self) -> dict[str, object]:
        """Return a stable JSON-compatible row."""

        return {
            "frame_index": self.frame_index,
            "timestamp": self.timestamp,
            "capture_ms": self.capture_ms,
            "preprocess_ms": self.preprocess_ms,
            "inference_ms": self.inference_ms,
            "control_ms": self.control_ms,
            "decision_ms": self.decision_ms,
            "probability": self.probability,
            "previous_action": int(self.previous_action),
            "action": int(self.action),
            "control_applied": self.control_applied,
            "deadline_missed": self.deadline_missed,
            "held_probability": self.held_probability,
            "press_probability": self.press_probability,
            "release_probability": self.release_probability,
        }


@dataclass(frozen=True, slots=True)
class LiveControlStep:
    """One live runtime decision plus optional preview artifacts."""

    record: LiveControlFrame
    preview_frame: FrameCaptureRecord | None
    input_event: RawInputEvent | None
    terminal_events: tuple[RawTerminalEvent, ...] = ()


@dataclass(frozen=True, slots=True)
class RuntimeDecisionConfig:
    """Threshold configuration used to convert probabilities into actions."""

    mode: str
    threshold: float
    press_threshold: float | None = None
    release_threshold: float | None = None

    def to_json_dict(self) -> dict[str, object]:
        """Return a stable JSON-compatible representation."""

        return {
            "decision_mode": self.mode,
            "threshold": self.threshold,
            "press_threshold": self.press_threshold,
            "release_threshold": self.release_threshold,
        }


def main() -> int:
    parser = _make_parser()
    args = parser.parse_args()
    try:
        report = _run(args)
    except (OSError, ValueError, VoxterContractError, FrameCaptureError) as exc:
        print(f"live Stage 1 control failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(report, indent=2, sort_keys=True))
    if args.fail_on_deadline_miss and report["missed_deadline_count"] != 0:
        return 1
    return 0


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--target-hz", type=float, default=60.0)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--geometry", default="1920,0 1920x1080")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--policy-stage",
        choices=("stage1", "stage2"),
        default="stage1",
        help="Checkpoint family to load.",
    )
    parser.add_argument(
        "--policy-runtime",
        choices=("torch", "onnx"),
        default="torch",
        help="Inference runtime. ONNX is supported for exported Stage 2 models.",
    )
    parser.add_argument("--onnx-provider", default="CPUExecutionProvider")
    parser.add_argument("--onnx-intra-op-threads", type=int, default=1)
    parser.add_argument("--onnx-inter-op-threads", type=int, default=1)
    parser.add_argument("--threshold", type=float)
    parser.add_argument(
        "--decision-mode",
        choices=("single-threshold", "hysteresis", "transition-heads"),
        default=None,
        help=(
            "Decision decoder. Stage 2 can use transition-heads to decode "
            "press/release auxiliary outputs."
        ),
    )
    parser.add_argument("--press-threshold", type=float)
    parser.add_argument("--release-threshold", type=float)
    parser.add_argument("--portal-source-types", type=int, default=1)
    parser.add_argument("--portal-cursor-mode", type=int, default=1)
    parser.add_argument("--portal-timeout", type=int, default=20)
    parser.add_argument("--apply-control", action="store_true")
    parser.add_argument("--confirm")
    parser.add_argument("--control-device", default="/dev/uinput")
    parser.add_argument("--key-code", type=int, default=17)
    parser.add_argument("--max-deadline-misses", type=int, default=3)
    parser.add_argument("--allow-longer-duration", action="store_true")
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--preview-name", default="preview.mp4")
    parser.add_argument("--preview-frames-dir", default="preview_frames")
    parser.add_argument("--terminal-event-device")
    parser.add_argument("--terminal-key-code", type=int, default=78)
    parser.add_argument("--terminal-type", default="death")
    parser.add_argument("--active-start-key-code", type=int, default=96)
    parser.add_argument(
        "--active-on-start",
        action="store_true",
        help="Emit an active_start terminal event at the first captured frame.",
    )
    parser.add_argument(
        "--ocr-attempt-roi",
        help=(
            "Enable OCR attempt detection with ROI x,y,width,height in captured "
            "frame coordinates. Attempt-number increases emit death and "
            "active_start terminal events."
        ),
    )
    parser.add_argument("--ocr-attempt-command", default="tesseract")
    parser.add_argument("--ocr-attempt-psm", type=int, default=7)
    parser.add_argument("--ocr-attempt-scale", type=int, default=3)
    parser.add_argument("--ocr-attempt-every-frames", type=int, default=30)
    parser.add_argument("--ocr-attempt-min-change-ms", type=float, default=750.0)
    parser.add_argument("--ocr-attempt-timeout-ms", type=float, default=1000.0)
    parser.add_argument("--ocr-attempt-key-code", type=int, default=0)
    parser.add_argument("--rejected-tail-ms", type=float, default=350.0)
    parser.add_argument("--rejected-skip-ms", type=float, default=1500.0)
    parser.add_argument("--include-frames", action="store_true")
    parser.add_argument("--fail-on-deadline-miss", action="store_true")
    return parser


def _run(args: argparse.Namespace) -> dict[str, object]:
    _validate_args(args)
    policy = _load_policy(args)
    metadata = policy.metadata
    decision_config = _make_decision_config(args, default_threshold=metadata.threshold)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "control_log.jsonl"
    terminal_log_path = output_dir / "terminal_events.jsonl"
    summary_path = output_dir / "summary.json"
    preview_frames_dir = output_dir / args.preview_frames_dir
    if args.preview:
        preview_frames_dir.mkdir(parents=True, exist_ok=True)
    observation_config = ObservationConfig(
        width=metadata.observation_width,
        height=metadata.observation_height,
    )
    stacker = RollingFrameStack(
        FrameStackConfig(
            length=metadata.frame_stack_length,
            width=metadata.observation_width,
            height=metadata.observation_height,
        )
    )
    controller: UInputKeyboardControl | None = None
    capture: PipeWireGStreamerFrameCapture | None = None
    terminal_reader: InputEventReader | None = None
    frames: list[LiveControlFrame] = []
    preview_frames: list[FrameCaptureRecord] = []
    input_events: list[RawInputEvent] = []
    terminal_events: list[RawTerminalEvent] = []
    stop_reason = "duration_elapsed"
    run_id = output_dir.name
    ocr_detector = _make_ocr_detector(args)
    try:
        controller = _make_controller(args)
        terminal_reader = _make_terminal_reader(args, run_id=run_id)
        capture = PipeWireGStreamerFrameCapture(
            args.geometry,
            source_types=args.portal_source_types,
            cursor_mode=args.portal_cursor_mode,
            portal_request_timeout_s=args.portal_timeout,
            image_format="gray8",
            async_writes=False,
            output_width=metadata.observation_width,
            output_height=metadata.observation_height,
        )
        started_at = time.perf_counter()
        frame_index = 0
        with log_path.open("w", encoding="utf-8") as log_file:
            while True:
                if args.max_frames is not None and frame_index >= args.max_frames:
                    stop_reason = "max_frames_reached"
                    break
                if time.perf_counter() - started_at >= args.duration:
                    break
                step = _run_one_frame(
                    frame_index,
                    run_id=run_id,
                    capture=capture,
                    stacker=stacker,
                    observation_config=observation_config,
                    policy=policy,
                    decision_config=decision_config,
                    target_hz=args.target_hz,
                    controller=controller,
                    fallback_previous_action=(
                        frames[-1].action if frames else ActionState.RELEASED
                    ),
                    preview_frames_dir=preview_frames_dir if args.preview else None,
                    geometry=args.geometry,
                    key_code=args.key_code,
                    ocr_detector=ocr_detector,
                    active_on_start=getattr(args, "active_on_start", False),
                )
                record = step.record
                frames.append(record)
                if step.preview_frame is not None:
                    preview_frames.append(step.preview_frame)
                if step.input_event is not None:
                    input_events.append(step.input_event)
                if step.terminal_events:
                    terminal_events.extend(step.terminal_events)
                    _reset_policy_on_terminal_events(
                        policy,
                        list(step.terminal_events),
                    )
                if terminal_reader is not None:
                    terminal_reader.read_available()
                    new_terminal_events = terminal_reader.pop_terminal_events()
                    terminal_events.extend(new_terminal_events)
                    _reset_policy_on_terminal_events(policy, new_terminal_events)
                log_file.write(json.dumps(record.to_json_dict(), sort_keys=True) + "\n")
                if _deadline_miss_limit_reached(frames, args.max_deadline_misses):
                    stop_reason = "deadline_miss_limit_reached"
                    break
                frame_index += 1
    finally:
        try:
            if terminal_reader is not None:
                terminal_reader.close()
            if controller is not None:
                controller.release()
                controller.close()
        finally:
            if capture is not None:
                capture.close()

    preview_result: PreviewGenerationResult | None = None
    if args.preview:
        preview_result = generate_capture_preview(
            output_dir,
            preview_frames,
            input_events,
            terminal_events,
            fps=args.target_hz,
            preview_name=args.preview_name,
            rejected_tail_s=args.rejected_tail_ms / 1000,
            rejected_skip_s=args.rejected_skip_ms / 1000,
        )
    if (
        args.terminal_event_device is not None
        or ocr_detector is not None
        or terminal_events
    ):
        terminal_log_path.write_text(
            "".join(
                json.dumps(event.to_json_dict(), sort_keys=True) + "\n"
                for event in terminal_events
            ),
            encoding="utf-8",
        )

    report = _summary_report(
        frames,
        terminal_events=terminal_events,
        checkpoint=metadata.to_json_dict(),
        target_hz=args.target_hz,
        output_dir=output_dir,
        log_path=log_path,
        terminal_log_path=(
            terminal_log_path
            if (
                args.terminal_event_device is not None
                or ocr_detector is not None
                or terminal_events
            )
            else None
        ),
        control_mode="uinput" if args.apply_control else "none",
        stop_reason=stop_reason,
        preview_result=preview_result,
        decision_config=decision_config,
        rejected_tail_s=args.rejected_tail_ms / 1000,
        rejected_skip_s=args.rejected_skip_ms / 1000,
        include_frames=args.include_frames,
    )
    summary_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _load_policy(args: argparse.Namespace) -> Stage1ProbabilityPolicy:
    if args.policy_stage == "stage1":
        return cast(
            Stage1ProbabilityPolicy,
            load_stage1_policy(
                args.checkpoint,
                device=args.device,
                threshold=args.threshold,
            ),
        )
    if args.policy_stage == "stage2":
        if getattr(args, "policy_runtime", "torch") == "onnx":
            return cast(
                Stage1ProbabilityPolicy,
                load_stage2_onnx_policy(
                    args.checkpoint,
                    threshold=args.threshold,
                    config=Stage2OnnxPolicyConfig(
                        provider=getattr(
                            args,
                            "onnx_provider",
                            "CPUExecutionProvider",
                        ),
                        intra_op_num_threads=getattr(
                            args,
                            "onnx_intra_op_threads",
                            1,
                        ),
                        inter_op_num_threads=getattr(
                            args,
                            "onnx_inter_op_threads",
                            1,
                        ),
                    ),
                ),
            )
        return cast(
            Stage1ProbabilityPolicy,
            load_stage2_policy(
                args.checkpoint,
                device=args.device,
                threshold=args.threshold,
            ),
        )
    raise ValueError("policy-stage must be stage1 or stage2")


def _validate_args(args: argparse.Namespace) -> None:
    if args.duration <= 0:
        raise ValueError("duration must be positive")
    if args.apply_control and args.duration > 5 and not args.allow_longer_duration:
        raise ValueError(
            "live control duration is capped at 5 seconds unless "
            "--allow-longer-duration is set"
        )
    if args.target_hz <= 0:
        raise ValueError("target-hz must be positive")
    if getattr(args, "policy_stage", "stage1") not in {"stage1", "stage2"}:
        raise ValueError("policy-stage must be stage1 or stage2")
    if getattr(args, "policy_runtime", "torch") not in {"torch", "onnx"}:
        raise ValueError("policy-runtime must be torch or onnx")
    policy_runtime = getattr(args, "policy_runtime", "torch")
    if policy_runtime == "onnx" and args.policy_stage != "stage2":
        raise ValueError("ONNX runtime is currently supported for Stage 2 only")
    if getattr(args, "onnx_intra_op_threads", 1) <= 0:
        raise ValueError("onnx-intra-op-threads must be positive")
    if getattr(args, "onnx_inter_op_threads", 1) <= 0:
        raise ValueError("onnx-inter-op-threads must be positive")
    decision_mode = getattr(args, "decision_mode", None)
    if decision_mode not in {
        None,
        "single-threshold",
        "hysteresis",
        "transition-heads",
    }:
        raise ValueError(
            "decision-mode must be single-threshold, hysteresis, or transition-heads"
        )
    if decision_mode == "transition-heads" and args.policy_stage != "stage2":
        raise ValueError(
            "transition-heads decision mode requires --policy-stage stage2"
        )
    if args.threshold is not None and not 0.0 <= args.threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    if (args.press_threshold is None) != (args.release_threshold is None):
        raise ValueError(
            "press-threshold and release-threshold must be provided together"
        )
    if args.press_threshold is not None and not 0.0 <= args.press_threshold <= 1.0:
        raise ValueError("press-threshold must be between 0 and 1")
    if args.release_threshold is not None and not 0.0 <= args.release_threshold <= 1.0:
        raise ValueError("release-threshold must be between 0 and 1")
    if (
        args.press_threshold is not None
        and args.release_threshold is not None
        and args.release_threshold >= args.press_threshold
    ):
        raise ValueError("release-threshold must be lower than press-threshold")
    if args.max_frames is not None and args.max_frames <= 0:
        raise ValueError("max-frames must be positive when provided")
    if args.key_code <= 0:
        raise ValueError("key-code must be positive")
    if args.max_deadline_misses < 0:
        raise ValueError("max-deadline-misses must be non-negative")
    if Path(args.preview_frames_dir).name != args.preview_frames_dir:
        raise ValueError("preview-frames-dir must be a plain directory name")
    if args.terminal_key_code <= 0:
        raise ValueError("terminal-key-code must be positive")
    if args.active_start_key_code <= 0:
        raise ValueError("active-start-key-code must be positive")
    if args.terminal_key_code == args.key_code:
        raise ValueError("terminal-key-code must differ from key-code")
    if args.active_start_key_code == args.key_code:
        raise ValueError("active-start-key-code must differ from key-code")
    if args.active_start_key_code == args.terminal_key_code:
        raise ValueError("active-start-key-code must differ from terminal-key-code")
    if getattr(args, "ocr_attempt_roi", None) is not None:
        parse_roi(args.ocr_attempt_roi)
    if getattr(args, "ocr_attempt_psm", 7) <= 0:
        raise ValueError("ocr-attempt-psm must be positive")
    if getattr(args, "ocr_attempt_scale", 3) <= 0:
        raise ValueError("ocr-attempt-scale must be positive")
    if getattr(args, "ocr_attempt_every_frames", 30) <= 0:
        raise ValueError("ocr-attempt-every-frames must be positive")
    if getattr(args, "ocr_attempt_min_change_ms", 750.0) < 0:
        raise ValueError("ocr-attempt-min-change-ms must be non-negative")
    if getattr(args, "ocr_attempt_timeout_ms", 1000.0) <= 0:
        raise ValueError("ocr-attempt-timeout-ms must be positive")
    if getattr(args, "ocr_attempt_key_code", 0) < 0:
        raise ValueError("ocr-attempt-key-code must be non-negative")
    if args.terminal_type not in {"death", "reset", "completion"}:
        raise ValueError("terminal-type must be death, reset, or completion")
    if args.rejected_tail_ms < 0:
        raise ValueError("rejected-tail-ms must be non-negative")
    if args.rejected_skip_ms < 0:
        raise ValueError("rejected-skip-ms must be non-negative")
    if args.apply_control and args.confirm != "APPLY-CONTROL":
        raise ValueError(
            "--apply-control requires --confirm APPLY-CONTROL because this "
            "path injects OS keyboard input"
        )


def _make_decision_config(
    args: argparse.Namespace,
    *,
    default_threshold: float,
) -> RuntimeDecisionConfig:
    threshold = args.threshold if args.threshold is not None else default_threshold
    decision_mode = getattr(args, "decision_mode", None)
    if decision_mode == "transition-heads":
        return RuntimeDecisionConfig(
            mode="transition-heads",
            threshold=threshold,
            press_threshold=(
                args.press_threshold if args.press_threshold is not None else threshold
            ),
            release_threshold=(
                args.release_threshold
                if args.release_threshold is not None
                else threshold
            ),
        )
    if args.press_threshold is None:
        return RuntimeDecisionConfig(
            mode=decision_mode or "single-threshold",
            threshold=threshold,
        )
    return RuntimeDecisionConfig(
        mode=decision_mode or "hysteresis",
        threshold=threshold,
        press_threshold=args.press_threshold,
        release_threshold=args.release_threshold,
    )


def _make_controller(args: argparse.Namespace) -> UInputKeyboardControl | None:
    if not args.apply_control:
        return None
    return UInputKeyboardControl(
        UInputKeyboardConfig(
            device_path=args.control_device,
            key_code=args.key_code,
        )
    )


def _make_terminal_reader(
    args: argparse.Namespace,
    *,
    run_id: str,
) -> InputEventReader | None:
    if args.terminal_event_device is None:
        return None
    reader = InputEventReader(
        args.terminal_event_device,
        run_id=run_id,
        attempt_id=None,
        key_code=args.key_code,
        terminal_markers={
            args.terminal_key_code: args.terminal_type,
            args.active_start_key_code: "active_start",
        },
    )
    reader.open()
    return reader


def _make_ocr_detector(args: argparse.Namespace) -> AttemptOcrDetector | None:
    if getattr(args, "ocr_attempt_roi", None) is None:
        return None
    return AttemptOcrDetector(
        AttemptOcrConfig(
            roi=parse_roi(args.ocr_attempt_roi),
            command=getattr(args, "ocr_attempt_command", "tesseract"),
            psm=getattr(args, "ocr_attempt_psm", 7),
            scale=getattr(args, "ocr_attempt_scale", 3),
            every_n_frames=getattr(args, "ocr_attempt_every_frames", 30),
            min_change_interval_s=getattr(args, "ocr_attempt_min_change_ms", 750.0)
            / 1000,
            timeout_s=getattr(args, "ocr_attempt_timeout_ms", 1000.0) / 1000,
            key_code=getattr(args, "ocr_attempt_key_code", 0),
            emit_active_start_on_first_read=not getattr(
                args,
                "active_on_start",
                False,
            ),
        )
    )


def _run_one_frame(
    frame_index: int,
    *,
    run_id: str,
    capture: PipeWireGStreamerFrameCapture,
    stacker: RollingFrameStack,
    observation_config: ObservationConfig,
    policy: Stage1ProbabilityPolicy,
    decision_config: RuntimeDecisionConfig,
    target_hz: float,
    controller: UInputKeyboardControl | None,
    fallback_previous_action: ActionState,
    preview_frames_dir: Path | None,
    geometry: str,
    key_code: int,
    ocr_detector: AttemptOcrDetector | None,
    active_on_start: bool,
) -> LiveControlStep:
    capture_started_at = time.perf_counter()
    payload = capture.pull_frame_payload()
    capture_ms = (time.perf_counter() - capture_started_at) * 1000

    decision_started_at = time.perf_counter()
    preprocess_started_at = time.perf_counter()
    observation = _payload_to_observation(
        payload,
        observation_config=observation_config,
    )
    frame_stack = stacker.update(observation)
    preprocess_ms = (time.perf_counter() - preprocess_started_at) * 1000

    previous_action = (
        controller.current_action
        if controller is not None
        else fallback_previous_action
    )
    inference_started_at = time.perf_counter()
    head_probabilities: dict[str, float] | None = None
    if decision_config.mode == "transition-heads":
        predict_head_probabilities = getattr(policy, "predict_head_probabilities", None)
        if not callable(predict_head_probabilities):
            raise ValueError("transition-heads decision mode requires Stage 2 heads")
        head_probabilities = predict_head_probabilities(frame_stack.data)
        probability = head_probabilities["held_probability"]
    else:
        probability = policy.predict_probability(frame_stack.data)
    inference_ms = (time.perf_counter() - inference_started_at) * 1000

    action = _select_action_for_decision(
        probability=probability,
        head_probabilities=head_probabilities,
        previous_action=previous_action,
        decision_config=decision_config,
    )
    control_started_at = time.perf_counter()
    if controller is not None:
        controller.apply_action(action)
    control_ms = (time.perf_counter() - control_started_at) * 1000
    observe_action = getattr(policy, "observe_action", None)
    if callable(observe_action):
        observe_action(action)

    timestamp = time.time()
    decision_ms = (time.perf_counter() - decision_started_at) * 1000
    record = LiveControlFrame(
        frame_index=frame_index,
        timestamp=timestamp,
        capture_ms=capture_ms,
        preprocess_ms=preprocess_ms,
        inference_ms=inference_ms,
        control_ms=control_ms,
        decision_ms=decision_ms,
        probability=probability,
        previous_action=previous_action,
        action=action,
        control_applied=controller is not None,
        deadline_missed=decision_ms > (1000.0 / target_hz),
        held_probability=(
            head_probabilities["held_probability"]
            if head_probabilities is not None
            else None
        ),
        press_probability=(
            head_probabilities["press_probability"]
            if head_probabilities is not None
            else None
        ),
        release_probability=(
            head_probabilities["release_probability"]
            if head_probabilities is not None
            else None
        ),
    )
    preview_frame = _write_preview_frame(
        payload,
        record=record,
        run_id=run_id,
        geometry=geometry,
        preview_frames_dir=preview_frames_dir,
    )
    input_event = _input_event_for_action_change(
        record,
        run_id=run_id,
        key_code=key_code,
    )
    terminal_events: tuple[RawTerminalEvent, ...] = ()
    if active_on_start and frame_index == 0:
        terminal_events = (
            RawTerminalEvent(
                run_id=run_id,
                attempt_id=None,
                timestamp=record.timestamp,
                device="voxter-live-control",
                key_code=0,
                kind=InputEventKind.PRESS,
                terminal_type="active_start",
            ),
        )
    if ocr_detector is not None:
        if payload.image_format != "gray8":
            raise ValueError("OCR attempt detection requires gray8 frames")
        terminal_events = (
            *terminal_events,
            *tuple(
                ocr_detector.detect(
                    GrayFrame(
                        width=payload.frame_width,
                        height=payload.frame_height,
                        data=payload.data,
                    ),
                    frame_index=frame_index,
                    timestamp=record.timestamp,
                    run_id=run_id,
                )
            ),
        )
    return LiveControlStep(
        record=record,
        preview_frame=preview_frame,
        input_event=input_event,
        terminal_events=terminal_events,
    )


def _reset_policy_on_terminal_events(
    policy: Stage1ProbabilityPolicy,
    terminal_events: list[RawTerminalEvent],
) -> None:
    if not any(
        event.terminal_type in {"death", "reset", "completion"}
        for event in terminal_events
    ):
        return
    reset_state = getattr(policy, "reset_state", None)
    if callable(reset_state):
        reset_state()


def _select_action(
    probability: float,
    *,
    previous_action: ActionState,
    decision_config: RuntimeDecisionConfig,
) -> ActionState:
    """Convert a held-state probability into the next binary action."""

    if decision_config.mode == "single-threshold":
        return (
            ActionState.HELD
            if probability >= decision_config.threshold
            else ActionState.RELEASED
        )
    if (
        decision_config.press_threshold is None
        or decision_config.release_threshold is None
    ):
        raise ValueError("hysteresis mode requires press and release thresholds")
    if previous_action is ActionState.RELEASED:
        if probability >= decision_config.press_threshold:
            return ActionState.HELD
        return ActionState.RELEASED
    if probability <= decision_config.release_threshold:
        return ActionState.RELEASED
    return ActionState.HELD


def _select_action_for_decision(
    *,
    probability: float,
    head_probabilities: dict[str, float] | None,
    previous_action: ActionState,
    decision_config: RuntimeDecisionConfig,
) -> ActionState:
    """Convert the configured policy output into the next binary action."""

    if decision_config.mode != "transition-heads":
        return _select_action(
            probability,
            previous_action=previous_action,
            decision_config=decision_config,
        )
    if head_probabilities is None:
        raise ValueError("transition-heads mode requires head probabilities")
    if (
        decision_config.press_threshold is None
        or decision_config.release_threshold is None
    ):
        raise ValueError("transition-heads mode requires press and release thresholds")
    if previous_action is ActionState.RELEASED:
        if head_probabilities["press_probability"] >= decision_config.press_threshold:
            return ActionState.HELD
        return ActionState.RELEASED
    if head_probabilities["release_probability"] >= decision_config.release_threshold:
        return ActionState.RELEASED
    return ActionState.HELD


def _payload_to_observation(
    payload: PipeWireFramePayload,
    *,
    observation_config: ObservationConfig,
) -> Observation:
    if payload.image_format == "gray8":
        return preprocess_grayscale_observation(
            payload.data,
            source_width=payload.frame_width,
            source_height=payload.frame_height,
            config=observation_config,
        )
    if payload.image_format == "rgb":
        return preprocess_rgb_observation(
            payload.data,
            source_width=payload.frame_width,
            source_height=payload.frame_height,
            config=observation_config,
        )
    raise ValueError("live Stage 1 control requires raw RGB or gray8 frames")


def _deadline_miss_limit_reached(
    frames: list[LiveControlFrame],
    max_deadline_misses: int,
) -> bool:
    if max_deadline_misses == 0:
        return False
    return sum(1 for frame in frames if frame.deadline_missed) >= max_deadline_misses


def _write_preview_frame(
    payload: PipeWireFramePayload,
    *,
    record: LiveControlFrame,
    run_id: str,
    geometry: str,
    preview_frames_dir: Path | None,
) -> FrameCaptureRecord | None:
    if preview_frames_dir is None:
        return None
    if payload.image_format != "gray8":
        raise ValueError("live-control preview requires gray8 frames")
    frame_path = preview_frames_dir / f"{record.frame_index:06d}.pgm"
    frame_path.write_bytes(
        encode_gray_pgm(
            GrayFrame(
                width=payload.frame_width,
                height=payload.frame_height,
                data=payload.data,
            )
        )
    )
    return FrameCaptureRecord(
        run_id=run_id,
        attempt_id=None,
        frame_index=record.frame_index,
        timestamp=record.timestamp,
        frame_path=str(frame_path),
        action=record.action,
        action_sample_timestamp=record.timestamp,
        geometry=geometry,
        capture_duration_s=payload.capture_duration_s,
        capture_backend="live-control-pipewire-gstreamer-gray8",
        image_format="pgm",
        frame_width=payload.frame_width,
        frame_height=payload.frame_height,
        source_width=payload.source_width,
        source_height=payload.source_height,
        capture_resized=payload.capture_resized,
    )


def _input_event_for_action_change(
    record: LiveControlFrame,
    *,
    run_id: str,
    key_code: int,
) -> RawInputEvent | None:
    if record.previous_action == record.action:
        return None
    kind = (
        InputEventKind.PRESS
        if record.action is ActionState.HELD
        else InputEventKind.RELEASE
    )
    return RawInputEvent(
        run_id=run_id,
        attempt_id=None,
        timestamp=record.timestamp,
        device="voxter-live-control",
        key_code=key_code,
        kind=kind,
        action=record.action,
    )


def _summary_report(
    frames: list[LiveControlFrame],
    *,
    terminal_events: list[RawTerminalEvent],
    checkpoint: dict[str, object],
    target_hz: float,
    output_dir: Path,
    log_path: Path,
    terminal_log_path: Path | None,
    control_mode: str,
    stop_reason: str,
    preview_result: PreviewGenerationResult | None,
    decision_config: RuntimeDecisionConfig,
    rejected_tail_s: float,
    rejected_skip_s: float,
    include_frames: bool,
) -> dict[str, object]:
    if not frames:
        raise ValueError("at least one frame must be captured")
    actions = [frame.action for frame in frames]
    probabilities = [frame.probability for frame in frames]
    press_probabilities = [
        frame.press_probability
        for frame in frames
        if frame.press_probability is not None
    ]
    release_probabilities = [
        frame.release_probability
        for frame in frames
        if frame.release_probability is not None
    ]
    held_probabilities = [
        frame.held_probability for frame in frames if frame.held_probability is not None
    ]
    rejected_frame_count = _terminal_rejected_frame_count(
        frames,
        terminal_events,
        rejected_tail_s=rejected_tail_s,
        rejected_skip_s=rejected_skip_s,
    )
    first_terminal_s = _first_terminal_s(frames, terminal_events)
    phase_metrics = _phase_metrics(frames, terminal_events)
    payload: dict[str, object] = {
        "schema_version": "stage1-live-control-v1",
        "control_mode": control_mode,
        "action_space": "binary-held-state",
        **decision_config.to_json_dict(),
        "stop_reason": stop_reason,
        "checkpoint": checkpoint,
        "target_hz": target_hz,
        "tick_budget_ms": 1000.0 / target_hz,
        "output_dir": str(output_dir),
        "log_path": str(log_path),
        "terminal_log_path": str(terminal_log_path) if terminal_log_path else None,
        "preview_path": preview_result.preview_path if preview_result else None,
        "preview_frame_count": (
            preview_result.frame_count if preview_result is not None else 0
        ),
        "frame_count": len(frames),
        "terminal_event_count": len(terminal_events),
        **phase_metrics,
        "first_terminal_s": first_terminal_s,
        "survival_duration_s": first_terminal_s,
        "accepted_frame_count": len(frames) - rejected_frame_count,
        "rejected_frame_count": rejected_frame_count,
        "missed_deadline_count": sum(1 for frame in frames if frame.deadline_missed),
        "passed_tick_budget": all(not frame.deadline_missed for frame in frames),
        "held_frame_count": sum(1 for action in actions if action is ActionState.HELD),
        "released_frame_count": sum(
            1 for action in actions if action is ActionState.RELEASED
        ),
        "mean_probability": sum(probabilities) / len(probabilities),
        "capture": _timing_summary([frame.capture_ms for frame in frames]),
        "preprocess": _timing_summary([frame.preprocess_ms for frame in frames]),
        "inference": _timing_summary([frame.inference_ms for frame in frames]),
        "control": _timing_summary([frame.control_ms for frame in frames]),
        "decision": _timing_summary([frame.decision_ms for frame in frames]),
    }
    if held_probabilities:
        payload["mean_held_probability"] = sum(held_probabilities) / len(
            held_probabilities
        )
        payload["max_held_probability"] = max(held_probabilities)
    if press_probabilities:
        payload["mean_press_probability"] = sum(press_probabilities) / len(
            press_probabilities
        )
        payload["max_press_probability"] = max(press_probabilities)
    if release_probabilities:
        payload["mean_release_probability"] = sum(release_probabilities) / len(
            release_probabilities
        )
        payload["max_release_probability"] = max(release_probabilities)
    if include_frames:
        payload["frames"] = [frame.to_json_dict() for frame in frames]
    return payload


def _phase_metrics(
    frames: list[LiveControlFrame],
    terminal_events: list[RawTerminalEvent],
) -> dict[str, object]:
    sorted_events = sorted(terminal_events, key=lambda event: event.timestamp)
    active_start_events = [
        event for event in sorted_events if event.terminal_type == "active_start"
    ]
    death_events = [event for event in sorted_events if event.terminal_type == "death"]
    first_frame_timestamp = min(frame.timestamp for frame in frames)
    first_active_timestamp = (
        active_start_events[0].timestamp if active_start_events else None
    )
    first_death_after_active_timestamp = None
    if first_active_timestamp is not None:
        for event in death_events:
            if event.timestamp >= first_active_timestamp:
                first_death_after_active_timestamp = event.timestamp
                break

    active_by_frame = _active_flags_by_frame(frames, sorted_events)
    active_frames = [
        frame
        for frame, is_active in zip(frames, active_by_frame, strict=True)
        if is_active
    ]
    intro_frames = [
        frame
        for frame, is_active in zip(frames, active_by_frame, strict=True)
        if not is_active
    ]
    intro_transition_count = 0
    active_transition_count = 0
    for previous_frame, frame, is_active in zip(
        frames,
        frames[1:],
        active_by_frame[1:],
        strict=False,
    ):
        if previous_frame.action == frame.action:
            continue
        if is_active:
            active_transition_count += 1
        else:
            intro_transition_count += 1

    first_active_start_s = (
        None
        if first_active_timestamp is None
        else max(0.0, first_active_timestamp - first_frame_timestamp)
    )
    first_death_after_active_s = (
        None
        if first_active_timestamp is None or first_death_after_active_timestamp is None
        else max(0.0, first_death_after_active_timestamp - first_active_timestamp)
    )
    return {
        "active_start_count": len(active_start_events),
        "death_count": len(death_events),
        "first_active_start_s": first_active_start_s,
        "first_death_after_active_s": first_death_after_active_s,
        "intro_frame_count": len(intro_frames),
        "active_frame_count": len(active_frames),
        "intro_held_frame_count": sum(
            1 for frame in intro_frames if frame.action is ActionState.HELD
        ),
        "active_held_frame_count": sum(
            1 for frame in active_frames if frame.action is ActionState.HELD
        ),
        "intro_action_transition_count": intro_transition_count,
        "active_action_transition_count": active_transition_count,
    }


def _active_flags_by_frame(
    frames: list[LiveControlFrame],
    terminal_events: list[RawTerminalEvent],
) -> list[bool]:
    active_flags: list[bool] = []
    event_index = 0
    is_active = False
    for frame in frames:
        while (
            event_index < len(terminal_events)
            and terminal_events[event_index].timestamp <= frame.timestamp
        ):
            terminal_type = terminal_events[event_index].terminal_type
            if terminal_type == "active_start":
                is_active = True
            elif terminal_type in {"death", "reset", "completion"}:
                is_active = False
            event_index += 1
        active_flags.append(is_active)
    return active_flags


def _terminal_rejected_frame_count(
    frames: list[LiveControlFrame],
    terminal_events: list[RawTerminalEvent],
    *,
    rejected_tail_s: float,
    rejected_skip_s: float,
) -> int:
    if not terminal_events:
        return 0
    rejected_indexes: set[int] = set()
    for event in terminal_events:
        if event.terminal_type == "active_start":
            continue
        start_s = event.timestamp - rejected_tail_s
        end_s = event.timestamp + rejected_skip_s
        for frame in frames:
            if start_s <= frame.timestamp <= end_s:
                rejected_indexes.add(frame.frame_index)
    return len(rejected_indexes)


def _first_terminal_s(
    frames: list[LiveControlFrame],
    terminal_events: list[RawTerminalEvent],
) -> float | None:
    stop_events = [
        event for event in terminal_events if event.terminal_type != "active_start"
    ]
    if not stop_events:
        return None
    first_frame_timestamp = min(frame.timestamp for frame in frames)
    first_terminal_timestamp = min(event.timestamp for event in stop_events)
    return max(0.0, first_terminal_timestamp - first_frame_timestamp)


def _timing_summary(values: list[float]) -> dict[str, object]:
    sorted_values = sorted(values)
    return {
        "count": len(sorted_values),
        "min_ms": sorted_values[0],
        "mean_ms": sum(sorted_values) / len(sorted_values),
        "p50_ms": _percentile(sorted_values, 0.50),
        "p95_ms": _percentile(sorted_values, 0.95),
        "p99_ms": _percentile(sorted_values, 0.99),
        "max_ms": sorted_values[-1],
    }


def _percentile(sorted_values: list[float], fraction: float) -> float:
    index = min(len(sorted_values) - 1, round((len(sorted_values) - 1) * fraction))
    return sorted_values[index]


if __name__ == "__main__":
    raise SystemExit(main())
