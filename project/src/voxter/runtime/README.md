# Runtime

Runtime code owns the deployed real-time control loop.

The loop sequence is:

1. capture frame
2. preprocess frame
3. update frame stack or recurrent state
4. run policy inference
5. threshold probability into held/released action
6. apply input state through `control`
7. log runtime data
8. repeat at the target rate

Runtime must preserve the real-time contract:

```text
capture + preprocess + inference + input <= tick interval
```

Deadline behavior, dropped-frame handling, fail-safe behavior, thresholds, and
hysteresis settings should be explicit configuration rather than hidden defaults.

The current live-control tool lives under `tools/live_stage1_control.py`. Its
name is historical; it supports both `--policy-stage stage1` and
`--policy-stage stage2`.

For Stage 2, runtime must feed back the final selected binary action to the
policy adapter and reset recurrent state on terminal markers such as death,
reset, or completion.

## Benchmarking

`voxter.runtime.benchmark` measures the runtime-shaped path:

```text
capture -> preprocess -> policy -> control
```

The current benchmark accepts injected stages so tests can exercise deadline
accounting without a desktop session. The `tools/runtime_benchmark.py` adapter
can run a synthetic path or a PipeWire capture path with preprocessing, policy,
and control stubs.

Loop modes are separated:

- `fixed-rate`: schedule cycles at the target rate and charge frame waiting to
  the cycle budget
- `frame-driven`: start each decision when a frame is received and report frame
  wait separately from decision latency

PipeWire benchmark modes are separated:

- `runtime`: pull an in-memory frame payload and do not write frames
- `recording`: write temporary frame files through the capture backend

Passing this benchmark means the measured stage skeleton stayed inside the tick
budget. It does not prove gameplay competence, trained-policy latency, or safe
OS input injection.
