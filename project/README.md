# Voxter

Voxter is the final project for the Machine Learning Theory course and the
strongest applied portfolio artifact in this repository. It is a real-time
visuomotor model demo: the system observes a game window, predicts keyboard
actions from recent frames, and can inject those actions through `/dev/uinput`
for a live showcase.

## What This Demonstrates

| Foundation | Evidence in this project |
| --- | --- |
| Experimental design | Data capture, causal alignment, preprocessing contracts, model/runtime separation, and supported-versus-unsupported claims in [`notebooks/voxter.ipynb`](notebooks/voxter.ipynb). |
| Neural networks | CNN and recurrent policy framing in [`src/voxter/policy/README.md`](src/voxter/policy/README.md). |
| Optimization | Behavioral-cloning and transition-sensitive training workflow in [`src/voxter/training/README.md`](src/voxter/training/README.md). |
| Evaluation | Offline metrics, transition timing, online gameplay metrics, and runtime deadline concerns in [`src/voxter/evaluation/README.md`](src/voxter/evaluation/README.md). |
| Systems constraints | Real-time loop, ONNX runtime contract, capture/preprocess/inference/input budget, and demo logs. |

## Artifact Status

This is a course final project preserved as portfolio evidence. It is more
engineered than the workshop and midterm notebooks, but it should still be read
as an academic final project rather than a maintained production package.

Supported evidence includes:

- a theory-synchronized project notebook,
- source-module documentation for capture, preprocessing, policy, training,
  evaluation, runtime, and control boundaries,
- an exported ONNX model and benchmark artifacts,
- a recorded showcase video,
- a demo run summary and logs.

Unsupported claims are intentionally not implied: this repository does not claim
general-purpose game playing, robust deployment across desktop environments, or
production-safe OS input automation.

## Best Review Path

1. Read [`notebooks/voxter.ipynb`](notebooks/voxter.ipynb) for the project
   thesis, contracts, model architecture, runtime budget, qualitative showcase,
   and limitations.
2. Read [`src/voxter/README.md`](src/voxter/README.md) for module boundaries.
3. Inspect [`assets/showcase.mp4`](assets/showcase.mp4) for recorded behavior.
4. Inspect [`models/voxter/voxter_benchmark.json`](models/voxter/voxter_benchmark.json)
   and [`demo-run/summary.json`](demo-run/summary.json) for runtime evidence.

## Files

- [`notebooks/voxter.ipynb`](notebooks/voxter.ipynb): main project narrative and
  evidence notebook.
- [`src/voxter/`](src/voxter/): source modules and boundary documentation.
- [`models/voxter/voxter.onnx`](models/voxter/voxter.onnx): exported runtime
  model.
- [`models/voxter/voxter.onnx.data`](models/voxter/voxter.onnx.data): external
  model weights used by ONNX Runtime.
- [`models/voxter/voxter.onnx.json`](models/voxter/voxter.onnx.json): runtime
  metadata.
- [`models/voxter/voxter_benchmark.json`](models/voxter/voxter_benchmark.json):
  local latency benchmark.
- [`tools/run_voxter_live.py`](tools/run_voxter_live.py): live demo runner.
- [`tools/benchmark_voxter.py`](tools/benchmark_voxter.py): model latency check.
- [`assets/showcase.mp4`](assets/showcase.mp4): recorded showcase video.
- [`demo-run/`](demo-run/): preserved demo run logs and summary.

## Setup

```bash
python -m pip install -e ".[onnx]"
```

The live demo also needs the local desktop capture stack, `tesseract`, and
permission to write to `/dev/uinput`.

## Benchmark

```bash
python tools/benchmark_voxter.py \
  models/voxter/voxter.onnx \
  --output models/voxter/benchmark-check.json
```

## Live Demo

```bash
python tools/run_voxter_live.py \
  --output-dir demo-run \
  --duration 60 \
  --allow-longer-duration \
  --target-hz 60 \
  --max-deadline-misses 1000 \
  --geometry "1920,0 1920x1080" \
  --decision-mode transition-heads \
  --press-threshold 0.50 \
  --release-threshold 0.30 \
  --ocr-attempt-roi 0,25,125,35 \
  --ocr-attempt-every-frames 60 \
  --ocr-attempt-timeout-ms 1000 \
  --active-on-start \
  --apply-control \
  --confirm APPLY-CONTROL \
  models/voxter/voxter.onnx
```

Use the live command only when the target game window is visible and focused.
