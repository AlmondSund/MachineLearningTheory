# Voxter

Voxter is the final project for the Machine Learning Theory course. It is a
real-time visuomotor modeling project: the system observes a game window,
predicts keyboard actions from recent frames, and includes tooling for local
runtime experiments.

## Project Guide

| Area | Files |
| --- | --- |
| Main notebook | [`notebooks/voxter.ipynb`](notebooks/voxter.ipynb) |
| Source-module overview | [`src/voxter/README.md`](src/voxter/README.md) |
| Policy notes | [`src/voxter/policy/README.md`](src/voxter/policy/README.md) |
| Training notes | [`src/voxter/training/README.md`](src/voxter/training/README.md) |
| Evaluation notes | [`src/voxter/evaluation/README.md`](src/voxter/evaluation/README.md) |
| Runtime notes | [`src/voxter/runtime/README.md`](src/voxter/runtime/README.md) |
| Demo summary | [`demo-run/summary.json`](demo-run/summary.json) |

## Topics In The Project

- Data capture and causal alignment.
- Preprocessing contracts for frame observations and action labels.
- CNN and recurrent policy stages.
- Behavioral-cloning training workflow.
- ONNX export and runtime metadata.
- Runtime budget, latency checks, and live-control logs.
- Project limitations and local runtime notes.
- Optional recorded run: [`assets/showcase.mp4`](assets/showcase.mp4).

## Material Status

This directory keeps the final project as course material. It has more
engineering structure than the workshop and midterm folders, but it should not
be read as a maintained production package or a general-purpose game-playing
system.

The live-control path touches desktop capture and OS input injection. Use it
only in a local environment where the target game window is visible, focused,
and safe to control.

## Files

- [`notebooks/voxter.ipynb`](notebooks/voxter.ipynb): main project notebook.
- [`src/voxter/`](src/voxter/): source modules and boundary documentation.
- [`models/voxter/voxter.onnx`](models/voxter/voxter.onnx): exported runtime
  model.
- [`models/voxter/voxter.onnx.data`](models/voxter/voxter.onnx.data): external
  model weights used by ONNX Runtime.
- [`models/voxter/voxter.onnx.json`](models/voxter/voxter.onnx.json): runtime
  metadata.
- [`models/voxter/voxter_benchmark.json`](models/voxter/voxter_benchmark.json):
  local latency benchmark.
- [`tools/run_voxter_live.py`](tools/run_voxter_live.py): local live-run tool.
- [`tools/benchmark_voxter.py`](tools/benchmark_voxter.py): model latency check.
- [`assets/showcase.mp4`](assets/showcase.mp4): optional recorded run.
- [`demo-run/`](demo-run/): preserved demo run logs and summary.

## Setup

```bash
python -m pip install -e ".[onnx]"
```

The local runtime path also needs the desktop capture stack, `tesseract`, and
permission to write to `/dev/uinput`.

## Benchmark

```bash
python tools/benchmark_voxter.py \
  models/voxter/voxter.onnx \
  --output models/voxter/benchmark-check.json
```

## Local Runtime Experiment

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
