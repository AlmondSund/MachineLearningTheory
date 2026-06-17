# Voxter

Voxter is a real-time visuomotor model demo. It observes the game window,
predicts keyboard actions from recent frames, and can inject those actions
through `/dev/uinput` for a live showcase.

## Files

- `models/voxter/voxter.onnx`: exported runtime model.
- `models/voxter/voxter.onnx.data`: external model weights used by ONNX Runtime.
- `models/voxter/voxter.onnx.json`: runtime metadata.
- `models/voxter/voxter_benchmark.json`: local latency benchmark.
- `tools/run_voxter_live.py`: live demo runner.
- `tools/benchmark_voxter.py`: model latency check.
- `assets/showcase.mp4`: recorded showcase video.

## Setup

```bash
python -m pip install -e ".[onnx]"
```

The live demo also needs the local desktop capture stack, `tesseract`, and
permission to write to `/dev/uinput`.

## Benchmark

```bash
python tools/benchmark_voxter.py   models/voxter/voxter.onnx   --output models/voxter/benchmark-check.json
```

## Live Demo

```bash
python tools/run_voxter_live.py   --output-dir demo-run   --duration 60   --allow-longer-duration   --target-hz 60   --max-deadline-misses 1000   --geometry "1920,0 1920x1080"   --decision-mode transition-heads   --press-threshold 0.50   --release-threshold 0.30   --ocr-attempt-roi 0,25,125,35   --ocr-attempt-every-frames 60   --ocr-attempt-timeout-ms 1000   --active-on-start   --apply-control   --confirm APPLY-CONTROL   models/voxter/voxter.onnx
```

Use the live command only when the target game window is visible and focused.
