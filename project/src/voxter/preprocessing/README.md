# Preprocessing

Preprocessing code converts raw captures into model-ready observations and
aligned labels.

Expected responsibilities include:

- stable crop selection when non-game UI must be removed
- resizing to the configured model resolution
- grayscale conversion and pixel normalization
- optional icon standardization when implemented safely
- frame-stack construction
- label alignment after measured system-delay correction
- invalid segment removal, such as death aftermath or reset screens

Preprocessing must not introduce future information or privileged metadata into
model inputs. Timing alignment decisions should be reproducible from raw logs and
configuration.

## Current Slice

`voxter.preprocessing.alignment` builds `aligned-manifest-v1` JSONL rows from a
raw capture directory. This first slice does not crop, resize, normalize, or
rewrite pixels; it preserves raw frame paths and capture metadata while deriving
the binary held-state label from `input_events.jsonl`.

`delta_sys` follows the theory-document sign convention:

```text
label_t = action_log_{t + delta_sys}
```

A positive `delta_sys` therefore uses a later frame index in the same
run/attempt as the label source. Samples whose aligned label index falls outside
the current run/attempt are discarded.

`voxter.preprocessing.calibration` scores candidate `delta_sys` values by
comparing frame-sampled held state with event-reconstructed labels. This is a
log-level calibration check; final visual delay calibration still needs visual
evidence or an explicit visual marker.

`voxter.preprocessing.observation` defines the first model-observation
contract, `observation-v1`: raw RGB bytes are converted to deterministic
one-channel grayscale `uint8` observations in height-width layout. The runtime
benchmark can use this path with PipeWire raw RGB payloads, but the preferred
live benchmark path asks GStreamer for `GRAY8` frames and validates the
resulting grayscale observation without a Python per-pixel RGB conversion loop.

`voxter.preprocessing.stack` defines the first Stage 1 policy-input contract,
`frame-stack-v1`: a fixed number of grayscale `observation-v1` frames are
concatenated oldest-to-newest in `khw` layout. Warm-up repeats the first
observation so every runtime step has a fixed-size stack without using future
frames.

`voxter.preprocessing.stage1_dataset` materializes the first offline Stage 1
behavior-cloning dataset from a raw capture directory. It reuses
`aligned-manifest-v1` labels, writes `observation-v1` grayscale byte payloads,
writes `frame-stack-v1` binary payloads, and emits a `stage1-manifest-v1` JSONL
manifest plus a `stage1-dataset-summary-v1` JSON summary with binary class
counts and weighted-BCE class weights. This slice intentionally supports only
PGM/gray8 frame files, matching the current realtime evidence path; JPEG/PNG
decoding remains a separate adapter.
