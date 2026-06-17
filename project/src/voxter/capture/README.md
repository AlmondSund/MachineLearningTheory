# Capture

Capture code owns screen acquisition, input-state logging, timestamps, and
system-delay calibration.

This module is allowed to perform external side effects such as reading the
screen, observing input state, writing capture logs, and measuring latency.
Keep those side effects out of preprocessing, policy, and evaluation code.

Capture outputs should support causal dataset construction:

- raw frame or raw frame path
- binary held-state input
- timestamp or game-tick index
- run and attempt identity
- optional metadata for analysis and evaluation

System-level delay must be measured and represented explicitly. A positive
model-theory `delta_sys` means the action log lags behind the visual frame, so
labels are taken from a later log index after correction.

Raw capture should keep frame rows and input events as separate streams:

- `frames.jsonl` records frame paths, frame indexes, frame timestamps, capture
  geometry, capture timing, sampled held state, and the timestamp at which that
  held state was sampled. It also records image format, encoded frame
  dimensions, source dimensions, and whether capture-side resizing was applied.
- `input_events.jsonl` records key press/release events at input-device
  resolution.

Frame and input-event timestamps must use the same clock domain. The current
Linux event reader records evdev wall-clock timestamps, so frame rows also use
wall-clock timestamps while monotonic time is reserved for measuring durations
and deadlines.

Input events are the authoritative source for transitions. Frame-sampled held
state is useful metadata, but fast taps may occur between frame samples.

Transient frame-pull failures are counted as dropped frames in
`capture_summary.json`. They do not create placeholder frame rows; the frame
manifest contains only frames actually written to disk.

The `grim` capture adapter is an offline/debug backend. It is not enough
evidence for a real-time 60 Hz runtime backend unless timing measurements fit
the full control-cycle budget.

The `pipewire` backend uses xdg-desktop-portal, PipeWire, GStreamer, and an
`appsink` reader to pull frames from a persistent stream. It supports compressed
JPEG output for practical live recording and raw PPM output for short
diagnostic captures. It queues frame bytes to a bounded background writer so
disk writes do not directly block the frame-pull loop; if the queue fills, the
session counts the slot as a dropped frame instead of inventing a manifest row.
Use capture-side downscaling for real-time tests because model inputs do not
need full 1920x1080 frames and JPEG encoding at full resolution can cause
latency spikes. Treat downscaled captures as model-resolution or diagnostic
capture runs unless the dataset manifest explicitly records that the immutable
raw source was capture-side resized.
It requires PyGObject from the system Python environment or a virtual
environment created with `--system-site-packages`.
