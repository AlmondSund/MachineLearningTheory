# Voxter Source Modules

The `voxter` package is organized around the project contracts in
`research/model-theory`.

## Module Boundaries

`capture/` owns screen capture, input-state logging, timestamps, and calibration
data collection. This is an external side-effect boundary.

`preprocessing/` owns deterministic conversion from captured frames and logs to
model observations, frame stacks, and aligned labels.

`policy/` owns model definitions and policy inference semantics, including
reactive CNN policies and memory-based policies such as CNN plus GRU.

`training/` owns behavioral cloning, sequential behavioral cloning, and
reinforcement fine-tuning workflows.

`evaluation/` owns offline imitation metrics, transition-timing metrics, online
gameplay metrics, and transfer benchmarks.

`runtime/` owns the deployed real-time control loop and deadline behavior.

`control/` owns input execution: applying the binary held/released action to the
game or operating system.

## Boundary Rule

The policy should consume observations, previous actions, and its own recurrent
state. It should not consume future frames, symbolic level scripts, fixed click
timestamps, or privileged game metadata for the main real-time transfer claim.
