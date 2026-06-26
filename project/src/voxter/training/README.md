# Training

Training code owns optimization workflows and experiment execution.

Expected workflows include:

- Stage 1 reactive behavioral cloning
- Stage 2 sequential behavioral cloning with contiguous sequence windows
- Stage 3 reinforcement fine-tuning from an imitation-initialized policy
- class weighting or sampler logic for binary action imbalance
- checkpointing and training metrics

Training data should be split by trajectory, section, seed, or section family.
Do not evaluate generalization with random frame-level splits.

Offline loss and accuracy are not sufficient for judging gameplay competence.
Training workflows should produce artifacts that can be checked online through
the runtime and evaluation modules.

Implemented modules:

- `stage1_data.py`: dependency-light Stage 1 dataset loading and smoke reports.
- `stage1_torch.py`: optional-PyTorch Stage 1 CNN training, metrics, and
  checkpoint writing.
- `stage2_data.py`: Stage 2 contiguous sequence-window loading from Stage 1
  datasets with previous-action rows.
- `stage2_torch.py`: optional-PyTorch Stage 2 MobileNetV3+GRU training,
  metrics, and checkpoint writing. The legacy scratch CNN+GRU model remains an
  explicit model option.
- `stage3_rollout.py`: dependency-light Stage 3 rollout and reward contracts
  for converting live-control logs into reinforcement-learning artifacts.

Current operational commands:

```bash
python tools/smoke_stage2_data.py data/datasets/phase-a-baseline-20260527-*-stage1-k8-320x180 \
  --sequence-length 32 \
  --batch-size 4 \
  --max-batches 4
python tools/train_stage2.py data/datasets/phase-a-baseline-20260527-*-stage1-k8-320x180 \
  --output-dir data/experiments/stage2/phase-a-k8-320x180-cpu-1epoch \
  --run-id phase-a-k8-320x180-cpu-1epoch \
  --epochs 1 \
  --batch-size 4 \
  --sequence-length 32 \
  --model-name stage2-mobilenetv3-small-gru \
  --hidden-size 64 \
  --transition-weight-multiplier 4.0 \
  --transition-window-radius 2 \
  --transition-aux-loss-weight 1.0 \
  --device cpu
```

The current full Stage 2 training target is Kaggle GPU through
`kaggle/voxter-stage2-training.ipynb`.
Stage 2 loss upweights press/release transition windows, trains auxiliary
press/release heads, and validation reports closed-loop metrics so runtime
feedback collapse is visible before live control.

Build the first Stage 3 fixed-policy rollout artifact from a live-control run:

```bash
python tools/build_stage3_rollout.py \
  data/experiments/stage2/live-control-stage2-onnx-apply-60hz-5s-headlog \
  --run-id stage3-fixed-policy-smoke
```

This writes `stage3_rollout.jsonl` and `stage3_rollout_summary.json` beside the
live-control logs by default. The artifact contains rewards, terminal flags,
action/probability rows, timing flags, and observation references when preview
frames are present. It does not perform policy optimization.

Run the first conservative offline Stage 3 update from rollout rows:

```bash
python tools/train_stage3.py \
  data/experiments/stage3/fixed-policy-terminal-smoke-60hz/stage3_rollout.jsonl \
  --checkpoint "data/experiments/stage2/phase-a-k8-320x180-kaggle-stage2-mobilenet-gru (1)/checkpoint.pt" \
  --output-dir data/experiments/stage3/offline-pg-terminal-smoke \
  --run-id stage3-offline-pg-terminal-smoke \
  --epochs 1 \
  --learning-rate 1e-6 \
  --kl-weight 0.05 \
  --device cpu
```

The trainer reconstructs the same rolling `K,H,W` frame stacks used by live
control, computes discounted returns inside each episode, and applies a small
Bernoulli policy-gradient update with a KL anchor to the source Stage 2 policy.
It writes a Stage 2-compatible `checkpoint.pt` plus Stage 3 `metrics.json`.
