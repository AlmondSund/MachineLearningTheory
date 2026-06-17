# Policy

Policy code owns the model contract and inference semantics.

The primary action output is the probability that the input should be held:

```text
p_t = P(action_held = 1 | observation history)
```

The runtime layer converts this probability into a binary held/released input
state using a threshold or hysteresis rule.

Expected model stages:

- Stage 1: CNN encoder plus binary head over a short frame stack
- Stage 2: CNN encoder plus recurrent memory, initially GRU, using previous
  action as part of the sequence input
- Stage 3: imitation-initialized policy fine-tuned with reinforcement learning

Implemented policy adapters:

- `stage1_policy.py`: loads `stage1-training-v1` checkpoints and predicts one
  held-state probability per frame stack.
- `stage2_policy.py`: loads `stage2-training-v1` checkpoints, keeps GRU hidden
  state internally, observes the final applied action, and predicts one
  held-state probability per frame stack.

Policy code should not perform screen capture, physical input application, or
dataset mutation.
