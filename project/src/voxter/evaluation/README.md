# Evaluation

Evaluation code owns metrics and benchmark protocols.

Offline evaluation should include binary cross-entropy, accuracy, balanced
accuracy, held-input precision and recall, transition precision and recall,
press/release timing error, and breakdowns by mode, Ghost Mode, section, or
held-out generator case when metadata is available.

Online evaluation is the behavioral test of the project. It should measure
survival time, progress, section completion, full-level completion, death
location distribution, attempts to completion, held-out generator performance,
standard-level transfer, and runtime deadline misses.

Frame-level accuracy can be misleading because rare transition errors can cause
death. Treat transition timing and online progress as first-class metrics.
