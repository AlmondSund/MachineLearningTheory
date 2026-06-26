# Knowledge Portfolio

This guide reads the repository by demonstrated knowledge rather than by course
folder. It is meant for reviewers who want to know what the artifacts prove and
which files are the best evidence.

## Portfolio Thesis

The course artifacts collectively show foundations in:

- optimization and gradient-based learning,
- statistical learning and kernel methods,
- linear and regularized models,
- neural network modeling,
- evaluation and diagnostics,
- experimental design from notebook-scale studies to an applied final project.

The repository is not a gallery of polished standalone products. It is a
curated record of course work with enough explanation to make the evidence
readable outside the classroom.

## Topic Evidence

### Optimization

Best evidence:

- [`midterms/02/notebooks/03-manual-nn.ipynb`](midterms/02/notebooks/03-manual-nn.ipynb)
- [`project/notebooks/voxter.ipynb`](project/notebooks/voxter.ipynb)
- [`project/src/voxter/training/README.md`](project/src/voxter/training/README.md)

What it demonstrates:

- Derivation and implementation of forward propagation, backpropagation, and
  update-sign conventions.
- Gradient checking before trusting the manual neural network implementation.
- Loss curves, train/validation/test separation, residual diagnostics, and
  comparison against a high-level dense network.
- Behavioral-cloning objectives for a real-time policy, including class
  imbalance and transition-sensitive training concerns.

Artifact status: course midterm solution and course final project evidence.

### Statistical Learning And Kernels

Best evidence:

- [`workshops/02-rkhs/notebook.ipynb`](workshops/02-rkhs/notebook.ipynb)
- [`midterms/01/notebooks/02-rkhs.ipynb`](midterms/01/notebooks/02-rkhs.ipynb)
- [`midterms/01/notebooks/03-geometry.ipynb`](midterms/01/notebooks/03-geometry.ipynb)

What it demonstrates:

- Work with feature-space statistics, RKHS representations, centered Gram
  matrices, and empirical mean embeddings.
- Distribution comparison with linear and Gaussian RBF kernels.
- Matrix construction for pairwise distances and interpretation of kernel
  geometry, symmetry, diagonal entries, and positive semidefiniteness.

Artifact status: course workshop and midterm solution evidence.

### Linear Models

Best evidence:

- [`workshops/01-regressors/notebook.ipynb`](workshops/01-regressors/notebook.ipynb)
- [`midterms/01/notebooks/01-regression.ipynb`](midterms/01/notebooks/01-regression.ipynb)
- [`workshops/03-model-comparison/notebook.ipynb`](workshops/03-model-comparison/notebook.ipynb)

What it demonstrates:

- Regression under additive white Gaussian noise.
- Polynomial design matrices and feature-space regression.
- OLS, Ridge, ML, and MAP estimators.
- Comparison of linear, regularized, nonlinear, probabilistic, and ensemble
  regression models.

Artifact status: course workshop and midterm solution evidence.

### Neural Networks

Best evidence:

- [`midterms/02/notebooks/03-manual-nn.ipynb`](midterms/02/notebooks/03-manual-nn.ipynb)
- [`workshops/04-classification/notebook.ipynb`](workshops/04-classification/notebook.ipynb)
- [`project/notebooks/voxter.ipynb`](project/notebooks/voxter.ipynb)
- [`project/src/voxter/policy/README.md`](project/src/voxter/policy/README.md)

What it demonstrates:

- A manual neural network for nonlinear regression with explicit dimensions,
  update rules, diagnostics, and comparison to a high-level implementation.
- CNN classification on MNIST and Fashion-MNIST with feature-map inspection.
- A final project model contract for a visuomotor policy, including reactive
  CNN and recurrent CNN plus GRU policy stages.

Artifact status: course workshop, course midterm solution, and final project
evidence.

### Evaluation And Diagnostics

Best evidence:

- [`workshops/03-model-comparison/notebook.ipynb`](workshops/03-model-comparison/notebook.ipynb)
- [`midterms/02/notebooks/02-rt-detr-l.ipynb`](midterms/02/notebooks/02-rt-detr-l.ipynb)
- [`project/src/voxter/evaluation/README.md`](project/src/voxter/evaluation/README.md)
- [`project/models/voxter/voxter_benchmark.json`](project/models/voxter/voxter_benchmark.json)
- [`project/demo-run/summary.json`](project/demo-run/summary.json)

What it demonstrates:

- Model comparison across assumptions, flexibility, and empirical behavior.
- Object-detection evaluation framing for the AMIA challenge, including class
  imbalance, geometry, dataset conversion checks, qualitative predictions, and
  submission behavior.
- Awareness that frame-level accuracy can be misleading for control; transition
  timing, online progress, latency, and deadline misses are treated as
  first-class evidence.

Artifact status: course workshop, course midterm solution, and final project
evaluation evidence.

### Experimental Work

Best evidence:

- [`workshops/05-time-series/notebook.ipynb`](workshops/05-time-series/notebook.ipynb)
- [`project/notebooks/voxter.ipynb`](project/notebooks/voxter.ipynb)
- [`project/src/voxter/preprocessing/README.md`](project/src/voxter/preprocessing/README.md)
- [`project/src/voxter/runtime/README.md`](project/src/voxter/runtime/README.md)
- [`project/assets/showcase.mp4`](project/assets/showcase.mp4)

What it demonstrates:

- Synthetic time-series generation from decomposition, ACF-based window
  selection, and next-step forecasting setup.
- End-to-end experiment definition for real-time visual control: data capture,
  causal alignment, preprocessing, model architecture, ONNX runtime contract,
  latency budget, and qualitative demo evidence.
- Explicit separation between supported claims and unsupported claims.

Artifact status: course workshop and final project evidence.

## Artifact Catalog

| Artifact | Status | Demonstrates | Best use for a reviewer |
| --- | --- | --- | --- |
| [`workshops/01-regressors/`](workshops/01-regressors/) | Course workshop | Linear regression, regularization, Bayesian estimator framing | Quick evidence of classical regression foundations. |
| [`workshops/02-rkhs/`](workshops/02-rkhs/) | Course workshop | RKHS statistics, Gram matrices, mean embeddings | Kernel/statistical learning evidence. |
| [`workshops/03-model-comparison/`](workshops/03-model-comparison/) | Course workshop | Comparative model reasoning and empirical diagnostics | Breadth across regression model families. |
| [`workshops/04-classification/`](workshops/04-classification/) | Course workshop | CNN classification and feature-map inspection | Neural network classification evidence. |
| [`workshops/05-time-series/`](workshops/05-time-series/) | Course workshop | Decomposition, ACF, forecasting setup | Time-series experiment evidence. |
| [`midterms/01/`](midterms/01/) | Course exam artifact | Regression, RKHS, kernel geometry | Formal derivation and solution evidence. |
| [`midterms/02/`](midterms/02/) | Course exam artifact | Detection, manual neural network, challenge writeup | Applied evaluation and optimization evidence. |
| [`project/`](project/) | Course final project and portfolio project | Real-time visuomotor learning pipeline | Strongest applied evidence in the repo. |

## Reading Strategy

For a fast review, read the top-level [`README.md`](README.md), then skim
[`project/README.md`](project/README.md), then open the best-evidence notebooks
for the foundations you care about.

For a course-evidence review, start with [`course/README.md`](course/README.md),
then use [`midterms/README.md`](midterms/README.md) and
[`workshops/README.md`](workshops/README.md) to inspect the artifacts in their
original academic context.
