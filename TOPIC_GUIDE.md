# Topic Guide

This guide groups the course materials by topic. It is a navigation aid for the
notebooks, midterms, workshops, and final project, not a replacement for the
original course structure.

## Optimization And Training

Start with:

- [`midterms/02/notebooks/03-manual-nn.ipynb`](midterms/02/notebooks/03-manual-nn.ipynb)
- [`project/notebooks/voxter.ipynb`](project/notebooks/voxter.ipynb)
- [`project/src/voxter/training/README.md`](project/src/voxter/training/README.md)

Covered material:

- Forward propagation, backpropagation, gradient checks, and update-sign
  conventions.
- Loss curves, train/validation/test splits, residual diagnostics, and
  comparison with a high-level dense model.
- Behavioral cloning, transition-sensitive training, and class-imbalance
  considerations in the final project.

## Statistical Learning And Kernels

Start with:

- [`workshops/02-rkhs/notebook.ipynb`](workshops/02-rkhs/notebook.ipynb)
- [`midterms/01/notebooks/02-rkhs.ipynb`](midterms/01/notebooks/02-rkhs.ipynb)
- [`midterms/01/notebooks/03-geometry.ipynb`](midterms/01/notebooks/03-geometry.ipynb)

Covered material:

- Sample statistics in feature spaces and RKHS representations.
- Centered Gram matrices and empirical mean embeddings.
- Linear and Gaussian RBF kernels for distribution comparison.
- Pairwise distance matrices, Gram geometry, symmetry, diagonal entries, and
  positive semidefiniteness.

## Linear And Regularized Models

Start with:

- [`workshops/01-regressors/notebook.ipynb`](workshops/01-regressors/notebook.ipynb)
- [`midterms/01/notebooks/01-regression.ipynb`](midterms/01/notebooks/01-regression.ipynb)
- [`workshops/03-model-comparison/notebook.ipynb`](workshops/03-model-comparison/notebook.ipynb)

Covered material:

- Regression under additive white Gaussian noise.
- Polynomial design matrices and feature-space regression.
- OLS, Ridge, ML, and MAP estimators.
- Comparison of linear, regularized, nonlinear, probabilistic, and ensemble
  regression models.

## Neural Networks

Start with:

- [`midterms/02/notebooks/03-manual-nn.ipynb`](midterms/02/notebooks/03-manual-nn.ipynb)
- [`workshops/04-classification/notebook.ipynb`](workshops/04-classification/notebook.ipynb)
- [`project/notebooks/voxter.ipynb`](project/notebooks/voxter.ipynb)
- [`project/src/voxter/policy/README.md`](project/src/voxter/policy/README.md)

Covered material:

- A manual neural network for nonlinear regression.
- CNN classification on MNIST and Fashion-MNIST.
- Feature-map inspection by class.
- Reactive CNN and recurrent CNN plus GRU policy stages in the final project.

## Model Assessment And Diagnostics

Start with:

- [`workshops/03-model-comparison/notebook.ipynb`](workshops/03-model-comparison/notebook.ipynb)
- [`midterms/02/notebooks/02-rt-detr-l.ipynb`](midterms/02/notebooks/02-rt-detr-l.ipynb)
- [`project/src/voxter/evaluation/README.md`](project/src/voxter/evaluation/README.md)
- [`project/models/voxter/voxter_benchmark.json`](project/models/voxter/voxter_benchmark.json)
- [`project/demo-run/summary.json`](project/demo-run/summary.json)

Covered material:

- Model comparison across assumptions, flexibility, and empirical behavior.
- Object-detection setup for the AMIA challenge, including class imbalance,
  geometry, dataset conversion checks, qualitative predictions, and limitations.
- Runtime latency, deadline misses, transition timing, and online behavior in a
  real-time control setting.

## Experimental Work

Start with:

- [`workshops/05-time-series/notebook.ipynb`](workshops/05-time-series/notebook.ipynb)
- [`project/notebooks/voxter.ipynb`](project/notebooks/voxter.ipynb)
- [`project/src/voxter/preprocessing/README.md`](project/src/voxter/preprocessing/README.md)
- [`project/src/voxter/runtime/README.md`](project/src/voxter/runtime/README.md)
- [`project/assets/showcase.mp4`](project/assets/showcase.mp4)

Covered material:

- Synthetic time-series construction from decomposition components.
- ACF-based window selection and next-step forecasting setup.
- Data capture, causal alignment, preprocessing, model architecture, ONNX
  runtime, latency budget, and qualitative demo output for the final project.

## Material Types

| Material | Type | Notes |
| --- | --- | --- |
| [`workshops/01-regressors/`](workshops/01-regressors/) | Course workshop | Regression and estimator comparison. |
| [`workshops/02-rkhs/`](workshops/02-rkhs/) | Course workshop | RKHS statistics and Gram-matrix work. |
| [`workshops/03-model-comparison/`](workshops/03-model-comparison/) | Course workshop | Regression model comparison and diagnostics. |
| [`workshops/04-classification/`](workshops/04-classification/) | Course workshop | CNN classification exercises. |
| [`workshops/05-time-series/`](workshops/05-time-series/) | Course workshop | Time-series decomposition and forecasting. |
| [`midterms/01/`](midterms/01/) | Midterm material | Regression, RKHS, and kernel geometry. |
| [`midterms/02/`](midterms/02/) | Midterm material | Detection, manual neural network, and challenge work. |
| [`project/`](project/) | Final project | Voxter real-time visuomotor model demo. |

## Suggested Reading

For a quick overview, start with [`README.md`](README.md), then use this guide
to jump to the topic you care about.

For the original course organization, use [`course/README.md`](course/README.md),
[`workshops/README.md`](workshops/README.md), and
[`midterms/README.md`](midterms/README.md).
