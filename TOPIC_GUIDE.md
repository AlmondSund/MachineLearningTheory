# Topic Guide

This guide groups the course materials by topic. It is a navigation aid for the
notebooks, midterms, workshops, and final project, not a replacement for the
original course structure.

## Optimization And Training

Primary:

- [`midterms/02/notebooks/03-manual-nn.ipynb`](midterms/02/notebooks/03-manual-nn.ipynb)

Secondary:

- [`project/src/voxter/training/README.md`](project/src/voxter/training/README.md)

Notes:

- Forward propagation, backpropagation, gradient checks, and update-sign
  conventions.
- Loss curves, train/validation/test splits, residual diagnostics, and
  comparison with a high-level dense model.
- The final project includes behavioral-cloning and transition-sensitive
  training notes, but the manual neural network notebook is the cleaner entry
  point for this topic.

## Statistical Learning And Kernels

Primary:

- [`midterms/01/notebooks/02-rkhs.ipynb`](midterms/01/notebooks/02-rkhs.ipynb)

Secondary:

- [`workshops/02-rkhs/notebook.ipynb`](workshops/02-rkhs/notebook.ipynb)
- [`midterms/01/notebooks/03-geometry.ipynb`](midterms/01/notebooks/03-geometry.ipynb)

Notes:

- Sample statistics in feature spaces and RKHS representations.
- Centered Gram matrices and empirical mean embeddings.
- Linear and Gaussian RBF kernels for distribution comparison.
- Kernel geometry and positive semidefiniteness are covered in the geometry
  notebook.

## Linear And Regularized Models

Primary:

- [`workshops/01-regressors/notebook.ipynb`](workshops/01-regressors/notebook.ipynb)

Secondary:

- [`workshops/03-model-comparison/notebook.ipynb`](workshops/03-model-comparison/notebook.ipynb)

Notes:

- Regression under additive white Gaussian noise.
- Polynomial design matrices and feature-space regression.
- OLS, Ridge, ML, and MAP estimators.
- The model-comparison workshop broadens the view to other regression families.

## Neural Networks

Primary:

- [`midterms/02/notebooks/03-manual-nn.ipynb`](midterms/02/notebooks/03-manual-nn.ipynb)

Secondary:

- [`workshops/04-classification/notebook.ipynb`](workshops/04-classification/notebook.ipynb)

Notes:

- A manual neural network for nonlinear regression.
- CNN classification on MNIST and Fashion-MNIST.
- Feature-map inspection by class.
- The final project also includes policy-model notes, but it is better read in
  the project context than as the main neural-network entry point.

## Model Assessment And Diagnostics

Primary:

- [`workshops/03-model-comparison/notebook.ipynb`](workshops/03-model-comparison/notebook.ipynb)

Secondary:

- [`midterms/02/notebooks/02-rt-detr-l.ipynb`](midterms/02/notebooks/02-rt-detr-l.ipynb)

Notes:

- Model comparison across assumptions, flexibility, and empirical behavior.
- Object-detection setup for the AMIA challenge, including class imbalance,
  geometry, dataset conversion checks, qualitative predictions, and limitations.
- The final project has runtime and evaluation notes, but they are project
  documentation rather than the main path for this topic.

## Experimental Work

Primary:

- [`workshops/05-time-series/notebook.ipynb`](workshops/05-time-series/notebook.ipynb)

Secondary:

- [`project/notebooks/voxter.ipynb`](project/notebooks/voxter.ipynb)

Notes:

- Synthetic time-series construction from decomposition components.
- ACF-based window selection and next-step forecasting setup.
- The final project covers data capture, causal alignment, preprocessing, model
  structure, runtime constraints, and project limitations.
- Optional recorded run: [`project/assets/showcase.mp4`](project/assets/showcase.mp4).

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
| [`project/`](project/) | Final project | Real-time visuomotor modeling project. |

## Suggested Reading

For a quick overview, start with [`README.md`](README.md), then use this guide
to jump to the topic you care about.

For the original course organization, use [`course/README.md`](course/README.md),
[`workshops/README.md`](workshops/README.md), and
[`midterms/README.md`](midterms/README.md).
