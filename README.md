# Machine Learning Theory Portfolio

This repository is a portfolio layer over my Machine Learning Theory course
work at Universidad Nacional de Colombia. The original notebooks, midterms,
workshops, and final project are preserved as course evidence; the added
navigation explains what those artifacts demonstrate about my ML foundations.

## What This Repository Proves

| Foundation | Evidence | What to look for |
| --- | --- | --- |
| Optimization | [`midterms/02/notebooks/03-manual-nn.ipynb`](midterms/02/notebooks/03-manual-nn.ipynb), [`project/notebooks/voxter.ipynb`](project/notebooks/voxter.ipynb) | Manual backpropagation, gradient checks, loss curves, behavioral-cloning objectives, runtime-driven training constraints. |
| Statistical learning | [`workshops/02-rkhs/notebook.ipynb`](workshops/02-rkhs/notebook.ipynb), [`midterms/01/notebooks/02-rkhs.ipynb`](midterms/01/notebooks/02-rkhs.ipynb) | RKHS mean embeddings, empirical Gram matrices, kernelized distribution comparison, sample statistics in feature space. |
| Linear models | [`workshops/01-regressors/notebook.ipynb`](workshops/01-regressors/notebook.ipynb), [`midterms/01/notebooks/01-regression.ipynb`](midterms/01/notebooks/01-regression.ipynb) | OLS, Ridge, ML/MAP estimators, polynomial feature design, bias-variance and noise-aware regression reasoning. |
| Neural networks | [`workshops/04-classification/notebook.ipynb`](workshops/04-classification/notebook.ipynb), [`midterms/02/notebooks/03-manual-nn.ipynb`](midterms/02/notebooks/03-manual-nn.ipynb), [`project/`](project/) | CNN classification, hand-derived neural network training, sequence-aware visuomotor policy modeling. |
| Evaluation | [`workshops/03-model-comparison/notebook.ipynb`](workshops/03-model-comparison/notebook.ipynb), [`midterms/02/notebooks/02-rt-detr-l.ipynb`](midterms/02/notebooks/02-rt-detr-l.ipynb), [`project/demo-run/summary.json`](project/demo-run/summary.json) | Model comparison, qualitative and quantitative diagnostics, detection metrics, runtime latency and live-control evidence. |
| Experimental work | [`workshops/05-time-series/notebook.ipynb`](workshops/05-time-series/notebook.ipynb), [`project/notebooks/voxter.ipynb`](project/notebooks/voxter.ipynb), [`project/assets/showcase.mp4`](project/assets/showcase.mp4) | End-to-end experiment design, data contracts, forecasting setup, real-time demo artifacts, supported-versus-unsupported claims. |

For the fuller evidence map, see [`PORTFOLIO.md`](PORTFOLIO.md).

## Best Entry Points

1. [`PORTFOLIO.md`](PORTFOLIO.md): topic-first guide for a reviewer who wants
   to understand the ML foundations demonstrated here.
2. [`project/README.md`](project/README.md): final project overview for Voxter,
   the most complete applied artifact in the repository.
3. [`midterms/README.md`](midterms/README.md): exam evidence with derivations,
   kernels, detection, and manual neural network work.
4. [`workshops/README.md`](workshops/README.md): practical course exercises
   across regression, RKHS, model comparison, classification, and time series.

## Artifact Status

This repository intentionally keeps two kinds of evidence:

- Course artifacts: notebooks, PDFs, workshop outputs, and midterm solutions
  created for academic evaluation. They prioritize derivations, completeness,
  and traceability over product polish.
- Portfolio project artifact: [`project/`](project/) is the final project and
  the strongest applied evidence. It contains a usable README, model/runtime
  structure, demo output, and recorded showcase, but it is still preserved as
  course evidence rather than presented as a maintained production package.

## Repository Map

- [`course/`](course/): course context, conventions, and how to read the repo as
  academic evidence.
- [`workshops/`](workshops/): practical notebooks grouped by ML topic.
- [`midterms/`](midterms/): original exam statements and solution artifacts.
- [`project/`](project/): final project, Voxter, a real-time visuomotor model
  demo.
- [`data/`](data/): local dataset and heavyweight artifact conventions.

## Running Notebooks

For notebook work that uses Jupyter frontends or `ipywidgets`, install the
project with the notebook extra:

```bash
python -m pip install -e ".[notebooks]"
./.venv/bin/jupyter lab
```

The notebooks are evidence artifacts. Some may depend on local data, generated
outputs, or course-specific execution context that is not required for reading
the portfolio narrative.
