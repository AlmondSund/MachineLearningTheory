# Course Evidence Context

This repository preserves work from the Machine Learning Theory course at
Universidad Nacional de Colombia. The top-level portfolio layer explains what
the artifacts demonstrate; this file explains how to read them as course
evidence.

## How To Read The Repository

- The artifacts are organized by academic activity: workshops, midterms, and
  final project.
- Notebooks often include derivations, exploratory cells, diagnostics, and
  course-specific assumptions in one place.
- Original statements and PDFs are preserved when available so the evidence can
  be traced back to the assignment context.
- The final project has the most complete engineering structure, but it remains
  a course final project rather than a maintained production system.

## Evidence Boundaries

| Area | Where it appears | Evidence type |
| --- | --- | --- |
| Classical regression and linear models | [`workshops/01-regressors/`](../workshops/01-regressors/), [`midterms/01/notebooks/01-regression.ipynb`](../midterms/01/notebooks/01-regression.ipynb) | Workshop and exam solution evidence. |
| RKHS and statistical learning | [`workshops/02-rkhs/`](../workshops/02-rkhs/), [`midterms/01/notebooks/02-rkhs.ipynb`](../midterms/01/notebooks/02-rkhs.ipynb) | Workshop and exam solution evidence. |
| Model comparison and diagnostics | [`workshops/03-model-comparison/`](../workshops/03-model-comparison/) | Practical comparison exercise. |
| Neural networks | [`workshops/04-classification/`](../workshops/04-classification/), [`midterms/02/notebooks/03-manual-nn.ipynb`](../midterms/02/notebooks/03-manual-nn.ipynb) | Workshop and exam solution evidence. |
| Applied experimental ML | [`project/`](../project/) | Final project and strongest portfolio artifact. |

## Organization Principles

- [`midterms/`](../midterms/) contains one folder per midterm with the original
  statement and related solution artifacts.
- [`workshops/`](../workshops/) contains guided practical material organized by
  topic.
- [`project/`](../project/) contains the final project, including notebook,
  model/runtime artifacts, demo output, and source-module documentation.
- [`data/`](../data/) documents local conventions for datasets and heavier
  artifacts.

## Naming

- Midterms use numeric folders such as `01` and `02`.
- Workshops use a numeric prefix plus a short topic slug when a topic slug is
  available.
- Original source files stay close to the statement or activity they belong to.
