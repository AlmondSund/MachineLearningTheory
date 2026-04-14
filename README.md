# Machine Learning Theory

Repository for the Machine Learning Theory course at Universidad Nacional de Colombia.

This repository is organized around the full course, not a single notebook or a single exam. The current materials include the first exam statement and two workshops, and the structure now leaves room for future derivations, implementations, notes, and reproducible artifacts.

## Repository Map

- `course/`: course context, organization, and repository conventions.
- `exams/`: exam statements, structured solution workspaces, and derived artifacts.
- `workshops/`: workshop notebooks grouped one workshop per directory.
- `notes/`: lecture notes, summaries, and study material.
- `data/`: local data conventions for datasets and heavier artifacts.
- `src/`: reusable Python support code for future solutions and experiments.

## Current Material

- `exams/01/`: first exam for `2026-1`, focused on kernel methods, feature spaces, RKHS embeddings, and matrix formulations.
- `workshops/01-regressors/`: notebook on regressors and polynomial denoising under AWGN.
- `workshops/02-rkhs/`: notebook on empirical statistics in feature space and RKHS, including MNIST-based experiments.

## Working Conventions

- Put original exam statements under `exams/<nn>/statement/`.
- Put derivations, code, and generated outputs for each exam under `exams/<nn>/solutions/`.
- Keep workshop notebooks in their own folders so each one can later grow notes, helper code, and outputs without forcing another repo-wide reorganization.
- Add reusable code to `src/` instead of embedding everything inside notebooks once the material starts to stabilize.

## Next Step

The first exam statement is now located at `exams/01/statement/main.pdf`. The repository is ready for solving it in a structured way without further layout churn.
