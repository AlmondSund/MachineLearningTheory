# Machine Learning Theory

Repository for the Machine Learning Theory course at Universidad Nacional de Colombia.

This repository is organized around the full course, not a single notebook or a single exam. The current materials include the first exam statement and two workshops, and the structure now leaves room for future derivations, implementations, notes, and reproducible artifacts.

## Repository Map

- `course/`: course context, organization, and repository conventions.
- `exams/`: exam statements plus one notebook solution per exam.
- `workshops/`: workshop notebooks grouped one workshop per directory.
- `notes/`: lecture notes, summaries, and study material.
- `data/`: local data conventions for datasets and heavier artifacts.
- `src/`: reusable Python support code for future solutions and experiments.

## Current Material

- `exams/01/`: first exam for `2026-1`, with the original statement in `main.pdf` and the full English solution in `notebook.ipynb`.
- `workshops/01-regressors/`: notebook on regressors and polynomial denoising under AWGN.
- `workshops/02-rkhs/`: notebook on empirical statistics in feature space and RKHS, including MNIST-based experiments.

## Working Conventions

- For each exam, keep the original statement as `exams/<nn>/main.pdf`.
- For each exam, keep the full solution as one self-contained notebook at `exams/<nn>/notebook.ipynb`.
- Keep workshop notebooks in their own folders so each one can later grow notes, helper code, and outputs without forcing another repo-wide reorganization.
- Add reusable code to `src/` instead of embedding everything inside notebooks once the material starts to stabilize.

## Next Step

The first exam now lives in `exams/01/` with only two files: `main.pdf` and `notebook.ipynb`.
