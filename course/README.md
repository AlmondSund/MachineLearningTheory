# Course Structure

This repository is intended to hold the full Machine Learning Theory course material for Universidad Nacional de Colombia.

## Organization Principles

- `exams/` contains one folder per exam with the original statement PDF and one notebook solution.
- `workshops/` contains exploratory or guided practical material.
- `notes/` contains lecture-oriented study material and distilled summaries.
- `src/` is reserved for code that becomes reusable across notebooks, exams, or experiments.

## Naming

- Exams use numeric folders such as `01`, `02`, `03`.
- Workshops use a numeric prefix plus a short topic slug.
- Keep original source files close to the statement or activity they belong to.

## Intended Workflow

1. Read the original statement in `main.pdf`.
2. Solve the exam directly in `notebook.ipynb`.
3. Put only truly reusable cross-exam code in `src/`.
