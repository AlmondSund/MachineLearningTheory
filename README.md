# Machine Learning Theory

Course materials for Machine Learning Theory at Universidad Nacional de
Colombia. The repository keeps the original structure of the course while adding
topic-oriented navigation for the notebooks, midterms, workshops, and final
project.

## Start Here

- [`TOPIC_GUIDE.md`](TOPIC_GUIDE.md): materials grouped by topic, including
  optimization, statistical learning, linear models, neural networks, model
  assessment, and experimental work.
- [`course/`](course/): notes on how the repository is organized as course
  material.
- [`workshops/`](workshops/): guided practical notebooks.
- [`midterms/`](midterms/): midterm statements, solution notebooks, and related
  materials.
- [`project/`](project/): final project, Voxter, a real-time visuomotor model
  demo.

## Topics Covered

| Topic | Main materials |
| --- | --- |
| Optimization and training | [`midterms/02/notebooks/03-manual-nn.ipynb`](midterms/02/notebooks/03-manual-nn.ipynb), [`project/notebooks/voxter.ipynb`](project/notebooks/voxter.ipynb) |
| Statistical learning and kernels | [`workshops/02-rkhs/notebook.ipynb`](workshops/02-rkhs/notebook.ipynb), [`midterms/01/notebooks/02-rkhs.ipynb`](midterms/01/notebooks/02-rkhs.ipynb) |
| Linear and regularized models | [`workshops/01-regressors/notebook.ipynb`](workshops/01-regressors/notebook.ipynb), [`midterms/01/notebooks/01-regression.ipynb`](midterms/01/notebooks/01-regression.ipynb) |
| Neural networks | [`workshops/04-classification/notebook.ipynb`](workshops/04-classification/notebook.ipynb), [`midterms/02/notebooks/03-manual-nn.ipynb`](midterms/02/notebooks/03-manual-nn.ipynb), [`project/`](project/) |
| Model assessment and diagnostics | [`workshops/03-model-comparison/notebook.ipynb`](workshops/03-model-comparison/notebook.ipynb), [`midterms/02/notebooks/02-rt-detr-l.ipynb`](midterms/02/notebooks/02-rt-detr-l.ipynb) |
| Experimental work | [`workshops/05-time-series/notebook.ipynb`](workshops/05-time-series/notebook.ipynb), [`project/`](project/) |

## Repository Map

- [`course/`](course/): course context and repository conventions.
- [`workshops/`](workshops/): practical notebooks grouped by activity.
- [`midterms/`](midterms/): original exam statements and solution materials.
- [`project/`](project/): final course project and related runtime artifacts.
- [`data/`](data/): local dataset and heavyweight artifact conventions.

## Material Status

Most files in this repository are course materials. The notebooks may include
assignment framing, exploratory cells, local paths, generated outputs, and
course-specific assumptions. The final project has a more complete engineering
layout, but it is still kept here as part of the course record.

## Running Notebooks

For notebook work that uses Jupyter frontends or `ipywidgets`, install the
project with the notebook extra:

```bash
python -m pip install -e ".[notebooks]"
./.venv/bin/jupyter lab
```

Some notebooks may depend on local data or generated artifacts that are not
required for reading the repository.
