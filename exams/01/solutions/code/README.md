# Exam 01 Code

`solve_exam_01.py` implements the numerical parts of the exam in English:

- regularized least squares for the polynomial feature map `phi(x) = [1, x, x^2]^T`,
- a small empirical mean-embedding check for linear and RBF kernels,
- empirical MMD between MNIST digit classes in RKHS,
- sanity checks for the squared-distance and Gaussian-kernel matrix identities.

Run it with the repository-local environment:

```bash
.venv/bin/python exams/01/solutions/code/solve_exam_01.py
```

The script writes its outputs to `exams/01/solutions/artifacts/numeric_results.json`.
