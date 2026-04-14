from __future__ import annotations

import argparse
import gzip
import json
import shutil
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUTPUT_PATH = REPO_ROOT / "exams" / "01" / "solutions" / "artifacts" / "numeric_results.json"
MNIST_DATA_DIR = REPO_ROOT / "data" / "raw" / "mnist"
MNIST_BASE_URLS = (
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
    "http://yann.lecun.com/exdb/mnist/",
)


@dataclass(frozen=True)
class PairScore:
    digit_a: int
    digit_b: int
    mmd2_biased: float


def build_polynomial_design_matrix(x_samples: np.ndarray) -> np.ndarray:
    """Builds the design matrix for phi(x) = [1, x, x^2]^T."""
    x_vector = np.asarray(x_samples, dtype=np.float64).reshape(-1)
    return np.column_stack(
        (
            np.ones_like(x_vector),
            x_vector,
            x_vector**2,
        )
    )


def solve_regularized_least_squares(
    design_matrix: np.ndarray,
    targets: np.ndarray,
    lambda_reg: float,
) -> np.ndarray:
    """Solves (Phi^T Phi + lambda I) w = Phi^T t."""
    if lambda_reg < 0.0:
        raise ValueError("lambda_reg must be non-negative.")

    phi = np.asarray(design_matrix, dtype=np.float64)
    t = np.asarray(targets, dtype=np.float64).reshape(-1)
    gram = phi.T @ phi
    rhs = phi.T @ t
    system = gram + lambda_reg * np.eye(phi.shape[1], dtype=np.float64)
    return np.linalg.solve(system, rhs)


def solve_kernelized_regularized_least_squares(
    design_matrix: np.ndarray,
    targets: np.ndarray,
    lambda_reg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Solves the dual system and maps it back to the primal weights."""
    if lambda_reg < 0.0:
        raise ValueError("lambda_reg must be non-negative.")

    phi = np.asarray(design_matrix, dtype=np.float64)
    t = np.asarray(targets, dtype=np.float64).reshape(-1)
    kernel_matrix = phi @ phi.T
    alpha = np.linalg.solve(
        kernel_matrix + lambda_reg * np.eye(phi.shape[0], dtype=np.float64),
        t,
    )
    return alpha, phi.T @ alpha


def pairwise_squared_distances(x_matrix: np.ndarray) -> np.ndarray:
    """Computes the full squared Euclidean distance matrix."""
    x = np.asarray(x_matrix, dtype=np.float64)
    squared_norms = np.sum(x * x, axis=1, keepdims=True)
    distances = squared_norms + squared_norms.T - 2.0 * (x @ x.T)
    return np.maximum(distances, 0.0)


def rbf_kernel_matrix(x_matrix: np.ndarray, gamma: float) -> np.ndarray:
    """Builds the Gaussian/RBF Gram matrix."""
    if gamma <= 0.0:
        raise ValueError("gamma must be strictly positive.")
    return np.exp(-gamma * pairwise_squared_distances(x_matrix))


def cross_rbf_kernel_matrix(x_matrix: np.ndarray, y_matrix: np.ndarray, gamma: float) -> np.ndarray:
    """Builds the cross-kernel matrix between two sample sets."""
    if gamma <= 0.0:
        raise ValueError("gamma must be strictly positive.")

    x = np.asarray(x_matrix, dtype=np.float64)
    y = np.asarray(y_matrix, dtype=np.float64)
    x_norms = np.sum(x * x, axis=1, keepdims=True)
    y_norms = np.sum(y * y, axis=1, keepdims=True)
    distances = x_norms + y_norms.T - 2.0 * (x @ y.T)
    distances = np.maximum(distances, 0.0)
    return np.exp(-gamma * distances)


def biased_mmd2_from_gram(k_xx: np.ndarray, k_yy: np.ndarray, k_xy: np.ndarray) -> float:
    """Returns the biased empirical MMD^2 estimate."""
    m = k_xx.shape[0]
    n = k_yy.shape[0]
    return float(
        np.sum(k_xx) / (m * m)
        + np.sum(k_yy) / (n * n)
        - 2.0 * np.sum(k_xy) / (m * n)
    )


def median_heuristic_gamma(x_matrix: np.ndarray) -> float:
    """Chooses gamma = 1 / median(nonzero squared distances)."""
    distances = pairwise_squared_distances(x_matrix)
    nonzero = distances[distances > 0.0]
    if nonzero.size == 0:
        raise ValueError("Median heuristic requires at least two distinct samples.")
    return float(1.0 / np.median(nonzero))


def ensure_mnist_file(filename: str) -> Path:
    """Downloads one MNIST gzip file into data/raw/mnist if missing."""
    destination = MNIST_DATA_DIR / filename
    if destination.exists():
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None

    for base_url in MNIST_BASE_URLS:
        try:
            with urllib.request.urlopen(base_url + filename, timeout=60) as response:
                with destination.open("wb") as output_handle:
                    shutil.copyfileobj(response, output_handle)
            return destination
        except Exception as exc:  # pragma: no cover - network fallback path
            last_error = exc
            if destination.exists():
                destination.unlink()

    raise RuntimeError(f"Failed to download MNIST file {filename!r}.") from last_error


def load_mnist_images(path: Path) -> np.ndarray:
    """Loads an IDX image file compressed with gzip."""
    with gzip.open(path, "rb") as handle:
        magic = int.from_bytes(handle.read(4), "big")
        if magic != 2051:
            raise ValueError(f"Unexpected image magic number {magic}.")
        count = int.from_bytes(handle.read(4), "big")
        rows = int.from_bytes(handle.read(4), "big")
        cols = int.from_bytes(handle.read(4), "big")
        data = np.frombuffer(handle.read(), dtype=np.uint8)
    return data.reshape(count, rows * cols).astype(np.float64) / 255.0


def load_mnist_labels(path: Path) -> np.ndarray:
    """Loads an IDX label file compressed with gzip."""
    with gzip.open(path, "rb") as handle:
        magic = int.from_bytes(handle.read(4), "big")
        if magic != 2049:
            raise ValueError(f"Unexpected label magic number {magic}.")
        count = int.from_bytes(handle.read(4), "big")
        data = np.frombuffer(handle.read(), dtype=np.uint8)
    return data.reshape(count)


def load_mnist_training_set() -> tuple[np.ndarray, np.ndarray]:
    """Loads the normalized MNIST training split."""
    images = load_mnist_images(ensure_mnist_file("train-images-idx3-ubyte.gz"))
    labels = load_mnist_labels(ensure_mnist_file("train-labels-idx1-ubyte.gz"))
    if images.shape[0] != labels.shape[0]:
        raise ValueError("MNIST image and label counts do not match.")
    return images, labels


def select_balanced_mnist_subset(
    images: np.ndarray,
    labels: np.ndarray,
    samples_per_class: int,
    seed: int,
) -> dict[int, np.ndarray]:
    """Selects a reproducible balanced subset for each digit."""
    if samples_per_class < 2:
        raise ValueError("samples_per_class must be at least 2.")

    rng = np.random.default_rng(seed)
    by_digit: dict[int, np.ndarray] = {}

    for digit in range(10):
        indices = np.flatnonzero(labels == digit)
        if indices.size < samples_per_class:
            raise ValueError(f"Digit {digit} has only {indices.size} samples.")
        chosen = rng.choice(indices, size=samples_per_class, replace=False)
        by_digit[digit] = images[np.sort(chosen)]

    return by_digit


def polynomial_regression_example(lambda_reg: float) -> dict[str, Any]:
    """Builds and solves the feature-space example from Question 1.3."""
    x_samples = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float64)
    targets = 1.0 + 0.5 * x_samples + x_samples**2
    design_matrix = build_polynomial_design_matrix(x_samples)
    primal_weights = solve_regularized_least_squares(design_matrix, targets, lambda_reg=lambda_reg)
    alpha, dual_weights = solve_kernelized_regularized_least_squares(
        design_matrix,
        targets,
        lambda_reg=lambda_reg,
    )
    predictions = design_matrix @ primal_weights

    return {
        "lambda_reg": float(lambda_reg),
        "x_samples": x_samples.tolist(),
        "targets": targets.tolist(),
        "design_matrix": design_matrix.tolist(),
        "estimated_weights": primal_weights.tolist(),
        "predictions": predictions.tolist(),
        "max_abs_primal_dual_difference": float(np.max(np.abs(primal_weights - dual_weights))),
        "alpha_norm": float(np.linalg.norm(alpha)),
    }


def mean_embedding_demo() -> dict[str, Any]:
    """Provides a concrete numerical check for Question 2.1."""
    x_samples = np.array(
        [
            [0.0, 1.0],
            [1.0, 0.0],
            [2.0, 1.0],
        ],
        dtype=np.float64,
    )
    z_query = np.array([1.0, 2.0], dtype=np.float64)
    sample_mean = np.mean(x_samples, axis=0)
    linear_eval = float(sample_mean @ z_query)
    direct_linear_eval = float(np.mean(x_samples @ z_query))

    gamma = 0.5
    rbf_values = np.exp(-gamma * np.sum((x_samples - z_query) ** 2, axis=1))
    rbf_eval = float(np.mean(rbf_values))

    return {
        "samples": x_samples.tolist(),
        "query": z_query.tolist(),
        "linear_sample_mean": sample_mean.tolist(),
        "linear_eval_via_mean": linear_eval,
        "linear_eval_direct_kernel_average": direct_linear_eval,
        "rbf_gamma": gamma,
        "rbf_eval_at_query": rbf_eval,
    }


def mnist_mmd_experiment(samples_per_class: int, seed: int) -> dict[str, Any]:
    """Computes pairwise RKHS discrepancies between MNIST digit classes."""
    images, labels = load_mnist_training_set()
    balanced = select_balanced_mnist_subset(
        images,
        labels,
        samples_per_class=samples_per_class,
        seed=seed,
    )

    pooled_for_bandwidth = np.vstack([balanced[digit][: min(20, samples_per_class)] for digit in range(10)])
    gamma = median_heuristic_gamma(pooled_for_bandwidth)

    pair_scores: list[PairScore] = []
    for digit_a in range(10):
        for digit_b in range(digit_a + 1, 10):
            x_class = balanced[digit_a]
            y_class = balanced[digit_b]
            k_xx = rbf_kernel_matrix(x_class, gamma=gamma)
            k_yy = rbf_kernel_matrix(y_class, gamma=gamma)
            k_xy = cross_rbf_kernel_matrix(x_class, y_class, gamma=gamma)
            pair_scores.append(
                PairScore(
                    digit_a=digit_a,
                    digit_b=digit_b,
                    mmd2_biased=biased_mmd2_from_gram(k_xx, k_yy, k_xy),
                )
            )

    pair_scores_sorted = sorted(pair_scores, key=lambda item: item.mmd2_biased)
    smallest_pairs = [asdict(item) for item in pair_scores_sorted[:5]]
    largest_pairs = [asdict(item) for item in pair_scores_sorted[-5:][::-1]]

    return {
        "samples_per_class": samples_per_class,
        "gamma": gamma,
        "smallest_pairs": smallest_pairs,
        "largest_pairs": largest_pairs,
    }


def kernel_geometry_sanity(gamma: float, seed: int) -> dict[str, Any]:
    """Numerically checks the identities used in Question 3."""
    images, labels = load_mnist_training_set()
    balanced = select_balanced_mnist_subset(images, labels, samples_per_class=5, seed=seed)
    x_small = np.vstack([balanced[0], balanced[1]])

    distances_formula = pairwise_squared_distances(x_small)
    distances_direct = np.sum((x_small[:, None, :] - x_small[None, :, :]) ** 2, axis=2)
    kernel_matrix = rbf_kernel_matrix(x_small, gamma=gamma)
    symmetrized = 0.5 * (kernel_matrix + kernel_matrix.T)
    eigenvalues = np.linalg.eigvalsh(symmetrized)

    return {
        "distance_formula_max_abs_error": float(np.max(np.abs(distances_formula - distances_direct))),
        "kernel_diagonal_max_abs_error": float(np.max(np.abs(np.diag(kernel_matrix) - 1.0))),
        "kernel_symmetry_max_abs_error": float(np.max(np.abs(kernel_matrix - kernel_matrix.T))),
        "kernel_min_eigenvalue": float(np.min(eigenvalues)),
    }


def write_results(output_path: Path, payload: dict[str, Any]) -> None:
    """Serializes the numeric outputs to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Solve the numeric parts of Exam 01.")
    parser.add_argument("--samples-per-class", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--lambda-reg", type=float, default=0.1)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    polynomial_results = polynomial_regression_example(lambda_reg=args.lambda_reg)
    embedding_results = mean_embedding_demo()
    mmd_results = mnist_mmd_experiment(
        samples_per_class=args.samples_per_class,
        seed=args.seed,
    )
    geometry_results = kernel_geometry_sanity(
        gamma=mmd_results["gamma"],
        seed=args.seed,
    )

    payload = {
        "polynomial_regression_example": polynomial_results,
        "mean_embedding_demo": embedding_results,
        "mnist_mmd_experiment": mmd_results,
        "kernel_geometry_sanity": geometry_results,
    }
    write_results(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
