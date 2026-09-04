from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("Cannot normalize a zero or non-finite vector")
    return (vector / norm).astype(np.float32, copy=False)


@dataclass(frozen=True, slots=True)
class RepresentativeVector:
    role: str
    vector: np.ndarray
    weight: float


@dataclass(frozen=True, slots=True)
class AggregationResult:
    vectors: tuple[RepresentativeVector, ...]
    assignments: np.ndarray
    explained_variance_ratio: tuple[float, ...]
    pca_extreme_indices: tuple[int, int]
    cluster_strategy: str
    core_clip_count: int
    outlier_indices: tuple[int, ...]
    local_density_threshold: float | None


MIN_CLUSTER_FRACTION = 0.10
MAX_OUTLIER_FRACTION = 0.10
OUTLIER_MAD_MULTIPLIER = 3.5


def _ordered_clusters(
    centers: np.ndarray, assignments: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    counts = np.bincount(assignments, minlength=2)
    order = sorted(range(2), key=lambda index: (-int(counts[index]), tuple(centers[index, :8])))
    remap = {old: new for new, old in enumerate(order)}
    return centers[order], np.array([remap[int(value)] for value in assignments], dtype=int)


def _select_density_core(
    matrix: np.ndarray, max_outlier_fraction: float,
) -> tuple[np.ndarray, tuple[int, ...], float | None]:
    """Trim only robustly isolated clips, capped to preserve legitimate rare styles."""
    if max_outlier_fraction <= 0 or matrix.shape[0] < 8:
        return np.arange(matrix.shape[0]), (), None

    similarities = matrix @ matrix.T
    np.fill_diagonal(similarities, -np.inf)
    neighbour_count = min(matrix.shape[0] - 1, max(3, round(np.sqrt(matrix.shape[0]))))
    nearest = np.partition(similarities, -neighbour_count, axis=1)[:, -neighbour_count:]
    local_density = nearest.mean(axis=1)
    median = float(np.median(local_density))
    mad = float(np.median(np.abs(local_density - median)))
    robust_scale = max(1.4826 * mad, 1e-6)
    threshold = median - OUTLIER_MAD_MULTIPLIER * robust_scale
    candidates = np.flatnonzero(local_density < threshold)
    max_outliers = max(1, int(np.floor(matrix.shape[0] * max_outlier_fraction)))
    ranked = candidates[np.argsort(local_density[candidates], kind="stable")]
    outliers = tuple(int(index) for index in ranked[:max_outliers])
    mask = np.ones(matrix.shape[0], dtype=bool)
    mask[list(outliers)] = False
    return np.flatnonzero(mask), outliers, threshold


def _minimum_support_assignments(
    similarities_to_centers: np.ndarray,
) -> tuple[np.ndarray, bool]:
    assignments = np.argmax(similarities_to_centers, axis=1)
    minimum = max(1, int(np.ceil(assignments.size * MIN_CLUSTER_FRACTION)))
    counts = np.bincount(assignments, minlength=2)
    if np.min(counts) >= minimum:
        return assignments, False

    preference = similarities_to_centers[:, 0] - similarities_to_centers[:, 1]
    order = np.argsort(-preference, kind="stable")
    if counts[0] < minimum:
        assignments[:] = 1
        assignments[order[:minimum]] = 0
    else:
        assignments[:] = 0
        assignments[order[-minimum:]] = 1
    return assignments, True


def _spherical_kmeans(
    matrix: np.ndarray, max_iterations: int = 50
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Deterministic spherical k-means with a 10% minimum-support constraint."""
    if matrix.shape[0] == 1:
        return np.stack([matrix[0], matrix[0]]), np.array([0], dtype=int), False

    global_center = l2_normalize(matrix.mean(axis=0))
    first = int(np.argmax(matrix @ global_center))
    second = int(np.argmin(matrix @ matrix[first]))
    centers = np.stack([matrix[first], matrix[second]])
    assignments = np.full(matrix.shape[0], -1, dtype=int)
    constrained = False

    for _ in range(max_iterations):
        new_assignments, adjusted = _minimum_support_assignments(matrix @ centers.T)
        constrained = constrained or adjusted
        updated_centers = np.stack(
            [
                l2_normalize(matrix[new_assignments == cluster_index].mean(axis=0))
                for cluster_index in range(2)
            ]
        )
        if np.array_equal(new_assignments, assignments) and np.allclose(
            updated_centers, centers
        ):
            assignments = new_assignments
            centers = updated_centers
            break
        assignments = new_assignments
        centers = updated_centers

    centers, assignments = _ordered_clusters(centers, assignments)
    return centers, assignments, constrained


def aggregate_embeddings(
    embeddings: np.ndarray, *, max_outlier_fraction: float = 0.0
) -> AggregationResult:
    matrix = np.asarray(embeddings, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise ValueError("embeddings must be a non-empty 2D matrix")

    if not 0.0 <= max_outlier_fraction <= MAX_OUTLIER_FRACTION:
        raise ValueError("max_outlier_fraction must be between 0 and 0.10")

    normalized = np.stack([l2_normalize(row) for row in matrix])
    core_indices, outlier_indices, density_threshold = _select_density_core(
        normalized, max_outlier_fraction
    )
    core = normalized[core_indices]
    global_vector = l2_normalize(core.mean(axis=0))
    centers, core_assignments, constrained = _spherical_kmeans(core)
    if matrix.shape[0] == 1:
        cluster_strategy = "single_embedding_duplicate"
    elif constrained:
        cluster_strategy = "minimum_support_spherical_kmeans"
    else:
        cluster_strategy = "spherical_kmeans"
    assignments = np.full(matrix.shape[0], -1, dtype=int)
    assignments[core_indices] = core_assignments
    counts = np.bincount(core_assignments, minlength=2)
    weights = counts / core.shape[0]

    centered = normalized - normalized.mean(axis=0, keepdims=True)
    if matrix.shape[0] == 1 or np.allclose(centered, 0.0):
        variance: tuple[float, ...] = ()
        extremes = (0, 0)
    else:
        _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
        eigenvalues = singular_values**2
        total = float(eigenvalues.sum())
        variance = tuple(float(value / total) for value in eigenvalues[:3]) if total else ()
        projections = centered @ vh[0]
        extremes = (int(np.argmin(projections)), int(np.argmax(projections)))

    vectors = (
        RepresentativeVector("global", global_vector, 1.0),
        RepresentativeVector("cluster_1", centers[0], float(weights[0])),
        RepresentativeVector("cluster_2", centers[1], float(weights[1])),
    )
    return AggregationResult(
        vectors,
        assignments,
        variance,
        extremes,
        cluster_strategy,
        int(core.shape[0]),
        outlier_indices,
        density_threshold,
    )
