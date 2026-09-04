from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

GLOBAL_SIMILARITY_WEIGHT = 0.25
CLUSTER_SIMILARITY_WEIGHT = 0.75


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError("Musical-map vectors must be non-zero")
    return matrix / norms


@dataclass(frozen=True, slots=True)
class GlobalProjection:
    """A frozen 512D-to-2D transform fitted on a versioned reference catalogue."""

    center: np.ndarray
    components: np.ndarray
    scale: np.ndarray
    variance: tuple[float, float]
    version: str
    reference_count: int

    def transform(self, vectors: np.ndarray) -> np.ndarray:
        matrix = _normalize_rows(np.asarray(vectors, dtype=np.float32))
        raw = (matrix - self.center) @ self.components.T
        return np.tanh(raw / self.scale)


def fit_global_projection(vectors: list[list[float]] | np.ndarray) -> GlobalProjection:
    """Fit deterministic PCA once so future profiles do not move existing points."""
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2 or not matrix.shape[0]:
        raise ValueError("A projection requires at least one global vector")
    matrix = _normalize_rows(matrix)

    center = matrix.mean(axis=0)
    centered = matrix - center
    dimensions = matrix.shape[1]
    variance = [0.0, 0.0]
    if np.allclose(centered, 0.0):
        components = np.zeros((2, dimensions), dtype=np.float32)
        components[0, 0] = 1.0
        components[1, min(1, dimensions - 1)] = 1.0
    else:
        _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
        component_count = min(2, vh.shape[0])
        components = vh[:component_count].astype(np.float32)
        if component_count == 1:
            fallback = np.zeros((1, dimensions), dtype=np.float32)
            fallback[0, int(np.argmin(np.abs(components[0])))] = 1.0
            fallback -= (fallback @ components.T) @ components
            fallback /= max(float(np.linalg.norm(fallback)), 1e-12)
            components = np.vstack([components, fallback])
        eigenvalues = singular_values**2
        total = max(float(eigenvalues.sum()), 1e-12)
        variance[:component_count] = [
            float(value / total) for value in eigenvalues[:component_count]
        ]

    reference_coordinates = centered @ components.T
    for axis in range(2):
        anchor = int(np.argmax(np.abs(reference_coordinates[:, axis])))
        if reference_coordinates[anchor, axis] < 0:
            components[axis] *= -1
            reference_coordinates[:, axis] *= -1
    scale = np.maximum(np.max(np.abs(reference_coordinates), axis=0), 1e-6)
    digest = hashlib.sha256()
    for item in (center, components, scale):
        digest.update(np.asarray(item, dtype="<f4").tobytes())
    digest.update(str(matrix.shape[0]).encode())

    return GlobalProjection(
        center=center.astype(np.float32),
        components=components,
        scale=scale.astype(np.float32),
        variance=(variance[0], variance[1]),
        version=f"pca-{digest.hexdigest()[:12]}",
        reference_count=int(matrix.shape[0]),
    )


def profile_similarity(left: dict, right: dict) -> tuple[float, float, float]:
    """Blend global cosine with symmetric, weight-aware cluster coverage."""
    left_roles = left["_by_role"]
    right_roles = right["_by_role"]
    global_score = float(
        np.dot(
            _normalize_rows(np.asarray([left_roles["global"]["vector"]]))[0],
            _normalize_rows(np.asarray([right_roles["global"]["vector"]]))[0],
        )
    )
    left_clusters = _normalize_rows(
        np.asarray([left_roles["cluster_1"]["vector"], left_roles["cluster_2"]["vector"]])
    )
    right_clusters = _normalize_rows(
        np.asarray([right_roles["cluster_1"]["vector"], right_roles["cluster_2"]["vector"]])
    )
    pairwise = left_clusters @ right_clusters.T
    left_weights = np.asarray(
        [left_roles["cluster_1"]["weight"], left_roles["cluster_2"]["weight"]], dtype=float
    )
    right_weights = np.asarray(
        [right_roles["cluster_1"]["weight"], right_roles["cluster_2"]["weight"]], dtype=float
    )
    left_weights /= max(float(left_weights.sum()), 1e-12)
    right_weights /= max(float(right_weights.sum()), 1e-12)
    cluster_score = 0.5 * (
        float(left_weights @ np.max(pairwise, axis=1))
        + float(right_weights @ np.max(pairwise, axis=0))
    )
    blended = (
        GLOBAL_SIMILARITY_WEIGHT * global_score + CLUSTER_SIMILARITY_WEIGHT * cluster_score
    )
    return blended, global_score, cluster_score


def build_musical_map(
    profiles: list[dict], projection: GlobalProjection | None = None
) -> dict:
    """Project global vectors only; retain cluster vectors for high-dimensional ranking."""
    if not profiles:
        raise ValueError("At least one profile is required")

    normalized_profiles: list[dict] = []
    global_vectors: list[np.ndarray] = []
    dimensions: int | None = None
    for profile in profiles:
        by_role = {item["role"]: item for item in profile["vectors"]}
        if set(by_role) != {"global", "cluster_1", "cluster_2"}:
            raise ValueError("Each map profile needs global, cluster_1, and cluster_2 vectors")
        for role in ("global", "cluster_1", "cluster_2"):
            vector = np.asarray(by_role[role]["vector"], dtype=np.float32)
            if vector.ndim != 1:
                raise ValueError("Musical-map vectors must be one-dimensional")
            dimensions = dimensions or int(vector.size)
            if vector.size != dimensions:
                raise ValueError("All musical-map vectors must have the same dimensions")
        global_vectors.append(np.asarray(by_role["global"]["vector"], dtype=np.float32))
        normalized_profiles.append({**profile, "_by_role": by_role})

    fitted_here = projection is None
    active_projection = projection or fit_global_projection(np.stack(global_vectors))
    if active_projection.center.size != dimensions:
        raise ValueError("Map projection dimensions do not match the profile vectors")
    coordinates = active_projection.transform(np.stack(global_vectors))

    similarities = np.empty((len(normalized_profiles), len(normalized_profiles)), dtype=float)
    global_similarities = np.empty_like(similarities)
    cluster_similarities = np.empty_like(similarities)
    for left_index, left in enumerate(normalized_profiles):
        for right_index, right in enumerate(normalized_profiles):
            blended, global_score, cluster_score = profile_similarity(left, right)
            similarities[left_index, right_index] = blended
            global_similarities[left_index, right_index] = global_score
            cluster_similarities[left_index, right_index] = cluster_score

    output_profiles = []
    for index, profile in enumerate(normalized_profiles):
        nearest = None
        if len(normalized_profiles) > 1:
            candidate_scores = similarities[index].copy()
            candidate_scores[index] = -np.inf
            nearest_index = int(np.argmax(candidate_scores))
            nearest = {
                "id": normalized_profiles[nearest_index]["id"],
                "label": normalized_profiles[nearest_index]["label"],
                "similarity": round(float(candidate_scores[nearest_index]), 6),
                "global_similarity": round(
                    float(global_similarities[index, nearest_index]), 6
                ),
                "cluster_similarity": round(
                    float(cluster_similarities[index, nearest_index]), 6
                ),
            }
        output_profiles.append(
            {
                "id": profile["id"],
                "label": profile["label"],
                "points": {
                    "global": {
                        "x": round(float(coordinates[index, 0]), 6),
                        "y": round(float(coordinates[index, 1]), 6),
                        "weight": 1.0,
                    }
                },
                "nearest": nearest,
            }
        )

    return {
        "method": "collection_pca_v2" if fitted_here else "frozen_reference_pca_v1",
        "projection_version": active_projection.version,
        "projection_is_stable": not fitted_here,
        "reference_profile_count": active_projection.reference_count,
        "similarity_method": "quarter_global_three_quarter_weighted_cluster_v1",
        "relative_to_profile_count": len(profiles),
        "axis_explained_variance_ratio": [
            round(value, 6) for value in active_projection.variance
        ],
        "profiles": output_profiles,
    }
