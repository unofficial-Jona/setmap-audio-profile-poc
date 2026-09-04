import numpy as np
import pytest

from setmap_audio.aggregation import aggregate_embeddings


def test_exports_normalized_global_and_weighted_spherical_clusters():
    embeddings = np.array(
        [[1.0, 0.0], [0.98, 0.02], [0.9, 0.1], [0.0, 1.0]], dtype=np.float32
    )
    result = aggregate_embeddings(embeddings)

    assert [vector.role for vector in result.vectors] == ["global", "cluster_1", "cluster_2"]
    assert all(np.linalg.norm(vector.vector) == pytest.approx(1.0) for vector in result.vectors)
    assert result.vectors[1].weight == pytest.approx(0.75)
    assert result.vectors[2].weight == pytest.approx(0.25)
    assert sum(vector.weight for vector in result.vectors[1:]) == pytest.approx(1.0)
    assert result.explained_variance_ratio[0] > 0.8


def test_single_embedding_produces_a_zero_weight_second_cluster():
    result = aggregate_embeddings(np.array([[3.0, 4.0]], dtype=np.float32))
    assert np.allclose(result.vectors[0].vector, [0.6, 0.8])
    assert [item.weight for item in result.vectors[1:]] == [1.0, 0.0]


def test_density_trim_removes_isolated_embedding_before_kmeans():
    embeddings = np.array(
        [[1.0, offset] for offset in np.linspace(0.0, 0.2, 19)] + [[0.0, 1.0]],
        dtype=np.float32,
    )
    result = aggregate_embeddings(embeddings, max_outlier_fraction=0.1)

    assert 19 in result.outlier_indices
    assert len(result.outlier_indices) <= 2
    assert result.core_clip_count + len(result.outlier_indices) == len(embeddings)
    assert [item.weight for item in result.vectors[1:]] == pytest.approx([0.5, 0.5])


def test_rejects_zero_vectors():
    with pytest.raises(ValueError, match="zero"):
        aggregate_embeddings(np.zeros((2, 3), dtype=np.float32))


def test_outlier_trim_is_disabled_by_default():
    embeddings = np.array([[1.0, 0.01 * index] for index in range(10)] + [[0.0, 1.0]])
    result = aggregate_embeddings(embeddings)
    assert result.core_clip_count == len(embeddings)
    assert result.outlier_indices == ()
