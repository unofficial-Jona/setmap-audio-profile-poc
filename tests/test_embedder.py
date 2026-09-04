import numpy as np
import pytest

from setmap_audio.embedder import DeterministicFakeEmbedder


def test_fake_embedder_is_deterministic_and_normalized():
    rng = np.random.default_rng(42)
    audio = rng.normal(0, 0.1, 48_000).astype(np.float32)
    embedder = DeterministicFakeEmbedder()
    first = embedder.embed([audio], 48_000)
    second = embedder.embed([audio], 48_000)
    assert first.shape == (1, 512)
    assert np.allclose(first, second)
    assert np.linalg.norm(first[0]) == pytest.approx(1.0)
