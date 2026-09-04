from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Protocol

import numpy as np


class AudioEmbedder(Protocol):
    model_id: str
    revision: str
    load_duration_ms: int

    def embed(self, audio_windows: Sequence[np.ndarray], sample_rate: int) -> np.ndarray: ...


class ClapEmbedder:
    """Lazy, CPU-only wrapper around Hugging Face CLAP."""

    def __init__(self, model_id: str, revision: str, torch_threads: int = 4) -> None:
        self.model_id = model_id
        self.revision = revision
        self.torch_threads = torch_threads
        self.load_duration_ms = 0
        self._model = None
        self._processor = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import ClapModel, ClapProcessor

        started = time.perf_counter()
        torch.set_num_threads(max(1, self.torch_threads))
        self._processor = ClapProcessor.from_pretrained(self.model_id, revision=self.revision)
        self._model = ClapModel.from_pretrained(self.model_id, revision=self.revision).to("cpu")
        self._model.eval()
        self.load_duration_ms = round((time.perf_counter() - started) * 1000)

    def embed(self, audio_windows: Sequence[np.ndarray], sample_rate: int) -> np.ndarray:
        self._load()
        import torch

        assert self._processor is not None
        assert self._model is not None
        inputs = self._processor(
            audio=list(audio_windows),
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=True,
        )
        with torch.inference_mode():
            features = self._model.get_audio_features(**inputs)
        vectors = features.detach().cpu().numpy().astype(np.float32, copy=False)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.maximum(norms, 1e-12)


class DeterministicFakeEmbedder:
    """Fast non-CLAP backend for tests and UI demos. Never use its vectors for retrieval."""

    model_id = "fake/deterministic-audio"
    revision = "v1"
    load_duration_ms = 0

    def embed(self, audio_windows: Sequence[np.ndarray], sample_rate: int) -> np.ndarray:
        del sample_rate
        vectors = []
        for audio in audio_windows:
            reduced = np.asarray(audio, dtype=np.float32)[::16]
            spectrum = np.log1p(np.abs(np.fft.rfft(reduced, n=8192))).astype(np.float32)
            indices = np.linspace(1, spectrum.size - 1, 512).astype(int)
            vector = spectrum[indices]
            vector -= vector.mean()
            norm = np.linalg.norm(vector)
            vectors.append(vector / max(float(norm), 1e-12))
        return np.stack(vectors).astype(np.float32)


def create_embedder(
    backend: str, model_id: str, revision: str, torch_threads: int
) -> AudioEmbedder:
    if backend == "fake":
        return DeterministicFakeEmbedder()
    if backend == "clap":
        return ClapEmbedder(model_id, revision, torch_threads)
    raise ValueError("AS_EMBEDDER_BACKEND must be 'clap' or 'fake'")
