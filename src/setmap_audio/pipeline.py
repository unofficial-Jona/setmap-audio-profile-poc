from __future__ import annotations

import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import numpy as np

from .aggregation import aggregate_embeddings
from .audio import CLIP_SECONDS, SAMPLE_RATE, iter_decode_windows, probe_duration, sample_offsets
from .embedder import AudioEmbedder
from .filtering import analyze_window
from .schemas import AudioProfileExport

ProgressCallback = Callable[[str, float, str], None]


def _timestamp(seconds: float) -> str:
    minutes, secs = divmod(int(round(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def process_audio(
    path: Path,
    *,
    embedder: AudioEmbedder,
    interval_seconds: int,
    batch_size: int,
    max_duration_seconds: int,
    min_rms_dbfs: float,
    max_clipped_ratio: float,
    progress: ProgressCallback,
    embedding_outlier_max_fraction: float = 0.0,
) -> dict:
    started = time.perf_counter()
    stage_started = started
    stage_durations: dict[str, int] = {}
    progress("probing", 0.02, "Checking the audio")
    duration = probe_duration(path)
    stage_durations["probe"] = round((time.perf_counter() - stage_started) * 1000)
    if duration > max_duration_seconds + 0.5:
        raise ValueError(f"Audio is longer than the {max_duration_seconds // 60}-minute limit.")

    offsets = sample_offsets(duration, interval_seconds=interval_seconds)
    retained_offsets: list[float] = []
    filter_counts: Counter[str] = Counter()
    batches: list[np.ndarray] = []
    pending_audio: list[np.ndarray] = []
    embedding_ms = 0
    filtering_ms = 0

    for index, (offset, audio) in enumerate(iter_decode_windows(path, offsets)):
        filter_started = time.perf_counter()
        decision = analyze_window(
            audio, min_rms_dbfs=min_rms_dbfs, max_clipped_ratio=max_clipped_ratio
        )
        filtering_ms += round((time.perf_counter() - filter_started) * 1000)
        if decision.keep:
            retained_offsets.append(offset)
            pending_audio.append(audio)
        else:
            filter_counts[decision.reason or "unknown"] += 1

        is_batch_end = len(pending_audio) == batch_size or index == len(offsets) - 1
        if is_batch_end and pending_audio:
            progress(
                "embedding",
                0.06 + 0.84 * (index + 1) / max(1, len(offsets)),
                f"Reviewed {index + 1} of {len(offsets)} windows · kept {len(retained_offsets)}",
            )
            embedding_started = time.perf_counter()
            batches.append(embedder.embed(pending_audio, SAMPLE_RATE))
            embedding_ms += round((time.perf_counter() - embedding_started) * 1000)
            pending_audio.clear()

    if not batches:
        raise ValueError("No usable musical windows remained after signal-quality filtering.")

    stage_durations["decode_and_filter"] = round(
        (time.perf_counter() - stage_started) * 1000
    ) - stage_durations["probe"] - embedding_ms
    stage_durations["filter_compute"] = filtering_ms
    stage_durations["embed"] = embedding_ms
    progress("aggregating", 0.94, "Building global and cluster vectors")
    aggregation_started = time.perf_counter()
    embeddings = np.concatenate(batches, axis=0)
    aggregation = aggregate_embeddings(
        embeddings, max_outlier_fraction=embedding_outlier_max_fraction
    )
    stage_durations["aggregate"] = round((time.perf_counter() - aggregation_started) * 1000)

    vectors = [
        {
            "role": item.role,
            "weight": round(item.weight, 6),
            "vector": [round(float(value), 8) for value in item.vector],
        }
        for item in aggregation.vectors
    ]
    extremes = [
        {
            "side": side,
            "retained_clip_index": clip_index,
            "start_seconds": retained_offsets[clip_index],
            "timecode": _timestamp(retained_offsets[clip_index]),
        }
        for side, clip_index in zip(("low", "high"), aggregation.pca_extreme_indices, strict=True)
    ]

    stage_durations["total"] = round((time.perf_counter() - started) * 1000)
    global_array = aggregation.vectors[0].vector
    payload = {
        "schema_version": "1.0",
        "model": {
            "id": embedder.model_id,
            "revision": embedder.revision,
            "dimensions": int(embeddings.shape[1]),
            "device": "cpu",
            "backend": "fake" if embedder.model_id.startswith("fake/") else "clap",
            "load_duration_ms": embedder.load_duration_ms,
        },
        "source": {
            "duration_seconds": round(duration, 3),
            "sample_interval_seconds": interval_seconds,
            "window_seconds": CLIP_SECONDS,
            "windows_considered": len(offsets),
            "windows_retained": len(retained_offsets),
            "windows_filtered": len(offsets) - len(retained_offsets),
            "filter_counts": dict(sorted(filter_counts.items())),
        },
        "aggregation": {
            "method": "robust_spherical_kmeans",
            "version": "3",
            "cluster_strategy": aggregation.cluster_strategy,
            "outlier_method": (
                "knn_density_mad_3.5" if embedding_outlier_max_fraction else "disabled"
            ),
            "max_outlier_fraction": embedding_outlier_max_fraction,
            "input_clip_count": int(embeddings.shape[0]),
            "core_clip_count": aggregation.core_clip_count,
            "embedding_outlier_count": len(aggregation.outlier_indices),
            "local_density_threshold": (
                round(aggregation.local_density_threshold, 8)
                if aggregation.local_density_threshold is not None
                else None
            ),
            "vectors": vectors,
        },
        "diagnostics": {
            "pca_explained_variance_ratio": [
                round(value, 6) for value in aggregation.explained_variance_ratio
            ],
            "pca_extreme_snippets": extremes,
            "diversity": round(1.0 - float(np.mean(embeddings @ global_array)), 6),
        },
        "timing_ms": stage_durations,
    }
    return AudioProfileExport.model_validate(payload).model_dump()
