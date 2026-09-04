from __future__ import annotations

import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .config import Settings
from .embedder import create_embedder
from .pipeline import process_audio
from .telemetry import Telemetry


@dataclass(slots=True)
class Job:
    id: str
    status: str = "queued"
    stage: str = "queued"
    progress: float = 0.0
    message: str = "Waiting for the CPU worker"
    result: dict[str, Any] | None = None
    error: str | None = None
    created_at: float = field(default_factory=time.time)

    def public(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("created_at", None)
        if self.status != "complete":
            payload.pop("result", None)
        return payload


class JobManager:
    def __init__(self, settings: Settings, telemetry: Telemetry) -> None:
        self.settings = settings
        self.telemetry = telemetry
        self.embedder = create_embedder(
            settings.embedder_backend,
            settings.model_id,
            settings.model_revision,
            settings.torch_threads,
        )
        self.jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="clap-cpu")

    def submit(
        self,
        path: Path,
        *,
        interval_seconds: int,
        embedding_outlier_max_fraction: float | None = None,
    ) -> Job:
        outlier_fraction = (
            self.settings.embedding_outlier_max_fraction
            if embedding_outlier_max_fraction is None
            else embedding_outlier_max_fraction
        )
        job = Job(id=str(uuid.uuid4()))
        with self._lock:
            self.jobs[job.id] = job
        self.telemetry.record(
            job.id,
            "job_submitted",
            properties={
                "interval_seconds": interval_seconds,
                "embedding_outlier_max_fraction": outlier_fraction,
                "vector_count": 3,
            },
        )
        self._executor.submit(self._run, job.id, path, interval_seconds, outlier_fraction)
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self.jobs.get(job_id)

    def _update(self, job_id: str, stage: str, progress: float, message: str) -> None:
        with self._lock:
            job = self.jobs[job_id]
            job.status = "processing"
            job.stage = stage
            job.progress = round(float(progress), 3)
            job.message = message

    def _run(
        self, job_id: str, path: Path, interval_seconds: int, outlier_fraction: float
    ) -> None:
        started = time.perf_counter()
        try:
            result = process_audio(
                path,
                embedder=self.embedder,
                interval_seconds=interval_seconds,
                batch_size=self.settings.batch_size,
                max_duration_seconds=self.settings.max_duration_seconds,
                min_rms_dbfs=self.settings.min_rms_dbfs,
                max_clipped_ratio=self.settings.max_clipped_ratio,
                embedding_outlier_max_fraction=outlier_fraction,
                progress=lambda stage, progress, message: self._update(
                    job_id, stage, progress, message
                ),
            )
            with self._lock:
                job = self.jobs[job_id]
                job.status = "complete"
                job.stage = "complete"
                job.progress = 1.0
                job.message = "Audio profile ready"
                job.result = result
            self.telemetry.record(
                job_id,
                "job_completed",
                duration_ms=round((time.perf_counter() - started) * 1000),
                properties={
                    "windows_considered": result["source"]["windows_considered"],
                    "windows_retained": result["source"]["windows_retained"],
                    "windows_filtered": result["source"]["windows_filtered"],
                    "filter_counts": result["source"]["filter_counts"],
                    "aggregation_core_clip_count": result["aggregation"]["core_clip_count"],
                    "embedding_outlier_count": result["aggregation"][
                        "embedding_outlier_count"
                    ],
                    "duration_seconds": result["source"]["duration_seconds"],
                    "timing_ms": result["timing_ms"],
                    "model_load_ms": result["model"]["load_duration_ms"],
                    "batch_size": self.settings.batch_size,
                    "torch_threads": self.settings.torch_threads,
                    "interval_seconds": interval_seconds,
                    "embedding_outlier_max_fraction": outlier_fraction,
                },
            )
        except Exception as exc:
            path.unlink(missing_ok=True)
            with self._lock:
                job = self.jobs[job_id]
                job.status = "failed"
                job.stage = "failed"
                job.message = "Processing failed"
                job.error = str(exc)
            self.telemetry.record(
                job_id,
                "job_failed",
                duration_ms=round((time.perf_counter() - started) * 1000),
                properties={"error_type": type(exc).__name__},
            )
        finally:
            path.unlink(missing_ok=True)
