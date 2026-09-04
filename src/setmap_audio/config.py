from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc


def _env_path(name: str) -> Path | None:
    value = os.getenv(name, "").strip()
    return Path(value).expanduser().resolve() if value else None


@dataclass(frozen=True, slots=True)
class Settings:
    model_id: str = os.getenv("AS_MODEL_ID", "laion/larger_clap_music")
    model_revision: str = os.getenv(
        "AS_MODEL_REVISION", "a0b4534a14f58e20944452dff00a22a06ce629d1"
    )
    embedder_backend: str = os.getenv("AS_EMBEDDER_BACKEND", "clap")
    torch_threads: int = _env_int("AS_TORCH_THREADS", 4)
    batch_size: int = _env_int("AS_BATCH_SIZE", 4)
    sample_interval_seconds: int = _env_int("AS_SAMPLE_INTERVAL_SECONDS", 25)
    max_duration_seconds: int = _env_int("AS_MAX_DURATION_SECONDS", 14400)
    max_upload_mib: int = _env_int("AS_MAX_UPLOAD_MIB", 1024)
    telemetry_enabled: bool = os.getenv("AS_TELEMETRY_ENABLED", "0") not in {"0", "false", "False"}
    data_dir: Path = Path(os.getenv("AS_DATA_DIR", "./data")).resolve()
    initial_profile_export: Path | None = _env_path("AS_INITIAL_PROFILE_EXPORT")
    min_rms_dbfs: float = _env_float("AS_MIN_RMS_DBFS", -48.0)
    max_clipped_ratio: float = _env_float("AS_MAX_CLIPPED_RATIO", 0.02)
    embedding_outlier_max_fraction: float = _env_float(
        "AS_EMBEDDING_OUTLIER_MAX_FRACTION", 0.0
    )

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mib * 1024 * 1024
