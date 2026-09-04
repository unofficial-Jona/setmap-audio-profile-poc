from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .config import Settings
from .jobs import JobManager
from .musical_map import GlobalProjection, build_musical_map, fit_global_projection
from .schemas import MusicalMapRequest
from .telemetry import Telemetry

PACKAGE_DIR = Path(__file__).resolve().parent
STATIC_DIR = PACKAGE_DIR / "static"
ALLOWED_EXTENSIONS = {".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg", ".opus"}


def _load_initial_profiles(path: Path | None) -> list[dict]:
    if path is None:
        return []
    if not path.is_file():
        raise ValueError("AS_INITIAL_PROFILE_EXPORT does not point to a file")
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("The initial profile export is not valid JSON") from exc
    profiles = payload.get("profiles", []) if isinstance(payload, dict) else []
    if not isinstance(profiles, list):
        raise ValueError("The initial profile export needs a profiles array")
    return [
        profile
        for profile in profiles[:100]
        if isinstance(profile, dict)
        and isinstance(profile.get("result") or profile.get("profile"), dict)
        and (profile.get("result") or profile.get("profile", {}))
        .get("aggregation", {})
        .get("vectors")
    ]


def _fit_reference_projection(profiles: list[dict]) -> GlobalProjection | None:
    vectors = []
    for record in profiles:
        result = record.get("result") or record.get("profile", {})
        by_role = {item.get("role"): item for item in result["aggregation"]["vectors"]}
        if by_role.get("global", {}).get("vector"):
            vectors.append(by_role["global"]["vector"])
    return fit_global_projection(vectors) if len(vectors) >= 3 else None


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    telemetry = Telemetry(settings.data_dir, settings.telemetry_enabled)
    jobs = JobManager(settings, telemetry)
    initial_profile_error: str | None = None
    try:
        initial_profile_records = _load_initial_profiles(settings.initial_profile_export)
        map_projection = _fit_reference_projection(initial_profile_records)
    except ValueError as exc:
        initial_profile_records = []
        map_projection = None
        initial_profile_error = str(exc)

    app = FastAPI(title="SetMap Audio Profile", version="0.1.0")
    app.state.settings = settings
    app.state.jobs = jobs
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/health")
    async def health() -> dict:
        return {
            "status": "ok",
            "model_loaded": getattr(jobs.embedder, "_model", None) is not None,
            "device": "cpu",
            "backend": settings.embedder_backend,
            "model_id": jobs.embedder.model_id,
            "model_revision": jobs.embedder.revision,
            "map_projection": {
                "method": "frozen_reference_pca_v1" if map_projection else "collection_pca_v2",
                "stable": map_projection is not None,
                "version": map_projection.version if map_projection else None,
                "reference_profile_count": map_projection.reference_count if map_projection else 0,
            },
        }

    @app.get("/api/initial-profiles")
    async def initial_profiles() -> dict:
        """Load an optional local batch export into the browser-side profile library."""
        if initial_profile_error:
            raise HTTPException(422, initial_profile_error)
        return {"profiles": initial_profile_records}

    @app.post("/api/jobs", status_code=202)
    async def create_job(
        audio: Annotated[UploadFile, File()],
        interval_seconds: Annotated[int | None, Form()] = None,
        embedding_outlier_max_fraction: Annotated[float | None, Form()] = None,
    ) -> dict:
        interval_seconds = interval_seconds or settings.sample_interval_seconds
        suffix = Path(audio.filename or "audio").suffix.lower()
        if suffix not in ALLOWED_EXTENSIONS:
            raise HTTPException(415, "Use MP3, WAV, FLAC, M4A, AAC, OGG, or Opus audio.")
        if interval_seconds not in {20, 25, 30}:
            raise HTTPException(422, "interval_seconds must be 20, 25, or 30")
        outlier_fraction = (
            settings.embedding_outlier_max_fraction
            if embedding_outlier_max_fraction is None
            else embedding_outlier_max_fraction
        )
        if outlier_fraction not in {0.0, 0.05, 0.1}:
            raise HTTPException(422, "embedding_outlier_max_fraction must be 0, 0.05, or 0.10")
        size = 0
        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                prefix="setmap-", suffix=suffix, delete=False
            ) as temp:
                temp_path = Path(temp.name)
                while chunk := await audio.read(1024 * 1024):
                    size += len(chunk)
                    if size > settings.max_upload_bytes:
                        raise HTTPException(413, f"Upload exceeds {settings.max_upload_mib} MiB.")
                    temp.write(chunk)
        except Exception:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
            raise

        assert temp_path is not None
        job = jobs.submit(
            temp_path,
            interval_seconds=interval_seconds,
            embedding_outlier_max_fraction=outlier_fraction,
        )
        return {"job_id": job.id, "status_url": f"/api/jobs/{job.id}"}

    @app.get("/api/jobs/{job_id}")
    async def get_job(job_id: str) -> dict:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "Job not found")
        return job.public()

    @app.get("/api/jobs/{job_id}/export")
    async def export_job(job_id: str) -> JSONResponse:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "Job not found")
        if job.status != "complete" or job.result is None:
            raise HTTPException(409, "Job is not complete")
        return JSONResponse(
            job.result,
            headers={
                "Content-Disposition": f'attachment; filename="audio-profile-{job_id}.json"'
            },
        )

    @app.post("/api/musical-map")
    async def musical_map(request: MusicalMapRequest) -> dict:
        try:
            return build_musical_map(
                [profile.model_dump() for profile in request.profiles],
                projection=map_projection,
            )
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    return app


app = create_app()


def run() -> None:
    import uvicorn

    print(json.dumps({"url": "http://127.0.0.1:8000", "device": "cpu"}))
    uvicorn.run("setmap_audio.app:app", host="127.0.0.1", port=8000, reload=False)
