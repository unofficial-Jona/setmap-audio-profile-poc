import time
from pathlib import Path

from setmap_audio.config import Settings
from setmap_audio.jobs import JobManager


class NullTelemetry:
    def record(self, *args, **kwargs):
        pass


def test_failed_job_removes_temporary_audio(monkeypatch, tmp_path: Path):
    audio = tmp_path / "upload.mp3"
    audio.write_bytes(b"not audio")
    settings = Settings(embedder_backend="fake", data_dir=tmp_path)
    manager = JobManager(settings, NullTelemetry())

    def fail(*args, **kwargs):
        raise RuntimeError("sanitized test failure")

    monkeypatch.setattr("setmap_audio.jobs.process_audio", fail)
    job = manager.submit(audio, interval_seconds=25)
    deadline = time.time() + 2
    while manager.get(job.id).status not in {"failed", "complete"} and time.time() < deadline:
        time.sleep(0.01)

    assert manager.get(job.id).status == "failed"
    assert not audio.exists()
