from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

SAMPLE_RATE = 48_000
CLIP_SECONDS = 10.0


class AudioError(ValueError):
    pass


def require_ffmpeg() -> None:
    missing = [binary for binary in ("ffmpeg", "ffprobe") if shutil.which(binary) is None]
    if missing:
        raise RuntimeError(f"Missing required command(s): {', '.join(missing)}")


def probe_duration(path: Path) -> float:
    require_ffmpeg()
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(command, capture_output=True, check=False, text=True, timeout=30)
    if result.returncode != 0:
        raise AudioError("The uploaded file could not be decoded as audio.")
    try:
        duration = float(json.loads(result.stdout)["format"]["duration"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise AudioError("The audio duration could not be determined.") from exc
    if not np.isfinite(duration) or duration <= 0:
        raise AudioError("The audio duration is invalid.")
    return duration


def sample_offsets(
    duration_seconds: float,
    *,
    clip_seconds: float = CLIP_SECONDS,
    interval_seconds: float = 30.0,
) -> list[float]:
    """Return deterministic, evenly spaced starts while always covering the final section."""
    if duration_seconds <= clip_seconds:
        return [0.0]
    if interval_seconds <= 0:
        raise ValueError("interval_seconds must be positive")

    last_start = max(0.0, duration_seconds - clip_seconds)
    offsets = list(np.arange(0.0, last_start + 1e-6, interval_seconds, dtype=float))
    if not offsets or last_start - offsets[-1] >= clip_seconds:
        offsets.append(last_start)
    return [round(float(offset), 3) for offset in offsets]


def _read_exact(stream, byte_count: int) -> bytes:
    chunks: list[bytes] = []
    remaining = byte_count
    while remaining:
        chunk = stream.read(min(remaining, 256 * 1024))
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def iter_decode_windows(
    path: Path, offsets: list[float], clip_seconds: float = CLIP_SECONDS
):
    """Stream one ffmpeg decode and retain only requested windows in memory."""
    if not offsets:
        return
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        "-f",
        "f32le",
        "pipe:1",
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert process.stdout is not None
    current_sample = 0
    window_samples = round(clip_seconds * SAMPLE_RATE)
    try:
        for start_seconds in offsets:
            target_sample = round(start_seconds * SAMPLE_RATE)
            skip_samples = max(0, target_sample - current_sample)
            skipped = _read_exact(process.stdout, skip_samples * 4)
            current_sample += len(skipped) // 4
            raw = _read_exact(process.stdout, window_samples * 4)
            current_sample += len(raw) // 4
            audio = np.frombuffer(raw, dtype="<f4").copy()
            if audio.size < SAMPLE_RATE:
                raise AudioError(f"Decoded audio near {start_seconds:.1f} seconds is too short.")
            yield start_seconds, audio
    finally:
        process.stdout.close()
        if process.poll() is None:
            process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
