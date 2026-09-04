from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class FilterDecision:
    keep: bool
    reason: str | None
    metrics: dict[str, float]


def analyze_window(
    audio: np.ndarray,
    *,
    min_rms_dbfs: float = -48.0,
    max_clipped_ratio: float = 0.02,
) -> FilterDecision:
    """Reject only obvious signal failures with conservative deterministic thresholds."""
    samples = np.asarray(audio, dtype=np.float32)
    if samples.ndim != 1 or samples.size == 0 or not np.isfinite(samples).all():
        return FilterDecision(False, "invalid", {})

    rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
    rms_dbfs = float(20 * np.log10(max(rms, 1e-12)))
    peak = float(np.max(np.abs(samples)))
    clipped_ratio = float(np.mean(np.abs(samples) >= 0.999))
    dc_offset = float(abs(np.mean(samples)))

    frame_size = 2048
    usable = samples[: (samples.size // frame_size) * frame_size]
    if usable.size:
        frames = usable.reshape(-1, frame_size).astype(np.float64)
        frame_rms = np.sqrt(np.mean(frames**2, axis=1))
        active_ratio = float(np.mean(frame_rms > 10 ** ((min_rms_dbfs - 6) / 20)))
    else:
        active_ratio = 1.0

    reduced = samples[::8]
    zero_crossing_rate = float(np.mean(np.signbit(reduced[1:]) != np.signbit(reduced[:-1])))
    spectrum = np.abs(np.fft.rfft(reduced[: min(reduced.size, 65_536)])) ** 2
    spectral_flatness = float(
        np.exp(np.mean(np.log(spectrum + 1e-12))) / max(float(np.mean(spectrum)), 1e-12)
    )
    metrics = {
        "rms_dbfs": round(rms_dbfs, 3),
        "peak": round(peak, 6),
        "clipped_ratio": round(clipped_ratio, 6),
        "active_ratio": round(active_ratio, 6),
        "dc_offset": round(dc_offset, 6),
        "zero_crossing_rate": round(zero_crossing_rate, 6),
        "spectral_flatness": round(spectral_flatness, 6),
    }

    if rms_dbfs < min_rms_dbfs:
        return FilterDecision(False, "low_energy", metrics)
    if active_ratio < 0.2:
        return FilterDecision(False, "mostly_silent", metrics)
    if clipped_ratio > max_clipped_ratio:
        return FilterDecision(False, "clipped", metrics)
    if dc_offset > 0.2:
        return FilterDecision(False, "dc_offset", metrics)
    if spectral_flatness > 0.92 and zero_crossing_rate > 0.35:
        return FilterDecision(False, "broadband_noise", metrics)
    return FilterDecision(True, None, metrics)
