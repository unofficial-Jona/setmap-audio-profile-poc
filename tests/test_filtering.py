import numpy as np

from setmap_audio.filtering import analyze_window


def test_filters_silence():
    decision = analyze_window(np.zeros(48_000, dtype=np.float32))
    assert not decision.keep
    assert decision.reason == "low_energy"


def test_filters_heavily_clipped_audio():
    decision = analyze_window(np.ones(48_000, dtype=np.float32))
    assert not decision.keep
    assert decision.reason == "clipped"


def test_keeps_a_clean_tone():
    time = np.arange(48_000 * 2) / 48_000
    audio = (0.2 * np.sin(2 * np.pi * 440 * time)).astype(np.float32)
    decision = analyze_window(audio)
    assert decision.keep
    assert decision.reason is None
