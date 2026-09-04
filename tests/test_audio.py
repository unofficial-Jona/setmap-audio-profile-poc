import pytest

from setmap_audio.audio import sample_offsets


def test_short_track_has_one_window():
    assert sample_offsets(7.5) == [0.0]


def test_sampling_includes_final_section_when_gap_is_large():
    assert sample_offsets(112, interval_seconds=30) == [0.0, 30.0, 60.0, 90.0, 102.0]


def test_sampling_does_not_duplicate_nearby_tail():
    assert sample_offsets(70, interval_seconds=30) == [0.0, 30.0, 60.0]


def test_short_tail_does_not_create_an_overlapping_stream_window():
    assert sample_offsets(18, interval_seconds=25) == [0.0]


def test_interval_must_be_positive():
    with pytest.raises(ValueError):
        sample_offsets(20, interval_seconds=0)
