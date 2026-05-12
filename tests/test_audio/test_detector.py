"""Tests for audio chunking utils — PR 1"""
import numpy as np
import pytest

from cc_tool.audio.utils import chunk_audio, normalize_waveform


def test_chunk_audio_basic():
    sr = 16000
    duration_sec = 5.0
    waveform = np.zeros(int(sr * duration_sec), dtype=np.float32)
    chunks = chunk_audio(waveform, sr, window_sec=1.0, stride_sec=0.5)

    # With 5s audio, 1s window, 0.5s stride: 9 full windows
    assert len(chunks) == 9
    for start_sec, end_sec, chunk in chunks:
        assert end_sec > start_sec
        assert len(chunk) == sr  # 1s window = 16000 samples


def test_chunk_audio_short_clip():
    """Audio shorter than one window → no chunks."""
    sr = 16000
    waveform = np.zeros(int(sr * 0.3), dtype=np.float32)
    chunks = chunk_audio(waveform, sr, window_sec=1.0, stride_sec=0.5)
    assert len(chunks) == 0


def test_normalize_waveform_range():
    waveform = np.array([0, 100, -200, 50], dtype=np.float32)
    result = normalize_waveform(waveform)
    assert result.max() <= 1.0
    assert result.min() >= -1.0


def test_normalize_waveform_silent():
    """Silent waveform should return all zeros without division error."""
    waveform = np.zeros(1000, dtype=np.float32)
    result = normalize_waveform(waveform)
    assert np.all(result == 0.0)
