"""Tests for audio extractor — PR 1"""
import os
import struct
import wave
import pytest

from cc_tool.audio.extractor import extract_audio


def _make_dummy_wav(path: str, duration_sec: float = 2.0, sample_rate: int = 16000) -> None:
    """Create a minimal silent WAV file for testing."""
    n_samples = int(duration_sec * sample_rate)
    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack("<" + "h" * n_samples, *([0] * n_samples)))


def test_extract_audio_missing_file(tmp_path):
    """Should raise FileNotFoundError for non-existent input."""
    with pytest.raises(FileNotFoundError):
        extract_audio(str(tmp_path / "nonexistent.mp4"))
