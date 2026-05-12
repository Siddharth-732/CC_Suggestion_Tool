"""
Audio utilities — chunking and resampling helpers (PR 1)
"""
import numpy as np


def chunk_audio(
    waveform: np.ndarray,
    sample_rate: int,
    window_sec: float = 1.0,
    stride_sec: float = 0.5,
) -> list[tuple[float, float, np.ndarray]]:
    """
    Split a waveform into overlapping windows for sliding-window inference.

    Args:
        waveform:    1-D numpy array of audio samples (mono, float32)
        sample_rate: Sample rate of the audio (e.g. 16000)
        window_sec:  Duration of each window in seconds
        stride_sec:  Stride between windows in seconds

    Returns:
        List of (start_sec, end_sec, chunk_array) tuples
    """
    window_samples = int(window_sec * sample_rate)
    stride_samples = int(stride_sec * sample_rate)
    total_samples = len(waveform)

    chunks = []
    start = 0
    while start + window_samples <= total_samples:
        end = start + window_samples
        start_sec = start / sample_rate
        end_sec = end / sample_rate
        chunks.append((start_sec, end_sec, waveform[start:end]))
        start += stride_samples

    return chunks


def normalize_waveform(waveform: np.ndarray) -> np.ndarray:
    """Normalize audio to [-1.0, 1.0] float32."""
    waveform = waveform.astype(np.float32)
    max_val = np.max(np.abs(waveform))
    if max_val > 0:
        waveform = waveform / max_val
    return waveform
