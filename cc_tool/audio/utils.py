import numpy as np

def chunk_audio(waveform, sample_rate, window_sec=1.0, stride_sec=0.5):
    """Split audio into overlapping windows for YAMNet."""
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

def normalize_waveform(waveform):
    """Normalize audio to [-1.0, 1.0]."""
    waveform = waveform.astype(np.float32)
    max_val = np.max(np.abs(waveform))
    if max_val > 0:
        waveform = waveform / max_val
    return waveform
