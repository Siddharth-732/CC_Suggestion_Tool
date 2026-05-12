"""
YAMNet-based Sound Event Detector (PR 1)

YAMNet is a pre-trained deep net that predicts 521 audio event classes
from AudioSet. We filter to non-speech classes and apply a confidence threshold.
"""
from __future__ import annotations

import numpy as np
import soundfile as sf
import tensorflow_hub as hub

from cc_tool.audio.models import AudioEvent
from cc_tool.audio.utils import chunk_audio, normalize_waveform

# AudioSet class indices that are SPEECH-related — we exclude these
# (class 0 = Speech, 1 = Male speech, 2 = Female speech, etc.)
SPEECH_CLASS_INDICES = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

# YAMNet TF Hub URL
YAMNET_MODEL_URL = "https://tfhub.dev/google/yamnet/1"


class SoundEventDetector:
    """
    Wraps YAMNet for sliding-window non-speech audio event detection.

    Usage:
        detector = SoundEventDetector(confidence_threshold=0.5)
        events = detector.detect("path/to/audio.wav")
    """

    def __init__(
        self,
        confidence_threshold: float = 0.5,
        window_sec: float = 1.0,
        stride_sec: float = 0.5,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.window_sec = window_sec
        self.stride_sec = stride_sec
        self._model = None
        self._class_names: list[str] = []

    def _load_model(self) -> None:
        """Lazy-load the YAMNet model on first use."""
        if self._model is None:
            print("[SoundEventDetector] Loading YAMNet model from TF Hub...")
            self._model = hub.load(YAMNET_MODEL_URL)
            # YAMNet provides class names via the model's class_map_path()
            import csv, urllib.request
            class_map_path = self._model.class_map_path().numpy().decode()
            with urllib.request.urlopen(class_map_path) as f:
                reader = csv.DictReader(
                    line.decode("utf-8") for line in f
                )
                self._class_names = [row["display_name"] for row in reader]
            print(f"[SoundEventDetector] Model loaded. {len(self._class_names)} classes.")

    def detect(self, wav_path: str) -> list[AudioEvent]:
        """
        Run sound event detection on a WAV file.

        Args:
            wav_path: Path to a 16kHz mono WAV file.

        Returns:
            List of AudioEvent objects above the confidence threshold,
            sorted by start time.
        """
        self._load_model()

        waveform, sample_rate = sf.read(wav_path, dtype="float32")
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)  # stereo → mono
        waveform = normalize_waveform(waveform)

        chunks = chunk_audio(waveform, sample_rate, self.window_sec, self.stride_sec)
        events: list[AudioEvent] = []

        for start_sec, end_sec, chunk in chunks:
            # YAMNet expects float32 waveform
            scores, embeddings, spectrogram = self._model(chunk)
            # scores shape: [num_frames, num_classes] — mean across frames
            mean_scores = scores.numpy().mean(axis=0)

            top_idx = int(np.argmax(mean_scores))
            top_score = float(mean_scores[top_idx])

            if top_idx in SPEECH_CLASS_INDICES:
                continue  # skip speech events
            if top_score < self.confidence_threshold:
                continue  # below threshold

            label = self._class_names[top_idx] if self._class_names else f"class_{top_idx}"
            events.append(
                AudioEvent(
                    label=label,
                    confidence=top_score,
                    start_sec=start_sec,
                    end_sec=end_sec,
                )
            )

        # Merge overlapping events with same label
        events = _merge_overlapping(events)
        return sorted(events, key=lambda e: e.start_sec)


def _merge_overlapping(events: list[AudioEvent]) -> list[AudioEvent]:
    """
    Merge consecutive events with the same label if their windows overlap.
    Keeps the highest confidence among merged events.
    """
    if not events:
        return events

    events = sorted(events, key=lambda e: e.start_sec)
    merged: list[AudioEvent] = [events[0]]

    for event in events[1:]:
        prev = merged[-1]
        if event.label == prev.label and event.start_sec <= prev.end_sec:
            # Extend the window, keep max confidence
            merged[-1] = AudioEvent(
                label=prev.label,
                confidence=max(prev.confidence, event.confidence),
                start_sec=prev.start_sec,
                end_sec=max(prev.end_sec, event.end_sec),
            )
        else:
            merged.append(event)

    return merged
