import numpy as np
import soundfile as sf
import tensorflow_hub as hub
import csv
import os
from cc_tool.audio.models import AudioEvent
from cc_tool.audio.utils import chunk_audio, normalize_waveform

# AudioSet indices for speech - we ignore these
SPEECH_INDICES = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

class SoundEventDetector:
    def __init__(self, confidence_threshold=0.3):
        self.confidence_threshold = confidence_threshold
        self._model = None
        self._class_names = []

    def _load_model(self):
        if self._model is None:
            print("Loading YAMNet model from TF Hub...")
            self._model = hub.load("https://tfhub.dev/google/yamnet/1")
            
            # Load class names - Fixed for Windows file paths
            class_map_path = self._model.class_map_path().numpy().decode()
            
            if os.path.exists(class_map_path):
                # If it's a local file path (common on Windows)
                with open(class_map_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    self._class_names = [row["display_name"] for row in reader]
            else:
                # If it's a URL
                import urllib.request
                with urllib.request.urlopen(class_map_path) as f:
                    reader = csv.DictReader(line.decode("utf-8") for line in f)
                    self._class_names = [row["display_name"] for row in reader]
            
            print(f"Model loaded with {len(self._class_names)} classes.")

    def detect(self, wav_path):
        self._load_model()
        waveform, sr = sf.read(wav_path, dtype="float32")
        if waveform.ndim > 1: waveform = waveform.mean(axis=1)
        waveform = normalize_waveform(waveform)

        chunks = chunk_audio(waveform, sr)
        raw_events = []

        print("Analyzing audio...")
        for start_sec, end_sec, chunk in chunks:
            scores, _, _ = self._model(chunk)
            mean_scores = scores.numpy().mean(axis=0)
            top_idx = int(np.argmax(mean_scores))
            top_score = float(mean_scores[top_idx])

            if top_idx not in SPEECH_INDICES and top_score >= self.confidence_threshold:
                raw_events.append(AudioEvent(
                    label=self._class_names[top_idx],
                    confidence=top_score,
                    start_sec=start_sec,
                    end_sec=end_sec
                ))

        return self._merge_events(raw_events)

    def _merge_events(self, events):
        if not events: return []
        events.sort(key=lambda x: x.start_sec)
        merged = [events[0]]
        for curr in events[1:]:
            prev = merged[-1]
            if curr.label == prev.label and curr.start_sec <= prev.end_sec:
                prev.end_sec = max(prev.end_sec, curr.end_sec)
                prev.confidence = max(prev.confidence, curr.confidence)
            else:
                merged.append(curr)
        return merged
