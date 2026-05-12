# Intelligent CC Suggestion Tool — Goal 1

## Sound Event Detection Module
This module automatically detects and classifies non-speech audio events (like honking, laughter, music) from a video file.

### Installation
1.  **FFmpeg:** Ensure FFmpeg is installed on your system.
2.  **Dependencies:** 
    ```bash
    pip install -r requirements.txt
    ```

### Usage
```python
from cc_tool.audio import extract_audio, SoundEventDetector

# 1. Extract audio from video
wav_path = extract_audio("video.mp4")

# 2. Detect events
detector = SoundEventDetector(confidence_threshold=0.3)
events = detector.detect(wav_path)

for e in events:
    print(f"[{e.start_sec}s - {e.end_sec}s] {e.label} ({e.confidence})")
```

### Files
- `cc_tool/audio/extractor.py`: Handles audio stripping from video.
- `cc_tool/audio/detector.py`: YAMNet model implementation.
- `cc_tool/audio/models.py`: Data structures for audio events.
- `cc_tool/audio/utils.py`: Chunking and normalization helpers.
