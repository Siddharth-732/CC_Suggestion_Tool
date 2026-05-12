# Intelligent Closed Caption (CC) Suggestion Tool

> **Organisation:** Planet Read | **Domain:** Education  
> An AI-powered Python pipeline that identifies moments in a video where a Closed Caption (CC) annotation is genuinely necessary and suggests contextually relevant CC text.

---

## How It Works

```
Video File
    │
    ├─► [PR 1] Audio Extractor + YAMNet Detector
    │         → AudioEvent list (label, confidence, timestamps)
    │
    ├─► [PR 2] Frame Extractor + MediaPipe Reaction Detector
    │         → ReactionResult list (reaction_type, confidence)
    │
    └─► [PR 3] CC Decision Engine + SRT/SLS Writer
              → output.srt / output.sls
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

> **Note:** FFmpeg must be installed separately and available on your `PATH`.  
> Windows: `winget install ffmpeg` | Linux: `sudo apt install ffmpeg`

### 2. Run the full pipeline
```bash
python pipeline.py --input video.mp4 --output output.srt
```

### 3. Options
| Flag | Default | Description |
|---|---|---|
| `--format` | `srt` | Output format: `srt` or `sls` |
| `--threshold` | `0.55` | Minimum score to emit a CC |
| `--audio-weight` | `0.6` | Weight for audio confidence |
| `--reaction-weight` | `0.4` | Weight for visual reaction confidence |
| `--audio-threshold` | `0.5` | Pre-filter: minimum audio event confidence |
| `--skip-vision` | off | Skip visual analysis (audio-only mode, faster) |
| `--json-summary` | off | Print annotations as JSON to stdout |

---

## Project Structure

```
cc_tool/
├── audio/          # PR 1 — Sound Event Detection (YAMNet)
├── vision/         # PR 2 — Speaker Reaction Detection (MediaPipe)
├── decision/       # PR 3 — CC Decision Engine
└── export/         # PR 3 — SRT/SLS output
pipeline.py         # CLI entry point
tests/              # Unit tests per module
```

---

## Running Tests
```bash
pip install pytest
pytest tests/ -v
```

---

## PR Strategy

| PR | Branch | Module | Goal |
|---|---|---|---|
| PR 1 | `feature/pr1-sound-event-detection` | `cc_tool/audio/` | Detect non-speech audio events |
| PR 2 | `feature/pr2-speaker-reaction-detection` | `cc_tool/vision/` | Detect speaker reactions (mid-point milestone) |
| PR 3 | `feature/pr3-cc-decision-engine` | `cc_tool/decision/` + `cc_tool/export/` | Decision engine + SRT/SLS output |

---

## Output Format (SRT Example)

```
1
00:00:04,200 --> 00:00:05,100
[honking]

2
00:00:12,000 --> 00:00:13,500
[laughter]
```

---

## Tech Stack
- **Audio Detection:** [YAMNet](https://tfhub.dev/google/yamnet/1) via TensorFlow Hub
- **Frame Extraction:** OpenCV
- **Reaction Detection:** MediaPipe (FaceMesh + Pose)
- **Output:** Standard `.srt` and `.sls` subtitle formats
- **Language:** Python 3.10+