from cc_tool.audio.extractor import extract_audio
from cc_tool.audio.detector import SoundEventDetector
import sys
import os

def test_goal_1(video_path):
    print(f"Testing Goal 1 with: {video_path}")
    
    # 1. Extract
    wav_path = extract_audio(video_path)
    print(f"Audio extracted to: {wav_path}")
    
    # 2. Detect
    detector = SoundEventDetector(confidence_threshold=0.3)
    events = detector.detect(wav_path)
    
    print(f"\nDetected {len(events)} Events:")
    for e in events:
        print(f"[{e.start_sec:0.2f}s - {e.end_sec:0.2f}s] {e.label} (Conf: {e.confidence:0.2f})")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_goal_1.py <path_to_video>")
    else:
        test_goal_1(sys.argv[1])
