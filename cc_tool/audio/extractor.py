import os
import subprocess
from pathlib import Path

def extract_audio(video_path: str, output_wav: str | None = None) -> str:
    """
    Extract audio track from video using FFmpeg.
    Assumes 'ffmpeg' is available in the system PATH.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    if output_wav is None:
        os.makedirs("outputs/audio", exist_ok=True)
        output_wav = f"outputs/audio/{video_path.stem}.wav"
    
    # Standard command using system 'ffmpeg'
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        str(output_wav)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"FFmpeg failed. Ensure FFmpeg is installed and in your PATH.\nError: {result.stderr}"
        )
        
    return str(output_wav)
