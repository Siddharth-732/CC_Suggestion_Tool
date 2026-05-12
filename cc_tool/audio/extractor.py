import os
import subprocess
from pathlib import Path

# Direct path to ffmpeg to bypass Windows Path issues
FFMPEG_PATH = r"C:\Users\siddh\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1.1-full_build\bin\ffmpeg.exe"

def extract_audio(video_path: str, output_wav: str | None = None) -> str:
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    if output_wav is None:
        os.makedirs("outputs/audio", exist_ok=True)
        output_wav = f"outputs/audio/{video_path.stem}.wav"
    
    # Use the absolute path to ffmpeg
    cmd = [
        FFMPEG_PATH, "-y",
        "-i", str(video_path),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        str(output_wav)
    ]
    
    # Check if ffmpeg exists at the hardcoded path
    if not os.path.exists(FFMPEG_PATH):
        # Fallback to just "ffmpeg" if the hardcoded one isn't there
        cmd[0] = "ffmpeg"

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg failed: {result.stderr}")
        
    return str(output_wav)
