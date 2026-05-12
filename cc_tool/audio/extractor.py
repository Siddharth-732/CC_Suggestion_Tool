"""
Audio Extractor — extracts audio track from a video file (PR 1)
"""
import os
import subprocess
from pathlib import Path


def extract_audio(video_path: str, output_wav: str | None = None) -> str:
    """
    Extract the audio track from a video file and save it as a 16kHz mono WAV.

    Args:
        video_path: Path to the input video file (.mp4, .mkv, etc.)
        output_wav: Optional path for the output WAV file.
                    Defaults to <video_name>.wav in a temp directory.

    Returns:
        Path to the extracted WAV file.

    Raises:
        FileNotFoundError: If the video file does not exist.
        RuntimeError: If ffmpeg fails to extract audio.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    if output_wav is None:
        os.makedirs("outputs/audio", exist_ok=True)
        output_wav = f"outputs/audio/{video_path.stem}.wav"

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vn",                   # no video
        "-acodec", "pcm_s16le",  # 16-bit PCM
        "-ar", "16000",          # 16kHz sample rate (required by YAMNet)
        "-ac", "1",              # mono
        str(output_wav),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg audio extraction failed:\n{result.stderr}"
        )

    return str(output_wav)
