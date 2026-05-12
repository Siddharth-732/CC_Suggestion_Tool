"""cc_tool.audio package — Sound Event Detection (PR 1)"""
from .detector import SoundEventDetector
from .extractor import extract_audio
from .models import AudioEvent

__all__ = ["SoundEventDetector", "extract_audio", "AudioEvent"]
