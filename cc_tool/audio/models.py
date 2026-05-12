"""
Intelligent CC Suggestion Tool — Audio Event Data Models (PR 1)
"""
from dataclasses import dataclass


@dataclass
class AudioEvent:
    """Represents a single detected non-speech audio event."""
    label: str           # e.g. "Honking", "Laughter", "Gunshot"
    confidence: float    # Model confidence [0.0, 1.0]
    start_sec: float     # Event start time in seconds
    end_sec: float       # Event end time in seconds

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "start_sec": round(self.start_sec, 3),
            "end_sec": round(self.end_sec, 3),
        }

    def __repr__(self) -> str:
        return (
            f"AudioEvent(label='{self.label}', confidence={self.confidence:.2f}, "
            f"start={self.start_sec:.2f}s, end={self.end_sec:.2f}s)"
        )
