"""
Intelligent CC Suggestion Tool — Audio Event Data Models
"""
from dataclasses import dataclass

@dataclass
class AudioEvent:
    label: str
    confidence: float
    start_sec: float
    end_sec: float

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "start_sec": round(self.start_sec, 3),
            "end_sec": round(self.end_sec, 3),
        }
