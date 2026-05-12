"""Tests for the CC Decision Combiner — PR 3"""
import pytest

from cc_tool.audio.models import AudioEvent
from cc_tool.vision.models import ReactionResult
from cc_tool.decision.combiner import combine_and_decide


def _make_event(label: str, conf: float, start: float = 0.0, end: float = 1.0) -> AudioEvent:
    return AudioEvent(label=label, confidence=conf, start_sec=start, end_sec=end)


def _make_reaction(idx: int, conf: float) -> ReactionResult:
    return ReactionResult(event_index=idx, reaction_type="head_turn", confidence=conf, frame_path=None)


def test_above_threshold_emits_cc():
    events = [_make_event("Honking", 0.9, 1.0, 2.0)]
    reactions = [_make_reaction(0, 0.8)]
    annotations = combine_and_decide(events, reactions, cc_threshold=0.55)
    assert len(annotations) == 1
    assert annotations[0].text == "[honking]"
    assert annotations[0].index == 1


def test_below_threshold_no_cc():
    # audio=0.3, reaction=0.2 → final = 0.6*0.3 + 0.4*0.2 = 0.18 + 0.08 = 0.26
    events = [_make_event("Rain", 0.3, 5.0, 6.0)]
    reactions = [_make_reaction(0, 0.2)]
    annotations = combine_and_decide(events, reactions, cc_threshold=0.55)
    assert len(annotations) == 0


def test_no_reactions_uses_audio_weight_only():
    # audio=0.9, reaction=0 → final = 0.6*0.9 = 0.54 (below 0.55)
    events = [_make_event("Gunshot", 0.9, 2.0, 3.0)]
    annotations = combine_and_decide(events, [], cc_threshold=0.55)
    assert len(annotations) == 0  # 0.54 < 0.55


def test_no_reactions_above_threshold_with_lower_gate():
    # audio=1.0, reaction=0 → final = 0.6 >= 0.5
    events = [_make_event("Explosion", 1.0, 3.0, 4.0)]
    annotations = combine_and_decide(events, [], cc_threshold=0.5)
    assert len(annotations) == 1


def test_multiple_events_mixed():
    events = [
        _make_event("Honking", 0.9, 1.0, 2.0),   # should pass
        _make_event("Wind",    0.2, 5.0, 6.0),    # should fail
        _make_event("Alarm",   0.8, 10.0, 11.0),  # should pass
    ]
    reactions = [
        _make_reaction(0, 0.7),  # strong reaction
        _make_reaction(1, 0.1),  # weak reaction
        _make_reaction(2, 0.0),  # no reaction
    ]
    annotations = combine_and_decide(events, reactions, cc_threshold=0.55)
    assert len(annotations) == 2
    assert annotations[0].text == "[honking]"
    assert annotations[1].text == "[alarm]"


def test_sequential_indices():
    events = [_make_event("Laughter", 0.8, i, i+1) for i in range(3)]
    reactions = [_make_reaction(i, 0.7) for i in range(3)]
    annotations = combine_and_decide(events, reactions, cc_threshold=0.4)
    assert [ann.index for ann in annotations] == [1, 2, 3]
