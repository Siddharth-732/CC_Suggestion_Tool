"""Tests for SRT writer — PR 3"""
import os
import pytest

from cc_tool.decision.models import CCAnnotation
from cc_tool.export.srt_writer import write_srt, _sec_to_srt_time


def _make_ann(idx: int, start: float, end: float, text: str) -> CCAnnotation:
    return CCAnnotation(
        index=idx, start_sec=start, end_sec=end, text=text,
        audio_conf=0.8, reaction_conf=0.7, final_score=0.74,
    )


def test_sec_to_srt_time_basic():
    assert _sec_to_srt_time(0.0) == "00:00:00,000"
    assert _sec_to_srt_time(61.5) == "00:01:01,500"
    assert _sec_to_srt_time(3661.123) == "01:01:01,123"


def test_write_srt_creates_file(tmp_path):
    anns = [
        _make_ann(1, 4.2, 5.1, "[honking]"),
        _make_ann(2, 12.0, 13.5, "[laughter]"),
    ]
    out = str(tmp_path / "output.srt")
    write_srt(anns, out)
    assert os.path.exists(out)


def test_write_srt_content(tmp_path):
    anns = [_make_ann(1, 4.2, 5.1, "[honking]")]
    out = str(tmp_path / "test.srt")
    write_srt(anns, out)

    content = open(out, encoding="utf-8").read()
    assert "1" in content
    assert "00:00:04,200 --> 00:00:05,100" in content
    assert "[honking]" in content


def test_write_srt_empty(tmp_path):
    out = str(tmp_path / "empty.srt")
    write_srt([], out)
    assert os.path.exists(out)
    assert open(out).read().strip() == ""
