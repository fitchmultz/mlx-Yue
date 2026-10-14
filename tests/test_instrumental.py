"""Instrumental conversion and portable preparation; no models or audio needed."""

from fractions import Fraction as F
import json
from pathlib import Path
import subprocess
import sys

import pytest

from lyra.music_tools.abc_tools import parse_abc
from lyra.music_tools.compile_score import compile_events
from lyra.music_tools.instrumental import load_prepared, lyric_tags
from lyra.music_tools.instrumentalize import convert_score


ABC = """X:1
T:
M:4/4
L:1/32
Q:1/4=92
V: Vocal clef=treble name="Vocal Melody" snm="Vocal"
V: Ins clef=treble name="Ins Melody" snm="Inst."
K:D
% verse
V: Vocal
"D"z8^F8-"A7"F16-|F8=F8z16|
V: Ins
G8z24|z16E8z8|
% outro
V: Vocal
M:5/8
K:Bb
"Bb"z4B8z8|
V: Ins
M:5/8
K:Bb
Z|
"""


def test_transfer_preserves_exact_sound_rests_meter_tempo_and_harmony():
    text, check = convert_score(ABC)
    score = parse_abc(text)
    assert score.voices["Vocal"].notes == []
    assert score.voices["Ins"].notes == [
        [F(0), 67, F(1)],
        [F(1), 66, F(4)],
        [F(5), 65, F(1)],
        [F(6), 64, F(1)],
        [F(17, 2), 70, F(1)],
    ]
    assert score.voices["Ins"].bars == [
        (F(0), F(4), (4, 4)),
        (F(4), F(4), (4, 4)),
        (F(8), F(5, 2), (5, 8)),
    ]
    assert score.voices["Ins"].time == F(21, 2)
    assert score.bpm == 92
    assert score.voices["Vocal"].chords == [(F(0), "D"), (F(2), "A7"), (F(4), "A7"), (F(8), "Bb")]
    assert score.voices["Ins"].keys == [(F(0), "D"), (F(8), "Bb")]
    assert lyric_tags(text) == "[Verse]\n\n[Outro]\n"
    assert check["affected_ins_notes"] == []


def test_overlap_clips_only_existing_ins_and_reports_retained_segments():
    source = ABC.replace("G8z24|z16E8z8|", "G32|A32|")
    text, check = convert_score(source, overlap="vocal")
    assert parse_abc(text).voices["Ins"].notes == [
        [F(0), 67, F(1)],
        [F(1), 66, F(4)],
        [F(5), 65, F(1)],
        [F(6), 69, F(2)],
        [F(17, 2), 70, F(1)],
    ]
    assert check["affected_ins_notes"] == [
        {
            "ins_note": 0,
            "onset": "0",
            "pitch": 67,
            "duration": "4",
            "retained_segments": [["0", "1"]],
        },
        {
            "ins_note": 1,
            "onset": "4",
            "pitch": 69,
            "duration": "4",
            "retained_segments": [["6", "2"]],
        },
    ]
    with pytest.raises(ValueError, match="overlap"):
        convert_score(source, overlap="error")


def test_melody_cover_removes_only_harmony():
    text, check = convert_score(ABC, keep_chords=False)
    score = parse_abc(text)
    assert score.voices["Vocal"].chords == []
    assert score.voices["Ins"].notes[-1] == [F(17, 2), 70, F(1)]
    assert check["chords"] == "removed for melody cover"


@pytest.mark.parametrize(
    "source",
    [
        ABC.replace("^F8", "(3F8"),
        ABC.replace('"D"z8^F8', '"D"z8[K:C]^F8').replace("G8z24", "G8[K:C]z24"),
    ],
)
def test_unsupported_notation_fails_instead_of_rewriting_destructively(source):
    with pytest.raises(ValueError):
        convert_score(source)


def test_compiler_preserves_fractional_notes_and_rearticulation_across_changes():
    text, check = compile_events(
        {
            "bpm": 90,
            "key": "C",
            "bars": [{"section": "intro"}, {"meter": "5/8", "key": "F#"}],
            "notes": [[0, "5/4", 60], ["5/4", "7/4", 60], [3, "7/2", 66]],
        }
    )
    score = parse_abc(text)
    assert score.voices["Ins"].notes == [
        [F(0), 60, F(5, 4)],
        [F(5, 4), 60, F(7, 4)],
        [F(3), 66, F(7, 2)],
    ]
    assert score.voices["Vocal"].notes == []
    assert score.voices["Ins"].time == F(13, 2)
    assert check["event_roundtrip"]
    with pytest.raises(ValueError, match="training grid"):
        compile_events({"bpm": 90, "bars": [{}], "notes": [[0, "1/3", 60]]})


def test_wrapper_prepares_verified_cover_without_overwriting_and_detects_tampering(tmp_path):
    source = tmp_path / "original.abc"
    source.write_text(ABC)
    output = tmp_path / "cover"
    wrapper = Path(__file__).resolve().parents[1] / "skills/yue2-music/scripts/instrumental.py"
    command = [
        sys.executable,
        str(wrapper),
        "cover",
        "--abc",
        str(source),
        "--keep-harmony",
        "--style",
        "Warm piano",
        "--prepare-only",
        "--output",
        str(output),
    ]
    result = subprocess.run(command, capture_output=True, text=True, cwd=tmp_path, timeout=30)
    assert result.returncode == 0, result.stderr
    request = load_prepared(output / "prepared")
    assert request["cot"] == "full"
    assert request["lyrics"] == "[Verse]\n\n[Outro]\n"
    assert "no singing" in request["style"]
    assert parse_abc(request["abc"]).voices["Vocal"].notes == []
    assert json.loads((output / "conversion/transfer.json").read_text())["output_ins_notes"] == 5
    original = source.read_bytes()
    repeated = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert repeated.returncode == 2
    assert source.read_bytes() == original
    assert load_prepared(output / "prepared") == request
    (output / "prepared/score.abc").write_text("tampered")
    with pytest.raises(ValueError, match="corrupt"):
        load_prepared(output / "prepared")
