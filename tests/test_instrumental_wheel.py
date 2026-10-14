"""The installed wheel, without source-tree paths or MLX, can convert scores."""

import json
import os
from pathlib import Path
import subprocess
import sys
import venv

from test_instrumental import ABC


def test_installed_wheel_has_working_instrumental_helpers_and_license(tmp_path):
    root = Path(__file__).resolve().parents[1]
    dist = tmp_path / "dist"
    subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(dist)],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    environment = tmp_path / "environment"
    venv.EnvBuilder(with_pip=False).create(environment)
    python = environment / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(python),
            "--no-deps",
            str(next(dist.glob("*.whl"))),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [
            str(python),
            "-I",
            "-c",
            """
import json
from pathlib import Path
import sys
from lyra.music_tools.abc_tools import parse_abc
from lyra.music_tools.compile_score import compile_events
from lyra.music_tools.instrumental import lyric_tags
from lyra.music_tools.instrumentalize import convert_score
import lyra

text, check = convert_score(sys.stdin.read())
score = parse_abc(text)
print(json.dumps({
    "vocal_notes": len(score.voices["Vocal"].notes),
    "ins_notes": len(score.voices["Ins"].notes),
    "lyrics": lyric_tags(text),
    "license": (Path(lyra.__file__).parent / "licenses/third-party/yue2-instrumental-MIT.txt").read_text(),
}))
""",
        ],
        input=ABC,
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    assert data["vocal_notes"] == 0
    assert data["ins_notes"] == 5
    assert data["lyrics"] == "[Verse]\n\n[Outro]\n"
    assert "Copyright (c) 2026 yue2-music contributors" in data["license"]
    for name in ("instrumentalize", "instrumental", "compile_score"):
        help_result = subprocess.run(
            [str(python), "-I", "-m", f"lyra.music_tools.{name}", "--help"],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert help_result.returncode == 0, help_result.stderr
