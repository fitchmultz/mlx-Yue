"""Skill script wrapper for RealAudio Tokenizer v4 operations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from common import find_python, resolve_project_root


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)

    p_tok = subparsers.add_parser("tokenize", help="Tokenize audio to semantic codes")
    p_tok.add_argument("audio", type=Path)
    p_tok.add_argument("--output", type=Path, required=True)

    p_rec = subparsers.add_parser("reconstruct", help="Reconstruct audio via NAR LoRA")
    p_rec.add_argument("audio", type=Path)
    p_rec.add_argument("--output", type=Path, required=True)
    p_rec.add_argument("--steps", type=int, default=32)
    p_rec.add_argument("--nar-scale", type=float, default=1.0)
    p_rec.add_argument("--seed", type=int, default=42)

    p_cmp = subparsers.add_parser("compare", help="Compare Stock vs Adapted 32/8-step")
    p_cmp.add_argument("audio", type=Path)
    p_cmp.add_argument("--output", type=Path, required=True)
    p_cmp.add_argument("--seed", type=int, default=42)

    args = parser.parse_args(argv)
    py = find_python()
    root = resolve_project_root()

    cmd = [str(py), "-m", "lyra.cli", "realaudio", args.action, str(args.audio), "--output", str(args.output)]
    if args.action == "reconstruct":
        cmd.extend(["--steps", str(args.steps), "--nar-scale", str(args.nar_scale), "--seed", str(args.seed)])
    elif args.action == "compare":
        cmd.extend(["--seed", str(args.seed)])

    return subprocess.call(cmd, cwd=str(root))


if __name__ == "__main__":
    sys.exit(main())
