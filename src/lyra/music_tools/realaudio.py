"""Real-audio tokenizer v4 CLI: tokenization, NAR LoRA reconstruction, and step comparisons."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import soundfile as sf

from ..measure import GPUExecution


def _load_paths():
    paths_file = Path("models/paths.json")
    if paths_file.exists():
        with open(paths_file, encoding="utf-8") as f:
            return json.load(f)
    return {
        "model": "models/mlx-Yue2-3B",
        "vae": "models/YuE2-Vae",
    }


def run_tokenize(args):
    from ..realaudio.pipeline import RealAudioPipeline

    paths = _load_paths()
    with GPUExecution(memory_budget_gib=args.memory_budget_gib) as guard:
        pipe = RealAudioPipeline.from_pretrained(
            model_dir=args.model or paths["model"],
            vae_dir=args.vae or paths["vae"],
            realaudio_dir=args.realaudio_dir,
            precision=args.precision,
            load_adapter=False,
        )
        tokens, meta = pipe.tokenize_audio(args.audio, output_dir=args.output)
        guard.check()

    print(json.dumps(meta, indent=2))
    print(f"Tokenized {meta['source_seconds']:.2f}s audio -> {len(tokens)} semantic tokens saved to {args.output}")
    return 0


def run_reconstruct(args):
    from ..realaudio.pipeline import RealAudioPipeline

    paths = _load_paths()
    with GPUExecution(memory_budget_gib=args.memory_budget_gib) as guard:
        pipe = RealAudioPipeline.from_pretrained(
            model_dir=args.model or paths["model"],
            vae_dir=args.vae or paths["vae"],
            realaudio_dir=args.realaudio_dir,
            precision=args.precision,
            load_adapter=True,
        )
        summary = pipe.reconstruct(
            audio_path=args.audio,
            output_dir=args.output,
            style=args.style,
            lyrics=args.lyrics,
            seed=args.seed,
            nar_scale=args.nar_scale,
            steps=args.steps,
        )
        guard.check()

    print(json.dumps(summary["reconstruction"], indent=2))
    print(f"Reconstruction finished: {summary['output_audio']} (RTF: {summary['reconstruction']['rtf']:.3f})")
    return 0


def run_compare(args):
    """Run a controlled comparison: Stock 32-step vs Adapted 32-step vs Adapted 8-step on same noise."""
    from ..realaudio.pipeline import RealAudioPipeline
    from ..pipeline import initial_noise

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    paths = _load_paths()

    with GPUExecution(memory_budget_gib=args.memory_budget_gib) as guard:
        pipe = RealAudioPipeline.from_pretrained(
            model_dir=args.model or paths["model"],
            vae_dir=args.vae or paths["vae"],
            realaudio_dir=args.realaudio_dir,
            precision=args.precision,
            load_adapter=True,
        )

        print(f"1/4 Tokenizing source audio: {args.audio} ...")
        tokens, tok_meta = pipe.tokenize_audio(args.audio, output_dir=out)
        np.save(str(out / "tokens.npy"), tokens.astype(np.int32))

        # Fix noise across all runs for exact comparability
        noise = initial_noise(len(tokens), args.seed)
        np.save(str(out / "noise.npy"), noise)

        runs = [
            {"name": "Stock_NAR_32step", "scale": 0.0, "steps": 32},
            {"name": "Adapted_NAR_32step", "scale": 1.0, "steps": 32},
            {"name": "Adapted_NAR_8step", "scale": 1.0, "steps": 8},
        ]

        results = []
        for i, run_cfg in enumerate(runs, start=2):
            tag = run_cfg["name"]
            print(f"{i}/4 Running {tag} (scale={run_cfg['scale']}, steps={run_cfg['steps']}) ...")
            audio, metrics = pipe.reconstruct_from_tokens(
                tokens=tokens,
                style=args.style,
                lyrics=args.lyrics,
                seed=args.seed,
                nar_scale=run_cfg["scale"],
                steps=run_cfg["steps"],
                noise=noise,
            )
            flac_path = out / f"{tag}.flac"
            sf.write(str(flac_path), audio, 48000, format="FLAC", subtype="PCM_24")
            metrics["audio_file"] = flac_path.name
            metrics["name"] = tag
            results.append(metrics)
            guard.check()

    report = {
        "input_audio": str(args.audio),
        "tokenization": tok_meta,
        "runs": results,
    }
    with open(out / "comparison_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Build an interactive listening comparison HTML
    html_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>RealAudio Tokenizer v4 MLX Comparison</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0f172a; color: #f8fafc; padding: 2rem; }}
        h1 {{ color: #38bdf8; font-size: 1.8rem; margin-bottom: 0.5rem; }}
        .meta {{ background: #1e293b; padding: 1rem; border-radius: 8px; margin-bottom: 1.5rem; font-size: 0.95rem; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 1.5rem; }}
        .card {{ background: #1e293b; border: 1px solid #334155; border-radius: 10px; padding: 1.25rem; }}
        .card h3 {{ margin-top: 0; color: #f1f5f9; }}
        .badge {{ display: inline-block; padding: 0.25rem 0.5rem; border-radius: 4px; font-size: 0.8rem; font-weight: bold; margin-bottom: 0.75rem; }}
        .badge-stock {{ background: #475569; color: #f8fafc; }}
        .badge-adapted {{ background: #0284c7; color: #f8fafc; }}
        .badge-fast {{ background: #16a34a; color: #f8fafc; }}
        audio {{ width: 100%; margin-top: 1rem; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 1rem; font-size: 0.9rem; }}
        th, td {{ padding: 0.5rem; text-align: left; border-bottom: 1px solid #334155; }}
        th {{ color: #94a3b8; }}
    </style>
</head>
<body>
    <h1>RealAudio Tokenizer v4 - MLX 试听对比</h1>
    <div class="meta">
        <p><strong>输入音频:</strong> {args.audio} ({tok_meta.get('source_seconds', 0):.2f}s, {len(tokens)} tokens)</p>
        <p><strong>测试环境:</strong> Apple Silicon (MLX Native) | 相同固定噪声 PCG64</p>
    </div>
    <div class="grid">
"""
    for r in results:
        badge_cls = "badge-stock" if r["nar_scale"] == 0 else ("badge-fast" if r["steps"] == 8 else "badge-adapted")
        html_content += f"""
        <div class="card">
            <span class="badge {badge_cls}">{r['name']}</span>
            <h3>{r['name']}</h3>
            <table>
                <tr><th>求解步数</th><td>{r['steps']} 步</td></tr>
                <tr><th>NAR LoRA</th><td>{'已加载 (scale=1.0)' if r['nar_scale'] > 0 else '未应用 (Stock)'}</td></tr>
                <tr><th>NAR 耗时</th><td>{r['nar_time_seconds']:.2f} s</td></tr>
                <tr><th>VAE 耗时</th><td>{r['vae_time_seconds']:.2f} s</td></tr>
                <tr><th>总重建耗时</th><td>{r['total_reconstruction_seconds']:.2f} s</td></tr>
                <tr><th>RTF</th><td><strong>{r['rtf']:.3f}</strong></td></tr>
            </table>
            <audio controls src="{r['audio_file']}"></audio>
        </div>
"""
    html_content += """
    </div>
</body>
</html>
"""
    with open(out / "index.html", "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"Comparison completed successfully! Output saved to: {out}")
    print(f"Report: {out / 'index.html'}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)

    # Tokenize
    p_tok = subparsers.add_parser("tokenize", help="Encode audio recording to semantic tokens")
    p_tok.add_argument("audio", type=Path, help="Input audio path")
    p_tok.add_argument("--output", type=Path, required=True, help="Directory to save tokens.npy/json")
    p_tok.add_argument("--model", type=Path, default=None)
    p_tok.add_argument("--vae", type=Path, default=None)
    p_tok.add_argument("--realaudio-dir", type=Path, default=Path("models/realaudio-v4/converted"))
    p_tok.add_argument("--precision", default="bf16")
    p_tok.add_argument("--memory-budget-gib", type=float, default=24)

    # Reconstruct
    p_rec = subparsers.add_parser("reconstruct", help="Encode audio and reconstruct via NAR LoRA + VAE")
    p_rec.add_argument("audio", type=Path, help="Input audio path")
    p_rec.add_argument("--output", type=Path, required=True, help="Output directory")
    p_rec.add_argument("--model", type=Path, default=None)
    p_rec.add_argument("--vae", type=Path, default=None)
    p_rec.add_argument("--realaudio-dir", type=Path, default=Path("models/realaudio-v4/converted"))
    p_rec.add_argument("--precision", default="bf16")
    p_rec.add_argument("--nar-scale", type=float, default=1.0)
    p_rec.add_argument("--steps", type=int, default=32)
    p_rec.add_argument("--seed", type=int, default=42)
    p_rec.add_argument("--style", default="acoustic, original recording")
    p_rec.add_argument("--lyrics", default="[instrumental]")
    p_rec.add_argument("--memory-budget-gib", type=float, default=24)

    # Compare
    p_cmp = subparsers.add_parser("compare", help="Compare Stock vs Adapted 32/8-step reconstruction")
    p_cmp.add_argument("audio", type=Path, help="Input audio path")
    p_cmp.add_argument("--output", type=Path, required=True, help="Output comparison directory")
    p_cmp.add_argument("--model", type=Path, default=None)
    p_cmp.add_argument("--vae", type=Path, default=None)
    p_cmp.add_argument("--realaudio-dir", type=Path, default=Path("models/realaudio-v4/converted"))
    p_cmp.add_argument("--precision", default="bf16")
    p_cmp.add_argument("--seed", type=int, default=42)
    p_cmp.add_argument("--style", default="acoustic, original recording")
    p_cmp.add_argument("--lyrics", default="[instrumental]")
    p_cmp.add_argument("--memory-budget-gib", type=float, default=24)

    # Convert
    p_conv = subparsers.add_parser("convert", help="Convert upstream .pt weights to safetensors")
    p_conv.add_argument("--upstream-dir", type=Path, default=Path("models/realaudio-v4/upstream"))
    p_conv.add_argument("--output", type=Path, default=Path("models/realaudio-v4/converted"))
    p_conv.add_argument("--head-dtype", default="float32")
    p_conv.add_argument("--nar-dtype", default="bfloat16")

    args = parser.parse_args(argv)

    if args.action == "tokenize":
        return run_tokenize(args)
    elif args.action == "reconstruct":
        return run_reconstruct(args)
    elif args.action == "compare":
        return run_compare(args)
    elif args.action == "convert":
        from ..realaudio.convert import convert_all
        convert_all(args.upstream_dir, args.output, head_dtype=args.head_dtype, nar_dtype=args.nar_dtype)
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
