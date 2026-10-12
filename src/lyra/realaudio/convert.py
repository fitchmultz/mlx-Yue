"""Convert upstream PyTorch realaudio v4 weights into clean MLX safetensors artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

UPSTREAM_REPO = "Mothersuperior/yue2-mothersuperior-realaudio-tokenizer-v4"
UPSTREAM_REVISION = "f2278a2e005dc4ecc421c53a0929f62b3aeb2280"

PINNED_FILES = {
    "tokenizer_head": {
        "filename": "tokenizer_head_joint_v4.pt",
        "sha256": "d23c4f757a05f031134b8471ec84245ec2338966516e1a9e26a17ff300a5f87e",
        "size": 171305291,
    },
    "nar_lora": {
        "filename": "nar_lora_joint_v4.pt",
        "sha256": "df175dbf9405a8e15b2c3f8dbdcc97303575f763787f227b03029020e28102fe",
        "size": 140626123,
    },
}

TOKENIZER_HEAD_CONFIG = {
    "architecture": "MLXRealAudioTokenizerHead",
    "d_in": 1024,
    "d_model": 512,
    "n_heads": 8,
    "dim_feedforward": 2048,
    "num_layers": 8,
    "vocab_size": 32768,
    "window_size": 512,
    "hop_size": 256,
    "instnorm": True,
    "instnorm_eps": 1e-5,
    "upstream_repo": UPSTREAM_REPO,
    "upstream_revision": UPSTREAM_REVISION,
}

NAR_LORA_CONFIG = {
    "architecture": "MLXRealAudioNARLoRA",
    "rank": 32,
    "num_layers": 28,
    "hidden_size": 2048,
    "modules": [
        "nar_self_attn.q_proj",
        "nar_self_attn.k_proj",
        "nar_self_attn.v_proj",
        "nar_self_attn.o_proj",
        "nar_mlp.gate_proj",
        "nar_mlp.up_proj",
        "nar_mlp.down_proj",
    ],
    "io_replacement": ["vae2llm", "llm2vae"],
    "upstream_repo": UPSTREAM_REPO,
    "upstream_revision": UPSTREAM_REVISION,
}


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024 * 4):
            h.update(chunk)
    return h.hexdigest()


def convert_tokenizer_head(
    src_path: Path,
    out_dir: Path,
    dtype: str = "float32",
) -> tuple[Path, dict[str, Any]]:
    import torch

    out_dir.mkdir(parents=True, exist_ok=True)
    actual_hash = _file_sha256(src_path)
    expected_hash = PINNED_FILES["tokenizer_head"]["sha256"]
    if actual_hash != expected_hash:
        raise ValueError(
            f"Tokenizer head hash mismatch: expected {expected_hash}, got {actual_hash}"
        )

    pt_data = torch.load(src_path, map_location="cpu", weights_only=False)
    state = pt_data.get("model", pt_data)

    target_dtype = np.float32 if dtype == "float32" else np.float16
    tensors: dict[str, np.ndarray] = {}

    tensors["pos"] = state["pos"].numpy().astype(target_dtype)
    tensors["inp.weight"] = state["inp.weight"].numpy().astype(target_dtype)
    tensors["inp.bias"] = state["inp.bias"].numpy().astype(target_dtype)
    tensors["norm.weight"] = state["norm.weight"].numpy().astype(target_dtype)
    tensors["norm.bias"] = state["norm.bias"].numpy().astype(target_dtype)
    tensors["head.weight"] = state["head.weight"].numpy().astype(target_dtype)
    tensors["head.bias"] = state["head.bias"].numpy().astype(target_dtype)

    for i in range(TOKENIZER_HEAD_CONFIG["num_layers"]):
        pfx = f"enc.layers.{i}."
        dst_pfx = f"layers.{i}."
        tensors[f"{dst_pfx}norm1.weight"] = (
            state[f"{pfx}norm1.weight"].numpy().astype(target_dtype)
        )
        tensors[f"{dst_pfx}norm1.bias"] = (
            state[f"{pfx}norm1.bias"].numpy().astype(target_dtype)
        )
        tensors[f"{dst_pfx}norm2.weight"] = (
            state[f"{pfx}norm2.weight"].numpy().astype(target_dtype)
        )
        tensors[f"{dst_pfx}norm2.bias"] = (
            state[f"{pfx}norm2.bias"].numpy().astype(target_dtype)
        )
        tensors[f"{dst_pfx}in_proj_weight"] = (
            state[f"{pfx}self_attn.in_proj_weight"].numpy().astype(target_dtype)
        )
        tensors[f"{dst_pfx}in_proj_bias"] = (
            state[f"{pfx}self_attn.in_proj_bias"].numpy().astype(target_dtype)
        )
        tensors[f"{dst_pfx}out_proj.weight"] = (
            state[f"{pfx}self_attn.out_proj.weight"].numpy().astype(target_dtype)
        )
        tensors[f"{dst_pfx}out_proj.bias"] = (
            state[f"{pfx}self_attn.out_proj.bias"].numpy().astype(target_dtype)
        )
        tensors[f"{dst_pfx}linear1.weight"] = (
            state[f"{pfx}linear1.weight"].numpy().astype(target_dtype)
        )
        tensors[f"{dst_pfx}linear1.bias"] = (
            state[f"{pfx}linear1.bias"].numpy().astype(target_dtype)
        )
        tensors[f"{dst_pfx}linear2.weight"] = (
            state[f"{pfx}linear2.weight"].numpy().astype(target_dtype)
        )
        tensors[f"{dst_pfx}linear2.bias"] = (
            state[f"{pfx}linear2.bias"].numpy().astype(target_dtype)
        )

    out_file = out_dir / "tokenizer_head.safetensors"
    mx_tensors = {k: mx.array(v) for k, v in tensors.items()}
    mx.save_safetensors(str(out_file), mx_tensors)

    cfg = dict(TOKENIZER_HEAD_CONFIG)
    cfg["dtype"] = dtype
    cfg_file = out_dir / "tokenizer_config.json"
    with open(cfg_file, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    meta = {
        "safetensors": out_file.name,
        "config": cfg_file.name,
        "sha256": _file_sha256(out_file),
        "source_sha256": actual_hash,
        "num_tensors": len(tensors),
    }
    return out_file, meta


def convert_nar_lora(
    src_path: Path,
    out_dir: Path,
    dtype: str = "bfloat16",
) -> tuple[Path, dict[str, Any]]:
    import torch

    out_dir.mkdir(parents=True, exist_ok=True)
    actual_hash = _file_sha256(src_path)
    expected_hash = PINNED_FILES["nar_lora"]["sha256"]
    if actual_hash != expected_hash:
        raise ValueError(
            f"NAR LoRA hash mismatch: expected {expected_hash}, got {actual_hash}"
        )

    pt_data = torch.load(src_path, map_location="cpu", weights_only=False)
    rank = pt_data.get("rank", 32)
    lora_list = pt_data["lora"]
    io_dict = pt_data.get("io", {})

    tensors: dict[str, np.ndarray] = {}
    it = iter(lora_list)
    modules = NAR_LORA_CONFIG["modules"]
    for layer_idx in range(NAR_LORA_CONFIG["num_layers"]):
        for mod in modules:
            A = next(it).float().numpy()
            B = next(it).float().numpy()
            tensors[f"layers.{layer_idx}.{mod}.lora_a"] = A
            tensors[f"layers.{layer_idx}.{mod}.lora_b"] = B

    for io_name in ("vae2llm", "llm2vae"):
        if io_name in io_dict:
            for param_name, tensor in io_dict[io_name].items():
                tensors[f"{io_name}.{param_name}"] = tensor.float().numpy()

    out_file = out_dir / "nar_lora.safetensors"
    mx_dtype = mx.float32 if dtype == "float32" else mx.bfloat16
    mx_tensors = {k: mx.array(v, dtype=mx_dtype) for k, v in tensors.items()}
    mx.save_safetensors(str(out_file), mx_tensors)

    cfg = dict(NAR_LORA_CONFIG)
    cfg["rank"] = rank
    cfg["dtype"] = dtype
    cfg_file = out_dir / "nar_lora_config.json"
    with open(cfg_file, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    meta = {
        "safetensors": out_file.name,
        "config": cfg_file.name,
        "sha256": _file_sha256(out_file),
        "source_sha256": actual_hash,
        "num_tensors": len(tensors),
    }
    return out_file, meta


def convert_all(
    upstream_dir: Path,
    out_dir: Path,
    head_dtype: str = "float32",
    nar_dtype: str = "bfloat16",
) -> dict[str, Any]:
    head_src = upstream_dir / PINNED_FILES["tokenizer_head"]["filename"]
    nar_src = upstream_dir / PINNED_FILES["nar_lora"]["filename"]

    _, head_meta = convert_tokenizer_head(head_src, out_dir, dtype=head_dtype)
    _, nar_meta = convert_nar_lora(nar_src, out_dir, dtype=nar_dtype)

    manifest = {
        "upstream_repo": UPSTREAM_REPO,
        "upstream_revision": UPSTREAM_REVISION,
        "tokenizer_head": head_meta,
        "nar_lora": nar_meta,
    }
    manifest_file = out_dir / "manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert realaudio v4 weights to MLX")
    parser.add_argument(
        "--upstream-dir",
        type=Path,
        default=Path("models/realaudio-v4/upstream"),
        help="Directory with upstream .pt files",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("models/realaudio-v4/converted"),
        help="Output directory for MLX safetensors",
    )
    parser.add_argument("--head-dtype", default="float32", choices=["float32", "float16"])
    parser.add_argument("--nar-dtype", default="bfloat16", choices=["bfloat16", "float32"])
    args = parser.parse_args()

    manifest = convert_all(
        args.upstream_dir,
        args.out_dir,
        head_dtype=args.head_dtype,
        nar_dtype=args.nar_dtype,
    )
    print("Conversion complete:")
    print(json.dumps(manifest, indent=2))
