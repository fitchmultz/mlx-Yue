"""Native MLX RealAudio Tokenizer Head and audio encoding pipeline."""
from __future__ import annotations

import json
from math import gcd
from pathlib import Path
from typing import Callable

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from scipy.signal import resample_poly


class MLXEncoderLayer(nn.Module):
    """Native MLX implementation of PyTorch TransformerEncoderLayer (norm_first=True, GELU)."""

    def __init__(self, d_model: int = 512, nhead: int = 8, dim_feedforward: int = 2048):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.in_proj_weight = mx.zeros((3 * d_model, d_model))
        self.in_proj_bias = mx.zeros((3 * d_model,))
        self.out_proj = nn.Linear(d_model, d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

    def __call__(self, x: mx.array) -> mx.array:
        # Pre-norm Self-Attention
        nx = self.norm1(x)
        b, t, d = nx.shape
        qkv = nx @ self.in_proj_weight.T + self.in_proj_bias
        q, k, v = mx.split(qkv, 3, axis=-1)
        q = q.reshape(b, t, self.nhead, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(b, t, self.nhead, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(b, t, self.nhead, self.head_dim).transpose(0, 2, 1, 3)
        scale = 1.0 / (self.head_dim ** 0.5)
        attn = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
        attn = attn.transpose(0, 2, 1, 3).reshape(b, t, d)
        x = x + self.out_proj(attn)

        # Pre-norm FFN
        nx2 = self.norm2(x)
        ffn = self.linear2(nn.gelu(self.linear1(nx2)))
        return x + ffn


class MLXRealAudioTokenizerHead(nn.Module):
    """8-layer Transformer mapping MERT2 L20 features (1024) to 32768 YuE2 semantic codes."""

    def __init__(
        self,
        d_in: int = 1024,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 8,
        vocab_size: int = 32768,
        window_size: int = 512,
    ):
        super().__init__()
        self.d_in = d_in
        self.d_model = d_model
        self.window_size = window_size
        self.inp = nn.Linear(d_in, d_model)
        self.pos = mx.zeros((1, window_size, d_model))
        self.layers = [
            MLXEncoderLayer(d_model, nhead, 4 * d_model) for _ in range(num_layers)
        ]
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def __call__(self, x: mx.array) -> mx.array:
        seq_len = x.shape[1]
        h = self.inp(x) + self.pos[:, :seq_len]
        for layer in self.layers:
            h = layer(h)
        return self.head(self.norm(h))


def instance_norm(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Per-track instance normalization on float32 features."""
    x = x.astype(np.float32)
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    return (x - mean) / (std + eps)


def interpolate_to_25hz(mert_feats: np.ndarray, target_frames: int) -> np.ndarray:
    """Linear time interpolation matching PyTorch interpolate(mode='linear', align_corners=False)."""
    t_in, c = mert_feats.shape
    if t_in == target_frames:
        return mert_feats
    scale = np.float32(t_in / target_frames)
    out_idx = np.arange(target_frames, dtype=np.float32)
    in_idx = (out_idx + np.float32(0.5)) * scale - np.float32(0.5)
    in_idx = np.clip(in_idx, 0, t_in - 1)
    idx_floor = np.floor(in_idx).astype(np.int64)
    idx_ceil = np.clip(idx_floor + 1, 0, t_in - 1)
    weight = (in_idx - idx_floor)[:, None]
    return (1.0 - weight) * mert_feats[idx_floor] + weight * mert_feats[idx_ceil]


def predict_tokens(
    head: MLXRealAudioTokenizerHead,
    features: np.ndarray,
    window_size: int = 512,
    hop_size: int = 256,
    cancelled: Callable[[], bool] | None = None,
) -> np.ndarray:
    """Run windowed argmax token prediction with seamless stitching across windows."""
    t_total = len(features)
    out = np.zeros(t_total, dtype=np.int64)
    starts = list(range(0, max(1, t_total - window_size + 1), hop_size))
    if starts[-1] + window_size < t_total:
        starts.append(max(0, t_total - window_size))

    quarter = window_size // 4
    for s0 in starts:
        if cancelled is not None and cancelled():
            raise InterruptedError("Cancelled during RealAudio token prediction")
        xw = features[s0 : s0 + window_size]
        n = len(xw)
        if n < window_size:
            xw = np.pad(xw, ((0, window_size - n), (0, 0)))
        x_mlx = mx.array(xw[None], dtype=mx.float32)
        logits = head(x_mlx)[0, :n]
        preds = np.array(logits.argmax(axis=-1))

        lo = s0 + (0 if s0 == 0 else quarter)
        hi = s0 + n - (0 if s0 + n >= t_total else quarter)
        out[lo:hi] = preds[lo - s0 : hi - s0]

    return out


def load_tokenizer_head(model_dir: Path | str) -> MLXRealAudioTokenizerHead:
    """Load native MLX RealAudio Tokenizer Head from converted safetensors."""
    model_dir = Path(model_dir)
    safetensors_path = model_dir / "tokenizer_head.safetensors"
    config_path = model_dir / "tokenizer_config.json"

    if not safetensors_path.exists():
        raise FileNotFoundError(f"Tokenizer head weights missing: {safetensors_path}")

    cfg = {}
    if config_path.exists():
        with open(config_path, encoding="utf-8") as f:
            cfg = json.load(f)

    head = MLXRealAudioTokenizerHead(
        d_in=cfg.get("d_in", 1024),
        d_model=cfg.get("d_model", 512),
        nhead=cfg.get("n_heads", 8),
        num_layers=cfg.get("num_layers", 8),
        vocab_size=cfg.get("vocab_size", 32768),
        window_size=cfg.get("window_size", 512),
    )

    weights = list(mx.load(str(safetensors_path)).items())
    head.load_weights(weights, strict=True)
    return head


def encode_realaudio(
    waveform: np.ndarray,
    sr: int,
    mert_model,
    tokenizer_head: MLXRealAudioTokenizerHead,
    cancelled: Callable[[], bool] | None = None,
) -> tuple[np.ndarray, dict]:
    """Encode audio into YuE2 semantic tokens (0..32767) at 25 Hz.

    Pipeline:
    1. Downmix to mono and resample to 24 kHz.
    2. Extract MERT2 layer-20 hidden states in 30-second windows.
    3. Interpolate MERT representations to exact 25 Hz sequence length.
    4. Instance normalize features over the entire song.
    5. Run MLX Tokenizer Head with overlapping 512-frame windows.
    """
    if cancelled is not None and cancelled():
        raise InterruptedError("Cancelled before audio encoding")

    # Downmix to mono
    if waveform.ndim > 1:
        mono = waveform.mean(axis=-1 if waveform.shape[-1] <= 2 else 0)
    else:
        mono = waveform
    mono = mono.astype(np.float32)

    # Resample to 24 kHz
    if sr != 24000:
        g = gcd(sr, 24000)
        mono24 = resample_poly(mono, 24000 // g, sr // g).astype(np.float32)
    else:
        mono24 = mono

    total_samples = len(mono24)
    if total_samples < 24000:
        # Minimum 1 second: pad with reflection / edge if very short
        mono24 = np.pad(mono24, (0, 24000 - total_samples), mode="reflect")

    # MERT2 layer 20 feature extraction in 30s (720,000 samples) chunks
    chunk_size = 24000 * 30
    mert_feats_list = []
    for s in range(0, len(mono24), chunk_size):
        if cancelled is not None and cancelled():
            raise InterruptedError("Cancelled during MERT2 feature extraction")
        chunk = mono24[s : s + chunk_size]
        if len(chunk) < 24000 and mert_feats_list:
            # Drop trailing chunk less than 1s if we already have chunks, matching upstream
            break
        elif len(chunk) < 24000:
            chunk = np.pad(chunk, (0, 24000 - len(chunk)), mode="reflect")

        # Call MERT2 with target_layer=20
        chunk_mx = mx.array(chunk[None], dtype=mx.float32)
        # target_layer=20 returns the 20th conformer block output
        l20_out = mert_model(chunk_mx, target_layer=20, cancelled=cancelled)
        # Flatten batch and time if needed: shape is [1, T_frames, 1024]
        mert_feats_list.append(np.array(l20_out[0]))

    concat_mert = np.concatenate(mert_feats_list, axis=0)

    # Calculate exact 25 Hz target frame count
    t25 = int(round(total_samples / 24000.0 * 25.0))
    # Interpolate time axis to exact 25 Hz
    feats25 = interpolate_to_25hz(concat_mert, t25)

    # Instance normalization over the full track
    normed_feats = instance_norm(feats25)

    # Predict semantic codes
    tokens = predict_tokens(
        tokenizer_head,
        normed_feats,
        window_size=tokenizer_head.window_size,
        hop_size=256,
        cancelled=cancelled,
    )

    meta = {
        "source_samples": total_samples,
        "source_seconds": float(total_samples / 24000.0),
        "target_25hz_frames": t25,
        "token_count": len(tokens),
        "unique_tokens": int(len(np.unique(tokens))),
    }
    return tokens, meta
