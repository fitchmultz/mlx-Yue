"""End-to-end RealAudio Tokenizer and Acoustic Reconstruction Pipeline."""
from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

import mlx.core as mx
import numpy as np
import soundfile as sf

from ..pipeline import YuE2Pipeline
from ..transcription.model import resolve_models
from ..mert import MERT2, convert_mert_weights
from .tokenizer import load_tokenizer_head, encode_realaudio
from .adapter import NARLoRAAdapter, ARLoRAAdapter
from yue2.protocol import SongRequest, token_prefixes


def _resolve_mert(cache_dir: str | Path = "models/hf-cache") -> MERT2:
    """Load native MLX MERT2 model."""
    _, mert_dir = resolve_models(cache_dir=cache_dir, offline=True)
    with open(mert_dir / "config.json", encoding="utf-8") as f:
        cfg = json.load(f)
    mert = MERT2(cfg)
    raw_weights = mx.load(str(mert_dir / "model.safetensors"))
    weights = convert_mert_weights(raw_weights)
    mert.load_weights(weights, strict=True)
    return mert


class RealAudioPipeline:
    """Manages audio tokenization, adapter loading, and end-to-end reconstruction."""

    def __init__(
        self,
        yue_pipeline: YuE2Pipeline,
        tokenizer_head=None,
        mert_model=None,
        nar_adapter: NARLoRAAdapter | None = None,
        converted_dir: Path | str = "models/realaudio-v4/converted",
    ):
        self.yue_pipeline = yue_pipeline
        self.converted_dir = Path(converted_dir)
        self._tokenizer_head = tokenizer_head
        self._mert = mert_model
        self.nar_adapter = nar_adapter

    @classmethod
    def from_pretrained(
        cls,
        model_dir: Path | str = "models/converted",
        vae_dir: Path | str = "models/vae",
        realaudio_dir: Path | str = "models/realaudio-v4/converted",
        precision: str = "bf16",
        require_ac: bool = True,
        load_adapter: bool = True,
    ) -> "RealAudioPipeline":
        realaudio_path = Path(realaudio_dir)
        yue_pipe = YuE2Pipeline.from_pretrained(
            model_dir,
            vae=vae_dir,
            precision=precision,
            local_files_only=True,
            require_ac=require_ac,
        )

        nar_adapter = None
        if load_adapter and (realaudio_path / "nar_lora.safetensors").exists():
            nar_adapter = NARLoRAAdapter.from_safetensors(realaudio_path)

        return cls(yue_pipeline=yue_pipe, nar_adapter=nar_adapter, converted_dir=realaudio_path)

    def _ensure_mert(self):
        if self._mert is None:
            self._mert = _resolve_mert()
        return self._mert

    def _ensure_head(self):
        if self._tokenizer_head is None:
            self._tokenizer_head = load_tokenizer_head(self.converted_dir)
        return self._tokenizer_head

    def tokenize_audio(
        self,
        audio_path: Path | str,
        output_dir: Path | str | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Encode an audio recording into 25 Hz semantic tokens (0..32767)."""
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Input audio not found: {audio_path}")

        t0 = time.perf_counter()
        waveform, sr = sf.read(str(audio_path), dtype="float32")
        read_time = time.perf_counter() - t0

        mert = self._ensure_mert()
        head = self._ensure_head()

        t1 = time.perf_counter()
        tokens, meta = encode_realaudio(waveform, sr, mert, head)
        encode_time = time.perf_counter() - t1

        meta["read_seconds"] = read_time
        meta["encode_seconds"] = encode_time
        meta["audio_file"] = str(audio_path)

        if output_dir:
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            np.save(str(out / "tokens.npy"), tokens.astype(np.int32))
            with open(out / "tokens.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "meta": meta,
                        "tokens_preview": tokens[:100].tolist(),
                        "token_count": len(tokens),
                    },
                    f,
                    indent=2,
                )

        return tokens, meta

    def reconstruct_from_tokens(
        self,
        tokens: np.ndarray,
        style: str = "acoustic, original recording",
        lyrics: str = "[instrumental]",
        seed: int = 42,
        nar_scale: float = 1.0,
        ar_adapter: ARLoRAAdapter | None = None,
        ar_scale: float = 1.0,
        steps: int = 32,
        noise: np.ndarray | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reconstruct 48kHz audio from semantic tokens using NAR acoustic flow matching and VAE."""
        pipe = self.yue_pipeline
        nar_model = pipe._load_model(for_nar=True)

        if self.nar_adapter is not None:
            self.nar_adapter.apply(nar_model, scale=nar_scale)

        req = SongRequest(style=style, lyrics=lyrics, cot="off", seed=seed, id="realaudio")
        prefix = token_prefixes(req, pipe.tokenizer)

        t_start = time.perf_counter()
        token_list = [int(x) for x in tokens]

        from ..nar import synthesize as nar_synthesize

        # Synthesize NAR latents
        nar_start = time.perf_counter()
        latents = nar_synthesize(
            nar_model,
            prefix=prefix,
            codec=token_list,
            seed=seed,
            steps=steps,
            noise=noise,
            query_chunk_size=pipe.query_chunk_size,
        )
        nar_time = time.perf_counter() - nar_start

        # Decode VAE latents
        vae_start = time.perf_counter()
        audio = pipe.decode(latents)
        vae_time = time.perf_counter() - vae_start
        total_time = time.perf_counter() - t_start

        audio_seconds = float(len(audio) / 48000.0)
        rtf = total_time / audio_seconds if audio_seconds > 0 else 0.0

        metrics = {
            "tokens_count": len(tokens),
            "audio_seconds": audio_seconds,
            "steps": steps,
            "nar_scale": nar_scale,
            "nar_time_seconds": nar_time,
            "vae_time_seconds": vae_time,
            "total_reconstruction_seconds": total_time,
            "rtf": rtf,
            "sample_rate": 48000,
        }

        if self.nar_adapter is not None:
            self.nar_adapter.revert(nar_model)

        return audio, metrics

    def reconstruct(
        self,
        audio_path: Path | str,
        output_dir: Path | str,
        style: str = "acoustic, original recording",
        lyrics: str = "[instrumental]",
        seed: int = 42,
        nar_scale: float = 1.0,
        steps: int = 32,
    ) -> dict[str, Any]:
        """End-to-end encode and reconstruct audio with metrics."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        tokens, tok_meta = self.tokenize_audio(audio_path, output_dir=out)
        audio, rec_meta = self.reconstruct_from_tokens(
            tokens=tokens,
            style=style,
            lyrics=lyrics,
            seed=seed,
            nar_scale=nar_scale,
            steps=steps,
        )

        out_flac = out / "reconstructed.flac"
        sf.write(str(out_flac), audio, 48000, format="FLAC", subtype="PCM_24")

        summary = {
            "tokenization": tok_meta,
            "reconstruction": rec_meta,
            "output_audio": str(out_flac),
            "output_tokens": str(out / "tokens.npy"),
        }
        with open(out / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        return summary
