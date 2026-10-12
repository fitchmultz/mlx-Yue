"""NAR and AR LoRA adapter management for native MLX YuE2 models."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import mlx.core as mx



class NARLoRAAdapter:
    """Manages the rank-32 NAR LoRA and custom vae2llm / llm2vae IO weights."""

    def __init__(
        self,
        lora_weights: Mapping[str, mx.array],
        io_weights: Mapping[str, Mapping[str, mx.array]],
        rank: int = 32,
    ):
        self.lora_weights = dict(lora_weights)
        self.io_weights = dict(io_weights)
        self.rank = rank
        self._applied_scale: float | None = None
        self._backup_weights: dict[str, Any] = {}

    @classmethod
    def from_safetensors(cls, path_or_dir: Path | str) -> "NARLoRAAdapter":
        path = Path(path_or_dir)
        if path.is_dir():
            path = path / "nar_lora.safetensors"
        if not path.exists():
            raise FileNotFoundError(f"NAR LoRA file not found: {path}")

        raw = mx.load(str(path))
        lora_weights: dict[str, mx.array] = {}
        io_weights: dict[str, dict[str, mx.array]] = {"vae2llm": {}, "llm2vae": {}}

        for k, v in raw.items():
            if k.startswith("layers."):
                lora_weights[k] = v
            elif k.startswith("vae2llm."):
                param = k.split(".", 1)[1]
                io_weights["vae2llm"][param] = v
            elif k.startswith("llm2vae."):
                param = k.split(".", 1)[1]
                io_weights["llm2vae"][param] = v

        return cls(lora_weights=lora_weights, io_weights=io_weights)

    def apply(self, nar_model, scale: float = 1.0) -> None:
        """Apply LoRA deltas and IO replacements to the AcousticModel in memory.

        Guarantees:
        - Never modifies disk weights.
        - Backs up original stock weights on first application so revert() is lossless.
        - Cleanly re-applies if scale changes.
        """
        if self._applied_scale == scale and self._backup_weights:
            return

        if self._backup_weights:
            self.revert(nar_model)

        # 1. Backup and apply IO replacements
        for io_name in ("vae2llm", "llm2vae"):
            mod = getattr(nar_model, io_name, None)
            if mod is not None and io_name in self.io_weights:
                if f"{io_name}.weight" not in self._backup_weights:
                    self._backup_weights[f"{io_name}.weight"] = mod.weight
                    self._backup_weights[f"{io_name}.bias"] = mod.bias
                if scale > 0.0:
                    rep = self.io_weights[io_name]
                    if "weight" in rep:
                        mod.weight = rep["weight"].astype(mod.weight.dtype)
                    if "bias" in rep:
                        mod.bias = rep["bias"].astype(mod.bias.dtype)

        # 2. Backup and apply linear LoRA deltas
        # 28 layers, 7 modules each
        modules_map = [
            ("nar_self_attn", "q_proj"),
            ("nar_self_attn", "k_proj"),
            ("nar_self_attn", "v_proj"),
            ("nar_self_attn", "o_proj"),
            ("nar_mlp", "gate_proj"),
            ("nar_mlp", "up_proj"),
            ("nar_mlp", "down_proj"),
        ]

        for layer_idx, layer in enumerate(nar_model.model.layers):
            for sub_name, proj_name in modules_map:
                sub_mod = getattr(layer, sub_name)
                lin = getattr(sub_mod, proj_name)
                key_prefix = f"layers.{layer_idx}.{sub_name}.{proj_name}"
                backup_key = f"{key_prefix}.weight"

                if backup_key not in self._backup_weights:
                    self._backup_weights[backup_key] = lin.weight

                if scale != 0.0:
                    a_key = f"{key_prefix}.lora_a"
                    b_key = f"{key_prefix}.lora_b"
                    if a_key in self.lora_weights and b_key in self.lora_weights:
                        A = self.lora_weights[a_key]
                        B = self.lora_weights[b_key]
                        delta = (B @ A).astype(lin.weight.dtype)
                        lin.weight = self._backup_weights[backup_key] + (scale * delta)

        self._applied_scale = scale
        mx.eval(nar_model.parameters())

    def revert(self, nar_model) -> None:
        """Revert all modified layers to exact stock weights."""
        if not self._backup_weights:
            return

        for io_name in ("vae2llm", "llm2vae"):
            mod = getattr(nar_model, io_name, None)
            if mod is not None:
                w_key = f"{io_name}.weight"
                b_key = f"{io_name}.bias"
                if w_key in self._backup_weights:
                    mod.weight = self._backup_weights[w_key]
                if b_key in self._backup_weights:
                    mod.bias = self._backup_weights[b_key]

        modules_map = [
            ("nar_self_attn", "q_proj"),
            ("nar_self_attn", "k_proj"),
            ("nar_self_attn", "v_proj"),
            ("nar_self_attn", "o_proj"),
            ("nar_mlp", "gate_proj"),
            ("nar_mlp", "up_proj"),
            ("nar_mlp", "down_proj"),
        ]

        for layer_idx, layer in enumerate(nar_model.model.layers):
            for sub_name, proj_name in modules_map:
                sub_mod = getattr(layer, sub_name)
                lin = getattr(sub_mod, proj_name)
                backup_key = f"layers.{layer_idx}.{sub_name}.{proj_name}.weight"
                if backup_key in self._backup_weights:
                    lin.weight = self._backup_weights[backup_key]

        self._applied_scale = None
        self._backup_weights.clear()
        mx.eval(nar_model.parameters())


class ARLoRAAdapter:
    """Manages optional artist AR LoRAs for YuE2 AR foundation models."""

    def __init__(self, lora_weights: Mapping[str, mx.array]):
        self.lora_weights = dict(lora_weights)
        self._backup_weights: dict[str, Any] = {}
        self._applied_scale: float | None = None

    @classmethod
    def from_torch_checkpoint(cls, path: Path | str) -> "ARLoRAAdapter":
        import torch

        data = torch.load(path, map_location="cpu", weights_only=False)
        lora_list = data["lora"]
        it = iter(lora_list)
        lora_weights: dict[str, mx.array] = {}

        modules_map = [
            ("self_attn", "q_proj"),
            ("self_attn", "k_proj"),
            ("self_attn", "v_proj"),
            ("self_attn", "o_proj"),
            ("mlp", "gate_proj"),
            ("mlp", "up_proj"),
            ("mlp", "down_proj"),
        ]

        for layer_idx in range(28):
            for sub_name, proj_name in modules_map:
                A = next(it).float().numpy()
                B = next(it).float().numpy()
                k = f"layers.{layer_idx}.{sub_name}.{proj_name}"
                lora_weights[f"{k}.lora_a"] = mx.array(A)
                lora_weights[f"{k}.lora_b"] = mx.array(B)

        return cls(lora_weights=lora_weights)

    def apply(self, ar_model, scale: float = 1.0) -> None:
        """Fold LoRA into AR model linear layers (requires unquantized float linear weights)."""
        if getattr(ar_model, "precision", "") == "8bit":
            raise ValueError(
                "Cannot fold AR LoRA directly into quantized 8-bit model. "
                "Please run with BF16 precision when applying artist AR LoRAs."
            )

        if self._applied_scale == scale and self._backup_weights:
            return

        if self._backup_weights:
            self.revert(ar_model)

        modules_map = [
            ("self_attn", "q_proj"),
            ("self_attn", "k_proj"),
            ("self_attn", "v_proj"),
            ("self_attn", "o_proj"),
            ("mlp", "gate_proj"),
            ("mlp", "up_proj"),
            ("mlp", "down_proj"),
        ]

        for layer_idx, layer in enumerate(ar_model.layers):
            for sub_name, proj_name in modules_map:
                sub_mod = getattr(layer, sub_name)
                lin = getattr(sub_mod, proj_name)
                key_prefix = f"layers.{layer_idx}.{sub_name}.{proj_name}"
                backup_key = f"{key_prefix}.weight"

                if backup_key not in self._backup_weights:
                    self._backup_weights[backup_key] = lin.weight

                if scale != 0.0:
                    a_key = f"{key_prefix}.lora_a"
                    b_key = f"{key_prefix}.lora_b"
                    if a_key in self.lora_weights and b_key in self.lora_weights:
                        A = self.lora_weights[a_key]
                        B = self.lora_weights[b_key]
                        delta = (B @ A).astype(lin.weight.dtype)
                        lin.weight = self._backup_weights[backup_key] + (scale * delta)

        self._applied_scale = scale
        mx.eval(ar_model.parameters())

    def revert(self, ar_model) -> None:
        if not self._backup_weights:
            return

        modules_map = [
            ("self_attn", "q_proj"),
            ("self_attn", "k_proj"),
            ("self_attn", "v_proj"),
            ("self_attn", "o_proj"),
            ("mlp", "gate_proj"),
            ("mlp", "up_proj"),
            ("mlp", "down_proj"),
        ]

        for layer_idx, layer in enumerate(ar_model.layers):
            for sub_name, proj_name in modules_map:
                sub_mod = getattr(layer, sub_name)
                lin = getattr(sub_mod, proj_name)
                backup_key = f"layers.{layer_idx}.{sub_name}.{proj_name}.weight"
                if backup_key in self._backup_weights:
                    lin.weight = self._backup_weights[backup_key]

        self._applied_scale = None
        self._backup_weights.clear()
        mx.eval(ar_model.parameters())
