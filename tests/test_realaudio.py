"""Tests for realaudio v4 tokenization, MLX head, and NAR LoRA adapter."""
import numpy as np
import mlx.core as mx
import mlx.nn as nn

from lyra.realaudio.tokenizer import (
    MLXRealAudioTokenizerHead,
    instance_norm,
    interpolate_to_25hz,
    predict_tokens,
    load_tokenizer_head,
)
from lyra.realaudio.adapter import NARLoRAAdapter


def test_tokenizer_head_architecture():
    head = MLXRealAudioTokenizerHead(
        d_in=1024,
        d_model=512,
        nhead=8,
        num_layers=8,
        vocab_size=32768,
        window_size=512,
    )
    dummy_x = mx.random.normal((1, 64, 1024))
    out = head(dummy_x)
    assert out.shape == (1, 64, 32768)


def test_instance_norm():
    np.random.seed(42)
    feats = np.random.randn(200, 1024).astype(np.float32) * 5.0 + 3.0
    normed = instance_norm(feats)
    assert np.allclose(normed.mean(axis=0), 0.0, atol=1e-5)
    assert np.allclose(normed.std(axis=0), 1.0, atol=1e-4)


def test_interpolate_to_25hz():
    mert_feats = np.ones((100, 1024), dtype=np.float32)
    interp = interpolate_to_25hz(mert_feats, 200)
    assert interp.shape == (200, 1024)
    assert np.allclose(interp, 1.0)


def test_predict_tokens_stitching():
    head = MLXRealAudioTokenizerHead(
        d_in=1024,
        d_model=512,
        nhead=8,
        num_layers=2,  # lightweight for test
        vocab_size=1000,
        window_size=64,
    )
    # Test short input
    short_feats = np.random.randn(30, 1024).astype(np.float32)
    toks_short = predict_tokens(head, short_feats, window_size=64, hop_size=32)
    assert len(toks_short) == 30

    # Test long input crossing multiple windows
    long_feats = np.random.randn(150, 1024).astype(np.float32)
    toks_long = predict_tokens(head, long_feats, window_size=64, hop_size=32)
    assert len(toks_long) == 150
    assert (toks_long >= 0).all() and (toks_long < 1000).all()


def test_load_converted_tokenizer_head():
    head = load_tokenizer_head("models/realaudio-v4/converted")
    assert head.d_in == 1024
    assert head.d_model == 512
    assert head.window_size == 512
    assert len(head.layers) == 8


def test_nar_lora_adapter_apply_and_revert():
    adapter = NARLoRAAdapter.from_safetensors("models/realaudio-v4/converted")
    assert len(adapter.lora_weights) == 392
    assert "vae2llm" in adapter.io_weights
    assert "llm2vae" in adapter.io_weights

    # Create a mock AcousticModel with layers and IO modules
    class MockLinear(nn.Module):
        def __init__(self, in_f, out_f):
            super().__init__()
            self.weight = mx.zeros((out_f, in_f))
            self.bias = mx.zeros((out_f,))

    class MockLayer(nn.Module):
        def __init__(self):
            super().__init__()
            class Attn(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.q_proj = MockLinear(2048, 2048)
                    self.k_proj = MockLinear(2048, 1024)
                    self.v_proj = MockLinear(2048, 1024)
                    self.o_proj = MockLinear(2048, 2048)
            class MLP(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.gate_proj = MockLinear(2048, 6144)
                    self.up_proj = MockLinear(2048, 6144)
                    self.down_proj = MockLinear(6144, 2048)
            self.nar_self_attn = Attn()
            self.nar_mlp = MLP()

    class MockAcousticModel(nn.Module):
        def __init__(self):
            super().__init__()
            class BB(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.layers = [MockLayer() for _ in range(28)]
            self.model = BB()
            self.vae2llm = MockLinear(64, 2048)
            self.llm2vae = MockLinear(2048, 64)

    mock = MockAcousticModel()
    orig_q = mx.array(mock.model.layers[0].nar_self_attn.q_proj.weight)
    orig_vae = mx.array(mock.vae2llm.weight)

    # Apply scale=1.0
    adapter.apply(mock, scale=1.0)
    assert not np.array_equal(np.array(mock.model.layers[0].nar_self_attn.q_proj.weight), np.array(orig_q))
    assert not np.array_equal(np.array(mock.vae2llm.weight), np.array(orig_vae))

    # Revert
    adapter.revert(mock)
    assert np.array_equal(np.array(mock.model.layers[0].nar_self_attn.q_proj.weight), np.array(orig_q))
    assert np.array_equal(np.array(mock.vae2llm.weight), np.array(orig_vae))
