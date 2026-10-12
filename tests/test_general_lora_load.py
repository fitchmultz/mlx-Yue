from pathlib import Path
import pytest
from lyra.realaudio.adapter import NARLoRAAdapter, ARLoRAAdapter

def test_load_community_jpop_lora():
    path = Path('/Users/vanch/.cache/huggingface/hub/models--storagejuju--yue2-jpop-t4-lora/snapshots/16c181c15cf7d06b1b570a149624b1e78724baa6/yue2_jpop_t4.safetensors')
    if not path.exists():
        pytest.skip('J-pop LoRA not downloaded in cache')
    adapter = NARLoRAAdapter.from_safetensors(path)
    assert len(adapter.lora_weights) > 100
    # Check q_proj delta
    assert 'layers.0.nar_self_attn.q_proj.delta' in adapter.lora_weights
    assert adapter.lora_weights['layers.0.nar_self_attn.q_proj.delta'].shape == (2048, 2048)

def test_load_community_trbdr_lora():
    path = Path('/Users/vanch/.cache/huggingface/hub/models--becausereasons--yue2-trbdr-folk-troubadour/snapshots/078b19e52baebaa71463ea94afac5ce13abfe29a/trbdr_porch.safetensors')
    if not path.exists():
        pytest.skip('TRBDR LoRA not downloaded in cache')
    nar_adapter = NARLoRAAdapter.from_safetensors(path)
    assert len(nar_adapter.lora_weights) > 100
    assert 'vae2llm' in nar_adapter.io_weights
    assert 'diff_weight' in nar_adapter.io_weights['vae2llm']
    ar_adapter = ARLoRAAdapter.from_safetensors(path)
    assert len(ar_adapter.lora_weights) > 100
    assert 'layers.0.self_attn.q_proj.delta' in ar_adapter.lora_weights
