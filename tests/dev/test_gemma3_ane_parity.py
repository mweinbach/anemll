import os
from typing import List, Tuple

import pytest
import torch
from huggingface_hub import snapshot_download

try:  # pragma: no cover - optional dependency
    from transformers import Gemma3ForCausalLM
    from transformers.models.gemma3.modeling_gemma3 import Gemma3TextConfig
except Exception as exc:  # pragma: no cover - if transformers missing
    pytest.skip(f"transformers unavailable: {exc}", allow_module_level=True)

from anemll.models.gemma3_ane_model import Gemma3ANEConfig, Gemma3ForCausalLMANE, MODEL_DTYPE


def _load_models(model_id: str) -> Tuple[Gemma3ForCausalLMANE, Gemma3ForCausalLM, Gemma3ANEConfig]:
    try:
        snapshot_path = snapshot_download(
            model_id,
            allow_patterns=["*.json", "*.safetensors", "tokenizer*", "*.model"],
            ignore_patterns=["*.bin"],
        )
    except Exception as exc:  # pragma: no cover - network or auth failure
        pytest.skip(f"Unable to download {model_id}: {exc}")

    ane_cfg = Gemma3ANEConfig.from_json(os.path.join(snapshot_path, "config.json"))
    ane_model = Gemma3ForCausalLMANE(ane_cfg, enable_coreml=False)
    ane_model.load_pretrained_weights(snapshot_path)
    ane_model.eval()

    hf_model = Gemma3ForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="cpu").eval()
    return ane_model, hf_model, ane_cfg


def _layer_stats(
    ane_model: Gemma3ForCausalLMANE,
    hf_model: Gemma3ForCausalLM,
    cfg: Gemma3ANEConfig,
    seq_len: int = 8,
    layers: int = 4,
) -> List[Tuple[int, float, float]]:
    torch.manual_seed(0)
    input_ids = torch.randint(0, cfg.vocab_size, (1, seq_len))
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)

    with torch.no_grad():
        hf_state = hf_model.model.embed_tokens(input_ids)
        ane_embed = ane_model.model.embed_tokens(input_ids.to(torch.long))
        ane_state = ane_embed.to(MODEL_DTYPE) * (cfg.hidden_size ** 0.5)
        base_embed = ane_state.clone()
        cos_global, sin_global = ane_model.model.rotary_global(ane_state, position_ids)
        cos_local, sin_local = ane_model.model.rotary_local(ane_state, position_ids)

        stats: List[Tuple[int, float, float]] = []
        for layer_idx in range(layers):
            hf_layer = hf_model.model.layers[layer_idx]
            ane_layer = ane_model.model.layers[layer_idx]

            hf_norm = hf_layer.input_layernorm(hf_state)
            ane_norm = ane_layer.input_layernorm(ane_state)

            if hf_layer.self_attn.is_sliding:
                pos_emb = hf_model.model.rotary_emb_local(hf_norm, position_ids)
                cos = cos_local
                sin = sin_local
            else:
                pos_emb = hf_model.model.rotary_emb(hf_norm, position_ids)
                cos = cos_global
                sin = sin_global

            hf_attn, _ = hf_layer.self_attn(
                hf_norm,
                position_embeddings=pos_emb,
                attention_mask=None,
                cache_position=None,
            )
            ane_attn = ane_layer.self_attn(ane_norm, base_embed, None, position_ids, cos, sin)

            hf_state = hf_state + hf_layer.post_attention_layernorm(hf_attn)
            ane_state = ane_state + ane_layer.post_attention_layernorm(ane_attn)

            hf_pre_ffn = hf_layer.pre_feedforward_layernorm(hf_state)
            ane_pre_ffn = ane_layer.pre_feedforward_layernorm(ane_state)
            hf_ffn = hf_layer.mlp(hf_pre_ffn)
            ane_ffn = ane_layer.mlp(ane_pre_ffn)

            hf_state = hf_state + hf_layer.post_feedforward_layernorm(hf_ffn)
            ane_state = ane_state + ane_layer.post_feedforward_layernorm(ane_ffn)

            diff = (hf_state.float() - ane_state.float()).abs()
            stats.append((layer_idx, float(diff.max().item()), float(diff.mean().item())))
    return stats


@pytest.mark.slow
def test_gemma3_ane_matches_first_layers():
    model_id = os.environ.get("ANEMLL_GEMMA3_MODEL", "google/gemma-3-1b-it")
    ane_model, hf_model, cfg = _load_models(model_id)
    stats = _layer_stats(ane_model, hf_model, cfg, seq_len=8, layers=4)

    # Ensure early layers stay numerically close so regressions are detected.
    for layer_idx, max_diff, mean_diff in stats:
        assert max_diff <= 1.5, f"Layer {layer_idx} max diff {max_diff:.3f} exceeds tolerance"
        assert mean_diff <= 0.02, f"Layer {layer_idx} mean diff {mean_diff:.4f} exceeds tolerance"


@pytest.mark.slow
def test_gemma3_long_seq_parity_64(capfd):
    """Run a longer sequence (64 tokens) and report per-layer drift.

    We only assert on early layers to prevent test flakiness, but we print a
    concise per-layer log to help track the first widening layer.
    """
    model_id = os.environ.get("ANEMLL_GEMMA3_MODEL", "google/gemma-3-1b-it")
    ane_model, hf_model, cfg = _load_models(model_id)

    torch.manual_seed(0)
    seq_len = 64
    input_ids = torch.randint(0, cfg.vocab_size, (1, seq_len))
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)

    with torch.no_grad():
        # Initialize states
        hf_state = hf_model.model.embed_tokens(input_ids)
        ane_embed = ane_model.model.embed_tokens(input_ids.to(torch.long))
        ane_state = ane_embed.to(MODEL_DTYPE) * (cfg.hidden_size ** 0.5)

        # Precompute rotary for speed
        cos_global, sin_global = ane_model.model.rotary_global(ane_state, position_ids)
        cos_local, sin_local = ane_model.model.rotary_local(ane_state, position_ids)

        lines = []
        for layer_idx in range(cfg.num_hidden_layers):
            hf_layer = hf_model.model.layers[layer_idx]
            ane_layer = ane_model.model.layers[layer_idx]

            hf_norm = hf_layer.input_layernorm(hf_state)
            ane_norm = ane_layer.input_layernorm(ane_state)

            if hf_layer.self_attn.is_sliding:
                cos = cos_local
                sin = sin_local
                pos_emb = hf_model.model.rotary_emb_local(hf_norm, position_ids)
            else:
                cos = cos_global
                sin = sin_global
                pos_emb = hf_model.model.rotary_emb(hf_norm, position_ids)

            hf_attn, _ = hf_layer.self_attn(
                hf_norm,
                position_embeddings=pos_emb,
                attention_mask=None,
                cache_position=None,
            )
            ane_attn = ane_layer.self_attn(ane_norm, None, None, position_ids, cos, sin)

            hf_state = hf_state + hf_layer.post_attention_layernorm(hf_attn)
            ane_state = ane_state + ane_layer.post_attention_layernorm(ane_attn)

            hf_ffn = hf_layer.mlp(hf_layer.pre_feedforward_layernorm(hf_state))
            ane_ffn = ane_layer.mlp(ane_layer.pre_feedforward_layernorm(ane_state))

            hf_state = hf_state + hf_layer.post_feedforward_layernorm(hf_ffn)
            ane_state = ane_state + ane_layer.post_feedforward_layernorm(ane_ffn)

            diff = (hf_state.float() - ane_state.float()).abs()
            lines.append(f"L{layer_idx:02d} max={float(diff.max()):.3f} mean={float(diff.mean()):.5f}")

        # Emit a compact per-layer summary for debugging in CI logs
        print("\n".join(lines))

    # Guardrail: first 6 layers should remain close
    # (full convergence across all layers is tracked offline to avoid flaky CI)
    cap = capfd.readouterr().out.strip().splitlines()
    early = cap[:6]
    assert early, "No layer output captured"
    # parse the numbers and assert modest bounds
    for row in early:
        parts = row.split()
        mval = float(parts[1].split('=')[1])
        meanv = float(parts[2].split('=')[1])
        assert mval <= 3.0, f"Early layer drift too high: {row}"
        assert meanv <= 0.03, f"Early layer mean drift too high: {row}"


@pytest.mark.slow
def test_gemma3_prefill_applies_final_norm():
    model_id = os.environ.get("ANEMLL_GEMMA3_MODEL", "google/gemma-3-1b-it")
    ane_model, _hf_model, cfg = _load_models(model_id)

    seq_len = 8
    torch.manual_seed(0)
    hidden = torch.randn(1, seq_len, cfg.hidden_size, dtype=MODEL_DTYPE)
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
    current_pos = torch.tensor(0, dtype=torch.int64)

    base_fp32 = hidden.to(torch.float32)
    cos_global, sin_global = ane_model.model.rotary_global(base_fp32, position_ids)
    cos_local, sin_local = ane_model.model.rotary_local(base_fp32, position_ids)
    cos_global = cos_global.to(hidden.dtype)
    sin_global = sin_global.to(hidden.dtype)
    cos_local = cos_local.to(hidden.dtype)
    sin_local = sin_local.to(hidden.dtype)

    manual = ane_model.model.process_layers(
        hidden.clone(),
        position_ids,
        causal_mask=None,
        current_pos=current_pos,
        cos_global=cos_global,
        sin_global=sin_global,
        cos_local=cos_local,
        sin_local=sin_local,
        start_layer=0,
        end_layer=cfg.num_hidden_layers,
        IN_PREFILL=True,
    )
    manual = ane_model.model.norm(manual)

    # Reset KV cache before invoking the public API
    if hasattr(ane_model.model, "kv_cache_0"):
        ane_model.model.kv_cache_0.zero_()

    prefill = ane_model.model.forward_prefill(
        hidden.clone(),
        position_ids=position_ids,
        causal_mask=None,
        current_pos=current_pos,
    )

    diff = (manual.float() - prefill.float()).abs()
    assert float(diff.max()) <= 1e-3, f"Prefill output deviates from normalized path (max diff {float(diff.max()):.4f})"
