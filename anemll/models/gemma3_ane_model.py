"""Gemma 3 (text-only) ANE-friendly model.

Design:
- All Linear layers are expressed as Conv2d(kernel_size=1) to map cleanly to ANE.
- Multi-Query Attention (MQA): Q heads = num_attention_heads, KV heads = num_key_value_heads.
- Rotary embeddings with configurable theta; provides cached cos/sin.
- Unified KV cache with fixed `state_length` layout identical to Qwen/LLaMA paths.
- Layer order mirrors HF Gemma3: input LN → Attn → post_attn LN → residual,
  pre_ffn LN → MLP (GELU tanh) → post_ffn LN → residual.

Weights are loaded from HF checkpoints (Gemma3ForCausalLM) by reshaping Linear
weights to Conv2d [out, in, 1, 1].
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import safetensors.torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.gemma3.configuration_gemma3 import Gemma3TextConfig
from transformers.models.gemma3.modeling_gemma3 import (
    Gemma3RMSNorm,
    Gemma3RotaryEmbedding,
    apply_rotary_pos_emb,
    create_causal_mask,
    create_sliding_window_causal_mask,
    eager_attention_forward,
)


MODEL_DTYPE = torch.float16
TEST_DEVICE = "cpu"
CONTEXT_LENGTH = 512
STATE_LENGTH = 512

# Cache behavior (match Qwen/LLaMA pipelines)
FORCE_UNIFIED_CACHE = True
ENABLE_UNIFIED_CACHE = True

# LM head splits (keep consistent with chat/runtime)
ENABLE_CONV2D = True
ENABLE_VACAB_SPLIT = False
ENABLE_VACAB_SPLIT8 = True
ENABLE_VACAB_SPLIT16 = False
ENABLE_LOGITS2 = True


def gemma_gelu(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


def apply_hidden_activation(x: torch.Tensor, activation: str) -> torch.Tensor:
    if activation == "gelu_pytorch_tanh":
        return gemma_gelu(x)
    if activation in ("gelu", "gelu_new"):
        return F.gelu(x)
    if activation in ("relu", "relu_new"):
        return F.relu(x)
    if activation == "silu":
        return F.silu(x)
    raise ValueError(f"Unsupported activation for Gemma3 MLP: {activation}")


@dataclass
class Gemma3ANEConfig:
    architectures: list
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    rms_norm_eps: float
    rope_theta: float
    vocab_size: int
    max_position_embeddings: int
    context_length: int
    state_length: int
    sliding_window: int
    layer_types: List[str]
    query_pre_attn_scalar: float
    attn_logit_softcapping: Optional[float]
    attention_dropout: float
    attention_bias: bool
    hidden_activation: str
    rope_local_base_freq: Optional[float]
    rope_scaling: Optional[Dict]
    raw_config: Dict[str, Any]

    @classmethod
    def from_json(cls, path: str) -> "Gemma3ANEConfig":
        with open(path, "r") as f:
            j = json.load(f)
        # The HF config uses Gemma3TextConfig keys
        # Derive per-layer attention types from HF config if not provided
        layer_types = j.get("layer_types")
        if not layer_types:
            try:
                # Build a lightweight HF model to inspect which layers are sliding vs full
                from transformers import Gemma3ForCausalLM as _HFModel
                hf_cfg = Gemma3TextConfig(**j)
                hf_tmp = _HFModel(hf_cfg)
                layer_types = [
                    ("sliding_attention" if getattr(hf_tmp.model.layers[i].self_attn, "is_sliding", False) else "full_attention")
                    for i in range(hf_cfg.num_hidden_layers)
                ]
                # Free quickly
                del hf_tmp
            except Exception:
                # Fallback: assume sliding if sliding_window set
                sw = j.get("sliding_window", 0)
                layer_types = ["sliding_attention" if sw and sw > 0 else "full_attention"] * j["num_hidden_layers"]
        return cls(
            architectures=j.get("architectures", ["Gemma3ForCausalLM"]),
            hidden_size=j["hidden_size"],
            num_hidden_layers=j["num_hidden_layers"],
            num_attention_heads=j["num_attention_heads"],
            num_key_value_heads=j["num_key_value_heads"],
            head_dim=j.get("head_dim", j["hidden_size"] // max(1, j["num_attention_heads"])),
            intermediate_size=j["intermediate_size"],
            rms_norm_eps=j.get("rms_norm_eps", 1e-6),
            rope_theta=j.get("rope_theta", 1_000_000.0),
            vocab_size=j["vocab_size"],
            max_position_embeddings=j.get("max_position_embeddings", CONTEXT_LENGTH),
            context_length=j.get("context_length", CONTEXT_LENGTH),
            state_length=j.get("state_length", STATE_LENGTH),
            sliding_window=j.get("sliding_window", 0),
            layer_types=layer_types,
            query_pre_attn_scalar=j.get("query_pre_attn_scalar", j.get("head_dim", j["hidden_size"] // max(1, j["num_attention_heads"]))),
            attn_logit_softcapping=j.get("attn_logit_softcapping"),
            attention_dropout=j.get("attention_dropout", 0.0),
            attention_bias=j.get("attention_bias", False),
            hidden_activation=j.get("hidden_activation", "gelu_pytorch_tanh"),
            rope_local_base_freq=j.get("rope_local_base_freq"),
            rope_scaling=j.get("rope_scaling"),
            raw_config=j,
        )


def get_kv_cache_idx(layer_idx: int, num_layers: int, num_groups: int = 1) -> Tuple[int, int, int]:
    layers_per_group = num_layers // num_groups
    group_idx = layer_idx // layers_per_group
    layer_in_group_idx = layer_idx % layers_per_group
    return group_idx, layer_in_group_idx, layers_per_group


def apply_attn_softcap(attn: torch.Tensor, softcap: Optional[float]) -> torch.Tensor:
    if softcap is None:
        return attn
    capped = attn / softcap
    capped = torch.tanh(capped)
    return capped * softcap


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    b, h, t, d = x.shape
    return x.unsqueeze(2).expand(b, h, n_rep, t, d).reshape(b, h * n_rep, t, d)


class Gemma3AttentionANE(nn.Module):
    def __init__(
        self,
        cfg: Gemma3ANEConfig,
        hf_config: Gemma3TextConfig,
        layer_idx: int,
        rotary_global: Gemma3RotaryEmbedding,
        rotary_local: Gemma3RotaryEmbedding,
    ) -> None:
        super().__init__()
        self.hidden_size = cfg.hidden_size
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.num_kv_groups = max(1, self.num_heads // max(1, self.num_kv_heads))
        self.num_key_value_groups = self.num_kv_groups
        self.head_dim = cfg.head_dim
        self.layer_idx = layer_idx
        self.scale = 1.0 / math.sqrt(float(cfg.query_pre_attn_scalar))
        self.attn_logit_softcapping = cfg.attn_logit_softcapping
        self.dropout = cfg.attention_dropout
        self.rotary_global = rotary_global
        self.rotary_local = rotary_local
        self.hf_config = hf_config
        self.is_causal = True

        layer_types = cfg.layer_types or []
        if layer_idx < len(layer_types):
            self.attention_type = layer_types[layer_idx]
        else:
            self.attention_type = "sliding_attention" if cfg.sliding_window > 0 else "full_attention"
        self.is_sliding = self.attention_type == "sliding_attention"
        self.sliding_window = cfg.sliding_window if (self.is_sliding and cfg.sliding_window > 0) else None

        self.q_proj = nn.Conv2d(self.hidden_size, self.num_heads * self.head_dim, 1, bias=cfg.attention_bias, dtype=MODEL_DTYPE).to(TEST_DEVICE)
        self.k_proj = nn.Conv2d(self.hidden_size, self.num_kv_heads * self.head_dim, 1, bias=cfg.attention_bias, dtype=MODEL_DTYPE).to(TEST_DEVICE)
        self.v_proj = nn.Conv2d(self.hidden_size, self.num_kv_heads * self.head_dim, 1, bias=cfg.attention_bias, dtype=MODEL_DTYPE).to(TEST_DEVICE)
        self.o_proj = nn.Conv2d(self.num_heads * self.head_dim, self.hidden_size, 1, bias=cfg.attention_bias, dtype=MODEL_DTYPE).to(TEST_DEVICE)

        self.q_norm = Gemma3RMSNorm(self.head_dim, eps=cfg.rms_norm_eps)
        self.k_norm = Gemma3RMSNorm(self.head_dim, eps=cfg.rms_norm_eps)

    def project_qkv(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, t, _ = x.shape
        hs = x.permute(0, 2, 1).unsqueeze(2)
        q = self.q_proj(hs).view(b, self.num_heads, self.head_dim, t).permute(0, 1, 3, 2)
        k = self.k_proj(hs).view(b, self.num_kv_heads, self.head_dim, t).permute(0, 1, 3, 2)
        v = self.v_proj(hs).view(b, self.num_kv_heads, self.head_dim, t).permute(0, 1, 3, 2)
        return q, k, v

    def _select_rotary(self) -> Gemma3RotaryEmbedding:
        return self.rotary_local if self.is_sliding else self.rotary_global

    def _get_position_embeddings(self, hidden_states: torch.Tensor, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        rotary = self._select_rotary()
        if position_ids is None:
            raise ValueError("position_ids must be provided for rotary embeddings")
        if position_ids.dim() == 1:
            position_ids = position_ids.unsqueeze(0)
        cos, sin = rotary(hidden_states, position_ids)
        return cos, sin

    def get_new_kv_cache_prefill(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q, k, v = self.project_qkv(x)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = apply_rotary_pos_emb(q, k, cos.to(q.dtype), sin.to(q.dtype))
        return q, k, v

    @staticmethod
    def _slice_mask(attention_mask: Optional[torch.Tensor], q_len: int, k_len: int) -> Optional[torch.Tensor]:
        if attention_mask is None:
            return None
        return attention_mask[..., :q_len, :k_len]

    def _build_relative(self, query_positions: torch.Tensor, k_len: int, device: torch.device) -> torch.Tensor:
        if query_positions.dim() == 1:
            query_positions = query_positions.unsqueeze(0)
        q_pos = query_positions.to(device=device, dtype=torch.long)
        key_positions = torch.arange(k_len, device=device, dtype=torch.long)
        return q_pos.unsqueeze(-1) - key_positions.view(1, 1, k_len)

    def _build_causal_mask(self, rel: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        mask = torch.zeros((*rel.shape[:-1], rel.shape[-1]), device=rel.device, dtype=dtype)
        neg_inf = torch.finfo(dtype).min
        mask = mask.masked_fill(rel < 0, neg_inf)
        return mask.unsqueeze(1)

    def _build_sliding_mask(self, rel: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        if not (self.is_sliding and self.sliding_window and self.sliding_window > 0):
            return torch.zeros((*rel.shape[:-1], rel.shape[-1]), device=rel.device, dtype=dtype).unsqueeze(1)
        neg_inf = torch.finfo(dtype).min
        mask = torch.zeros((*rel.shape[:-1], rel.shape[-1]), device=rel.device, dtype=dtype)
        mask = mask.masked_fill(rel > (self.sliding_window - 1), neg_inf)
        return mask.unsqueeze(1)

    def _run_attention(
        self,
        query_positions: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        b, h, t, d = q.shape
        if query_positions is None:
            raise ValueError("query_positions required for attention mask generation")
        if query_positions.dim() == 1:
            query_positions = query_positions.unsqueeze(0)
        rel = self._build_relative(query_positions, k.shape[-2], q.device)
        if attention_mask is None:
            causal_mask = self._build_causal_mask(rel, q.dtype)
            sliding_mask = self._build_sliding_mask(rel, q.dtype)
            attention_mask = causal_mask + sliding_mask
        else:
            attention_mask = attention_mask.to(q.dtype)
        q_float = q.to(torch.float32)
        k_float = k.to(torch.float32)
        v_float = v.to(torch.float32)
        attn_mask_float = attention_mask.to(torch.float32) if attention_mask is not None else None
        attn_out, _ = eager_attention_forward(
            self,
            q_float,
            k_float,
            v_float,
            attn_mask_float,
            dropout=self.dropout if self.training else 0.0,
            scaling=self.scale,
            sliding_window=self.sliding_window,
            softcap=self.attn_logit_softcapping,
        )
        attn_out = attn_out.to(q.dtype).reshape(b, t, -1)
        attn_out = self.o_proj(attn_out.permute(0, 2, 1).unsqueeze(2)).squeeze(2).permute(0, 2, 1)
        return attn_out

    def forward(
        self,
        x: torch.Tensor,
        input_embeds: torch.Tensor,
        causal_mask: Optional[torch.Tensor],
        position_ids: torch.LongTensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        b, t, _ = x.shape
        q, k, v = self.project_qkv(x)
        q = self.q_norm(q)
        k = self.k_norm(k)
        cos = cos.to(q.dtype)
        sin = sin.to(q.dtype)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        causal_mask = self._slice_mask(causal_mask, t, k.shape[-2])
        return self._run_attention(position_ids, q, k, v, causal_mask)


class Gemma3MLPANE(nn.Module):
    def __init__(self, cfg: Gemma3ANEConfig, use_conv: bool = True) -> None:
        super().__init__()
        h = cfg.hidden_size
        i = cfg.intermediate_size
        self.gate_proj = nn.Conv2d(h, i, 1, bias=False, dtype=MODEL_DTYPE).to(TEST_DEVICE)
        self.up_proj = nn.Conv2d(h, i, 1, bias=False, dtype=MODEL_DTYPE).to(TEST_DEVICE)
        self.down_proj = nn.Conv2d(i, h, 1, bias=False, dtype=MODEL_DTYPE).to(TEST_DEVICE)
        self.activation = cfg.hidden_activation
        self.use_conv = use_conv

    def _linear(self, weight: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # weight: [out, in, 1, 1], x: [B, T, in]
        w = weight.view(weight.shape[0], weight.shape[1])
        y = torch.matmul(x, w.t().to(x.dtype))
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_conv:
            s = x.permute(0, 2, 1).unsqueeze(2).to(MODEL_DTYPE)
            a = self.gate_proj(s)
            b = self.up_proj(s)
            c = apply_hidden_activation(a, self.activation)
            y = self.down_proj(c * b)
            return y.squeeze(2).permute(0, 2, 1)

        # Parity path: operate directly in [B, T, H]
        x_fp = x.to(torch.float32)
        gate = self._linear(self.gate_proj.weight, x_fp)
        up = self._linear(self.up_proj.weight, x_fp)
        activated = apply_hidden_activation(gate, self.activation)
        hidden = activated * up
        down = self._linear(self.down_proj.weight, hidden)
        return down.to(x.dtype)


class Gemma3DecoderLayerANE(nn.Module):
    def __init__(
        self,
        cfg: Gemma3ANEConfig,
        hf_config: Gemma3TextConfig,
        layer_idx: int,
        rotary_global: Gemma3RotaryEmbedding,
        rotary_local: Gemma3RotaryEmbedding,
        use_conv_mlp: bool,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.self_attn = Gemma3AttentionANE(cfg, hf_config, layer_idx, rotary_global, rotary_local)
        self.mlp = Gemma3MLPANE(cfg, use_conv=use_conv_mlp)
        self.input_layernorm = Gemma3RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.post_attention_layernorm = Gemma3RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.pre_feedforward_layernorm = Gemma3RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.post_feedforward_layernorm = Gemma3RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)

    def forward(self, x: torch.Tensor, causal_mask: Optional[torch.Tensor], position_ids: torch.LongTensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x, residual, causal_mask, position_ids, cos, sin)
        x = self.post_attention_layernorm(x)
        x = residual + x

        residual = x
        x = self.pre_feedforward_layernorm(x)
        x = self.mlp(x)
        x = self.post_feedforward_layernorm(x)
        x = residual + x
        return x


class Gemma3ModelANE(nn.Module):
    def __init__(self, cfg: Gemma3ANEConfig, use_conv_mlp: bool = True) -> None:
        super().__init__()
        self.config = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size).to(TEST_DEVICE)
        self.embed_scale = cfg.hidden_size ** 0.5
        hf_cfg = Gemma3TextConfig(**cfg.raw_config)
        local_cfg_dict = dict(cfg.raw_config)
        local_theta = cfg.rope_local_base_freq if cfg.rope_local_base_freq is not None else cfg.rope_theta
        local_cfg_dict["rope_theta"] = local_theta
        local_cfg_dict["rope_scaling"] = {"rope_type": "default"}
        if local_cfg_dict.get("sliding_window"):
            local_cfg_dict["max_position_embeddings"] = local_cfg_dict["sliding_window"]
        local_cfg = Gemma3TextConfig(**local_cfg_dict)
        self.hf_config = hf_cfg
        self.rotary_global = Gemma3RotaryEmbedding(config=hf_cfg)
        self.rotary_local = Gemma3RotaryEmbedding(config=local_cfg)
        self.layers = nn.ModuleList([
            Gemma3DecoderLayerANE(cfg, hf_cfg, idx, self.rotary_global, self.rotary_local, use_conv_mlp)
            for idx in range(cfg.num_hidden_layers)
        ])
        self.norm = Gemma3RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.head_dim = cfg.head_dim

        if FORCE_UNIFIED_CACHE or ENABLE_UNIFIED_CACHE:
            cache_size = (2 * cfg.num_hidden_layers, cfg.num_key_value_heads, cfg.state_length, self.head_dim)
            self.register_buffer("kv_cache_0", torch.zeros(cache_size, dtype=MODEL_DTYPE, device=TEST_DEVICE))

    def process_layer_prefill(self, i: int, x: torch.Tensor, position_ids, causal_mask, current_pos, cos, sin, layer_offset, layer_mask=None):
        layer = self.layers[i]
        residual_input = x
        normed = layer.input_layernorm(x)
        normed = normed.to(MODEL_DTYPE)
        q_rot, k_rot, v = layer.self_attn.get_new_kv_cache_prefill(normed, cos, sin)
        if FORCE_UNIFIED_CACHE or ENABLE_UNIFIED_CACHE:
            kv = getattr(self, "kv_cache_0")
        else:
            raise NotImplementedError
        _, layer_in_group_idx, layers_per_group = get_kv_cache_idx(i, self.config.num_hidden_layers)
        key_idx = layer_in_group_idx
        value_idx = layer_in_group_idx + layers_per_group
        seq_length = q_rot.shape[2]
        start_pos = int(current_pos.item()) if torch.is_tensor(current_pos) else int(current_pos)
        kv[key_idx:key_idx + 1, :, start_pos:start_pos + seq_length, :] = k_rot[:1].to(kv.dtype)
        kv[value_idx:value_idx + 1, :, start_pos:start_pos + seq_length, :] = v[:1].to(kv.dtype)

        b, t, _ = x.shape
        k_full = kv[key_idx:key_idx + 1, :, : start_pos + seq_length, :]
        v_full = kv[value_idx:value_idx + 1, :, : start_pos + seq_length, :]
        if layer_mask is not None:
            mask = layer_mask[:, :, -t:, : k_full.shape[-2]]
        else:
            mask = layer.self_attn._slice_mask(causal_mask, t, k_full.shape[-2])
        pos_slice = position_ids[:, -t:] if position_ids is not None else None
        out = layer.self_attn._run_attention(pos_slice, q_rot, k_full, v_full, mask)
        # Post norms + MLP
        out = layer.post_attention_layernorm(out)
        x = x + out
        res = x
        x = layer.pre_feedforward_layernorm(x)
        x = layer.mlp(x)
        x = layer.post_feedforward_layernorm(x)
        x = res + x
        return x

    def process_layer_regular(self, i: int, x: torch.Tensor, position_ids, causal_mask, current_pos, cos, sin, layer_offset, layer_mask=None):
        layer = self.layers[i]
        # Single-token KV cache update and attention
        residual_input = x
        normed = layer.input_layernorm(x)
        normed = normed.to(MODEL_DTYPE)
        if position_ids is not None:
            if position_ids.dim() == 2:
                pos_token = position_ids[:, -1:]
            else:
                pos_token = position_ids[-1:].unsqueeze(0)
        else:
            pos_token = torch.tensor([[int(current_pos.item())]], dtype=torch.int64, device=normed.device)
        q, k_rot, v_raw = layer.self_attn.get_new_kv_cache_prefill(normed, cos, sin)
        b, t, _ = normed.shape
        k_rot_cache = k_rot
        # Update unified cache
        if FORCE_UNIFIED_CACHE or ENABLE_UNIFIED_CACHE:
            kv = getattr(self, "kv_cache_0")
        else:
            raise NotImplementedError
        group_idx, layer_in_group_idx, layers_per_group = get_kv_cache_idx(i, self.config.num_hidden_layers)
        key_idx = layer_in_group_idx
        value_idx = layer_in_group_idx + layers_per_group
        pos = int(current_pos.item())
        kv[key_idx:key_idx + 1, :, pos:pos + t, :] = k_rot_cache[:1].to(kv.dtype)
        kv[value_idx:value_idx + 1, :, pos:pos + t, :] = v_raw[:1].to(kv.dtype)
        # Attend over cache up to current_pos
        k_full = kv[key_idx:key_idx + 1, :, : pos + t, :]
        v_full = kv[value_idx:value_idx + 1, :, : pos + t, :]
        if layer_mask is not None:
            mask = layer_mask[:, :, -t:, : k_full.shape[-2]]
        else:
            mask = layer.self_attn._slice_mask(causal_mask, t, k_full.shape[-2])
        out = layer.self_attn._run_attention(pos_token, q, k_full, v_full, mask)
        out = layer.post_attention_layernorm(out)
        x = x + out
        res = x
        x = layer.pre_feedforward_layernorm(x)
        x = layer.mlp(x)
        x = layer.post_feedforward_layernorm(x)
        x = res + x
        return x

    def process_layers(
        self,
        x: torch.Tensor,
        position_ids,
        causal_mask,
        current_pos,
        cos_global,
        sin_global,
        cos_local,
        sin_local,
        start_layer=0,
        end_layer=None,
        IN_PREFILL=False,
    ):
        if end_layer is None:
            end_layer = len(self.layers)
        layer_offset = 0
        mask_is_mapping = isinstance(causal_mask, dict)
        base_causal_mask = None if mask_is_mapping else causal_mask

        for i in range(start_layer, end_layer):
            layer = self.layers[i]
            layer_mask = None
            if mask_is_mapping:
                layer_mask = causal_mask.get(layer.self_attn.attention_type)
            cos_src, sin_src = (cos_local, sin_local) if layer.self_attn.is_sliding else (cos_global, sin_global)
            seq_len = x.shape[1]
            cos_slice = cos_src[:, -seq_len:, :]
            sin_slice = sin_src[:, -seq_len:, :]
            if IN_PREFILL:
                pos_slice = position_ids[:, -seq_len:] if position_ids is not None else None
                x = self.process_layer_prefill(i, x, pos_slice, base_causal_mask, current_pos, cos_slice, sin_slice, layer_offset, layer_mask)
            else:
                x = self.process_layer_regular(i, x, position_ids, base_causal_mask, current_pos, cos_slice, sin_slice, layer_offset, layer_mask)
        return x

    def _prepare_attention_masks(self, inputs_embeds: torch.Tensor, position_ids: torch.Tensor, causal_mask):
        # If caller provided a ready 4D mask, wrap it into a mapping for both attention types
        if isinstance(causal_mask, torch.Tensor) and causal_mask.ndim == 4:
            return {"full_attention": causal_mask, "sliding_attention": causal_mask}
        if isinstance(causal_mask, dict):
            return causal_mask

        cache_position = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device, dtype=torch.long)
        mask_kwargs = {
            "config": self.hf_config,
            "input_embeds": inputs_embeds,
            "attention_mask": None,
            "cache_position": cache_position,
            "past_key_values": None,
            "position_ids": position_ids,
        }

        masks = {
            "full_attention": create_causal_mask(**mask_kwargs),
        }

        if self.config.sliding_window:
            masks["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)
        return masks

    def forward(self, input_ids: torch.LongTensor, causal_mask: torch.Tensor, position_ids: torch.LongTensor, current_pos: torch.LongTensor, IN_PREFILL: bool = False) -> torch.Tensor:
        x = self.embed_tokens(input_ids) * self.embed_scale
        base_fp32 = x.to(torch.float32)
        cos_global, sin_global = self.rotary_global(base_fp32, position_ids)
        cos_local, sin_local = self.rotary_local(base_fp32, position_ids)
        cos_global = cos_global.to(x.dtype)
        sin_global = sin_global.to(x.dtype)
        cos_local = cos_local.to(x.dtype)
        sin_local = sin_local.to(x.dtype)
        mask_map = self._prepare_attention_masks(x, position_ids, causal_mask)
        x = self.process_layers(x, position_ids, mask_map, current_pos, cos_global, sin_global, cos_local, sin_local, IN_PREFILL=IN_PREFILL)
        x = self.norm(x)
        return x

    def forward_prefill(
        self,
        hidden_states,
        position_ids=None,
        causal_mask=None,
        current_pos=None,
        start_layer=None,
        end_layer=None,
    ):
        if current_pos is None:
            current_pos = torch.tensor(0, dtype=torch.int64, device=hidden_states.device)
        if start_layer is None:
            start_layer = 0
        if end_layer is None:
            end_layer = len(self.layers)

        base_fp32 = hidden_states.to(torch.float32)
        cos_global, sin_global = self.rotary_global(base_fp32, position_ids)
        cos_local, sin_local = self.rotary_local(base_fp32, position_ids)
        cos_global = cos_global.to(hidden_states.dtype)
        sin_global = sin_global.to(hidden_states.dtype)
        cos_local = cos_local.to(hidden_states.dtype)
        sin_local = sin_local.to(hidden_states.dtype)

        mask_map = self._prepare_attention_masks(hidden_states, position_ids, causal_mask)

        hidden_states = self.process_layers(
            hidden_states,
            position_ids,
            mask_map,
            current_pos,
            cos_global,
            sin_global,
            cos_local,
            sin_local,
            start_layer,
            end_layer,
            IN_PREFILL=True,
        )

        if end_layer == len(self.layers):
            hidden_states = self.norm(hidden_states)

        return hidden_states

    # Weight loading
    def load_pretrained_weights(self, model_dir: str) -> bool:
        if not os.path.isdir(model_dir):
            raise FileNotFoundError(model_dir)
        state: Dict[str, torch.Tensor] = {}
        for f in os.listdir(model_dir):
            if f.endswith(".safetensors"):
                state.update(safetensors.torch.load_file(os.path.join(model_dir, f)))

        conv_state = {}
        for k, v in state.items():
            nk = k
            if nk.startswith("model."):
                nk = nk.replace("model.", "")
            if nk.startswith("language_model."):
                nk = nk.replace("language_model.", "")
            if nk == "lm_head.weight":
                continue
            # Linear -> Conv reshape for projections and MLP
            if any(p in nk for p in ["q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight", "gate_proj.weight", "up_proj.weight", "down_proj.weight"]):
                conv_state[nk] = v.view(v.shape[0], v.shape[1], 1, 1)
            else:
                conv_state[nk] = v

        missing, unexpected = self.load_state_dict(conv_state, strict=False)
        missing = [m for m in missing if "rotary" in m]
        if missing or unexpected:
            print("[gemma3-ane] missing:", missing)
            print("[gemma3-ane] unexpected:", unexpected)
        return True


class Gemma3ForCausalLMANE(nn.Module):
    config_class = Gemma3ANEConfig

    def __init__(self, cfg: Gemma3ANEConfig, enable_coreml: bool = True) -> None:
        super().__init__()
        self.config = cfg
        self.model = Gemma3ModelANE(cfg, use_conv_mlp=enable_coreml)
        self.enable_coreml = enable_coreml

        if ENABLE_CONV2D:
            if ENABLE_VACAB_SPLIT16:
                vs = cfg.vocab_size // 16
                rem = cfg.vocab_size % 16
                for i in range(16):
                    ss = vs + (1 if i < rem else 0)
                    setattr(self, f"lm_head16_{i+1}", nn.Conv2d(cfg.hidden_size, ss, 1, bias=False, dtype=MODEL_DTYPE).to(TEST_DEVICE))
            elif ENABLE_VACAB_SPLIT8:
                vs = cfg.vocab_size // 8
                rem = cfg.vocab_size % 8
                for i in range(8):
                    ss = vs + (1 if i < rem else 0)
                    setattr(self, f"lm_head8_{i+1}", nn.Conv2d(cfg.hidden_size, ss, 1, bias=False, dtype=MODEL_DTYPE).to(TEST_DEVICE))
            elif ENABLE_VACAB_SPLIT:
                self.lm_head2_1 = nn.Conv2d(cfg.hidden_size, cfg.vocab_size // 2, 1, bias=False, dtype=MODEL_DTYPE).to(TEST_DEVICE)
                self.lm_head2_2 = nn.Conv2d(cfg.hidden_size, cfg.vocab_size - cfg.vocab_size // 2, 1, bias=False, dtype=MODEL_DTYPE).to(TEST_DEVICE)
            else:
                self.lm_head1 = nn.Conv2d(cfg.hidden_size, cfg.vocab_size, 1, bias=False, dtype=MODEL_DTYPE).to(TEST_DEVICE)
        else:
            self.lm_head = nn.Conv2d(cfg.hidden_size, cfg.vocab_size, 1, bias=False, dtype=MODEL_DTYPE).to(TEST_DEVICE)

    def forward(self, input_ids, update_mask, position_ids, causal_mask, current_pos, IN_PREFILL: bool = False):
        hidden = self.model(input_ids, causal_mask, position_ids, current_pos, IN_PREFILL=False)
        if hasattr(self, "lm_head8_1"):
            hs = hidden.permute(0, 2, 1).unsqueeze(2)
            logits = [getattr(self, f"lm_head8_{i}")(hs).squeeze(2).transpose(1, 2) for i in range(1, 9)]
            if self.enable_coreml and ENABLE_LOGITS2:
                return tuple(logits)
            return torch.cat(logits, dim=2)
        elif hasattr(self, "lm_head2_1"):
            hs = hidden.permute(0, 2, 1).unsqueeze(2)
            l1 = self.lm_head2_1(hs).squeeze(2).transpose(1, 2)
            l2 = self.lm_head2_2(hs).squeeze(2).transpose(1, 2)
            if self.enable_coreml and ENABLE_LOGITS2:
                return l1, l2
            return torch.cat([l1, l2], dim=2)
        elif hasattr(self, "lm_head1"):
            hs = hidden.permute(0, 2, 1).unsqueeze(2)
            return self.lm_head1(hs).squeeze(2).transpose(1, 2)
        else:
            hs = hidden.permute(0, 2, 1).unsqueeze(2)
            return self.lm_head(hs).squeeze(2).transpose(1, 2)

    def forward_prefill(self, hidden_states, position_ids=None, causal_mask=None, current_pos=None, start_layer=None, end_layer=None):
        return self.model.forward_prefill(hidden_states, position_ids, causal_mask, current_pos, start_layer, end_layer)

    def load_pretrained_weights(self, model_dir: str) -> bool:
        if not self.model.load_pretrained_weights(model_dir):
            return False
        state: Dict[str, torch.Tensor] = {}
        for f in os.listdir(model_dir):
            if f.endswith(".safetensors"):
                state.update(safetensors.torch.load_file(os.path.join(model_dir, f)))
        w = state.get("lm_head.weight")
        if w is None:
            # Gemma3 ties output head to embeddings; fallback to embed_tokens.weight
            w = state.get("model.embed_tokens.weight") if "model.embed_tokens.weight" in state else state.get("embed_tokens.weight")
            if w is None:
                print("[gemma3-ane] Warning: lm_head.weight missing and no embed_tokens.weight found")
                return False
        w = w.view(w.shape[0], w.shape[1], 1, 1)
        if hasattr(self, "lm_head8_1"):
            vs = self.config.vocab_size // 8
            rem = self.config.vocab_size % 8
            sizes = [vs + (1 if i < rem else 0) for i in range(8)]
            splits = torch.split(w, sizes)
            for i, s in enumerate(splits, 1):
                getattr(self, f"lm_head8_{i}").weight.data.copy_(s)
        elif hasattr(self, "lm_head2_1"):
            v2 = self.config.vocab_size // 2
            s1, s2 = torch.split(w, [v2, self.config.vocab_size - v2])
            self.lm_head2_1.weight.data.copy_(s1)
            self.lm_head2_2.weight.data.copy_(s2)
        elif hasattr(self, "lm_head1"):
            self.lm_head1.weight.data.copy_(w)
        else:
            self.lm_head.weight.data.copy_(w)
        return True
