"""Phi-4 (Phi3 architecture) model implementation for ANEMLL.

This module mirrors the simplified inference-friendly implementations used by
LLaMA and Qwen so that we can run Apple Neural Engine conversions without
depending on the full Hugging Face `transformers` runtime.  Only the pieces that
are required by the CoreML converters are implemented.
"""

from __future__ import annotations

import math
import os
import json
from typing import Dict, Tuple

import safetensors.torch
import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL_DTYPE = torch.float16
TEST_DEVICE = "cpu"
CONTEXT_LENGTH = 512
STATE_LENGTH = 512

# Phi-4 exposes a single LM head (no vocab splitting)
SPLIT_LM_HEAD = 1
ENABLE_COREML = False


def _to_device_dtype(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to(device=TEST_DEVICE, dtype=MODEL_DTYPE)


class PhiConfig:
    """Lightweight configuration object mimicking Hugging Face config."""

    def __init__(self, **kwargs):
        self.architectures = kwargs.get("architectures", ["Phi3ForCausalLM"])
        self.attention_dropout = kwargs.get("attention_dropout", 0.0)
        self.bos_token_id = kwargs.get("bos_token_id", 199999)
        self.eos_token_id = kwargs.get("eos_token_id", 199999)
        self.hidden_act = kwargs.get("hidden_act", "silu")
        self.hidden_size = kwargs.get("hidden_size", 3072)
        self.initializer_range = kwargs.get("initializer_range", 0.02)
        self.intermediate_size = kwargs.get("intermediate_size", 8192)
        self.max_position_embeddings = kwargs.get("max_position_embeddings", 4096)
        self.model_type = kwargs.get("model_type", "phi3")
        self.num_attention_heads = kwargs.get("num_attention_heads", 24)
        self.num_hidden_layers = kwargs.get("num_hidden_layers", 32)
        self.num_key_value_heads = kwargs.get("num_key_value_heads", 8)
        self.rope_theta = kwargs.get("rope_theta", 10000.0)
        self.rope_scaling = kwargs.get("rope_scaling")
        self.partial_rotary_factor = kwargs.get("partial_rotary_factor", 1.0)
        self.tie_word_embeddings = kwargs.get("tie_word_embeddings", True)
        self.vocab_size = kwargs.get("vocab_size", 200064)
        self.sliding_window = kwargs.get("sliding_window", None)
        self.use_cache = kwargs.get("use_cache", True)
        self.context_length = kwargs.get("context_length", CONTEXT_LENGTH)
        self.state_length = max(
            kwargs.get("state_length", STATE_LENGTH),
            self.context_length,
            kwargs.get("original_max_position_embeddings", self.max_position_embeddings),
        )
        self.rope_scaling = kwargs.get("rope_scaling")
        if self.rope_scaling:
            self.rope_scaling.setdefault("rope_type", self.rope_scaling.get("type", "longrope"))

    @classmethod
    def from_json(cls, json_file: str) -> "PhiConfig":
        with open(json_file, "r") as f:
            data = json.load(f)
        return cls(**data)


class PhiRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return (self.weight * hidden_states).to(input_dtype)


class PhiRotaryEmbedding(nn.Module):
    """Rotary embedding implementation specialised for Phi."""

    def __init__(self, config: PhiConfig) -> None:
        super().__init__()
        head_dim = config.hidden_size // config.num_attention_heads
        self.rotary_dim = int(head_dim * config.partial_rotary_factor)
        inv_freq = 1.0 / (
            config.rope_theta ** (torch.arange(0, self.rotary_dim, 2).float() / self.rotary_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, position_ids: torch.LongTensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if position_ids.dim() == 1:
            position_ids = position_ids.unsqueeze(0)
        position_ids = position_ids.to(self.inv_freq.device)
        freqs = torch.einsum("bl,j->blj", position_ids.float(), self.inv_freq.float())
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        return cos.to(MODEL_DTYPE), sin.to(MODEL_DTYPE)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    rotary_dim = cos.shape[-1]
    # Align cos/sin to match (batch, heads, seq_len, rotary_dim)
    while cos.dim() < q.dim():
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    q_rot = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_rot = (k_rot * cos) + (rotate_half(k_rot) * sin)

    q_embed = torch.cat((q_rot, q_pass), dim=-1)
    k_embed = torch.cat((k_rot, k_pass), dim=-1)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    bsz, n_kv, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].repeat(1, 1, n_rep, 1, 1)
    return hidden_states.view(bsz, n_kv * n_rep, seq_len, head_dim)


class PhiMLP(nn.Module):
    def __init__(self, config: PhiConfig) -> None:
        super().__init__()
        hidden = config.hidden_size
        intermediate = config.intermediate_size
        self.gate_up_proj = nn.Conv2d(hidden, 2 * intermediate, 1, bias=False, dtype=MODEL_DTYPE)
        self.down_proj = nn.Conv2d(intermediate, hidden, 1, bias=False, dtype=MODEL_DTYPE)
        self.act = nn.SiLU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hs = hidden_states.permute(0, 2, 1).unsqueeze(2).to(MODEL_DTYPE)
        gate_up = self.gate_up_proj(hs)
        gate, up = gate_up.chunk(2, dim=1)
        activated = self.act(gate) * up
        out = self.down_proj(activated)
        return out.squeeze(2).permute(0, 2, 1)


class PhiAttention(nn.Module):
    def __init__(self, config: PhiConfig) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scale = 1 / math.sqrt(self.head_dim)

        q_dim = self.num_heads * self.head_dim
        kv_dim = self.num_kv_heads * self.head_dim

        self.q_proj = nn.Conv2d(config.hidden_size, q_dim, 1, bias=False, dtype=MODEL_DTYPE)
        self.k_proj = nn.Conv2d(config.hidden_size, kv_dim, 1, bias=False, dtype=MODEL_DTYPE)
        self.v_proj = nn.Conv2d(config.hidden_size, kv_dim, 1, bias=False, dtype=MODEL_DTYPE)
        self.o_proj = nn.Conv2d(q_dim, config.hidden_size, 1, bias=False, dtype=MODEL_DTYPE)
        self.rotary_emb = PhiRotaryEmbedding(config)

    def _project(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hs = hidden_states.permute(0, 2, 1).unsqueeze(2).to(MODEL_DTYPE)
        q = self.q_proj(hs).reshape(hidden_states.size(0), self.num_heads, hidden_states.size(1), self.head_dim)
        k = self.k_proj(hs).reshape(hidden_states.size(0), self.num_kv_heads, hidden_states.size(1), self.head_dim)
        v = self.v_proj(hs).reshape(hidden_states.size(0), self.num_kv_heads, hidden_states.size(1), self.head_dim)
        return q, k, v

    def forward(self, hidden_states: torch.Tensor, causal_mask: torch.Tensor, position_ids: torch.LongTensor) -> torch.Tensor:
        bsz, seq_len, _ = hidden_states.shape
        q, k, v = self._project(hidden_states)

        cos, sin = self.rotary_emb(position_ids)
        cos = cos.to(hidden_states.dtype)
        sin = sin.to(hidden_states.dtype)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        n_rep = self.num_heads // self.num_kv_heads
        k = repeat_kv(k, n_rep)
        v = repeat_kv(v, n_rep)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if causal_mask is not None:
            causal_slice = causal_mask[:, :, :seq_len, :seq_len]
            attn_weights = attn_weights + causal_slice.to(attn_weights.dtype)
        attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.permute(0, 1, 3, 2).reshape(bsz, -1, seq_len)
        attn_output = attn_output.unsqueeze(2)
        output = self.o_proj(attn_output)
        return output.squeeze(2).permute(0, 2, 1)

    def get_new_kv_cache(self, hidden_states: torch.Tensor, current_pos: torch.Tensor, rotary_emb) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q, k, v = self._project(hidden_states)
        cos, sin = rotary_emb
        cos = cos.to(hidden_states.dtype)
        sin = sin.to(hidden_states.dtype)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        return q, k, v

    def get_new_kv_cache_prefill(self, hidden_states: torch.Tensor, current_pos: torch.Tensor, rotary_emb) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.get_new_kv_cache(hidden_states, current_pos, rotary_emb)

    def forward_regular(self, hidden_states: torch.Tensor, query_states: torch.Tensor, kv_cache_layer: Tuple[torch.Tensor, torch.Tensor], causal_mask: torch.Tensor, current_pos: torch.Tensor) -> torch.Tensor:
        key_cache, value_cache = kv_cache_layer
        n_rep = self.num_heads // self.num_kv_heads
        key_states = repeat_kv(key_cache.unsqueeze(0), n_rep)
        value_states = repeat_kv(value_cache.unsqueeze(0), n_rep)
        attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) * self.scale
        if causal_mask is not None:
            attn_weights = attn_weights + causal_mask.to(attn_weights.dtype)
        attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.permute(0, 1, 3, 2).reshape(hidden_states.size(0), -1, hidden_states.size(1))
        attn_output = attn_output.unsqueeze(2)
        output = self.o_proj(attn_output)
        return output.squeeze(2).permute(0, 2, 1)

    def forward_prefill(self, hidden_states: torch.Tensor, query_states: torch.Tensor, kv_cache_layer: Tuple[torch.Tensor, torch.Tensor], causal_mask: torch.Tensor) -> torch.Tensor:
        key_cache, value_cache = kv_cache_layer
        n_rep = self.num_heads // self.num_kv_heads
        key_states = repeat_kv(key_cache.unsqueeze(0), n_rep)
        value_states = repeat_kv(value_cache.unsqueeze(0), n_rep)
        attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) * self.scale
        if causal_mask is not None:
            attn_weights = attn_weights + causal_mask.to(attn_weights.dtype)
        attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.permute(0, 1, 3, 2).reshape(hidden_states.size(0), -1, hidden_states.size(1))
        attn_output = attn_output.unsqueeze(2)
        output = self.o_proj(attn_output)
        return output.squeeze(2).permute(0, 2, 1)


class PhiDecoderLayer(nn.Module):
    def __init__(self, config: PhiConfig) -> None:
        super().__init__()
        self.self_attn = PhiAttention(config)
        self.mlp = PhiMLP(config)
        self.input_layernorm = PhiRMSNorm(config.hidden_size, eps=1e-5)
        self.post_attention_layernorm = PhiRMSNorm(config.hidden_size, eps=1e-5)

    def forward(self, hidden_states: torch.Tensor, causal_mask: torch.Tensor, position_ids: torch.LongTensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, causal_mask, position_ids)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


def get_kv_cache_idx(layer_idx: int, num_layers: int) -> Tuple[int, int, int]:
    layers_per_group = num_layers
    group_idx = 0
    layer_in_group = layer_idx
    return group_idx, layer_in_group, layers_per_group


class PhiModel(nn.Module):
    def __init__(self, config: PhiConfig) -> None:
        super().__init__()
        self.config = config
        self.disable_kv_cache = False
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size).to(TEST_DEVICE)
        self.layers = nn.ModuleList([PhiDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = PhiRMSNorm(config.hidden_size, eps=1e-5)

        head_dim = config.hidden_size // config.num_attention_heads
        cache_shape = (
            2 * config.num_hidden_layers,
            config.num_key_value_heads,
            config.state_length,
            head_dim,
        )
        self.register_buffer("kv_cache_0", torch.zeros(cache_shape, dtype=MODEL_DTYPE, device=TEST_DEVICE))

    def get_rotary_embeddings_s(self, current_pos: torch.Tensor):
        position_ids = current_pos.reshape(-1).long()
        cos, sin = self.layers[0].self_attn.rotary_emb(position_ids)
        return cos.to(MODEL_DTYPE), sin.to(MODEL_DTYPE)

    def get_rotary_embedding_prefill(self, positions: torch.Tensor):
        cos, sin = self.layers[0].self_attn.rotary_emb(positions.long())
        return cos.to(MODEL_DTYPE), sin.to(MODEL_DTYPE)

    def process_layer_prefill(self, layer_idx: int, hidden_states: torch.Tensor, position_ids: torch.Tensor, causal_mask: torch.Tensor, current_pos: int, rotary_emb) -> torch.Tensor:
        layer = self.layers[layer_idx]
        normalized = layer.input_layernorm(hidden_states)
        query_states, key_states, value_states = layer.self_attn.get_new_kv_cache_prefill(normalized, current_pos, rotary_emb)

        kv_cache = self.kv_cache_0
        key_idx = layer_idx
        value_idx = layer_idx + self.config.num_hidden_layers
        seq_len = key_states.shape[2]
        kv_cache[key_idx:key_idx + 1, :, current_pos:current_pos + seq_len, :] = key_states
        kv_cache[value_idx:value_idx + 1, :, current_pos:current_pos + seq_len, :] = value_states

        key_cache = kv_cache[key_idx:key_idx + 1].squeeze(0)
        value_cache = kv_cache[value_idx:value_idx + 1].squeeze(0)
        causal = self._build_causal_mask(seq_len, torch.tensor(current_pos, device=hidden_states.device, dtype=torch.int64), hidden_states.device)
        attn_out = layer.self_attn.forward_prefill(normalized, query_states, (key_cache, value_cache), causal)
        hidden_states = hidden_states + attn_out
        post = layer.post_attention_layernorm(hidden_states)
        hidden_states = hidden_states + layer.mlp(post)
        return hidden_states

    def process_layer_regular(self, layer_idx: int, hidden_states: torch.Tensor, position_ids: torch.Tensor, causal_mask: torch.Tensor, current_pos: torch.Tensor, rotary_emb) -> torch.Tensor:
        layer = self.layers[layer_idx]
        normalized = layer.input_layernorm(hidden_states)
        seq_len = hidden_states.shape[1]
        query_states, key_states, value_states = layer.self_attn.get_new_kv_cache(normalized, current_pos, rotary_emb)

        if not self.disable_kv_cache:
            kv_cache = self.kv_cache_0
            key_idx = layer_idx
            value_idx = layer_idx + self.config.num_hidden_layers
            pos = int(current_pos.item()) if isinstance(current_pos, torch.Tensor) else int(current_pos)
            kv_cache[key_idx:key_idx + 1, :, pos:pos + seq_len, :] = key_states
            kv_cache[value_idx:value_idx + 1, :, pos:pos + seq_len, :] = value_states
            key_cache = kv_cache[key_idx:key_idx + 1].squeeze(0)
            value_cache = kv_cache[value_idx:value_idx + 1].squeeze(0)
            causal = self._build_causal_mask(seq_len, current_pos, hidden_states.device)
            attn_out = layer.self_attn.forward_regular(normalized, query_states, (key_cache, value_cache), causal, current_pos)
        else:
            attn_out = layer.self_attn(normalized, causal_mask, position_ids)
        hidden_states = hidden_states + attn_out
        post = layer.post_attention_layernorm(hidden_states)
        hidden_states = hidden_states + layer.mlp(post)
        return hidden_states

    def _build_causal_mask(self, seq_len: int, current_pos: torch.Tensor | int, device: torch.device) -> torch.Tensor:
        cache_len = self.config.state_length
        positions = torch.arange(cache_len, device=device, dtype=torch.int64)
        if isinstance(current_pos, torch.Tensor):
            base_pos = current_pos.to(torch.int64).reshape(1, 1)
        else:
            base_pos = torch.tensor([[current_pos]], device=device, dtype=torch.int64)
        offsets = torch.arange(seq_len, device=device, dtype=torch.int64).unsqueeze(1)
        limits = base_pos + offsets
        allowed = positions.unsqueeze(0) <= limits
        mask = torch.where(
            allowed,
            torch.zeros_like(allowed, dtype=MODEL_DTYPE),
            torch.full_like(allowed, float('-inf'), dtype=MODEL_DTYPE),
        )
        return mask.unsqueeze(0).unsqueeze(0)

    def process_layers(self, hidden_states: torch.Tensor, position_ids: torch.Tensor, causal_mask: torch.Tensor, current_pos: torch.Tensor, rotary_emb, start_layer: int = 0, end_layer: int | None = None, IN_PREFILL: bool = False) -> torch.Tensor:
        if end_layer is None:
            end_layer = len(self.layers)
        for idx in range(start_layer, end_layer):
            if IN_PREFILL:
                hidden_states = self.process_layer_prefill(idx, hidden_states, position_ids, causal_mask, int(current_pos), rotary_emb)
            else:
                hidden_states = self.process_layer_regular(idx, hidden_states, position_ids, causal_mask, current_pos, rotary_emb)
        return hidden_states

    def forward(self, input_ids: torch.Tensor, causal_mask: torch.Tensor, position_ids: torch.Tensor, current_pos: torch.Tensor, IN_PREFILL: bool = False) -> torch.Tensor:
        hidden = self.embed_tokens(input_ids)
        if IN_PREFILL:
            rotary = self.get_rotary_embedding_prefill(position_ids.squeeze(0))
        else:
            rotary = self.get_rotary_embeddings_s(current_pos)
        hidden = self.process_layers(hidden, position_ids, causal_mask, current_pos, rotary, 0, None, IN_PREFILL)
        hidden = self.norm(hidden)
        return hidden

    def forward_prefill(self, hidden_states: torch.Tensor, position_ids: torch.Tensor, causal_mask: torch.Tensor, current_pos: torch.Tensor) -> torch.Tensor:
        rotary = self.get_rotary_embedding_prefill(position_ids)
        hidden_states = self.process_layers(hidden_states, position_ids.unsqueeze(0), causal_mask, int(current_pos), rotary, IN_PREFILL=True)
        hidden_states = self.norm(hidden_states)
        return hidden_states

    def load_pretrained_weights(self, model_path: str) -> bool:
        if not os.path.isdir(model_path):
            raise FileNotFoundError(model_path)
        state_dict: Dict[str, torch.Tensor] = {}
        for file in os.listdir(model_path):
            if file.endswith(".safetensors"):
                state_dict.update(safetensors.torch.load_file(os.path.join(model_path, file)))

        new_state: Dict[str, torch.Tensor] = {}
        head_dim = self.config.hidden_size // self.config.num_attention_heads
        q_dim = self.config.num_attention_heads * head_dim
        kv_dim = self.config.num_key_value_heads * head_dim

        for k, v in state_dict.items():
            if k.startswith("model."):
                k = k[len("model."):]
            if k.endswith("self_attn.qkv_proj.weight"):
                layer_idx = int(k.split(".")[1])
                q, k_weight, v_weight = torch.split(v, [q_dim, kv_dim, kv_dim], dim=0)
                new_state[f"layers.{layer_idx}.self_attn.q_proj.weight"] = q.view(q_dim, self.config.hidden_size, 1, 1)
                new_state[f"layers.{layer_idx}.self_attn.k_proj.weight"] = k_weight.view(kv_dim, self.config.hidden_size, 1, 1)
                new_state[f"layers.{layer_idx}.self_attn.v_proj.weight"] = v_weight.view(kv_dim, self.config.hidden_size, 1, 1)
            elif k.endswith("self_attn.o_proj.weight"):
                layer_idx = int(k.split(".")[1])
                new_state[f"layers.{layer_idx}.self_attn.o_proj.weight"] = v.view(self.config.hidden_size, q_dim, 1, 1)
            elif k.endswith("mlp.down_proj.weight"):
                layer_idx = int(k.split(".")[1])
                new_state[f"layers.{layer_idx}.mlp.down_proj.weight"] = v.view(self.config.hidden_size, self.config.intermediate_size, 1, 1)
            elif k.endswith("mlp.gate_up_proj.weight"):
                layer_idx = int(k.split(".")[1])
                new_state[f"layers.{layer_idx}.mlp.gate_up_proj.weight"] = v.view(2 * self.config.intermediate_size, self.config.hidden_size, 1, 1)
            elif k == "embed_tokens.weight":
                new_state[k] = v
            elif k == "norm.weight":
                new_state[k] = v
            elif k.endswith("input_layernorm.weight"):
                new_state[k] = v
            elif k.endswith("post_attention_layernorm.weight"):
                new_state[k] = v
            elif k == "lm_head.weight":
                new_state["lm_head.weight"] = v.view(self.config.vocab_size, self.config.hidden_size, 1, 1)

        missing, unexpected = self.load_state_dict(new_state, strict=False)
        missing = [m for m in missing if not m.startswith("kv_cache_0") and "rotary_emb" not in m]
        if missing or unexpected:
            print("Missing", missing)
            print("Unexpected", unexpected)
        return not missing and not unexpected


class PhiForCausalLM(nn.Module):
    config_class = PhiConfig

    def __init__(self, config: PhiConfig, enable_coreml: bool = False, disable_kv_cache: bool = False) -> None:
        super().__init__()
        global ENABLE_COREML
        ENABLE_COREML = enable_coreml
        self.config = config
        self.model = PhiModel(config)
        self.model.disable_kv_cache = disable_kv_cache
        self.lm_head = nn.Conv2d(config.hidden_size, config.vocab_size, 1, bias=False, dtype=MODEL_DTYPE).to(TEST_DEVICE)

    def forward(self, input_ids: torch.Tensor, update_mask: torch.Tensor, position_ids: torch.Tensor, causal_mask: torch.Tensor, current_pos: torch.Tensor, IN_PREFILL: bool = False) -> torch.Tensor:
        hidden = self.model(input_ids, causal_mask, position_ids, current_pos, IN_PREFILL=IN_PREFILL)
        if hidden.dim() == 2:
            hidden = hidden.unsqueeze(0)
        hs = hidden.permute(0, 2, 1).unsqueeze(2)
        logits = self.lm_head(hs).squeeze(2).permute(0, 2, 1)
        return logits

    def forward_prefill(self, hidden_states: torch.Tensor, position_ids: torch.Tensor, causal_mask: torch.Tensor, current_pos: torch.Tensor) -> torch.Tensor:
        hidden = self.model.forward_prefill(hidden_states, position_ids, causal_mask, current_pos)
        hs = hidden.permute(0, 2, 1).unsqueeze(2)
        logits = self.lm_head(hs).squeeze(2).permute(0, 2, 1)
        return logits

    def load_pretrained_weights(self, model_path: str) -> bool:
        if not self.model.load_pretrained_weights(model_path):
            return False
        state_dict: Dict[str, torch.Tensor] = {}
        for file in os.listdir(model_path):
            if file.endswith(".safetensors"):
                state_dict.update(safetensors.torch.load_file(os.path.join(model_path, file)))
        if "lm_head.weight" in state_dict:
            self.lm_head.weight.data.copy_(state_dict["lm_head.weight"].view(self.config.vocab_size, self.config.hidden_size, 1, 1))
        return True


__all__ = [
    "PhiConfig",
    "PhiModel",
    "PhiForCausalLM",
    "MODEL_DTYPE",
    "TEST_DEVICE",
    "CONTEXT_LENGTH",
]
