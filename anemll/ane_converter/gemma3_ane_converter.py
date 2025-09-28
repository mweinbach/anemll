"""Split converter for Gemma 3 (text-only) ANE model.

Mirrors the Qwen/LLaMA split conversion:
 - Part 1: Embeddings
 - Part 2: Transformer (FFN) for generation (with unified KV state)
 - Part 2_prefill: Prefill path to seed KV cache for a batch
 - Part 3: LM head

Weights are loaded from a HF snapshot directory (safetensors).
"""

from __future__ import annotations

import argparse
import os
from typing import Optional, List

import numpy as np
import torch
import coremltools as ct
import coremltools.optimize as cto

from .environment import require_coreml
from .base_converter import BaseConverter
from .metadata import AddMetadata, ModelPart
from ..models.gemma3_ane_model import (
    Gemma3ANEConfig,
    Gemma3ForCausalLMANE,
    TEST_DEVICE,
    MODEL_DTYPE,
)


class Gemma3ANEConverter(BaseConverter):
    def __init__(self, model: Gemma3ForCausalLMANE, context_length: int = 512, batch_size: int = 64, lut_bits: Optional[int] = 4, num_chunks: int = 1) -> None:
        super().__init__(model)
        self.context_length = context_length
        self.batch_size = batch_size
        self.lut_bits = lut_bits
        self.num_chunks = num_chunks
        self.converted_model = None

    @staticmethod
    def GetTransformerStates(model, part=None, prefix="model."):
        head_dim = model.model.head_dim
        num_layers = model.config.num_hidden_layers
        states = [
            ct.StateType(
                wrapped_type=ct.TensorType(
                    shape=(2 * num_layers, model.config.num_key_value_heads, model.config.state_length, head_dim),
                    dtype=np.float16,
                ),
                name=f"{prefix}kv_cache_0",
            )
        ]
        return states

    def postprocess(self, num_workers=None):
        if self.converted_model is not None and self.lut_bits is not None:
            config = cto.coreml.OptimizationConfig(
                global_config=cto.coreml.OpPalettizerConfig(
                    mode="kmeans", nbits=self.lut_bits, granularity="per_grouped_channel", group_size=8, num_kmeans_workers=(num_workers if num_workers is not None else 1)
                )
            )
            self.converted_model = cto.coreml.palettize_weights(self.converted_model, config)

    def convert(self, part: str = "all") -> ct.models.MLModel | List[ct.models.MLModel]:
        require_coreml()
        self.preprocess()
        if part in ("all", "full", "123"):
            return [self.convert_part_1(self.model), self.convert_part_2(self.model), self.convert_part_3(self.model)]
        if part in ("1", "embeddings"):
            return self.convert_part_1(self.model)
        if part == "2":
            return self.convert_part_2(self.model)
        if part in ("2_prefill", "prefill"):
            return self.convert_part_2_prefill(self.model)
        if part == "3":
            return self.convert_part_3(self.model)
        raise ValueError(part)

    def convert_part_1(self, model: Gemma3ForCausalLMANE) -> ct.models.MLModel:
        import torch.nn as nn
        class Emb(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m
            def forward(self, input_ids):
                return self.m.model.embed_tokens(input_ids)
        import torch.nn as nn
        w = Emb(model)
        w.eval()
        x = torch.zeros((1, 1), dtype=torch.int32, device=TEST_DEVICE)
        traced = torch.jit.trace(w, (x,))
        ml = ct.convert(
            traced,
            inputs=[ct.TensorType(name="input_ids", shape=x.shape, dtype=np.int32)],
            outputs=[ct.TensorType(name="hidden_states", dtype=np.float16)],
            compute_precision=ct.precision.FLOAT16,
            compute_units=ct.ComputeUnit.CPU_AND_NE,
            minimum_deployment_target=ct.target.iOS18,
            convert_to="mlprogram",
        )
        if self.lut_bits:
            self.converted_model = ml
            self.postprocess()
            ml = self.converted_model
        return ml

    def convert_part_3(self, model: Gemma3ForCausalLMANE) -> ct.models.MLModel:
        import torch.nn as nn
        class Head(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m
            def forward(self, hidden_states):
                hs = hidden_states.permute(0, 2, 1).unsqueeze(2)
                if hasattr(self.m, "lm_head8_1"):
                    outs = [getattr(self.m, f"lm_head8_{i}")(hs).squeeze(2).transpose(1, 2) for i in range(1, 9)]
                    return tuple(outs)
                return self.m.lm_head1(hs).squeeze(2).transpose(1, 2)
        import torch.nn as nn
        w = Head(model)
        w.eval()
        h = torch.zeros((1, 1, model.config.hidden_size), dtype=torch.float16, device=TEST_DEVICE)
        traced = torch.jit.trace(w, (h,))
        outputs = [ct.TensorType(name=f"logits{i}", dtype=np.float16) for i in range(1, 9)] if hasattr(model, "lm_head8_1") else [ct.TensorType(name="logits", dtype=np.float16)]
        ml = ct.convert(
            traced,
            inputs=[ct.TensorType(name="hidden_states", shape=h.shape, dtype=np.float16)],
            outputs=outputs,
            compute_precision=ct.precision.FLOAT16,
            compute_units=ct.ComputeUnit.CPU_AND_NE,
            minimum_deployment_target=ct.target.iOS18,
            convert_to="mlprogram",
        )
        if self.lut_bits:
            self.converted_model = ml
            self.postprocess()
            ml = self.converted_model
        return ml

    def convert_part_2(self, model: Gemma3ForCausalLMANE) -> ct.models.MLModel:
        import torch.nn as nn
        class FFN(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m
            def forward(self, hidden_states, position_ids, causal_mask, current_pos):
                if position_ids.dim() == 1:
                    position_ids = position_ids.unsqueeze(0)
                cos_global, sin_global = self.m.model.rotary_global(hidden_states, position_ids)
                cos_local, sin_local = self.m.model.rotary_local(hidden_states, position_ids)
                # Process all transformer layers and apply final norm
                hidden_states = self.m.model.process_layers(
                    hidden_states,
                    position_ids,
                    causal_mask,
                    current_pos,
                    cos_global,
                    sin_global,
                    cos_local,
                    sin_local,
                    IN_PREFILL=False,
                )
                return self.m.model.norm(hidden_states)
        import torch.nn as nn
        w = FFN(model)
        w.eval()
        h = torch.zeros((1, 1, model.config.hidden_size), dtype=torch.float16, device=TEST_DEVICE)
        pid = torch.zeros((1,), dtype=torch.int32, device=TEST_DEVICE)
        mask = torch.zeros((1, 1, 1, self.context_length), dtype=torch.float16, device=TEST_DEVICE)
        cur = torch.zeros((1,), dtype=torch.int32, device=TEST_DEVICE)
        traced = torch.jit.trace(w, (h, pid, mask, cur))
        ml = ct.convert(
            traced,
            inputs=[
                ct.TensorType(name="hidden_states", shape=h.shape, dtype=np.float16),
                ct.TensorType(name="position_ids", shape=pid.shape, dtype=np.int32),
                ct.TensorType(name="causal_mask", shape=mask.shape, dtype=np.float16),
                ct.TensorType(name="current_pos", shape=cur.shape, dtype=np.int32),
            ],
            outputs=[ct.TensorType(name="output_hidden_states", dtype=np.float16)],
            states=self.GetTransformerStates(model, prefix="m.model."),
            compute_precision=ct.precision.FLOAT16,
            compute_units=ct.ComputeUnit.CPU_AND_NE,
            minimum_deployment_target=ct.target.iOS18,
            convert_to="mlprogram",
        )
        if self.lut_bits:
            self.converted_model = ml
            self.postprocess(num_workers=8)
            ml = self.converted_model
        return ml

    def convert_part_2_prefill(self, model: Gemma3ForCausalLMANE) -> ct.models.MLModel:
        import torch.nn as nn
        class Prefill(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m
                self.states = Gemma3ANEConverter.GetTransformerStates(m, prefix="m.model.")
            def forward(self, hidden_states, position_ids, causal_mask, current_pos):
                return self.m.model.forward_prefill(hidden_states, position_ids, causal_mask, current_pos)
        import torch.nn as nn
        w = Prefill(model)
        w.eval()
        h = torch.zeros((1, self.batch_size, model.config.hidden_size), dtype=torch.float16, device=TEST_DEVICE)
        # position_ids must be 2D [batch, seq] for Gemma rotary embeddings
        pid = torch.zeros((1, self.batch_size), dtype=torch.int32, device=TEST_DEVICE)
        mask = torch.zeros((1, 1, self.batch_size, self.context_length), dtype=torch.float16, device=TEST_DEVICE)
        cur = torch.zeros((1,), dtype=torch.int32, device=TEST_DEVICE)
        traced = torch.jit.trace(w, (h, pid, mask, cur))
        ml = ct.convert(
            traced,
            inputs=[
                ct.TensorType(name="hidden_states", shape=h.shape, dtype=np.float16),
                ct.TensorType(name="position_ids", shape=pid.shape, dtype=np.int32),
                ct.TensorType(name="causal_mask", shape=mask.shape, dtype=np.float16),
                ct.TensorType(name="current_pos", shape=cur.shape, dtype=np.int32),
            ],
            outputs=[ct.TensorType(name="output_hidden_states", dtype=np.float16)],
            states=w.states,
            compute_precision=ct.precision.FLOAT16,
            compute_units=ct.ComputeUnit.CPU_AND_NE,
            minimum_deployment_target=ct.target.iOS18,
            convert_to="mlprogram",
        )
        if self.lut_bits:
            self.converted_model = ml
            self.postprocess(num_workers=None)
            ml = self.converted_model
        return ml


def parse_args():
    ap = argparse.ArgumentParser(description="Gemma3 ANE converter")
    ap.add_argument("--model", type=str, required=True, help="Path to HF snapshot (directory)")
    ap.add_argument("--prefix", type=str, default="gemma3", help="Output filename prefix")
    ap.add_argument("--context-length", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lut", type=int, default=None)
    ap.add_argument("--output", type=str, default=".")
    ap.add_argument("--part", type=str, choices=["1", "2", "2_prefill", "3", "all", "full", "prefill", "embeddings"], default="all")
    # Accept --chunk for compatibility with convert_model.sh (ignored; Gemma 3 ANE path isn't chunked yet)
    ap.add_argument("--chunk", type=int, default=1)
    return ap.parse_args()


def test_conversion(model_path: str, prefix: str, context_length: int, batch_size: int, lut_bits: Optional[int], output_dir: str, part: str) -> ct.models.MLModel | List[ct.models.MLModel]:
    cfg = Gemma3ANEConfig.from_json(os.path.join(model_path, "config.json"))
    cfg.context_length = context_length
    cfg.state_length = max(cfg.state_length, context_length)
    model = Gemma3ForCausalLMANE(cfg, enable_coreml=True)
    model.load_pretrained_weights(model_path)
    conv = Gemma3ANEConverter(model, context_length=context_length, batch_size=batch_size, lut_bits=lut_bits)
    ml = conv.convert(part=part)
    os.makedirs(output_dir, exist_ok=True)
    mdls = ml if isinstance(ml, list) else [ml]
    for i, m in enumerate(mdls):
        AddMetadata(m, {
            "context_length": context_length,
            "batch_size": batch_size if part in ["2_prefill", "prefill"] else None,
            "lut_bits": lut_bits,
            "num_chunks": 1,
            "chunk_no": None,
            "split_part": (ModelPart.FULL.value if part in ["full", "all", "123"] else part),
        })
        fname = f"{prefix}"
        if part in ["1", "embeddings"]:
            fname += "_embeddings"
        elif part == "3":
            fname += "_lm_head"
        elif part in ["2", "2_prefill"]:
            base = "FFN" if part == "2" else "prefill"
            fname += f"_{base}"
        if lut_bits is not None:
            fname += f"_lut{lut_bits}"
        if part in ["2", "2_prefill"]:
            fname += f"_chunk_01of01"
        fname += ".mlpackage"
        out = os.path.join(output_dir, fname)
        m.save(out)
    return ml


def main():  # pragma: no cover - CLI
    args = parse_args()
    try:
        pmap = {"full": "all", "embeddings": "1", "prefill": "2_prefill"}
        test_conversion(model_path=args.model, prefix=args.prefix, context_length=args.context_length, batch_size=args.batch_size, lut_bits=args.lut, output_dir=args.output, part=pmap.get(args.part, args.part))
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise SystemExit(1)


if __name__ == "__main__":
    main()
