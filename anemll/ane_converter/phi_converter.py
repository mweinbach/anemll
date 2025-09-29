"""Converter for Phi-4 (Phi3 architecture) models."""

from __future__ import annotations

import argparse
import os
from typing import List, Optional

import numpy as np
import torch
import coremltools as ct
import coremltools.optimize as cto

from .base_converter import BaseConverter
from ..models.phi_model import (
    PhiForCausalLM,
    PhiConfig,
    MODEL_DTYPE,
    TEST_DEVICE,
    CONTEXT_LENGTH,
)


class PhiConverter(BaseConverter):
    """Handles conversion of Phi models into CoreML split components."""

    model_cls = PhiForCausalLM

    def __init__(
        self,
        model: PhiForCausalLM,
        context_length: int = CONTEXT_LENGTH,
        batch_size: int = 64,
        lut_bits: Optional[int] = None,
        num_chunks: int = 1,
    ) -> None:
        super().__init__(model)
        self.context_length = context_length
        self.batch_size = batch_size
        self.lut_bits = lut_bits
        self.num_chunks = num_chunks
        self.converted_model = None

    def postprocess(self, num_workers: Optional[int] = None) -> None:
        if self.converted_model is not None and self.lut_bits is not None:
            config = cto.coreml.OptimizationConfig(
                global_config=cto.coreml.OpPalettizerConfig(
                    mode="kmeans",
                    nbits=self.lut_bits,
                    granularity="per_grouped_channel",
                    group_size=8,
                    num_kmeans_workers=num_workers or 1,
                )
            )
            self.converted_model = cto.coreml.palettize_weights(self.converted_model, config)

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------
    def convert_embeddings(self, model: PhiForCausalLM) -> ct.models.MLModel:
        class EmbeddingsWrapper(torch.nn.Module):
            def __init__(self, inner: PhiForCausalLM) -> None:
                super().__init__()
                self.embed = inner.model.embed_tokens

            def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
                return self.embed(input_ids).to(MODEL_DTYPE)

        wrapper = EmbeddingsWrapper(model)
        wrapper.eval()
        sample = torch.zeros((1, 1), dtype=torch.int32, device=TEST_DEVICE)
        traced = torch.jit.trace(wrapper, sample)
        input_shape = ct.EnumeratedShapes(
            shapes=[[1, 1], [1, self.batch_size]],
            default=[1, 1],
        )
        mlmodel = ct.convert(
            traced,
            inputs=[ct.TensorType(name="input_ids", shape=input_shape, dtype=np.int32)],
            outputs=[ct.TensorType(name="hidden_states", dtype=np.float16)],
            compute_precision=ct.precision.FLOAT16,
            compute_units=ct.ComputeUnit.CPU_AND_NE,
            minimum_deployment_target=ct.target.iOS18,
            convert_to="mlprogram",
        )
        if self.lut_bits:
            self.converted_model = mlmodel
            self.postprocess(num_workers=8)
            mlmodel = self.converted_model
        return mlmodel

    # ------------------------------------------------------------------
    # LM Head
    # ------------------------------------------------------------------
    def convert_lm_head(self, model: PhiForCausalLM) -> ct.models.MLModel:
        class LMHeadWrapper(torch.nn.Module):
            def __init__(self, inner: PhiForCausalLM) -> None:
                super().__init__()
                self.head = inner.lm_head

            def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
                hs = hidden_states.permute(0, 2, 1).unsqueeze(2)
                out = self.head(hs)
                return out.squeeze(2).permute(0, 2, 1)

        wrapper = LMHeadWrapper(model)
        wrapper.eval()
        sample = torch.zeros((1, 1, model.config.hidden_size), dtype=MODEL_DTYPE, device=TEST_DEVICE)
        traced = torch.jit.trace(wrapper, sample)
        mlmodel = ct.convert(
            traced,
            inputs=[ct.TensorType(name="hidden_states", shape=sample.shape, dtype=np.float16)],
            outputs=[ct.TensorType(name="output_logits", dtype=np.float16)],
            compute_precision=ct.precision.FLOAT16,
            compute_units=ct.ComputeUnit.CPU_AND_NE,
            minimum_deployment_target=ct.target.iOS18,
            convert_to="mlprogram",
        )
        if self.lut_bits:
            self.converted_model = mlmodel
            self.postprocess(num_workers=8)
            mlmodel = self.converted_model
        return mlmodel

    # ------------------------------------------------------------------
    # Transformer chunks
    # ------------------------------------------------------------------
    def _chunk_indices(self, total_layers: int) -> List[tuple[int, int]]:
        if self.num_chunks <= 1:
            return [(0, total_layers)]
        per_chunk = total_layers // self.num_chunks
        indices = []
        for idx in range(self.num_chunks):
            start = idx * per_chunk
            end = total_layers if idx == self.num_chunks - 1 else (idx + 1) * per_chunk
            indices.append((start, end))
        return indices

    def convert_ffn_chunk(self, model: PhiForCausalLM, chunk_idx: int, start: int, end: int) -> ct.models.MLModel:
        class FFNWrapper(torch.nn.Module):
            def __init__(self, inner: PhiForCausalLM, start_layer: int, end_layer: int, context_length: int) -> None:
                super().__init__()
                self.model = inner.model
                self.start = start_layer
                self.end = end_layer
                self.context_length = context_length
                self.states = self.model.kv_cache_0  # reuse buffer for tracing

            def forward(self, hidden_states, position_ids, causal_mask, current_pos):
                rotary = self.model.get_rotary_embeddings_s(current_pos)
                out = self.model.process_layers(
                    hidden_states,
                    position_ids,
                    causal_mask,
                    current_pos,
                    rotary,
                    start_layer=self.start,
                    end_layer=self.end,
                    IN_PREFILL=False,
                )
                return self.model.norm(out)

        wrapper = FFNWrapper(model, start, end, self.context_length)
        wrapper.eval()
        hidden = torch.zeros((1, 1, model.config.hidden_size), dtype=MODEL_DTYPE, device=TEST_DEVICE)
        position_ids = torch.zeros((1,), dtype=torch.int32, device=TEST_DEVICE)
        causal_mask = torch.zeros((1, 1, 1, self.context_length), dtype=MODEL_DTYPE, device=TEST_DEVICE)
        current_pos = torch.zeros((1,), dtype=torch.int32, device=TEST_DEVICE)
        traced = torch.jit.trace(wrapper, (hidden, position_ids, causal_mask, current_pos))
        mlmodel = ct.convert(
            traced,
            inputs=[
                ct.TensorType(name="hidden_states", shape=hidden.shape, dtype=np.float16),
                ct.TensorType(name="position_ids", shape=position_ids.shape, dtype=np.int32),
                ct.TensorType(name="causal_mask", shape=causal_mask.shape, dtype=np.float16),
                ct.TensorType(name="current_pos", shape=current_pos.shape, dtype=np.int32),
            ],
            outputs=[ct.TensorType(name="output_hidden_states", dtype=np.float16)],
            compute_precision=ct.precision.FLOAT16,
            compute_units=ct.ComputeUnit.CPU_AND_NE,
            minimum_deployment_target=ct.target.iOS18,
            convert_to="mlprogram",
        )
        if self.lut_bits:
            self.converted_model = mlmodel
            self.postprocess(num_workers=None if self.num_chunks > 1 else 8)
            mlmodel = self.converted_model
        return mlmodel

    def convert_prefill_chunk(self, model: PhiForCausalLM, chunk_idx: int, start: int, end: int) -> ct.models.MLModel:
        class PrefillWrapper(torch.nn.Module):
            def __init__(self, inner: PhiForCausalLM, start_layer: int, end_layer: int, batch_size: int, context_length: int) -> None:
                super().__init__()
                self.model = inner.model
                self.start = start_layer
                self.end = end_layer
                self.batch_size = batch_size
                self.context_length = context_length

            def forward(self, hidden_states, position_ids, causal_mask, current_pos):
                rotary = self.model.get_rotary_embedding_prefill(position_ids)
                out = self.model.process_layers(
                    hidden_states,
                    position_ids.unsqueeze(0),
                    causal_mask,
                    int(current_pos.item()) if current_pos.numel() else 0,
                    rotary,
                    start_layer=self.start,
                    end_layer=self.end,
                    IN_PREFILL=True,
                )
                return out

        wrapper = PrefillWrapper(model, start, end, self.batch_size, self.context_length)
        wrapper.eval()
        hidden = torch.zeros((1, self.batch_size, model.config.hidden_size), dtype=MODEL_DTYPE, device=TEST_DEVICE)
        position_ids = torch.arange(self.batch_size, dtype=torch.int32, device=TEST_DEVICE)
        causal_mask = torch.zeros((1, 1, self.batch_size, self.context_length), dtype=MODEL_DTYPE, device=TEST_DEVICE)
        current_pos = torch.zeros((1,), dtype=torch.int32, device=TEST_DEVICE)
        traced = torch.jit.trace(wrapper, (hidden, position_ids, causal_mask, current_pos))
        mlmodel = ct.convert(
            traced,
            inputs=[
                ct.TensorType(name="hidden_states", shape=hidden.shape, dtype=np.float16),
                ct.TensorType(name="position_ids", shape=position_ids.shape, dtype=np.int32),
                ct.TensorType(name="causal_mask", shape=causal_mask.shape, dtype=np.float16),
                ct.TensorType(name="current_pos", shape=current_pos.shape, dtype=np.int32),
            ],
            outputs=[ct.TensorType(name="output_hidden_states", dtype=np.float16)],
            compute_precision=ct.precision.FLOAT16,
            compute_units=ct.ComputeUnit.CPU_AND_NE,
            minimum_deployment_target=ct.target.iOS18,
            convert_to="mlprogram",
        )
        if self.lut_bits:
            self.converted_model = mlmodel
            self.postprocess(num_workers=None if self.num_chunks > 1 else 8)
            mlmodel = self.converted_model
        return mlmodel

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def convert(self, part: str = "full") -> ct.models.MLModel | List[ct.models.MLModel]:
        if part in ("1", "embeddings"):
            return self.convert_embeddings(self.model)
        if part in ("3", "lm_head"):
            return self.convert_lm_head(self.model)
        if part in ("2", "ffn"):
            outputs = []
            for idx, (start, end) in enumerate(self._chunk_indices(self.model.config.num_hidden_layers)):
                outputs.append(self.convert_ffn_chunk(self.model, idx, start, end))
            return outputs
        if part in ("2_prefill", "prefill"):
            outputs = []
            for idx, (start, end) in enumerate(self._chunk_indices(self.model.config.num_hidden_layers)):
                outputs.append(self.convert_prefill_chunk(self.model, idx, start, end))
            return outputs
        if part in ("all", "123", "full"):
            embeds = self.convert_embeddings(self.model)
            transformer = self.convert("2")
            lm_head = self.convert_lm_head(self.model)
            return [embeds, *transformer, lm_head]
        raise ValueError(f"Unsupported part: {part}")


# ------------------------------------------------------------------
# CLI utilities
# ------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Phi model to CoreML components")
    parser.add_argument("--model", required=True, help="Path to the model directory")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--prefix", default="phi", help="Filename prefix")
    parser.add_argument("--context-length", type=int, default=CONTEXT_LENGTH)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lut", type=int, default=None)
    parser.add_argument("--chunk", type=int, default=1)
    parser.add_argument("--part", choices=["1", "2", "2_prefill", "3", "all", "full", "prefill", "embeddings"], default="all")
    return parser.parse_args()


def load_model(model_path: str, context_length: int, batch_size: int, prefix: str) -> PhiForCausalLM:
    config = PhiConfig.from_json(os.path.join(model_path, "config.json"))
    config.context_length = context_length
    config.state_length = max(config.state_length, context_length)
    model = PhiForCausalLM(config, enable_coreml=True)
    model.load_pretrained_weights(model_path)
    model.eval()
    return model


def save_output(models, output_dir: str, prefix: str, part: str, lut_bits: Optional[int], num_chunks: int) -> None:
        if not isinstance(models, list):
            models = [models]
        os.makedirs(output_dir, exist_ok=True)
        for idx, m in enumerate(models):
            name = prefix
            if part in ("1", "embeddings"):
                name += "_embeddings"
            elif part in ("3", "lm_head"):
                name += "_lm_head"
            elif part in ("2", "2_prefill", "ffn", "prefill"):
                suffix = "FFN" if part in ("2", "ffn") else "prefill"
                name += f"_{suffix}"
                if lut_bits is not None:
                    name += f"_lut{lut_bits}"
                name += f"_chunk_{idx+1:02d}of{num_chunks:02d}"
            if part not in ("2", "2_prefill", "ffn", "prefill") and lut_bits is not None:
                name += f"_lut{lut_bits}"
            name += ".mlpackage"
            path = os.path.join(output_dir, name)
            m.save(path)


def main() -> None:
    args = parse_args()
    model = load_model(args.model, args.context_length, args.batch_size, args.prefix)
    converter = PhiConverter(
        model=model,
        context_length=args.context_length,
        batch_size=args.batch_size,
        lut_bits=args.lut,
        num_chunks=args.chunk,
    )
    part_map = {"full": "all", "embeddings": "1", "prefill": "2_prefill"}
    part = part_map.get(args.part, args.part)
    result = converter.convert(part=part)
    save_output(result, args.output, args.prefix, part, args.lut, args.chunk)


if __name__ == "__main__":
    main()
