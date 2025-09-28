"""CoreML/ANE converter for Gemma 3 (text-only).

This path performs a single-shot conversion of the Hugging Face
`Gemma3ForCausalLM` graph into a Core ML mlprogram that can run on the Apple
Neural Engine. Unlike the LLaMA/Qwen pipelines, we do not split the graph into
embeddings/FFN/LM-head parts or expose KV-cache states yet. The output is a
single `.mlpackage` that produces logits for a fixed input shape.

Intended for: google/gemma-3-1b-it
"""

from __future__ import annotations

import argparse
import os
import subprocess
from typing import Optional

import numpy as np
import torch
import coremltools as ct
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from .environment import require_coreml
from .base_converter import BaseConverter
from .metadata import AddMetadata, ModelPart


class Gemma3Converter(BaseConverter):
    def __init__(
        self,
        model: torch.nn.Module,
        *,
        context_length: int = 128,
    ) -> None:
        super().__init__(model)
        self.context_length = int(context_length)
        self.converted_model: Optional[ct.models.MLModel] = None

    def convert(self) -> ct.models.MLModel:
        """Trace a thin wrapper and convert to Core ML."""
        require_coreml()

        class Wrapper(torch.nn.Module):
            def __init__(self, model: torch.nn.Module):
                super().__init__()
                self.model = model

            def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
                out = self.model(input_ids=input_ids, attention_mask=attention_mask)
                return out.logits

        wrapper = Wrapper(self.model)
        wrapper.eval()

        # Static shapes for tracing; batch=1, seq_len=context_length
        seq = self.context_length
        sample_input_ids = torch.zeros((1, seq), dtype=torch.int32)
        sample_attention = torch.ones((1, seq), dtype=torch.int32)

        traced = torch.jit.trace(wrapper, (sample_input_ids, sample_attention))

        mlmodel = ct.convert(
            traced,
            inputs=[
                ct.TensorType(name="input_ids", shape=sample_input_ids.shape, dtype=np.int32),
                ct.TensorType(name="attention_mask", shape=sample_attention.shape, dtype=np.int32),
            ],
            outputs=[ct.TensorType(name="logits", dtype=np.float16)],
            compute_precision=ct.precision.FLOAT16,
            compute_units=ct.ComputeUnit.CPU_AND_NE,
            minimum_deployment_target=ct.target.iOS18,
            convert_to="mlprogram",
        )

        self.converted_model = mlmodel
        return mlmodel

    def convert_lm_head(self) -> ct.models.MLModel:
        """Convert only the final normalization + LM projection.

        Input:  hidden_states [1, 1, hidden]
        Output: logits [1, 1, vocab]
        """
        require_coreml()

        # Expect huggingface Gemma3ForCausalLM with .model.norm and .lm_head
        norm = getattr(self.model, "model").norm
        lm_head = getattr(self.model, "lm_head")

        class HeadWrapper(torch.nn.Module):
            def __init__(self, norm: torch.nn.Module, head: torch.nn.Module):
                super().__init__()
                self.norm = norm
                self.head = head

            def forward(self, hidden_states: torch.Tensor):
                x = self.norm(hidden_states)
                return self.head(x)

        wrapper = HeadWrapper(norm, lm_head)
        wrapper.eval()

        hidden = torch.zeros((1, 1, getattr(self.model.config, "hidden_size", 1152)), dtype=torch.float32)
        traced = torch.jit.trace(wrapper, (hidden,))

        mlmodel = ct.convert(
            traced,
            inputs=[ct.TensorType(name="hidden_states", shape=hidden.shape, dtype=np.float32)],
            outputs=[ct.TensorType(name="logits", dtype=np.float16)],
            compute_precision=ct.precision.FLOAT16,
            compute_units=ct.ComputeUnit.CPU_AND_NE,
            minimum_deployment_target=ct.target.iOS18,
            convert_to="mlprogram",
        )

        self.converted_model = mlmodel
        return mlmodel


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert Gemma 3 (text-only) to Core ML")
    p.add_argument("--model", type=str, required=True, help="HF model id or local path")
    p.add_argument("--output", type=str, required=True, help="Output directory")
    p.add_argument("--prefix", type=str, default="gemma3", help="Output filename prefix")
    p.add_argument("--context-length", type=int, default=128, help="Sequence length for tracing (full) or ignored for head-only")
    p.add_argument("--part", type=str, choices=["full", "3", "lm_head"], default="full", help="Convert full model or only LM head")
    return p.parse_args()


def main() -> None:  # pragma: no cover - CLI
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    print(f"Loading HF model: {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    cfg = AutoConfig.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32, device_map=None)
    model.eval()

    print("Converting to Core ML (ANE)…")
    conv = Gemma3Converter(model, context_length=args.context_length)
    if args.part in ("3", "lm_head"):
        mlmodel = conv.convert_lm_head()
    else:
        mlmodel = conv.convert()

    AddMetadata(
        mlmodel,
        {
            "context_length": args.context_length,
            "batch_size": 1,
            "lut_bits": None,
            "split_part": ModelPart.FULL.value,
            "function_names": ["logits"],
        },
    )

    suffix = "_lm_head" if args.part in ("3", "lm_head") else ""
    out_pkg = os.path.join(args.output, f"{args.prefix}{suffix}.mlpackage")
    print(f"Saving model to: {out_pkg}")
    mlmodel.save(out_pkg)

    # Try to compile for best runtime and to allow specifying compute units at load time
    try:
        print("Compiling with xcrun coremlcompiler…")
        subprocess.run([
            "xcrun",
            "coremlcompiler",
            "compile",
            out_pkg,
            args.output,
        ], check=True)
        print("Compilation successful.")
    except Exception as e:
        print(f"Warning: Compilation failed or unavailable: {e}")

    print("Done.")


if __name__ == "__main__":
    main()
