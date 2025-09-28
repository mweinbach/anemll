"""Gemma 3 text-only wrapper for ANEMLL.

This lightweight wrapper uses Hugging Face Transformers to load and run
`google/gemma-3-1b-it` for basic text generation. It mirrors the interface
pattern of other model wrappers and avoids CoreML/ANE specifics. Conversion to
CoreML is tracked separately.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List

import torch
from transformers import (
    AutoTokenizer,
    AutoConfig,
    Gemma3ForCausalLM,
)

from .base_model import BaseModel


DEFAULT_DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


@dataclass
class Gemma3Metadata:
    context_length: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    hidden_size: int
    vocab_size: int


class Gemma3Model(BaseModel):
    """Wrapper around Hugging Face's text-only Gemma 3 causal LM."""

    def __init__(self, config: AutoConfig, *, device: Optional[str] = None):
        super().__init__(config)
        self.config = config
        self.device = device or DEFAULT_DEVICE
        self.model: Optional[Gemma3ForCausalLM] = None
        self.tokenizer = None
        self.metadata: Optional[Gemma3Metadata] = None

    @classmethod
    def from_pretrained(
        cls,
        model_id: str = "google/gemma-3-1b-it",
        *,
        device: Optional[str] = None,
    ) -> "Gemma3Model":
        config = AutoConfig.from_pretrained(model_id)
        inst = cls(config, device=device)
        inst.load_pretrained_weights(model_id)
        return inst

    def load_pretrained_weights(self, model_id: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        # Always load on CPU first to avoid dtype/device issues; move later
        self.model = Gemma3ForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            device_map=None,
        )
        self.model.eval()
        if self.device != "cpu":
            self.model.to(self.device)
        # Populate minimal metadata for tests/telemetry
        text_cfg = getattr(self.config, "text_config", self.config)
        self.metadata = Gemma3Metadata(
            context_length=int(getattr(text_cfg, "max_position_embeddings", 8192)),
            num_hidden_layers=int(getattr(text_cfg, "num_hidden_layers", 16)),
            num_attention_heads=int(getattr(text_cfg, "num_attention_heads", 16)),
            num_key_value_heads=int(getattr(text_cfg, "num_key_value_heads", getattr(text_cfg, "num_attention_heads", 16))),
            hidden_size=int(getattr(text_cfg, "hidden_size", 2048)),
            vocab_size=int(getattr(text_cfg, "vocab_size", 128256)),
        )

    def preprocess(self):
        pass

    def validate(self):
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Gemma3Model is not loaded")
        return True

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 32,
        temperature: float = 0.7,
        top_p: float = 0.95,
        stop_tokens: Optional[List[str]] = None,
    ) -> str:
        self.validate()
        assert self.tokenizer is not None and self.model is not None

        # Use chat template if available
        if hasattr(self.tokenizer, "apply_chat_template"):
            messages = [
                {"role": "user", "content": prompt},
            ]
            model_inputs = self.tokenizer.apply_chat_template(
                messages, return_tensors="pt", add_generation_prompt=True
            )
        else:
            model_inputs = self.tokenizer(prompt, return_tensors="pt").input_ids

        input_ids = model_inputs.to(self.device)

        output_ids = self.model.generate(
            input_ids,
            do_sample=True if temperature > 0 else False,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )[0]

        text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        if stop_tokens:
            for tok in stop_tokens:
                if tok in text:
                    text = text.split(tok, 1)[0]
                    break
        return text

