"""Gemma 3n model scaffolding for ANEMLL.

This module wraps Hugging Face's Gemma3n checkpoint layout and exposes a thin
interface compatible with the existing conversion pipeline.  The implementation
focuses on structured weight loading and metadata exposure so the converter can
reason about sliding-window attention, AltUp / Laurel blocks, and multi-modal
components before CoreML translation is implemented.

The actual Apple Neural Engine specific kernels still need to be implemented;
for now we retain the Hugging Face modules in ``self.reference_model`` for
parity checks while providing helpers to inspect tensor shapes and config
parameters inside ANEMLL.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from transformers import Gemma3nForConditionalGeneration, Gemma3nConfig

from .base_model import BaseModel


DEFAULT_DEVICE = "cpu"
DEFAULT_DTYPE = torch.bfloat16


@dataclass
class Gemma3nMetadata:
    """Lightweight metadata mirror used by the conversion pipeline."""

    max_position_embeddings: int
    sliding_window: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    hidden_size: int
    head_dim: int
    vocab_size: int
    altup_active_idx: int
    final_logit_softcapping: float
    num_kv_shared_layers: int


class Gemma3nModel(BaseModel):
    """Wrapper around Hugging Face's Gemma3n text tower.

    The class loads the pretrained weights via ``safetensors`` using the
    Transformers implementation but deliberately keeps a reduced surface area so
    ANEMLL's converters can evolve incrementally.  Until CoreML primitives are
    implemented, ``forward`` simply delegates to the reference model which makes
    functional validation straightforward.
    """

    def __init__(self, config: Gemma3nConfig):
        super().__init__(config)
        self.config = config
        self.device = DEFAULT_DEVICE
        self.reference_model: Optional[Gemma3nForConditionalGeneration] = None
        self.metadata: Optional[Gemma3nMetadata] = None

    @classmethod
    def from_pretrained(cls, model_path: str, torch_dtype: torch.dtype = DEFAULT_DTYPE):
        """Create a wrapper instance from an on-disk Hugging Face snapshot."""

        resolved_config = Gemma3nConfig.from_pretrained(model_path)
        instance = cls(resolved_config)
        instance.load_pretrained_weights(model_path, torch_dtype=torch_dtype)
        return instance

    def load_pretrained_weights(
        self,
        model_path: str,
        enable_conv2d: bool = True,
        enable_vocab_split: bool = False,
        enable_vocab_split8: bool = True,
        enable_logits2: bool = True,
        enable_coreml: bool = False,
        mlp_up_split: int = 1,
        mlp_down_split: int = 1,
        enable_debug: bool = False,
        torch_dtype: torch.dtype = DEFAULT_DTYPE,
    ):
        """Load weights via Transformers and cache metadata for the converter."""

        snapshot = Path(model_path)
        if not snapshot.exists():
            raise FileNotFoundError(f"Gemma3n snapshot not found: {snapshot}")

        if enable_debug:
            print(f"Loading Gemma3n weights from {snapshot}")

        self.reference_model = Gemma3nForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map="cpu",
        )
        self.reference_model.eval()
        self.metadata = Gemma3nMetadata(
            max_position_embeddings=self.config.text_config.max_position_embeddings,
            sliding_window=self.config.text_config.sliding_window,
            num_hidden_layers=self.config.text_config.num_hidden_layers,
            num_attention_heads=self.config.text_config.num_attention_heads,
            num_key_value_heads=self.config.text_config.num_key_value_heads,
            hidden_size=self.config.text_config.hidden_size,
            head_dim=self.config.text_config.head_dim,
            vocab_size=self.config.text_config.vocab_size,
            altup_active_idx=self.config.text_config.altup_active_idx,
            final_logit_softcapping=self.config.text_config.final_logit_softcapping,
            num_kv_shared_layers=self.config.text_config.num_kv_shared_layers,
        )

    def preprocess(self):
        pass

    def validate(self):
        if self.reference_model is None:
            raise RuntimeError("Gemma3nModel weights not loaded")
        return True

    def forward(self, *args, **kwargs):
        if self.reference_model is None:
            raise RuntimeError("Gemma3nModel weights not loaded")
        return self.reference_model(*args, **kwargs)


