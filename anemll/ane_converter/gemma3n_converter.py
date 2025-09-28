"""Gemma 3n converter scaffolding.

The implementation currently focuses on configuration inspection and sanity
checks while we prototype the ANE-specific graph lowering for sliding-window
attention, AltUp blocks, and multi-modal towers.  Converters are expected to
raise ``NotImplementedError`` until the CoreML build steps are finalized.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Optional

import coremltools as ct

from .base_converter import BaseConverter
from ..models.gemma3n_model import Gemma3nModel


class Gemma3nConverter(BaseConverter):
    """Conversion driver for Gemma-3n checkpoints."""

    def __init__(self, model: Gemma3nModel, *, context_length: Optional[int] = None, batch_size: int = 1):
        super().__init__(model)
        self.context_length = context_length or model.config.text_config.max_position_embeddings
        self.batch_size = batch_size
        self.converted_model: Optional[ct.models.MLModel] = None

    def preprocess(self):
        if not isinstance(self.model, Gemma3nModel):
            raise TypeError("Gemma3nConverter expects a Gemma3nModel instance")
        self.model.validate()

    def convert(self, split_part: Optional[str] = None):
        self.preprocess()
        raise NotImplementedError(
            "Gemma3n conversion to CoreML is still under development. "
            "The converter scaffolding is present, but the graph lowering "
            "has not been implemented yet."
        )

    def summarize(self) -> dict[str, object]:
        """Return a summary useful for logging or unit tests."""
        metadata = self.model.metadata
        return {
            "context_length": self.context_length,
            "batch_size": self.batch_size,
            "metadata": asdict(metadata) if metadata else None,
        }
