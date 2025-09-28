"""Fast smoke test for Gemma 3 1B Instruct via Transformers.

This test intentionally performs a tiny generation (<= 8 tokens) using
`google/gemma-3-1b-it` to verify that our wrapper can load and run on the host
environment. It does not touch CoreML/ANE.
"""

from __future__ import annotations

import os
import sys
import time

import pytest


def test_gemma3_load_and_generate():
    # Import lazily to avoid importing Transformers if not installed
    try:
        from anemll.models.gemma3_model import Gemma3Model
    except Exception as e:  # pragma: no cover - env specific
        pytest.skip(f"Skipping Gemma 3 test: import failed: {e}")

    model_id = os.environ.get("ANEMLL_GEMMA3_MODEL", "google/gemma-3-1b-it")

    start = time.time()
    try:
        model = Gemma3Model.from_pretrained(model_id)
    except Exception as e:
        # Common causes: license not accepted, offline, missing transformers version
        pytest.skip(f"Skipping Gemma 3 test: could not load {model_id}: {e}")

    load_secs = time.time() - start
    assert model.metadata is not None
    assert model.metadata.vocab_size > 0
    assert model.metadata.hidden_size > 0

    # Tiny generation to validate end-to-end path
    prompt = "You are a helpful assistant.\nUser: Say hi in one sentence.\nAssistant:"
    try:
        text = model.generate(prompt, max_new_tokens=8, temperature=0.0)
    except Exception as e:
        pytest.fail(f"Gemma 3 generation failed: {e}")

    assert isinstance(text, str) and len(text) > 0
    # Ensure the test remains fast-ish on CI/local machines
    assert load_secs < 600, f"Model load took too long: {load_secs:.1f}s"

    # Basic sanity: the output should include at least one alphabetic character
    assert any(c.isalpha() for c in text)


if __name__ == "__main__":  # pragma: no cover
    # Allow running as a standalone smoke test: `python tests/dev/test_gemma3_model.py`
    import pytest

    sys.exit(pytest.main([__file__]))

