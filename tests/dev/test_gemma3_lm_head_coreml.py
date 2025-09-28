"""Smoke test: run Gemma 3 head on ANE and compare to HF logits.

This test loads `google/gemma-3-1b-it`, encodes a short prompt, computes the
last-token hidden state on CPU (HF), then projects it to logits via a CoreML
model converted by `anemll.ane_converter.gemma3_converter --part 3`.

The test asserts the top-5 tokens from CoreML and HF match at least 3/5.
"""

from __future__ import annotations

import os
import numpy as np
import torch
import pytest
import coremltools as ct
from transformers import AutoModelForCausalLM, AutoTokenizer


@pytest.mark.slow
def test_gemma3_head_coreml_topk_agrees(tmp_path):
    model_id = os.environ.get("ANEMLL_GEMMA3_MODEL", "google/gemma-3-1b-it")
    out_dir = os.environ.get("ANEMLL_GEMMA3_OUT", "outputs/gemma3")
    mlc_path = os.path.join(out_dir, "gemma3_1b_lm_head.mlmodelc")
    if not os.path.exists(mlc_path):
        pytest.skip("Compiled CoreML head not found; run gemma3_converter --part 3 first")

    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    model.eval()

    prompt = "User: Say hi in one short sentence.\nAssistant:"
    enc = tok(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True, return_dict=True)
    hidden_last = out.hidden_states[-1][:, -1:, :].to(torch.float32).cpu().numpy()

    # CoreML head inference
    mlm = ct.models.CompiledMLModel(mlc_path, ct.ComputeUnit.CPU_AND_NE)
    logits_ne = mlm.predict({"hidden_states": hidden_last})["logits"][0, 0]

    # HF head for reference
    with torch.no_grad():
        logits_hf = model.lm_head(model.model.norm(out.hidden_states[-1][:, -1:, :]))[0, 0].cpu().numpy()

    # Compare top-5 tokens overlap
    top5_ne = np.argsort(logits_ne)[-5:][::-1]
    top5_hf = np.argsort(logits_hf)[-5:][::-1]
    overlap = len(set(top5_ne.tolist()).intersection(set(top5_hf.tolist())))
    assert overlap >= 3, f"Top-5 overlap too small: {overlap}/5"

