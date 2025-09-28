Qwen 3 on ANE (0.6B, 1.7B, 4B)

This guide shows how to convert and compile Qwen 3 models for Apple Neural Engine using ANEMLL.

Quick start (Qwen3-1.7B):

- One‑shot test script
  - `./tests/conv/test_hf_model.sh Qwen/Qwen3-1.7B /tmp/qwen3-1.7b 2`
- Or use the staged converter directly
  - `./anemll/utils/convert_model.sh --model ~/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/<snap>`
    `--output /tmp/qwen3-1.7b --prefix qwen --context 1024 --batch 64 --chunk 2 --lut2 4 --lut3 6`

Notes

- Converter selection
  - Qwen3 uses the Qwen converter (`anemll.ane_converter.qwen_converter`)
  - Qwen2.5 uses the Qwen 2.5 converter (`anemll.ane_converter.qwen2_5_converter`)
  - The tooling detects Qwen3 even if `model_type` is `qwen2` by inspecting `architectures` (QwenForCausalLM)
- Tokenizer
  - iOS config uses `Qwen2Tokenizer` and `model_type: qwen3`
- Recommended settings
  - Context length: 512–1024 for best ANE performance
  - Chunking: `--chunk 2` for 1.7B; adjust if you hit compile limits
  - Quantization: `--lut2 4` (FFN/prefill), `--lut3 6` (LM head)
- Chat testing
  - `python tests/chat.py --meta /tmp/qwen3-1.7b/meta.yaml --prompt "Hello" --max-tokens 64`

