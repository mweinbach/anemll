# Gemma-3n E4B Conversion Notes

Date: 2025-09-23

> **Status:** Converter scaffolding checked in; CoreML lowering is not yet implemented.

## Repository Snapshot
- Model repo: `google/gemma-3n-E4B-it`
- Pulled files (local cache: `/tmp/gemma3n`):
  - `config.json`
  - `generation_config.json`
  - `tokenizer.json`
  - `tokenizer_config.json`
  - `model.safetensors.index.json` (lists 4 shard files)
- Architecture entry point: `Gemma3nForConditionalGeneration` (Transformers 4.53.0.dev0)

## Key Architectural Differences vs Existing Pipelines

| Feature | Gemma-3n | Current ANEMLL LLAMA/Qwen path |
|---------|----------|--------------------------------|
| Core blocks | AltUp predictor/corrector, Laurel residual augmentation, per-layer gating | Simple attention + MLP | 
| Attention | Mix of sliding-window and full layers (`layer_types` pattern) with 32K context, GQA (8 heads, 2 KV heads), KV-sharing across final 15 layers | Full attention only, no sliding window, unified KV cache |
| Positional encodings | Dual RoPE sets (global + local) with `rope_theta=1e6`, `sliding_window=512`, `rope_local_base_freq=1e4` | Single RoPE buffer derived from context length |
| Extra towers | Vision encoder (mobilenetv5_300m_enc) + audio conformer, additional special tokens | None |
| Output scaling | AltUp soft-capping (`final_logit_softcapping=30`) and correction scaling | Plain LM head |
| Per-layer inputs | Layer-specific embeddings (`hidden_size_per_layer_input=256`) gated each block | Not present |

Implication: simply routing Gemma weights into the LLaMA converter would skip all auxiliary paths and yield incorrect outputs.

## HF Weight Map Observations
- Text tower tensors live under `model.language_model.layers.*.*`
- AltUp assets: `altup.correction_coefs`, `altup.router_norm`, `altup.correct_output_scale`, etc.
- Laurel block weights: `laurel.linear_left/right`, `laurel.post_laurel_norm`
- Attention tensors split by layer type, require separate handling for sliding layers and global layers.
- Vision/audio towers stored under `model.vision_tower.*` and `model.audio_tower.*`; text-only conversion must either retain or drop them consistently with tokenizer special tokens (`boi_token_id`, `eoi_token_id`, etc.).

## Gaps in Current Conversion Stack
1. **Model implementation** – No PyTorch reference module that mirrors Gemma-3n Text/AltUp/Laurel behaviour inside `anemll/models/`.
2. **Converter logic** – Existing converters (`llama_converter`, `qwen_converter`, `qwen2_5_converter`) assume:
   - Uniform attention with simple KV cache layout
   - Single LM head and FFN blocks per layer
   - No per-layer auxiliary projections or multi-modal towers
3. **Runtime metadata** – `meta.yaml` schema has no fields for sliding-window parameters, dual RoPE buffers, AltUp scaling, or additional token ids.
4. **Inference runtime** – Swift/Python runners only manage standard KV caches. Sliding-window + KV-sharing requires different Core ML state layout.

## Proposed Implementation Plan
1. **Gemma3n Model Wrapper** (`anemll/models/gemma3n_model.py`)
   - Port HF `Gemma3nTextModel` logic: AltUp predictor/corrector, Laurel block, dual RoPE buffers, sliding attention, KV sharing.
   - Decide treatment of vision/audio towers (text-only vs full multimodal) and tokenizer alignment.
2. **ANE Converter** (`anemll/ane_converter/gemma3n_converter.py`)
   - Generate Core ML graphs for embeddings, attention (sliding + global), MLP, AltUp corrections, and per-layer gating.
   - Export correct states for shared-KV layers and sliding windows.
   - Support chunking / quantization strategy compatible with 32K context.
3. **CLI Integration**
   - Update `anemll/utils/convert_model.sh` to detect `model_type: gemma3n` and route to the new converter with sensible defaults (e.g., context 32K, batch tuned for sliding attention).
   - Extend dependency checks for any extra packages.
4. **Runtime Updates**
   - Extend `meta.yaml` to describe sliding window size, KV-sharing parameters, AltUp scaling coefficients, multi-modal token IDs.
   - Update Python/Swift inference runners to consume the new metadata and attach Core ML states appropriately.
5. **Testing**
   - Unit coverage in `tests/dev/` for layer primitives (AltUp, Laurel, sliding attention) to catch numerical drift.
   - End-to-end smoke conversion test (`tests/test_gemma3n_model.py`) that verifies chat generation parity vs HF (restricted to text path).
   - Optional integration tests for multi-modal inputs if those towers are supported.

## Outstanding Questions / Decisions
- Scope: text-only conversion (drop audio/vision) vs attempting full multi-modal export.
- Quantization strategy: confirm LUT viability on complex attention paths; evaluate FP16 fallback for critical layers.
- Performance constraints: sliding window requires different state slicing – need to validate ANE compatibility.
- Tokenizer special tokens: ensure `boa`, `boi`, `eoa`, `eoi`, `image/audio_token_id` align with runtime assumptions if towers are omitted.

## Next Steps
- Confirm desired scope (`text-only` vs `multimodal`).
- Allocate time for port (expect multi-day effort to mirror HF implementation and stand up conversion).
- Once scope/time approved, begin with model wrapper + converter prototypes before touching CLI/runtime.

