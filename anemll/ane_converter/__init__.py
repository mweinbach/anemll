"""ANE converter package."""

from .llama_converter import LlamaConverter
from .qwen_converter import QwenConverter
from .qwen2_5_converter import Qwen25Converter
from .gemma3_converter import Gemma3Converter
from .gemma3_ane_converter import Gemma3ANEConverter

# Optional converters: import defensively to avoid hard dependency issues
try:  # DeepSeek is LLaMA-based and may be incomplete in some worktrees
    from .deepseek_converter import DeepSeekConverter  # type: ignore
except Exception:  # pragma: no cover - optional/import-time variability
    DeepSeekConverter = None  # type: ignore

try:
    from .gemma3n_converter import Gemma3nConverter  # type: ignore
except Exception:  # pragma: no cover - optional/import-time variability
    Gemma3nConverter = None  # type: ignore

__all__ = [name for name, sym in (
    ("LlamaConverter", LlamaConverter),
    ("QwenConverter", QwenConverter),
    ("Qwen25Converter", Qwen25Converter),
    ("DeepSeekConverter", locals().get("DeepSeekConverter")),
    ("Gemma3Converter", Gemma3Converter),
    ("Gemma3ANEConverter", Gemma3ANEConverter),
    ("Gemma3nConverter", locals().get("Gemma3nConverter")),
) if sym is not None]
