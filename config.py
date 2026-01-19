"""
LLM Council Configuration

Add or remove council members by editing the COUNCIL_MODELS list.
Display names are optional - models without entries use their raw name.
"""

# All available Ollama models for the council
# VRAM requirements listed for reference
AVAILABLE_MODELS: list[str] = [
    "llama3.3:70b",     # ~40GB - Meta's flagship
    "qwen2.5:32b",      # ~20GB - Alibaba's multilingual model
    "gemma2:27b",       # ~17GB - Google's efficient model
    "deepseek-r1:32b",  # ~20GB - Reasoning specialist
    "mistral:7b",       # ~4GB  - Fast European model
]

# Default models enabled on startup (3 models)
DEFAULT_ENABLED_MODELS: list[str] = [
    "qwen2.5:32b",
    "gemma2:27b",
    "mistral:7b",
]

# Legacy alias for backward compatibility
COUNCIL_MODELS = AVAILABLE_MODELS

# Chairman configuration (synthesizes final answer)
CHAIRMAN_MODEL = "claude-opus-4-5-20251101"
CHAIRMAN_PROVIDER = "anthropic"

# Friendly display names for the UI
MODEL_DISPLAY_NAMES: dict[str, str] = {
    # Council members
    "llama3.3:70b": "Llama 3.3 70B — Meta's flagship",
    "qwen2.5:32b": "Qwen 2.5 32B — Alibaba's multilingual model",
    "gemma2:27b": "Gemma 2 27B — Google's efficient model",
    "mistral:7b": "Mistral 7B — Fast European model",
    # Additional models
    "deepseek-r1:32b": "DeepSeek R1 32B — Reasoning specialist",
    "llama3.2:3b": "Llama 3.2 3B — Lightweight fallback",
    # Chairman
    "claude-opus-4-5-20251101": "Claude Opus 4.5 — Chairman",
}


def get_display_name(model: str) -> str:
    """Get friendly display name for a model, or return raw name if not configured."""
    return MODEL_DISPLAY_NAMES.get(model, model)


def get_council_with_names() -> list[tuple[str, str]]:
    """Return list of (model_id, display_name) tuples for current council."""
    return [(m, get_display_name(m)) for m in COUNCIL_MODELS]
