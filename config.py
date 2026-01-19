"""
LLM Council Configuration

Add or remove council members by editing the COUNCIL_MODELS list.
Display names are optional - models without entries use their raw name.
"""

# All available Ollama models for the council
AVAILABLE_MODELS: list[str] = [
    "llama3.3:70b",     # ~40GB - Meta's flagship
    "qwen2.5:32b",      # ~20GB - Alibaba's multilingual model
    "gemma2:27b",       # ~17GB - Google's efficient model
    "deepseek-r1:32b",  # ~20GB - Reasoning specialist
    "mistral:7b",       # ~4GB  - Fast European model
]

# VRAM requirements in GB for each model (approximate)
MODEL_VRAM_GB: dict[str, float] = {
    "llama3.3:70b": 40.0,
    "qwen2.5:32b": 20.0,
    "gemma2:27b": 17.0,
    "deepseek-r1:32b": 20.0,
    "mistral:7b": 4.0,
    "llama3.2:3b": 2.0,
}

# Maximum VRAM to use concurrently (leave some headroom)
# Adjust this based on your GPU - set higher if you have more VRAM
MAX_CONCURRENT_VRAM_GB: float = 60.0

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


def get_model_vram(model: str) -> float:
    """Get VRAM requirement for a model in GB."""
    return MODEL_VRAM_GB.get(model, 10.0)  # Default 10GB if unknown


def create_vram_batches(models: list[str], max_vram: float = None) -> list[list[str]]:
    """
    Group models into batches that fit within VRAM limits.

    Strategy: Sort by VRAM descending, then greedily pack into batches.
    This ensures large models run alone if needed, while smaller models
    can run together.

    Args:
        models: List of model IDs to batch
        max_vram: Maximum VRAM per batch (defaults to MAX_CONCURRENT_VRAM_GB)

    Returns:
        List of batches, where each batch is a list of model IDs
    """
    if max_vram is None:
        max_vram = MAX_CONCURRENT_VRAM_GB

    if not models:
        return []

    # Sort by VRAM requirement descending (largest first)
    sorted_models = sorted(models, key=lambda m: get_model_vram(m), reverse=True)

    batches = []
    current_batch = []
    current_vram = 0.0

    for model in sorted_models:
        model_vram = get_model_vram(model)

        if current_vram + model_vram <= max_vram:
            # Fits in current batch
            current_batch.append(model)
            current_vram += model_vram
        else:
            # Start new batch
            if current_batch:
                batches.append(current_batch)
            current_batch = [model]
            current_vram = model_vram

    # Don't forget the last batch
    if current_batch:
        batches.append(current_batch)

    return batches
