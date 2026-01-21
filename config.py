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

# Context window limits in tokens for each model
# Used to determine if a model can handle uploaded documents
MODEL_CONTEXT_TOKENS: dict[str, int] = {
    "llama3.3:70b": 128000,   # 128K context
    "qwen2.5:32b": 32000,     # 32K context
    "gemma2:27b": 8000,       # 8K context (limited)
    "deepseek-r1:32b": 64000, # 64K context
    "mistral:7b": 32000,      # 32K context
    "llama3.2:3b": 128000,    # 128K context
}

# Approximate characters per token (conservative estimate)
# Most models average 3-4 chars/token; we use 3 to be safe
CHARS_PER_TOKEN: float = 3.0

# Safety margin for context usage (reserve for prompt + response)
# 0.7 means we only use 70% of context for the document
CONTEXT_SAFETY_MARGIN: float = 0.7

# Maximum VRAM to use concurrently (leave some headroom)
# Adjust this based on your GPU - set higher if you have more VRAM
MAX_CONCURRENT_VRAM_GB: float = 60.0

# Safety factor applied to VRAM estimates (1.20 = 20% buffer)
# Accounts for VRAM fragmentation, KV cache growth, and estimation errors
VRAM_SAFETY_FACTOR: float = 1.20

# Default models enabled on startup (3 models)
DEFAULT_ENABLED_MODELS: list[str] = [
    "qwen2.5:32b",
    "gemma2:27b",
    "mistral:7b",
]

# Model selection presets for quick configuration
MODEL_PRESETS: dict[str, dict] = {
    "fast": {
        "name": "⚡ Fast",
        "description": "Quick answers with smaller models",
        "models": ["gemma2:27b", "mistral:7b"],
    },
    "balanced": {
        "name": "⚖️ Balanced",
        "description": "Good tradeoff of speed and depth",
        "models": ["qwen2.5:32b", "gemma2:27b", "mistral:7b"],
    },
    "deep": {
        "name": "🔬 Deep",
        "description": "Thorough analysis with heavy hitters",
        "models": ["llama3.3:70b", "qwen2.5:32b", "deepseek-r1:32b"],
    },
}

# Example prompts for first-run usability
# Domain-specific examples to demonstrate the council's capabilities
EXAMPLE_PROMPTS: list[dict[str, str]] = [
    {
        "label": "🧬 ctDNA Surveillance",
        "prompt": "What is the current evidence for using ctDNA to guide watch-and-wait strategies in colorectal cancer patients who achieve clinical complete response after neoadjuvant therapy? What detection thresholds and monitoring intervals are recommended?",
    },
    {
        "label": "🤖 Robotic Platforms",
        "prompt": "Compare the da Vinci Xi, Hugo RAS, and Medtronic Hugo surgical robotic platforms for urologic procedures. What are the key differences in workspace, instrumentation, costs, and clinical outcomes data?",
    },
    {
        "label": "📄 Paper Critique",
        "prompt": "What are the key methodological considerations when critically appraising a randomized controlled trial? Provide a structured framework for evaluating internal validity, external validity, and risk of bias.",
    },
    {
        "label": "💊 Drug Interaction",
        "prompt": "A patient on warfarin for atrial fibrillation needs to start amiodarone for rhythm control. What drug interactions should be considered, what monitoring is required, and how should the warfarin dose be adjusted?",
    },
    {
        "label": "🔬 Research Design",
        "prompt": "What are the advantages and limitations of using propensity score matching versus inverse probability weighting in observational studies? When would you choose one approach over the other?",
    },
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


def get_model_vram(model: str, with_safety_factor: bool = True) -> float:
    """
    Get VRAM requirement for a model in GB.

    Args:
        model: Model ID
        with_safety_factor: If True, applies VRAM_SAFETY_FACTOR for conservative estimates

    Returns:
        VRAM in GB (with safety factor applied by default)
    """
    base_vram = MODEL_VRAM_GB.get(model, 10.0)  # Default 10GB if unknown
    if with_safety_factor:
        return base_vram * VRAM_SAFETY_FACTOR
    return base_vram


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


def get_model_context_limit(model: str) -> int:
    """
    Get context window limit in tokens for a model.

    Args:
        model: Model ID

    Returns:
        Context limit in tokens (default 32000 if unknown)
    """
    return MODEL_CONTEXT_TOKENS.get(model, 32000)


def get_model_max_chars(model: str) -> int:
    """
    Get maximum document characters a model can handle.

    Applies safety margin to reserve space for prompt and response.

    Args:
        model: Model ID

    Returns:
        Maximum characters for document content
    """
    context_tokens = get_model_context_limit(model)
    usable_tokens = int(context_tokens * CONTEXT_SAFETY_MARGIN)
    return int(usable_tokens * CHARS_PER_TOKEN)


def check_model_context_fit(model: str, char_count: int) -> dict:
    """
    Check if a model can handle a document of given length.

    Args:
        model: Model ID
        char_count: Number of characters in the document

    Returns:
        Dict with:
            - fits: bool - True if document fits in context
            - max_chars: int - Maximum chars the model can handle
            - utilization: float - Percentage of context used (0-100+)
            - warning: str | None - Warning message if approaching limit
    """
    max_chars = get_model_max_chars(model)
    utilization = (char_count / max_chars * 100) if max_chars > 0 else 100

    fits = char_count <= max_chars
    warning = None

    if not fits:
        warning = f"Document ({char_count:,} chars) exceeds {get_display_name(model)} context limit ({max_chars:,} chars)"
    elif utilization > 80:
        warning = f"Document uses {utilization:.0f}% of {get_display_name(model)} context - may truncate"

    return {
        "fits": fits,
        "max_chars": max_chars,
        "utilization": utilization,
        "warning": warning,
    }


def filter_models_by_context(models: list[str], char_count: int) -> tuple[list[str], list[dict]]:
    """
    Filter models that can handle a document of given length.

    Args:
        models: List of model IDs to check
        char_count: Number of characters in the document

    Returns:
        Tuple of:
            - List of models that can handle the document
            - List of exclusion details for models that can't
    """
    compatible = []
    excluded = []

    for model in models:
        check = check_model_context_fit(model, char_count)
        if check["fits"]:
            compatible.append(model)
        else:
            excluded.append({
                "model": model,
                "display_name": get_display_name(model),
                "max_chars": check["max_chars"],
                "reason": check["warning"],
            })

    return compatible, excluded
