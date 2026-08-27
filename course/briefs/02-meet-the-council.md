# Module 2: Meet the council

### Teaching Arc
- **Metaphor:** **A medical tumor board.** Different specialists with different training (a surgeon, a radiologist, a pathologist) each look at the same case and add their angle. None of them is "best" — they're each *differently shaped*. Some are fast and broad; some are deep and slow.
- **Opening hook:** "When you check models in the UI — Llama, Qwen, Gemma, Mistral — those aren't just names. Each one is a different brain with a different size, training, and personality."
- **Key insight:** The council isn't one piece of software. It's six characters with different jobs: five **local LLMs** that live on your GPU and one **remote Chairman** (Claude) that lives in the cloud. Plus a few supporting cast members (the Gradio UI, the engine, the session storage).
- **"Why should I care?":** When you're directing AI coding tools, you constantly make this call: *which model should do this work?* This module gives you a concrete mental model — local vs. remote, big vs. small, generalist vs. specialist — that maps to every real-world AI app you'll build.

### Code Snippets (pre-extracted)

**Snippet A — Display names and roles** (`config.py` lines 113-124):

```python
MODEL_DISPLAY_NAMES: dict[str, str] = {
    "llama3.3:70b": "Llama 3.3 70B — Meta's flagship",
    "qwen2.5:32b": "Qwen 2.5 32B — Alibaba's multilingual model",
    "gemma2:27b": "Gemma 2 27B — Google's efficient model",
    "mistral:7b": "Mistral 7B — Fast European model",
    "deepseek-r1:32b": "DeepSeek R1 32B — Reasoning specialist",
    "llama3.2:3b": "Llama 3.2 3B — Lightweight fallback",
    "claude-opus-4-5-20251101": "Claude Opus 4.5 — Chairman",
}
```

**Snippet B — Quick-pick presets that group models by use case** (`config.py` lines 62-78):

```python
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
```

**Snippet C — Default council on startup** (`config.py` lines 55-59):

```python
DEFAULT_ENABLED_MODELS: list[str] = [
    "qwen2.5:32b",
    "gemma2:27b",
    "mistral:7b",
]
```

**Snippet D — Chairman config (lives in the cloud)** (`config.py` lines 108-110):

```python
CHAIRMAN_MODEL = "claude-opus-4-5-20251101"
CHAIRMAN_PROVIDER = "anthropic"
```

### Screens (5)
1. **The cast list** — Icon-Label Rows for each character: 5 local LLMs + Claude (Chairman). Each row says: name, training origin (Meta/Alibaba/Google/Mistral/Anthropic), size, where it runs (your GPU vs. cloud API). Use the actor color variables `--color-actor-1..5`.
2. **Local vs. remote — the most important distinction** — two-column compare: Local (Ollama, runs on your hardware, free per call, private, slower for big models) vs. Remote (Anthropic API, runs in their datacenter, costs $$ per call, smarter on average). Tooltip API. Tooltip Ollama.
3. **Sizes you can feel** — pattern cards showing VRAM cost. "Llama 3.3 70B uses ~40GB of GPU memory — that's basically the whole car. Mistral 7B uses ~4GB — it fits in a glove compartment." Tease module 4 (VRAM batching) here.
4. **The presets explained** — Code↔English translation of Snippet B. English explains *why* each preset exists: Fast for cheap throwaway questions, Balanced for everyday work, Deep when correctness matters more than speed. Connect to vibe-coder decision-making.
5. **Quiz**

### Interactive Elements
- [x] **Code↔English translation** — Snippet B (presets). Each line of English explains *the decision* the preset embodies, not the syntax.
- [x] **Code↔English translation #2** — Snippet A (display names dict). English: each line teaches the model + a one-sentence "personality."
- [x] **Drag-and-drop matching** — chips: `Llama 3.3 70B`, `Mistral 7B`, `DeepSeek R1 32B`, `Claude Opus 4.5`. Zones:
  - "I need step-by-step reasoning on a hard logic puzzle" → DeepSeek R1
  - "I want a draft answer in under 10 seconds" → Mistral 7B
  - "I need the biggest, most capable open model" → Llama 3.3 70B
  - "I need a final, polished synthesis with web search" → Claude Opus 4.5
- [x] **Pattern cards** — one card per model, with size, VRAM, and "best at" line. Use actor color variables.
- [x] **Quiz** — 3 scenario questions:
  - Q1: "Your friend asks 'why use Claude at the end if you already have 5 local models?' What's the best answer?" (Correct: "Claude is roughly the strongest single model available — better at long reasoning and combining multiple inputs than any of the local ones. We use the locals for *diversity*, then escalate to Claude for the final word." Wrong: "Local models are bad." / "Claude is required by law.")
  - Q2: "You build a similar app and want to keep all data on your own machine. Which member of the council can you NOT include?" (Correct: Claude Opus 4.5 — it's a cloud API. Tests the local/remote distinction.)
  - Q3: "You add a 7th model that's a 40B Chinese-language specialist. Where in the codebase do you register it?" (Correct: `config.py` — `AVAILABLE_MODELS`, `MODEL_VRAM_GB`, `MODEL_CONTEXT_TOKENS`, `MODEL_DISPLAY_NAMES`. Tests where the *roster* lives so they can steer AI to the right file.)
- [x] **Callout** — "Aha! moment": "Knowing which model to pick is one of the most underrated skills in modern AI engineering. Almost every production app is really a *router* deciding which model gets which request."
- [x] **Glossary tooltips** — LLM (if not already done in M1), GPU, VRAM, API, model weights, inference, open source, parameters (the "70B" number), Anthropic, Ollama, training.

### Reference Files to Read
- `references/interactive-elements.md` → Icon-Label Rows, Pattern/Feature Cards, Code↔English Translation Blocks, Drag-and-Drop Matching, Multiple-Choice Quizzes, Callout Boxes, Glossary Tooltips
- `references/content-philosophy.md` → all of it (especially Metaphors First and Show, Don't Tell — list-of-models is the WORST place for paragraphs)
- `references/gotchas.md` → all of it
- `references/design-system.md` → actor color variables `--color-actor-1..5`

### Connections
- **Previous module:** Module 1 introduced the 3-stage pipeline and the panel-of-judges idea. This module gives names to the judges.
- **Next module:** Module 3 — Speaking in parallel. We'll explain *how* multiple models actually run at the same time without stepping on each other. This module just establishes the cast; the next module shows them in motion.
- **Tone/style notes:**
  - Continue calling models by full display name on first mention per module.
  - Accent color is teal `#2A7B9B`. Do not introduce other accents.
  - **Hold off on async/asyncio explanation** — that's module 3. Here just say "they all run at the same time" and tooltip.
  - **Hold off on VRAM math** — that's module 4. Here just give an intuitive sense of size (small, medium, huge).
