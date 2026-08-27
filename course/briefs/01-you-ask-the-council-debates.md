# Module 1: You ask, the council debates

### Teaching Arc
- **Metaphor:** A **panel of expert judges on a competition show**. Each judge watches the performance independently, scores it, then a head judge synthesizes the panel's takes into the final verdict. Models = judges; Claude = head judge.
- **Opening hook:** "Imagine you type a hard medical question into a box, hit Submit, and instead of one AI answering, three AIs answer at the same time — then they review each other's work, then a fourth, smarter AI reads everything and writes the final answer. That's what happens here."
- **Key insight:** A single LLM is a single opinion. The Council pipeline trades a few minutes of compute for a multi-source answer with explicit agreement/disagreement signals — so the user knows when to trust it.
- **"Why should I care?":** Vibe coders ship apps that call a single AI and pray. Understanding the *council pattern* unlocks a more reliable way to use AI: parallel attempts + cross-checking + synthesis. This module shows you the shape of that pattern end-to-end before we zoom in.

### Code Snippets (pre-extracted)

**Snippet A — The whole journey at a glance** (`council.py`, conceptual — pulled verbatim from `CLAUDE.md` flow doc and matching `council.py` logger banners around lines 264-268, 472-476, 987-996):

```python
# Stage 1: parallel initial responses
logger.info("STAGE 1: INITIAL RESPONSES")
logger.info(f"Council members: {len(models)}")

# Stage 2: parallel peer reviews
logger.info("STAGE 2: PEER REVIEWS")
logger.info(f"Each of {len(responses)} models reviewing {len(responses)-1} responses")

# Stage 2.5: consensus calculation (variance math on parsed scores)

# Stage 3: chairman synthesis
logger.info("STAGE 3: CHAIRMAN SYNTHESIS")
logger.info(f"Chairman: {get_display_name(CHAIRMAN_MODEL)}")
```

**Snippet B — The actual prompt sent to each council member** (`council.py` lines 392-397):

```python
prompt = f"""You are a member of an expert council deliberating on the following question.
Provide your best, most thoughtful answer. Be thorough but concise.

QUESTION: {question}

YOUR RESPONSE:"""
```

**Snippet C — Council models declared in config** (`config.py` lines 9-15):

```python
AVAILABLE_MODELS: list[str] = [
    "llama3.3:70b",     # ~40GB - Meta's flagship
    "qwen2.5:32b",      # ~20GB - Alibaba's multilingual model
    "gemma2:27b",       # ~17GB - Google's efficient model
    "deepseek-r1:32b",  # ~20GB - Reasoning specialist
    "mistral:7b",       # ~4GB  - Fast European model
]
```

### Screens (5)
1. **The button you click** — opening hook with screenshot-style description of clicking Submit. Establishes user perspective.
2. **Three stages at a glance** — visual flow: Stage 1 (parallel responses) → Stage 2 (peer reviews) → Stage 3 (synthesis). Use Numbered Step Cards or Flow Diagrams.
3. **Why a council and not just one AI?** — the practical problem: one model can be wrong/biased/hallucinate. Panel = redundancy + cross-check. Two short reasons in a callout.
4. **Trace one question through** — Data Flow Animation: user → Gradio → Council Engine → Ollama (×3 parallel) → Reviews (×3 parallel) → Claude Chairman → final answer back to user.
5. **The prompt they see + quiz** — show Snippet B in a Code↔English translation, then quiz.

### Interactive Elements
- [x] **Code↔English translation** — Snippet B (the council prompt). English: "Tell them they're an expert on a panel… give them the question… ask them to be thorough but brief." Plus a separate translation for Snippet C explaining each model line.
- [x] **Quiz** — 3 scenario questions:
  - Q1: "You ask the council the same question twice and get two slightly different final answers. Why?" (Correct: Local LLMs are non-deterministic — they sample from probabilities. Different runs, different samples. Wrong options: "The code is broken" / "The chairman is rotating.")
  - Q2: "You want a *faster* answer and don't need maximum quality. Which stage would you most want to shorten?" (Correct: Stage 1 — initial responses are the longest because they generate from scratch in parallel; reviews and synthesis read what's already there. Tests whether they understood the stages.)
  - Q3: "Your friend asks 'is this just an ensemble like in machine learning?' What's the best answer?" (Correct: "Sort of — but instead of averaging predictions, each model writes a free-text answer, the others critique them by name, and a smarter chairman synthesizes. It's an *ensemble that talks*." Tests conceptual mapping.)
- [x] **Data flow animation** — actors: User → Gradio UI → Council Engine → 3 Local LLMs → Reviews → Claude Chairman → Final Answer. Steps:
  1. "User clicks Submit with question" — highlight User
  2. "Gradio sends question to Council Engine" — packet from User → Engine
  3. "Engine fans out to 3 local LLMs in parallel" — packet to 3 LLMs (use any of the 3 actor highlights; describe as fan-out)
  4. "Each LLM streams an answer back" — highlight 3 LLMs
  5. "Engine sends all 3 answers back to each LLM (anonymized) for review" — packet round trip
  6. "Each LLM scores the others on 3 criteria" — highlight 3 LLMs again
  7. "Engine bundles everything and asks Claude to synthesize" — packet to Claude
  8. "Claude writes the final answer" — highlight Claude
  9. "User reads the final answer with a consensus score" — packet back to User
- [x] **Group chat animation** — show the *idea* of the council talking: 4 short messages where the engine asks each model the same question, then a 5th where Claude says "I'll synthesize." (Sets up the conversation framing for later modules; the heavy chat anim is in Module 5.)
- [x] **Callout box** — "Aha! moment": "Engineers call this pattern **ensembling with cross-evaluation**. Most production AI systems eventually grow into something like this once a single-model answer isn't good enough."

### Reference Files to Read
- `references/interactive-elements.md` → Numbered Step Cards, Code↔English Translation Blocks, Multiple-Choice Quizzes, Message Flow / Data Flow Animation, Group Chat Animation, Callout Boxes, Glossary Tooltips
- `references/design-system.md` → tokens for spacing, accent (we use teal: `#2A7B9B`), actor color variables `--color-actor-1..4`
- `references/content-philosophy.md` → all of it
- `references/gotchas.md` → all of it

### Connections
- **Previous module:** None — this is the opener. Spend the first screen reassuring a non-technical reader.
- **Next module:** Meet the council — names, sizes, and roles of each LLM. We'll start putting faces to the abstract "Model 1/2/3" boxes.
- **Tone/style notes:**
  - Course-wide actor convention: **Llama 3.3 70B** (Meta's flagship), **Qwen 2.5 32B** (Alibaba's multilingual), **Gemma 2 27B** (Google's efficient), **DeepSeek R1 32B** (reasoning), **Mistral 7B** (fast European), **Claude Opus 4.5** (Chairman). Use full names on first mention, then short names.
  - Accent color is teal (`#2A7B9B`). Do not introduce other accents.
  - When referencing async/parallel, do NOT explain yet — just say "in parallel" and tooltip if needed. Module 3 owns the deep async explanation.
  - **Glossary tooltips needed on first use:** LLM, Ollama, VRAM (defer detail to module 4), parallel, streaming, synthesis, prompt, token, ensemble, Gradio, Anthropic, Claude.
