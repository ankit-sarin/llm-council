# Module 4: Fitting giants in a box — VRAM batching

### Teaching Arc
- **Metaphor:** **Packing a moving truck.** You have a truck with a fixed capacity (60GB of GPU memory). Your furniture comes in different sizes — a giant 40GB sofa (Llama 70B), some 20GB armchairs (Qwen, DeepSeek), a 17GB dresser (Gemma), and a 4GB lamp (Mistral). You load the biggest things first, then squeeze smaller items into the gaps. If everything fits in one trip, you go once. If not, you do multiple trips (batches), and you let the truck cool off between trips because rapid loading/unloading wears on the engine.
- **Opening hook:** "When you click the 🔬 Deep preset, you're asking three big models to run at the same time — together about 81GB. Your GPU has roughly 100GB total, but the council caps itself at 60GB. So it splits them into trips. Here's the algorithm that decides which model rides in which trip."
- **Key insight:** GPU memory is finite. The council uses a **greedy bin-packing** algorithm (sort by size descending, then pack each model into the first batch with room) to fit big models inside a memory budget. It adds a 20% safety buffer to every estimate and a cooldown between batches.
- **"Why should I care?":** This is THE pattern for running heavy AI workloads on limited hardware. Every time you build something that needs to run a big model and the AI assistant says "we'll need to batch this" or "this will OOM" (out-of-memory), they mean this. You'll be able to read the algorithm, sanity-check its choices, and tweak the budget.

### Code Snippets (pre-extracted)

**Snippet A — The decision: run all in parallel, or split into batches** (`council.py` lines 350-363):

```python
from config import MAX_CONCURRENT_VRAM_GB

# Calculate total VRAM needed for all models
total_vram = sum(get_model_vram(m) for m in models)

# Only use batching if total VRAM exceeds the limit
if total_vram <= MAX_CONCURRENT_VRAM_GB:
    # All models fit - run in parallel (single batch)
    batches = [models]
    execution_mode = "PARALLEL"
else:
    # Need to batch to avoid VRAM exhaustion
    batches = create_vram_batches(models)
    execution_mode = "BATCHED"
```

**Snippet B — The greedy bin-packing algorithm** (`config.py` lines 154-200):

```python
def create_vram_batches(models: list[str], max_vram: float = None) -> list[list[str]]:
    """
    Group models into batches that fit within VRAM limits.

    Strategy: Sort by VRAM descending, then greedily pack into batches.
    This ensures large models run alone if needed, while smaller models
    can run together.
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
```

**Snippet C — The 20% safety buffer** (`config.py` lines 50-52, 137-151):

```python
# Safety factor applied to VRAM estimates (1.20 = 20% buffer)
# Accounts for VRAM fragmentation, KV cache growth, and estimation errors
VRAM_SAFETY_FACTOR: float = 1.20

def get_model_vram(model: str, with_safety_factor: bool = True) -> float:
    base_vram = MODEL_VRAM_GB.get(model, 10.0)  # Default 10GB if unknown
    if with_safety_factor:
        return base_vram * VRAM_SAFETY_FACTOR
    return base_vram
```

**Snippet D — Cooldown between batches scales with how heavy the batch was** (`council.py` lines 110-129):

```python
COOLDOWN_BASE_SECONDS = 1.0  # Minimum cooldown between batches
COOLDOWN_PER_10GB_SECONDS = 0.5  # Additional seconds per 10GB of VRAM in completed batch

def calculate_batch_cooldown(batch: list[str]) -> float:
    """
    Calculate VRAM-aware cooldown time after a batch completes.

    Larger batches using more VRAM need longer cooldowns for GPU memory to clear.
    """
    batch_vram = sum(get_model_vram(m) for m in batch)
    cooldown = COOLDOWN_BASE_SECONDS + (batch_vram / 10.0) * COOLDOWN_PER_10GB_SECONDS
    return round(cooldown, 1)
```

**Snippet E — Per-model VRAM costs** (`config.py` lines 18-25):

```python
MODEL_VRAM_GB: dict[str, float] = {
    "llama3.3:70b": 40.0,
    "qwen2.5:32b": 20.0,
    "gemma2:27b": 17.0,
    "deepseek-r1:32b": 20.0,
    "mistral:7b": 4.0,
    "llama3.2:3b": 2.0,
}
```

### Screens (5)
1. **The moving truck** — set up the metaphor. Visual: items of different sizes vs. a truck with a capacity line.
2. **The decision in 8 lines** — Code↔English of Snippet A. English explains "add up the size, compare to the limit, decide one trip or many." Use **a worked example** below it: "Deep preset = 70B + 32B + 32B = 40+20+20 = 80GB. With 20% safety: 96GB. Limit is 60GB. So: BATCHED." (Clarify: the totals shown use the safety factor.)
3. **The packing algorithm, step by step** — Code↔English of Snippet B. Use a small **drag-and-drop interactive** OR a flow diagram showing the greedy packing in action with the Deep preset:
   - Sorted: [70B (48GB w/ safety), 32B (24GB), 32B (24GB)]
   - Batch 1: try 70B alone (48GB ≤ 60GB) ✓ → [70B]
   - Batch 2: try 32B (24GB ≤ 60GB) ✓ → [32B]; add 32B (24+24=48 ≤ 60) ✓ → [32B, 32B]
   - Result: 2 batches: [[70B], [32B, 32B]]
4. **The safety net: 20% buffer + cooldown** — Code↔English of Snippet C and D side by side OR Pattern Cards explaining each safeguard. Why 20%? Because the listed VRAM is the minimum; actual usage grows as the conversation gets longer (KV cache). Why cooldown? GPU memory takes a moment to actually free after a model unloads.
5. **Quiz**

### Interactive Elements
- [x] **Code↔English translation** — Snippet A (parallel-or-batch decision).
- [x] **Code↔English translation #2** — Snippet B (the greedy algorithm) OR D (the cooldown). Pick the one that fits the page best.
- [x] **Drag-and-drop matching** OR **interactive worked example** — show 4 model chips (`Llama 70B`, `Qwen 32B`, `Gemma 27B`, `Mistral 7B`) and 2 batch zones (Batch 1: ≤60GB, Batch 2: ≤60GB). User has to pack them. After they pack, the system checks. (If drag-and-drop is too heavy, do it as a static flow diagram with the algorithm tracing through.)
- [x] **Quiz** — 3 scenario questions:
  - Q1: "You bump `MAX_CONCURRENT_VRAM_GB` from 60 to 100. What's the most likely consequence?" (Correct: "More models fit in one parallel batch, so total wall-clock time drops — until the day one of your prompts pushes the KV cache past 100GB and you get an out-of-memory crash. Headroom matters." Wrong: "Models get faster individually." / "Nothing changes." / "The safety factor breaks.")
  - Q2: "You select 4 models with VRAM costs [40, 20, 20, 17] (without safety factor). The limit is 60GB and the safety factor is 1.20. Will the run be parallel or batched, and how many batches?" (Correct: "Batched. With safety: 48+24+24+20.4 = 116.4GB. Greedy pack: Batch 1 = [70B (48GB)] alone since adding any next item exceeds 60. Batch 2 = [32B (24), 32B (24)] = 48. Batch 3 = [27B (20.4)]. Three batches." Tests the algorithm.)
  - Q3: "Your friend's AI app keeps OOM-crashing on a fancy GPU. Based on this module, what's the first question to ask?" (Correct: "What's the *peak* memory it allocates, and does the code reserve any safety margin? Most OOM crashes are 'the estimate was right under ideal conditions but the real run grew past the limit.' A safety factor is the standard fix." Trains the diagnostic instinct.)
- [x] **Callout** — "Aha! moment": "**Greedy bin-packing** isn't unique to GPUs — it's the same algorithm used to load airplanes, plan delivery trucks, and schedule cloud jobs. Once you see it, you'll see it everywhere."
- [x] **Callout** — "Safety factors are everywhere": "Engineers add safety margins to *every* number that came from an estimate. Aircraft cables are rated 2× their max load. Bridges 3×. AI memory budgets 1.2×. If a number in your code is exact, that's usually a smell."
- [x] **Glossary tooltips** — GB, VRAM, GPU memory, KV cache, bin-packing, greedy algorithm, OOM (out of memory), batch, cooldown, safety factor, fragmentation, parallel, concurrent.

### Reference Files to Read
- `references/interactive-elements.md` → Code↔English Translation Blocks, Drag-and-Drop Matching, Pattern/Feature Cards, Multiple-Choice Quizzes, Callout Boxes, Numbered Step Cards (for algorithm trace), Glossary Tooltips
- `references/content-philosophy.md` → all of it
- `references/gotchas.md` → all of it
- `references/design-system.md` → tokens

### Connections
- **Previous module:** Module 3 mentioned "models run in parallel unless VRAM exceeds the limit, in which case they batch." This module explains exactly when, how, and with what safeguards.
- **Next module:** Module 5 — Anonymous peer review. Now that we understand how all the models *run*, we look at what they do in Stage 2: review each other's work.
- **Tone/style notes:**
  - Accent: teal `#2A7B9B`.
  - **The moving-truck metaphor lives in this module only.** Different from kitchen (M3) and panel/tumor-board (M1/M2).
  - This is the most "algorithmic" module — lean into the worked example. A non-technical reader should be able to *trace the algorithm by hand* by the end.
  - Don't shame the user about hardware. Frame it as "every team that ships AI hits this; here's the standard pattern."
