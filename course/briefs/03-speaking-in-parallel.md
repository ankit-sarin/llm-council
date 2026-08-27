# Module 3: Speaking in parallel — async streaming

### Teaching Arc
- **Metaphor:** **A short-order kitchen with three line cooks.** If one cook makes pancakes, eggs, and bacon one at a time, breakfast is slow. If three cooks each take one item *simultaneously*, breakfast is fast — but only if they don't fight for the same pan. **`asyncio`** is the head cook coordinating who's doing what without stepping on each other. **Streaming** is each cook handing you bites of food as they're ready, instead of plating everything at once.
- **Opening hook:** "When you watch the Deliberation tab, you see three columns of text filling up at the same time — one token at a time, like a typewriter. That's not magic; it's two specific Python tricks called *async* and *streaming*."
- **Key insight:** Python normally does one thing at a time. **`asyncio`** lets it juggle many slow operations (like waiting on an LLM) without blocking. **Streaming** turns a slow "wait 30 seconds, get all the text" call into "see the text appear as it's generated." Together they're why the app *feels* fast even though each model takes 10–60 seconds.
- **"Why should I care?":** Every time you tell an AI coding tool "make this faster" and the AI says "we'll add async" or "let's stream the response" — this is what they mean. Knowing the shape lets you ask for it correctly and spot when an implementation is wrong (e.g., "I asked for streaming but the response still comes all at once — what's wrong?").

### Code Snippets (pre-extracted)

**Snippet A — Parallel execution: launch all models at once, gather when done** (`council.py` lines 423-426):

```python
# Run all models in this batch concurrently
tasks = [asyncio.create_task(run_model(m)) for m in batch]
await asyncio.gather(*tasks)
```

**Snippet B — Streaming tokens from one model, with a thread + queue bridge** (`council.py` lines 158-176, the core of `_call_ollama_streaming`):

```python
def run_stream():
    """Stream tokens from Ollama. Always signals 'done' via finally."""
    error_msg = None
    try:
        for chunk in ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            stream=True
        ):
            token = chunk.get("message", {}).get("content", "")
            if token:  # Only enqueue non-empty tokens
                token_queue.put(("token", token))
    except Exception as e:
        error_msg = str(e)
        token_queue.put(("error", error_msg))
    finally:
        # Always signal completion so the consumer loop can exit
        token_queue.put(("done", None))
```

**Snippet C — Two safety timeouts that keep the UI from hanging forever** (`council.py` lines 107-108, 188-197):

```python
STREAM_TOTAL_TIMEOUT_SECONDS = 600  # 10 minutes max for any single model response
STREAM_STALL_TIMEOUT_SECONDS = 120  # 2 minutes without receiving a token = stalled

# ... inside the consumer loop:
if elapsed > STREAM_TOTAL_TIMEOUT_SECONDS:
    logger.warning(f"[{get_display_name(model)}] Total timeout ({STREAM_TOTAL_TIMEOUT_SECONDS}s) exceeded")
    timed_out = True
    break

if time_since_last_token > STREAM_STALL_TIMEOUT_SECONDS:
    logger.warning(f"[{get_display_name(model)}] Stall timeout ({STREAM_STALL_TIMEOUT_SECONDS}s) - no tokens received")
    timed_out = True
    break
```

**Snippet D — The Stop button uses an asyncio.Event passed all the way down** (`council.py` lines 199-203):

```python
# Check if stop was requested
if stop_event is not None and stop_event.is_set():
    logger.info(f"[{get_display_name(model)}] Stop requested, halting stream")
    was_stopped = True
    break
```

### Screens (5)
1. **The line-cook kitchen** — open with the metaphor. Visual: 3 cooks vs. 1 cook. Without saying "Python," establish the *idea* that parallel work needs coordination.
2. **What async actually means** — short text + Code↔English translation of Snippet A. English: "Spin up a task per model… now wait for all of them to finish at the same time." Tooltip `asyncio`, `await`, `gather`, `task`, `coroutine`.
3. **Streaming, token by token** — Code↔English translation of Snippet B. English: "Start asking Ollama for an answer… for each tiny chunk that comes back, put it on a queue… when the model is done OR something breaks, signal 'done' no matter what." Explain WHY there's a thread: `ollama.chat()` is synchronous, but we want async — the thread + queue is the bridge.
4. **What can go wrong, and how the code defends against it** — Code↔English of Snippet C (timeouts) and Snippet D (stop event). Two failure modes: total runtime too long, or stalled (no tokens for 2 minutes). Plus user-initiated stop. Use **pattern cards** for each safeguard.
5. **Spot the bug + quiz**

### Interactive Elements
- [x] **Code↔English translation** — Snippet A (the parallel launch). Each line in plain English, emphasizing "what does `gather` actually do?"
- [x] **Code↔English translation #2** — Snippet B (the streaming bridge). 8 lines of English mapping to the inner thread function.
- [x] **Spot-the-bug** — give a version of Snippet B WITHOUT the `finally:` block (just the try/except), and ask "what's wrong?" Correct answer: "If `ollama.chat()` raises an exception type that isn't caught, OR if the loop returns normally but we forgot to signal 'done', the consumer waiting on the queue would hang forever. The `finally` block guarantees the 'done' signal *always* fires." The buggy line is the missing `finally`. Use the bug-challenge pattern.
- [x] **Quiz** — 3 scenario questions:
  - Q1: "You add a 4th model to the council. You expect total time to be roughly the same as 3 models. You're shocked when it doubles. What's the likely cause?" (Correct: "The 4th model pushed total VRAM over the GPU's limit, so the engine switched from parallel to batched execution — now models run in waves instead of all at once. Module 4 covers this." Wrong: "Python is slow." / "asyncio doesn't scale.")
  - Q2: "You're watching the deliberation tab and one column has been frozen at 70% for 3 minutes. Without you doing anything, the text shows '[Response timed out]'. Why?" (Correct: "The stall timeout (`STREAM_STALL_TIMEOUT_SECONDS = 120`) fired — no tokens arrived for 2 minutes, so the engine assumed the model is stuck and stopped waiting. The model's output up to that point is kept." Tests understanding of the safeguard.)
  - Q3: "You're directing AI to add a 'cancel' button to a different streaming app you're building. Based on what you saw here, what's the cleanest pattern to ask for?" (Correct: "Pass an `asyncio.Event` (or a similar cancellation signal) from the UI all the way down into the streaming loop, and check `event.is_set()` between chunks. This is exactly what the council does." Wrong: "Kill the process." / "Add a try/except.")
- [x] **Callout** — "Aha! moment": "The `finally:` block is one of the most underrated tools in defensive code. Whenever you're bridging two systems — like a synchronous library to an async caller — *always* put the 'signal done' in `finally`. Otherwise one path forgets, and your code hangs forever."
- [x] **Callout** — "Why a thread *and* asyncio?": "asyncio is great when libraries are designed for it. `ollama.chat()` isn't — it blocks. The thread is the airlock: the thread does the blocking work, the queue ferries tokens out, the async loop stays responsive."
- [x] **Glossary tooltips** — async, asyncio, await, coroutine, task, thread, queue, blocking, non-blocking, synchronous, streaming, token, chunk, generator, race condition, deadlock, timeout, finally block, exception, daemon thread, event loop.

### Reference Files to Read
- `references/interactive-elements.md` → Code↔English Translation Blocks, "Spot the Bug" Challenge, Multiple-Choice Quizzes, Pattern/Feature Cards, Callout Boxes, Glossary Tooltips
- `references/content-philosophy.md` → all of it
- `references/gotchas.md` → all of it
- `references/design-system.md` → just for any extra tokens needed

### Connections
- **Previous module:** Module 2 introduced the cast of LLMs. Now we show them in motion.
- **Next module:** Module 4 — VRAM batching. We mentioned that parallel can fall back to batched when memory is tight. Module 4 explains *exactly* when and how.
- **Tone/style notes:**
  - Accent: teal `#2A7B9B`.
  - **The kitchen metaphor lives in this module only.** Do not call models "cooks" elsewhere; that would muddle the panel-of-judges framing from M1/M2.
  - Lean into the *practical* takeaway: vibe coders should be able to spot when their AI assistant has implemented streaming wrong (e.g., "the response still comes all at once" — that means the assistant collected all tokens before yielding, not actually streamed).
  - This is the heaviest "computer science" module. Use *more* tooltips and *shorter* paragraphs. Aim for 60–70% visual.
