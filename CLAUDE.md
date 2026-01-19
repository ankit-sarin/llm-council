# LLM Council

Multi-model debate system where local Ollama models deliberate on questions, review each other's responses anonymously, and Claude Opus 4.5 synthesizes the final answer.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Gradio Interface                        │
│  Tab 1: Ask Council  │  Tab 2: Deliberation  │  Tab 3: Settings │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   Council Engine (council.py)               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │  Stage 1    │→ │  Stage 2    │→ │  Stage 3    │         │
│  │  Responses  │  │  Reviews    │  │  Synthesis  │         │
│  │ (streaming) │  │ (streaming) │  │  (Claude)   │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│    Ollama    │      │    Ollama    │      │   Anthropic  │
│   Models     │      │   Models     │      │  Claude Opus │
│  (parallel)  │      │  (reviews)   │      │  (chairman)  │
└──────────────┘      └──────────────┘      └──────────────┘
```

## Project Structure

```
llm-council/
├── CLAUDE.md           # This file
├── requirements.txt    # Python dependencies
├── config.py           # Model configuration
├── council.py          # Async council engine with streaming
└── app.py              # Gradio interface (port 7861)
```

## Files

### config.py
Model configuration with VRAM requirements:
```python
AVAILABLE_MODELS = [
    "llama3.3:70b",     # ~40GB - Meta's flagship
    "qwen2.5:32b",      # ~20GB - Alibaba's multilingual model
    "gemma2:27b",       # ~17GB - Google's efficient model
    "deepseek-r1:32b",  # ~20GB - Reasoning specialist
    "mistral:7b",       # ~4GB  - Fast European model
]

DEFAULT_ENABLED_MODELS = ["qwen2.5:32b", "gemma2:27b", "mistral:7b"]
CHAIRMAN_MODEL = "claude-opus-4-5-20251101"
```

### council.py
Async engine with streaming callbacks:
- `stream_initial_responses(question, models, callback)` - Yields as each model completes
- `stream_peer_reviews(question, responses, callback)` - Yields as each review completes
- `get_chairman_synthesis(question, responses, reviews)` - Claude Opus final synthesis

### app.py
Gradio interface with 3 tabs:
- **Tab 1: Ask the Council** - Model checkboxes, question input, final answer
- **Tab 2: Council Deliberation** - Live view of responses/reviews as they arrive
- **Tab 3: Settings** - Chairman info, verbose toggle

## Data Classes

```python
@dataclass
class ModelResponse:
    model: str
    content: str
    elapsed_seconds: float

@dataclass
class PeerReview:
    reviewer: str
    rankings: list[dict]
    analysis: str

@dataclass
class CouncilResult:
    question: str
    responses: dict[str, ModelResponse]
    reviews: dict[str, PeerReview]
    synthesis: str
    total_elapsed: float
```

## Council Flow

### Stage 1: Initial Responses (Streaming)
- All Ollama models run in parallel via `asyncio.as_completed()`
- UI updates as each model finishes with ✅ indicator
- Shows elapsed time per model

### Stage 2: Peer Reviews (Streaming)
- Each model reviews OTHER models' responses (anonymized as "Response A", "Response B")
- Models rank by: accuracy, insight, completeness (1-10 scale)
- UI updates as each review completes

### Stage 3: Chairman Synthesis
- Claude Opus 4.5 receives all responses with model names + all reviews
- Produces:
  1. Synthesized best answer
  2. Key contributors and their insights
  3. Areas of consensus
  4. Areas of disagreement and resolution

## Key Implementation Patterns

### Streaming with asyncio.as_completed
```python
async def stream_initial_responses(question, models, callback):
    tasks = [asyncio.create_task(call_with_id(model)) for model in models]
    for coro in asyncio.as_completed(tasks):
        model, response = await coro
        await callback(model, response, all_responses)
```

### Sync Ollama wrapped for async
```python
async def _call_ollama(model: str, prompt: str) -> ModelResponse:
    response = await asyncio.to_thread(
        ollama.chat,
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    return ModelResponse(model=model, content=response["message"]["content"], ...)
```

### Gradio generator for live updates
```python
def run_council_generator(question, selected_models):
    result_queue = queue.Queue()
    # Async runner pushes to queue
    # Generator yields from queue for Gradio streaming
    while True:
        msg_type, data = result_queue.get(timeout=600)
        if msg_type == "done":
            break
        yield data
```

### Anonymization for peer review
```python
def _anonymize_responses(responses, exclude_model):
    other_responses = {k: v for k, v in responses.items() if k != exclude_model}
    mapping = {}
    for i, (model, response) in enumerate(other_responses.items()):
        letter = chr(65 + i)  # A, B, C, ...
        mapping[letter] = model
    return formatted_text, mapping
```

## Environment Variables

```bash
ANTHROPIC_API_KEY=sk-ant-...  # Required for chairman
```

## Commands

```bash
# Run the app
python app.py
# Runs on http://0.0.0.0:7861

# Test Ollama connectivity
ollama list
```

## UI Status Indicators

During processing, the UI shows:
```
Stage 1/3: Gathering responses (2/3 complete)...
- Qwen 2.5 32B: ✅ 45.2s
- Gemma 2 27B: ✅ 52.1s
- Mistral 7B: ⏳ working...
```

## Dependencies

Key packages from requirements.txt:
- `gradio==6.3.0` - Web interface
- `ollama==0.6.1` - Local model API
- `anthropic==0.76.0` - Claude API
- `asyncio` - Parallel execution
