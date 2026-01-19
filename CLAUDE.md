# LLM Council

Multi-model debate system where local Ollama models deliberate on questions, review each other's responses anonymously, and Claude Opus 4.5 synthesizes the final answer.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Gradio Interface                        │
│  [Question Input] → [Debate View] → [Review View] → [Final] │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      Council Engine                         │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐        │
│  │ Stage 1 │→ │ Stage 2 │→ │ Stage 3 │→ │ Stage 4 │        │
│  │ Debate  │  │ Review  │  │ Rebut   │  │Synthesis│        │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘        │
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
├── CLAUDE.md
├── requirements.txt
├── app.py                 # Gradio interface entry point
├── council/
│   ├── __init__.py
│   ├── engine.py          # Council orchestration logic
│   ├── models.py          # Pydantic models for responses
│   └── clients/
│       ├── __init__.py
│       ├── ollama.py      # Async Ollama client wrapper
│       └── anthropic.py   # Async Anthropic client wrapper
└── prompts/
    ├── debate.txt         # Initial response prompt
    ├── review.txt         # Anonymous peer review prompt
    ├── rebuttal.txt       # Response to reviews prompt
    └── synthesis.txt      # Chairman synthesis prompt
```

## Available Ollama Models

Council members (run parallel via asyncio):
- `deepseek-r1:32b` - Strong reasoning
- `qwen2.5:32b` - Balanced general purpose
- `gemma2:27b` - Good instruction following
- `mistral:7b` - Fast, good for quick debates
- `llama3.3:70b` - Large, high quality (slower)
- `llama3.2:3b` - Lightweight fallback

## Code Conventions

### Async Patterns

All model calls must be async. Use `asyncio.gather()` for parallel execution:

```python
async def run_debate(question: str, models: list[str]) -> list[Response]:
    tasks = [get_model_response(model, question) for model in models]
    return await asyncio.gather(*tasks)
```

### Response Models

Use Pydantic for all data structures:

```python
class Response(BaseModel):
    model: str
    content: str
    stage: Literal["debate", "review", "rebuttal", "synthesis"]
    timestamp: datetime = Field(default_factory=datetime.now)

class Review(BaseModel):
    reviewer: str
    target_id: str  # anonymized: "Response A", "Response B", etc.
    critique: str
    score: int = Field(ge=1, le=10)
```

### Anonymization

When passing responses for review, strip model names:

```python
def anonymize_responses(responses: list[Response]) -> dict[str, str]:
    return {f"Response {chr(65+i)}": r.content for i, r in enumerate(responses)}
```

### Client Wrappers

Ollama client (council/clients/ollama.py):
```python
import ollama

async def query_ollama(model: str, prompt: str) -> str:
    response = await asyncio.to_thread(
        ollama.chat,
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"]
```

Anthropic client (council/clients/anthropic.py):
```python
from anthropic import Anthropic

client = Anthropic()

async def query_claude(prompt: str, system: str = "") -> str:
    response = await asyncio.to_thread(
        client.messages.create,
        model="claude-opus-4-5-20251101",
        max_tokens=4096,
        system=system,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text
```

## Council Flow

### Stage 1: Initial Debate
Each Ollama model answers the question independently (parallel).

### Stage 2: Peer Review
Each model reviews all other responses (anonymized as "Response A", "Response B", etc.). Models cannot review their own response.

### Stage 3: Rebuttal (optional)
Models can respond to critiques of their answers.

### Stage 4: Chairman Synthesis
Claude Opus 4.5 receives:
- Original question
- All responses with model names
- All reviews
- Any rebuttals

Produces final synthesized answer highlighting consensus, disagreements, and best reasoning.

## Gradio Interface

Use `gr.Blocks` for multi-stage display:

```python
with gr.Blocks() as app:
    question = gr.Textbox(label="Question")
    submit = gr.Button("Convene Council")

    with gr.Tab("Debate"):
        debate_output = gr.Markdown()
    with gr.Tab("Reviews"):
        review_output = gr.Markdown()
    with gr.Tab("Final Synthesis"):
        synthesis_output = gr.Markdown()

    submit.click(
        fn=run_council,
        inputs=[question],
        outputs=[debate_output, review_output, synthesis_output]
    )
```

## Environment Variables

```bash
ANTHROPIC_API_KEY=sk-ant-...  # Required for chairman
OLLAMA_HOST=http://localhost:11434  # Default Ollama endpoint
COUNCIL_MODELS=deepseek-r1:32b,qwen2.5:32b,gemma2:27b  # Comma-separated
```

## Commands

```bash
# Run the app
python app.py

# Test Ollama connectivity
ollama list

# Run with specific models
COUNCIL_MODELS=mistral:7b,llama3.2:3b python app.py
```

## Error Handling

Wrap model calls with timeouts and fallbacks:

```python
async def safe_query(model: str, prompt: str, timeout: int = 120) -> str:
    try:
        return await asyncio.wait_for(
            query_ollama(model, prompt),
            timeout=timeout
        )
    except asyncio.TimeoutError:
        return f"[{model} timed out]"
    except Exception as e:
        return f"[{model} error: {e}]"
```

## Testing a Single Model

```python
import ollama
response = ollama.chat(model="mistral:7b", messages=[
    {"role": "user", "content": "Hello"}
])
print(response["message"]["content"])
```
