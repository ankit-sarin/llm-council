# LLM Council

Multi-model debate system where local Ollama models deliberate on questions, review each other's responses anonymously, and Claude Opus 4.5 synthesizes the final answer.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           Gradio Interface                               │
│  Tab 1: Ask Council │ Tab 2: Deliberation │ Tab 3: Settings │ Tab 4: History │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
             ┌──────────┐    ┌──────────┐    ┌──────────┐
             │  File    │    │  VRAM    │    │ Session  │
             │  Upload  │    │ Batching │    │ Storage  │
             └──────────┘    └──────────┘    └──────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     Council Engine (council.py)                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │  Stage 1    │→ │  Stage 2    │→ │  Consensus  │→ │  Stage 3    │    │
│  │  Responses  │  │  Reviews    │  │ Calculation │  │  Synthesis  │    │
│  │ (parallel)  │  │ (parallel)  │  │             │  │  (Claude)   │    │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
          ┌─────────────────────────┼─────────────────────────┐
          ▼                         ▼                         ▼
┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐
│      Ollama      │      │      Ollama      │      │    Anthropic     │
│      Models      │      │      Models      │      │   Claude Opus    │
│   (parallel)     │      │    (reviews)     │      │   (chairman)     │
└──────────────────┘      └──────────────────┘      └──────────────────┘
```

## Project Structure

```
llm-council/
├── CLAUDE.md           # This file
├── requirements.txt    # Python dependencies
├── config.py           # Model configuration + VRAM batching
├── council.py          # Async council engine with streaming
├── app.py              # Gradio interface (port 7861)
└── sessions/           # Saved council sessions (JSON)
```

## Files

### config.py
Model configuration with VRAM requirements and batching:
```python
AVAILABLE_MODELS = [
    "llama3.3:70b",     # ~40GB - Meta's flagship
    "qwen2.5:32b",      # ~20GB - Alibaba's multilingual model
    "gemma2:27b",       # ~17GB - Google's efficient model
    "deepseek-r1:32b",  # ~20GB - Reasoning specialist
    "mistral:7b",       # ~4GB  - Fast European model
]

# VRAM requirements per model (GB)
MODEL_VRAM_GB = {
    "llama3.3:70b": 40.0,
    "qwen2.5:32b": 20.0,
    "gemma2:27b": 17.0,
    "deepseek-r1:32b": 20.0,
    "mistral:7b": 4.0,
}

MAX_CONCURRENT_VRAM_GB = 60.0  # Only batch if total exceeds this

DEFAULT_ENABLED_MODELS = ["qwen2.5:32b", "gemma2:27b", "mistral:7b"]
CHAIRMAN_MODEL = "claude-opus-4-5-20251101"
```

### council.py
Async engine with parallel-first execution and streaming:
- `stream_initial_responses_live(question, models, callbacks, document_text)` - Parallel responses (batches only if VRAM overflows)
- `stream_peer_reviews_live(question, responses, callbacks)` - Parallel reviews (batches only if VRAM overflows)
- `get_chairman_synthesis(question, responses, reviews, consensus, document_filename)` - Claude synthesis
- `calculate_consensus(reviews, responses)` - Agreement score from peer review variance
- `create_vram_batches(models)` - Groups models to fit within VRAM limits (used only when needed)

### app.py
Gradio interface with 4 tabs:
- **Tab 1: Ask the Council** - File upload, model checkboxes, question input, final answer, consensus meter
- **Tab 2: Council Deliberation** - Live view of responses/reviews, consensus meter
- **Tab 3: Settings** - Chairman info, GPU memory management, VRAM config
- **Tab 4: History** - Browse past sessions, export to Markdown

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
class ConsensusScore:
    score: float           # 0-100, higher = more agreement
    level: str             # "high", "medium", "low"
    description: str       # Human-readable description
    disagreement_areas: list[str]  # Specific disagreements

@dataclass
class ResponseComposition:
    council_contribution: float  # 0-100, percentage from council models
    chairman_independent: float  # 0-100, percentage from chairman's own analysis
    web_search_used: float       # 0-100, percentage from web search
    web_searches_performed: list[str]  # Search queries performed
    chairman_insights: list[str]       # Key insights added by chairman

@dataclass
class CouncilResult:
    question: str
    responses: dict[str, ModelResponse]
    reviews: dict[str, PeerReview]
    synthesis: str
    total_elapsed: float
    consensus: ConsensusScore | None
```

## Features

### Chairman Independent Thinking & Web Search
The chairman (Claude Opus 4.5) can now:
- Add independent insights beyond just summarizing council responses
- Perform web searches to verify facts or get current information
- Mark contributions with `[CHAIRMAN'S INSIGHT]` or `[WEB SOURCE]` tags

Response composition is tracked and displayed in a meter showing:
- **Council Models** (blue): Percentage derived from local model responses
- **Chairman Independent** (purple): Percentage from chairman's own analysis
- **Web Search** (orange): Percentage from web search results

### Stop Deliberation
Users can stop deliberation at any time using the "Stop Deliberation" button:
1. Click "⏹️ Stop Deliberation" during any stage
2. Choose between:
   - **Clear Session**: Discard all progress and start fresh
   - **Synthesize Now**: Have the chairman synthesize with available data

When stopped early, the chairman uses `get_chairman_early_synthesis()` which:
- Works with whatever responses have been collected
- Adds more independent analysis to compensate for limited council input
- Can perform web searches to supplement incomplete data

### Parallel-First Execution
Models run in parallel by default for maximum speed. Batching only activates when total VRAM would exceed the limit:
```python
total_vram = sum(get_model_vram(m) for m in models)

if total_vram <= MAX_CONCURRENT_VRAM_GB:  # 60GB default
    # All models fit - run in parallel (single batch)
    batches = [models]
    execution_mode = "PARALLEL"
else:
    # Need to batch to avoid VRAM exhaustion
    batches = create_vram_batches(models)
    execution_mode = "BATCHED"
```

| Selection | Total VRAM | Mode | Result |
|-----------|------------|------|--------|
| Default 3 models | 41GB | PARALLEL | All stream simultaneously |
| Add llama3.3:70b | 81GB | BATCHED | Split into batches |

### Consensus Tracking
Measures agreement between reviewers:
- Parses scores from review text (Accuracy, Insight, Completeness)
- Calculates variance across reviewers for each response
- High variance = low consensus, Low variance = high consensus
- Visual meter shows score 0-100 with color coding (green/orange/red)

### File Upload
Analyze documents with the council:
- **Supported formats:** .txt, .md, .pdf, .docx
- **PDF extraction:** pdfplumber (better for tables)
- **Word extraction:** python-docx (paragraphs + tables)
- **Limits:** 10MB file size, 150K character extraction
- **Warnings:** Long documents (>100K chars), truncation notice
- **Gemma 2 exclusion:** Automatically excluded for documents >30K chars (8K context limit)

### Session History
All sessions saved to `sessions/` as JSON:
```json
{
  "id": "20240115_143022",
  "timestamp": "2024-01-15T14:30:22",
  "question": "...",
  "models": ["Qwen 2.5 32B", "Gemma 2 27B"],
  "responses": {...},
  "reviews": {...},
  "consensus": {"score": 75, "level": "high", ...},
  "synthesis": "...",
  "document": {"filename": "report.pdf", "char_count": 15000, "preview": "..."}
}
```

## Council Flow

### Stage 1: Initial Responses (Parallel Streaming)
- All models run in parallel if total VRAM fits within limit (60GB)
- Falls back to VRAM-aware batching only when necessary
- 2-second delay between batches (if batched) for GPU memory clearing
- UI updates as each model finishes with ✅ indicator

### Stage 2: Peer Reviews (Parallel Streaming)
- Each model reviews OTHER models' responses (anonymized as "Response A", "Response B")
- Models rank by: accuracy, insight, completeness (1-10 scale)
- Same parallel-first execution as Stage 1

### Stage 2.5: Consensus Calculation
- Parse scores from all reviews
- Calculate variance to determine agreement level
- Display consensus meter in UI

### Stage 3: Chairman Synthesis
- Claude Opus 4.5 receives all responses + reviews + consensus info
- Can add independent insights beyond council responses (marked with `[CHAIRMAN'S INSIGHT]`)
- Can perform web searches for verification/current info (marked with `[WEB SOURCE]`)
- When consensus is low, emphasizes exploring disagreements
- Notes when responses were based on an uploaded document
- Produces:
  1. Synthesized best answer
  2. Key contributors and their insights
  3. Areas of consensus
  4. Areas of disagreement and resolution
  5. Chairman's independent additions and web searches
- Returns both synthesis text and ResponseComposition metrics

## Key Implementation Patterns

### Parallel-First Execution
```python
async def stream_initial_responses_live(question, models, callbacks, document_text=None):
    total_vram = sum(get_model_vram(m) for m in models)

    if total_vram <= MAX_CONCURRENT_VRAM_GB:
        batches = [models]  # All parallel
    else:
        batches = create_vram_batches(models)  # [[70B], [32B, 7B], [27B]]

    for batch in batches:
        tasks = [asyncio.create_task(run_model(m)) for m in batch]
        await asyncio.gather(*tasks)
        if len(batches) > 1:
            await asyncio.sleep(2)  # Let GPU memory clear between batches
```

### Document-Aware Prompts
```python
if document_text:
    prompt = f"""Based on the following document, answer this question:
    QUESTION: {question}
    DOCUMENT: {document_text}"""
else:
    prompt = f"""You are a member of an expert council...
    QUESTION: {question}"""
```

### Consensus Calculation
```python
def calculate_consensus(reviews, responses):
    # Parse "Response A: Accuracy=8, Insight=7, Completeness=9"
    # Calculate variance across reviewers
    # Low variance (< 4) = high consensus
    # High variance (> 8) = low consensus
    return ConsensusScore(score=75, level="high", ...)
```

### Session Storage
```python
def save_session(question, responses, reviews, synthesis, consensus, models,
                 document_filename=None, document_char_count=None):
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = SESSIONS_DIR / f"session_{session_id}.json"
    json.dump(session_data, filepath)
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

## UI Components

### Tab 1: Ask the Council
- File upload (optional) - .txt, .md, .pdf, .docx
- File info display with preview
- Question input
- Model checkboxes
- Submit button + Stop Deliberation button
- Stop confirmation options (Clear Session / Synthesize Now)
- Chairman's final answer
- Consensus meter
- Response composition meter

### Tab 3: Settings
- GPU Memory Management section
  - "Clear GPU Memory" button (restarts Ollama)
  - VRAM configuration display
- Model information table

### Tab 4: History
- Session dropdown (date | consensus | question preview)
- Refresh button
- Export to Markdown button
- Session details accordion

## Dependencies

Key packages from requirements.txt:
- `gradio==6.3.0` - Web interface
- `ollama==0.6.1` - Local model API
- `anthropic==0.76.0` - Claude API
- `pdfplumber==0.11.9` - PDF text extraction
- `python-docx==1.2.0` - Word document extraction
- `asyncio` - Parallel execution
