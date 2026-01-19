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
│  │ (batched)   │  │ (batched)   │  │             │  │  (Claude)   │    │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
          ┌─────────────────────────┼─────────────────────────┐
          ▼                         ▼                         ▼
┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐
│      Ollama      │      │      Ollama      │      │    Anthropic     │
│      Models      │      │      Models      │      │   Claude Opus    │
│  (VRAM batches)  │      │    (reviews)     │      │   (chairman)     │
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

MAX_CONCURRENT_VRAM_GB = 45.0  # Max VRAM per batch

DEFAULT_ENABLED_MODELS = ["qwen2.5:32b", "gemma2:27b", "mistral:7b"]
CHAIRMAN_MODEL = "claude-opus-4-5-20251101"
```

### council.py
Async engine with VRAM-aware batching and streaming:
- `stream_initial_responses_live(question, models, callbacks, document_text)` - Batched responses with live tokens
- `stream_peer_reviews_live(question, responses, callbacks)` - Batched reviews with live tokens
- `get_chairman_synthesis(question, responses, reviews, consensus, document_filename)` - Claude synthesis
- `calculate_consensus(reviews, responses)` - Agreement score from peer review variance
- `create_vram_batches(models)` - Groups models to fit within VRAM limits

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
class CouncilResult:
    question: str
    responses: dict[str, ModelResponse]
    reviews: dict[str, PeerReview]
    synthesis: str
    total_elapsed: float
    consensus: ConsensusScore | None
```

## Features

### VRAM-Aware Batching
Prevents GPU memory exhaustion by running models in batches:
```python
def create_vram_batches(models, max_vram=45.0):
    # Sort by VRAM descending (largest first)
    # Greedily pack into batches that fit within max_vram
    # Example: [70B] -> [32B, 7B] -> [27B]
```

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
- **Limits:** 5MB file size, 50k character extraction
- **Warnings:** Long documents (>30k chars), truncation notice

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

### Stage 1: Initial Responses (Batched Streaming)
- Models grouped into VRAM-aware batches
- Each batch runs in parallel, batches run sequentially
- 2-second delay between batches for GPU memory clearing
- UI updates as each model finishes with ✅ indicator

### Stage 2: Peer Reviews (Batched Streaming)
- Each model reviews OTHER models' responses (anonymized as "Response A", "Response B")
- Models rank by: accuracy, insight, completeness (1-10 scale)
- Batched execution same as Stage 1

### Stage 2.5: Consensus Calculation
- Parse scores from all reviews
- Calculate variance to determine agreement level
- Display consensus meter in UI

### Stage 3: Chairman Synthesis
- Claude Opus 4.5 receives all responses + reviews + consensus info
- When consensus is low, emphasizes exploring disagreements
- Notes when responses were based on an uploaded document
- Produces:
  1. Synthesized best answer
  2. Key contributors and their insights
  3. Areas of consensus
  4. Areas of disagreement and resolution

## Key Implementation Patterns

### VRAM Batching
```python
async def stream_initial_responses_live(question, models, callbacks, document_text=None):
    batches = create_vram_batches(models)  # [[70B], [32B, 7B], [27B]]

    for batch in batches:
        tasks = [asyncio.create_task(run_model(m)) for m in batch]
        await asyncio.gather(*tasks)
        await asyncio.sleep(2)  # Let GPU memory clear
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
- Submit button
- Chairman's final answer
- Consensus meter

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
