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
├── app.py              # Gradio interface (PORT env, default 7861)
└── sessions/           # Saved council sessions (JSON)
```

## Files

### config.py
Model configuration with VRAM requirements, context limits, and batching:
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

# Context window limits per model (tokens)
MODEL_CONTEXT_TOKENS = {
    "llama3.3:70b": 128000,
    "qwen2.5:32b": 32000,
    "gemma2:27b": 8000,    # Limited context
    "deepseek-r1:32b": 64000,
    "mistral:7b": 32000,
}

MAX_CONCURRENT_VRAM_GB = 60.0   # Only batch if total exceeds this
VRAM_SAFETY_FACTOR = 1.20       # 20% buffer for conservative estimates
CHARS_PER_TOKEN = 3.0           # Conservative char-to-token estimate
CONTEXT_SAFETY_MARGIN = 0.7     # Reserve 30% for prompt + response

DEFAULT_ENABLED_MODELS = ["qwen2.5:32b", "gemma2:27b", "mistral:7b"]
CHAIRMAN_MODEL = "claude-opus-4-5-20251101"

# Model selection presets
MODEL_PRESETS = {
    "fast": {"name": "⚡ Fast", "models": ["gemma2:27b", "mistral:7b"]},
    "balanced": {"name": "⚖️ Balanced", "models": ["qwen2.5:32b", "gemma2:27b", "mistral:7b"]},
    "deep": {"name": "🔬 Deep", "models": ["llama3.3:70b", "qwen2.5:32b", "deepseek-r1:32b"]},
}

# Domain-specific example prompts for first-run usability
EXAMPLE_PROMPTS = [
    {"label": "🧬 ctDNA Surveillance", "prompt": "What is the current evidence for using ctDNA..."},
    {"label": "🤖 Robotic Platforms", "prompt": "Compare the da Vinci Xi, Hugo RAS, and..."},
    {"label": "📄 Paper Critique", "prompt": "What are the key methodological considerations..."},
    {"label": "💊 Drug Interaction", "prompt": "A patient on warfarin needs to start amiodarone..."},
    {"label": "🔬 Research Design", "prompt": "What are the advantages and limitations of..."},
]
```

### council.py
Async engine with parallel-first execution, streaming, and timeouts:
```python
# Timeout constants
STREAM_TOTAL_TIMEOUT_SECONDS = 600   # 10 min max per model
STREAM_STALL_TIMEOUT_SECONDS = 120   # 2 min without tokens = stalled
COOLDOWN_BASE_SECONDS = 1.0          # Minimum inter-batch delay
COOLDOWN_PER_10GB_SECONDS = 0.5      # Additional delay per 10GB VRAM
```

Key functions:
- `stream_initial_responses_live(question, models, callbacks, document_text, stop_event)` - Parallel responses with stop support
- `stream_peer_reviews_live(question, responses, callbacks, stop_event)` - Parallel reviews with stop support
- `_call_ollama_streaming(model, prompt, on_token, stop_event)` - Streams tokens with timeout and cancellation
- `get_chairman_synthesis(question, responses, reviews, consensus, document_filename)` - Claude synthesis
- `calculate_consensus(reviews, responses)` - Uses pre-parsed scores from PeerReview.rankings
- `enrich_rankings_with_scores(rankings, analysis)` - Parses accuracy/insight/completeness into rankings
- `calculate_batch_cooldown(batch)` - VRAM-aware delay between batches
- `create_vram_batches(models)` - Groups models to fit within VRAM limits

### app.py
Gradio interface with 4 tabs:
- **Tab 1: Ask the Council** - File upload, model presets, model checkboxes, example prompts, question input, final answer, consensus meter
- **Tab 2: Council Deliberation** - Live view of responses/reviews, consensus meter
- **Tab 3: Settings** - Chairman info, GPU memory management, VRAM config
- **Tab 4: History** - Browse past sessions, export to Markdown

Key constants:
```python
QUEUE_ITEM_TIMEOUT = 120    # 2 min wait for each queue item
TOTAL_TIMEOUT = 900         # 15 min max total streaming time
```

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

### Privacy & Session Saving
Sessions are **not saved by default** - users must explicitly click "Save Session to History" to save.

When a document is uploaded, a red warning banner appears:
- Warns users not to upload PHI/PII
- Explains that document contents are sent to AI models

Before saving, common identifiers are automatically redacted:
- Email addresses → `[EMAIL REDACTED]`
- Phone numbers → `[PHONE REDACTED]`
- SSNs → `[SSN REDACTED]`
- MRNs/Medical records → `[MRN REDACTED]`
- Dates of birth → `[DOB REDACTED]`
- Credit card numbers → `[CARD REDACTED]`

The `sessions/` directory is excluded via `.gitignore`.

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

**Implementation:** Stop uses `asyncio.Event` propagated through all streaming functions:
- `_call_ollama_streaming(model, prompt, on_token, stop_event)` - checks `stop_event.is_set()` in token loop
- `stream_initial_responses_live(..., stop_event)` - passes to all model calls
- `stream_peer_reviews_live(..., stop_event)` - passes to all reviewer calls
- Stopped responses are marked with `[Stopped by user]` suffix

When stopped early, the chairman uses `get_chairman_early_synthesis()` which:
- Works with whatever responses have been collected
- Adds more independent analysis to compensate for limited council input
- Can perform web searches to supplement incomplete data

### Model Selection Presets
Quick configuration buttons for common use cases:
- **⚡ Fast**: gemma2:27b + mistral:7b — Quick answers with smaller models
- **⚖️ Balanced**: qwen2.5:32b + gemma2:27b + mistral:7b — Good tradeoff of speed and depth
- **🔬 Deep**: llama3.3:70b + qwen2.5:32b + deepseek-r1:32b — Thorough analysis with heavy hitters

Selecting a preset automatically sets the model checkboxes.

### Example Prompts
Domain-specific example prompts improve first-run usability:
- 🧬 ctDNA Surveillance — Watch-and-wait strategies in colorectal cancer
- 🤖 Robotic Platforms — Comparison of da Vinci, Hugo, Medtronic systems
- 📄 Paper Critique — Framework for evaluating RCT methodology
- 💊 Drug Interaction — Warfarin-amiodarone management
- 🔬 Research Design — Propensity score matching vs inverse probability weighting

Selecting an example loads the full prompt into the question input.

### VRAM Management
VRAM estimates include a 20% safety factor (`VRAM_SAFETY_FACTOR = 1.20`) to avoid OOM:
```python
def get_model_vram(model, with_safety_factor=True):
    base_vram = MODEL_VRAM_GB.get(model, 10.0)
    return base_vram * VRAM_SAFETY_FACTOR if with_safety_factor else base_vram
```

Inter-batch cooldowns scale with batch VRAM usage:
```python
def calculate_batch_cooldown(batch):
    batch_vram = sum(get_model_vram(m) for m in batch)
    return COOLDOWN_BASE_SECONDS + (batch_vram / 10.0) * COOLDOWN_PER_10GB_SECONDS
    # 48GB batch → 1.0 + 2.4 = 3.4 seconds cooldown
```

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
Measures agreement between reviewers with coverage statistics:
- Scores are parsed into `PeerReview.rankings` via `enrich_rankings_with_scores()`
- Each ranking entry gains `accuracy`, `insight`, `completeness` numeric fields
- `calculate_consensus()` uses pre-parsed scores from rankings
- Coverage stats track how many reviewers and responses were successfully parsed
- `ConsensusScore.description` includes coverage info (e.g., "3/3 reviewers, 2/3 responses parsed")
- High variance = low consensus, Low variance = high consensus
- Visual meter shows score 0-100 with color coding (green/orange/red)

### File Upload
Analyze documents with the council:
- **Supported formats:** .txt, .md, .pdf, .docx
- **PDF extraction:** pdfplumber (better for tables)
- **Word extraction:** python-docx (paragraphs + tables)
- **Limits:** 10MB file size, 150K character extraction
- **Warnings:** Long documents (>100K chars), truncation notice
- **Context limits:** Models with insufficient context are automatically excluded (uses MODEL_CONTEXT_TOKENS)

Context limit helpers in config.py:
```python
get_model_context_limit(model) -> int         # Tokens for model
get_model_max_chars(model) -> int             # Max chars with safety margin
check_model_context_fit(model, chars) -> dict # Check if document fits
filter_models_by_context(models, chars) -> (compatible, excluded)
```

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
- All models run in parallel if total VRAM (with 20% safety factor) fits within limit (60GB)
- Falls back to VRAM-aware batching only when necessary
- VRAM-aware cooldowns between batches (larger batches wait longer)
- Timeouts: 10 min total, 2 min stall detection per model
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
async def stream_initial_responses_live(question, models, callbacks, document_text=None, stop_event=None):
    total_vram = sum(get_model_vram(m) for m in models)  # Includes 20% safety factor

    if total_vram <= MAX_CONCURRENT_VRAM_GB:
        batches = [models]  # All parallel
    else:
        batches = create_vram_batches(models)  # [[70B], [32B, 7B], [27B]]

    for batch in batches:
        if stop_event and stop_event.is_set():
            break
        tasks = [asyncio.create_task(run_model(m)) for m in batch]
        await asyncio.gather(*tasks)
        if len(batches) > 1:
            cooldown = calculate_batch_cooldown(batch)  # VRAM-aware delay
            await asyncio.sleep(cooldown)
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

### Score Enrichment and Consensus
```python
def enrich_rankings_with_scores(rankings, analysis):
    # Parse "Response A: Accuracy=8, Insight=7, Completeness=9" from analysis
    # Add numeric fields to each ranking: {"response_letter": "A", "accuracy": 8, ...}
    # Returns (enriched_rankings, responses_parsed_count)

def calculate_consensus(reviews, responses):
    # Uses pre-parsed scores from PeerReview.rankings
    # Calculates variance across reviewers for each response
    # Tracks coverage: how many reviewers and responses were parsed
    # Low variance (< 4) = high consensus, High variance (> 8) = low consensus
    return ConsensusScore(score=75, level="high", description="3/3 reviewers, 2/3 responses parsed")
```

### Session Storage
```python
def save_session(question, responses, reviews, synthesis, consensus, models,
                 document_filename=None, document_char_count=None):
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = SESSIONS_DIR / f"session_{session_id}.json"
    # PII redaction applied before saving
    json.dump(session_data, filepath)
```

### Streaming Robustness
Ollama streaming includes multiple safeguards:
```python
async def _call_ollama_streaming(model, prompt, on_token, stop_event=None):
    # Thread runs blocking ollama.chat() with stream=True
    # Queue bridges sync thread to async event loop
    # Timeouts prevent indefinite hangs:
    #   - STREAM_TOTAL_TIMEOUT_SECONDS (600s) for entire stream
    #   - STREAM_STALL_TIMEOUT_SECONDS (120s) between tokens
    # stop_event.is_set() checked in token loop for cancellation
    # Thread is daemon=True and joined with timeout
    # finally: block always puts ("done", None) to unblock consumer
    # Token extraction tolerates missing fields: chunk.get("message", {}).get("content", "")
```

App-level timeouts:
```python
QUEUE_ITEM_TIMEOUT = 120  # 2 min wait for each queue item
TOTAL_TIMEOUT = 900       # 15 min max total streaming time
# Transfer task wrapped in try/finally for cleanup
```

## Environment Variables

```bash
# Required for chairman
ANTHROPIC_API_KEY=sk-ant-...

# Required for authentication (app will not start without these)
LLM_COUNCIL_USER=your_username
LLM_COUNCIL_PASSWORD=your_password

# Optional: Custom port (default: 7861)
PORT=7861
```

## Graceful Shutdown

The app handles SIGTERM and SIGINT signals for clean shutdown:
- **Ctrl+C (SIGINT)**: Prints "SIGINT received. Shutting down gracefully..." and exits
- **kill -TERM (SIGTERM)**: Prints "SIGTERM received. Shutting down gracefully..." and exits

This enables proper shutdown in containerized environments (Docker, Kubernetes) and process managers.

## Port Conflict Recovery

On startup, `ensure_port_available(port)` checks whether the configured port is free:
- **Port free:** Startup continues normally
- **Stale LLM Council process:** Automatically sends SIGTERM, waits 5s, then SIGKILL if needed, and reclaims the port
- **Another process:** Prints a clear error identifying the PID and process name, then exits

The systemd service has restart limits (`StartLimitBurst=5` / `StartLimitIntervalSec=60`) to prevent infinite crash loops — after 5 failures in 60 seconds, systemd stops retrying. `RestartSec=10` gives the port time to release between attempts.

## Authentication

The app requires authentication to protect access. Users must log in before using the interface.

**Setup:**
```bash
export LLM_COUNCIL_USER='your_username'
export LLM_COUNCIL_PASSWORD='your_password'
```

Or add to `.env` file (already in `.gitignore`):
```
LLM_COUNCIL_USER=your_username
LLM_COUNCIL_PASSWORD=your_password
```

**Behavior:**
- If credentials are not set, the app prints an error and exits
- Users see a Gradio login screen before accessing the interface
- Single user/password pair (not multi-user)

## Commands

```bash
# Set credentials first
export LLM_COUNCIL_USER='admin'
export LLM_COUNCIL_PASSWORD='secret'

# Run the app (default port 7861)
python app.py

# Run on custom port
PORT=8080 python app.py

# Test Ollama connectivity
ollama list
```

## UI Components

### Tab 1: Ask the Council
- File upload (optional) - .txt, .md, .pdf, .docx
- PHI warning banner (shown when document uploaded)
- File info display with preview
- Model preset buttons (Fast / Balanced / Deep)
- Model checkboxes
- Example prompt dropdown
- Question input
- Submit button + Stop Deliberation button (elem_id="stop_btn")
- Stop confirmation options (Clear Session / Synthesize Now)
- Chairman's final answer
- Consensus meter (with coverage stats)
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
