"""
LLM Council Engine

Orchestrates multi-model debates with peer review and chairman synthesis.
"""

import asyncio
import logging
import os
import time
from dataclasses import dataclass

import ollama
from anthropic import Anthropic

from config import (
    CHAIRMAN_MODEL,
    COUNCIL_MODELS,
    get_display_name,
    get_model_vram,
    create_vram_batches,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class ModelResponse:
    """Response from a council member with metadata."""
    model: str
    content: str
    elapsed_seconds: float


@dataclass
class PeerReview:
    """Review of other responses by a council member."""
    reviewer: str
    rankings: list[dict]  # [{"response": "A", "accuracy": 8, "insight": 7, "completeness": 9}, ...]
    analysis: str


@dataclass
class ConsensusScore:
    """Consensus metrics from peer reviews."""
    score: float  # 0-100, higher = more agreement
    level: str  # "high", "medium", "low"
    description: str  # Human-readable description
    disagreement_areas: list[str]  # Specific areas where models disagreed


@dataclass
class ResponseComposition:
    """Tracks the source composition of the chairman's synthesis."""
    council_contribution: float  # 0-100, percentage from council model responses
    chairman_independent: float  # 0-100, percentage from chairman's own analysis
    web_search_used: float  # 0-100, percentage from web search
    web_searches_performed: list[str]  # List of search queries performed
    chairman_insights: list[str]  # Key insights added by chairman independently


@dataclass
class CouncilResult:
    """Complete results from a council session."""
    question: str
    responses: dict[str, ModelResponse]
    reviews: dict[str, PeerReview]
    synthesis: str
    total_elapsed: float
    consensus: ConsensusScore | None = None
    composition: ResponseComposition | None = None


# --- Ollama Calls ---

async def _call_ollama(model: str, prompt: str) -> ModelResponse:
    """Call a single Ollama model asynchronously (non-streaming)."""
    logger.info(f"[{get_display_name(model)}] Starting generation...")
    start = time.perf_counter()

    try:
        # ollama.chat is synchronous, so wrap in thread
        response = await asyncio.to_thread(
            ollama.chat,
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        elapsed = time.perf_counter() - start
        content = response["message"]["content"]
        logger.info(f"[{get_display_name(model)}] Done in {elapsed:.1f}s ({len(content)} chars)")
        return ModelResponse(model=model, content=content, elapsed_seconds=elapsed)

    except Exception as e:
        elapsed = time.perf_counter() - start
        logger.error(f"[{get_display_name(model)}] Failed after {elapsed:.1f}s: {e}")
        return ModelResponse(model=model, content=f"[Error: {e}]", elapsed_seconds=elapsed)


async def _call_ollama_streaming(
    model: str,
    prompt: str,
    on_token,
    stop_event: asyncio.Event | None = None
) -> ModelResponse:
    """
    Call Ollama with token streaming, calling on_token for each chunk.

    Args:
        model: The model name to call
        prompt: The prompt to send
        on_token: Callback function for each token
        stop_event: Optional asyncio.Event to signal stopping early
    """
    import queue
    import threading

    logger.info(f"[{get_display_name(model)}] Starting streaming generation...")
    start = time.perf_counter()
    content = ""
    was_stopped = False
    thread = None
    token_queue = queue.Queue()

    def run_stream():
        """Stream tokens from Ollama. Always signals 'done' via finally."""
        error_msg = None
        try:
            for chunk in ollama.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                stream=True
            ):
                token = chunk["message"]["content"]
                token_queue.put(("token", token))
        except Exception as e:
            error_msg = str(e)
            token_queue.put(("error", error_msg))
        finally:
            # Always signal completion so the consumer loop can exit
            token_queue.put(("done", None))

    try:
        thread = threading.Thread(target=run_stream, daemon=True)
        thread.start()

        while True:
            # Check if stop was requested
            if stop_event is not None and stop_event.is_set():
                logger.info(f"[{get_display_name(model)}] Stop requested, halting stream")
                was_stopped = True
                break

            # Non-blocking check with small timeout, then yield to event loop
            try:
                msg_type, data = token_queue.get(timeout=0.05)
                if msg_type == "done":
                    break
                elif msg_type == "error":
                    content = f"[Error: {data}]"
                    # Continue to drain queue until "done"
                elif msg_type == "token":
                    content += data
                    await on_token(model, content)
            except queue.Empty:
                # Yield to event loop to allow other tasks to run
                await asyncio.sleep(0.01)
                # Check if thread is still alive
                if not thread.is_alive() and token_queue.empty():
                    break

        elapsed = time.perf_counter() - start

        if was_stopped:
            content = content + "\n\n[Stopped by user]" if content else "[Stopped by user]"
            logger.info(f"[{get_display_name(model)}] Stopped after {elapsed:.1f}s ({len(content)} chars)")
        else:
            logger.info(f"[{get_display_name(model)}] Done in {elapsed:.1f}s ({len(content)} chars)")

        return ModelResponse(model=model, content=content, elapsed_seconds=elapsed)

    except Exception as e:
        elapsed = time.perf_counter() - start
        logger.error(f"[{get_display_name(model)}] Failed after {elapsed:.1f}s: {e}")
        return ModelResponse(model=model, content=f"[Error: {e}]", elapsed_seconds=elapsed)

    finally:
        # Always join the thread before returning
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
            if thread.is_alive():
                logger.warning(f"[{get_display_name(model)}] Thread did not terminate in time")


async def get_initial_responses(question: str, models: list[str] | None = None) -> dict[str, ModelResponse]:
    """
    Get initial responses from all council members in parallel.

    Args:
        question: The question to ask the council
        models: List of model names (defaults to COUNCIL_MODELS)

    Returns:
        Dict mapping model names to their responses
    """
    models = models or COUNCIL_MODELS

    logger.info("=" * 60)
    logger.info("STAGE 1: INITIAL RESPONSES")
    logger.info(f"Question: {question[:100]}{'...' if len(question) > 100 else ''}")
    logger.info(f"Council members: {len(models)}")
    logger.info("=" * 60)

    prompt = f"""You are a member of an expert council deliberating on the following question.
Provide your best, most thoughtful answer. Be thorough but concise.

QUESTION: {question}

YOUR RESPONSE:"""

    # Run all models in parallel
    tasks = [_call_ollama(model, prompt) for model in models]
    results = await asyncio.gather(*tasks)

    responses = {r.model: r for r in results}

    total_time = sum(r.elapsed_seconds for r in results)
    logger.info(f"Stage 1 complete. Total model time: {total_time:.1f}s")

    return responses


async def stream_initial_responses(question: str, models: list[str], callback):
    """
    Stream initial responses, calling callback as each model completes.

    Args:
        question: The question to ask the council
        models: List of model names
        callback: async function(model, response) called as each completes
    """
    logger.info("=" * 60)
    logger.info("STAGE 1: INITIAL RESPONSES (STREAMING)")
    logger.info(f"Question: {question[:100]}{'...' if len(question) > 100 else ''}")
    logger.info(f"Council members: {len(models)}")
    logger.info("=" * 60)

    prompt = f"""You are a member of an expert council deliberating on the following question.
Provide your best, most thoughtful answer. Be thorough but concise.

QUESTION: {question}

YOUR RESPONSE:"""

    # Create tasks with model tracking
    async def call_with_id(model):
        response = await _call_ollama(model, prompt)
        return model, response

    tasks = [asyncio.create_task(call_with_id(model)) for model in models]
    responses = {}

    # Yield results as they complete
    for coro in asyncio.as_completed(tasks):
        model, response = await coro
        responses[model] = response
        await callback(model, response, responses)

    logger.info(f"Stage 1 complete. {len(responses)} responses collected.")
    return responses


async def stream_initial_responses_live(
    question: str,
    models: list[str],
    on_token_callback,
    on_complete_callback,
    document_text: str | None = None,
    stop_event: asyncio.Event | None = None
):
    """
    Stream initial responses with live token updates.
    Runs all models in parallel unless total VRAM exceeds the limit,
    in which case VRAM-aware batching is used.

    Args:
        question: The question to ask the council
        models: List of model names
        on_token_callback: async function(partial_responses: dict) called on each token
        on_complete_callback: async function(model, response, all_responses) called when model completes
        document_text: Optional text from an uploaded document to analyze
        stop_event: Optional asyncio.Event to signal stopping early
    """
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

    logger.info("=" * 60)
    logger.info("STAGE 1: INITIAL RESPONSES (LIVE STREAMING)")
    logger.info(f"Question: {question[:100]}{'...' if len(question) > 100 else ''}")
    if document_text:
        logger.info(f"Document: {len(document_text)} characters provided")
    logger.info(f"Council members: {len(models)} | Total VRAM: ~{total_vram:.0f}GB | Limit: {MAX_CONCURRENT_VRAM_GB}GB")
    logger.info(f"Execution mode: {execution_mode} ({len(batches)} batch{'es' if len(batches) > 1 else ''})")
    if len(batches) > 1:
        for i, batch in enumerate(batches):
            batch_vram = sum(get_model_vram(m) for m in batch)
            logger.info(f"  Batch {i+1}: {[get_display_name(m) for m in batch]} (~{batch_vram:.0f}GB)")
    logger.info("=" * 60)

    # Build prompt - different format when document is provided
    if document_text:
        # When a document is uploaded, ask the council to analyze it
        prompt = f"""You are a member of an expert council. A document has been provided for analysis.
Based on the following document, answer this question thoughtfully and thoroughly.

QUESTION: {question}

DOCUMENT:
{document_text}

YOUR RESPONSE:"""
    else:
        # Standard prompt without document
        prompt = f"""You are a member of an expert council deliberating on the following question.
Provide your best, most thoughtful answer. Be thorough but concise.

QUESTION: {question}

YOUR RESPONSE:"""

    # Track partial and complete responses
    partial_responses = {m: "" for m in models}
    complete_responses = {}

    # Lock for thread-safe updates
    import threading
    lock = threading.Lock()

    async def on_token(model, content):
        with lock:
            partial_responses[model] = content
        await on_token_callback(partial_responses, complete_responses)

    async def run_model(model):
        response = await _call_ollama_streaming(model, prompt, on_token, stop_event)
        with lock:
            complete_responses[model] = response
            partial_responses[model] = response.content
        await on_complete_callback(model, response, complete_responses)
        return response

    # Run models - in parallel if single batch, or sequentially by batch
    for batch_idx, batch in enumerate(batches):
        if len(batches) == 1:
            logger.info(f"Running all {len(batch)} models in parallel...")
        else:
            logger.info(f"Running batch {batch_idx + 1}/{len(batches)}: {[get_display_name(m) for m in batch]}")

        # Run all models in this batch concurrently
        tasks = [asyncio.create_task(run_model(m)) for m in batch]
        await asyncio.gather(*tasks)

        # Small delay between batches to let GPU memory clear
        if batch_idx < len(batches) - 1:
            logger.info("Waiting for GPU memory to clear before next batch...")
            await asyncio.sleep(2)

    logger.info(f"Stage 1 complete. {len(complete_responses)} responses collected.")
    return complete_responses


async def stream_peer_reviews_live(
    question: str,
    responses: dict[str, ModelResponse],
    on_token_callback,
    on_complete_callback,
    stop_event: asyncio.Event | None = None
):
    """
    Stream peer reviews with live token updates.
    Runs all reviewers in parallel unless total VRAM exceeds the limit.

    Args:
        question: The original question
        responses: Dict of model responses from stage 1
        on_token_callback: async function(partial_reviews: dict) called on each token
        on_complete_callback: async function(reviewer, review, all_reviews) called when review completes
        stop_event: Optional asyncio.Event to signal stopping early
    """
    from config import MAX_CONCURRENT_VRAM_GB

    reviewers = list(responses.keys())

    # Calculate total VRAM needed
    total_vram = sum(get_model_vram(r) for r in reviewers)

    # Only use batching if total VRAM exceeds the limit
    if total_vram <= MAX_CONCURRENT_VRAM_GB:
        batches = [reviewers]
        execution_mode = "PARALLEL"
    else:
        batches = create_vram_batches(reviewers)
        execution_mode = "BATCHED"

    logger.info("=" * 60)
    logger.info("STAGE 2: PEER REVIEWS (LIVE STREAMING)")
    logger.info(f"Each of {len(responses)} models reviewing {len(responses)-1} responses")
    logger.info(f"Total VRAM: ~{total_vram:.0f}GB | Limit: {MAX_CONCURRENT_VRAM_GB}GB")
    logger.info(f"Execution mode: {execution_mode} ({len(batches)} batch{'es' if len(batches) > 1 else ''})")
    logger.info("=" * 60)

    partial_reviews = {m: "" for m in responses.keys()}
    complete_reviews = {}

    import threading
    lock = threading.Lock()

    async def review_model(reviewer):
        anonymized, mapping = _anonymize_responses(responses, exclude_model=reviewer)
        logger.info(f"[{get_display_name(reviewer)}] Reviewing responses: {list(mapping.keys())}")

        prompt = f"""You are reviewing responses from other council members on the following question.
Your task is to evaluate each response and rank them.

ORIGINAL QUESTION: {question}

RESPONSES TO REVIEW:
{anonymized}

Please evaluate each response on three criteria (1-10 scale):
- ACCURACY: How factually correct is the response?
- INSIGHT: How deep and valuable are the insights?
- COMPLETENESS: How thoroughly does it address the question?

Provide your evaluation in this exact format:

RANKINGS:
Response A: Accuracy=X, Insight=X, Completeness=X
Response B: Accuracy=X, Insight=X, Completeness=X
[continue for all responses]

ANALYSIS:
[2-3 sentences maximum. Focus only on key differentiators between responses.]"""

        async def on_token(model, content):
            with lock:
                partial_reviews[reviewer] = content
            await on_token_callback(partial_reviews, complete_reviews)

        response = await _call_ollama_streaming(reviewer, prompt, on_token, stop_event)

        rankings = []
        for letter in mapping.keys():
            rankings.append({
                "response": letter,
                "model": mapping[letter],
            })

        review = PeerReview(
            reviewer=reviewer,
            rankings=rankings,
            analysis=response.content,
        )

        with lock:
            complete_reviews[reviewer] = review
            partial_reviews[reviewer] = response.content

        await on_complete_callback(reviewer, review, complete_reviews)
        return review

    # Run reviewers - in parallel if single batch, or sequentially by batch
    for batch_idx, batch in enumerate(batches):
        if len(batches) == 1:
            logger.info(f"Running all {len(batch)} reviewers in parallel...")
        else:
            logger.info(f"Running review batch {batch_idx + 1}/{len(batches)}: {[get_display_name(m) for m in batch]}")

        tasks = [asyncio.create_task(review_model(r)) for r in batch]
        await asyncio.gather(*tasks)

        if batch_idx < len(batches) - 1:
            logger.info("Waiting for GPU memory to clear before next batch...")
            await asyncio.sleep(2)

    logger.info("Stage 2 complete.")
    return complete_reviews


async def stream_peer_reviews(question: str, responses: dict[str, ModelResponse], callback):
    """
    Stream peer reviews, calling callback as each reviewer completes.

    Args:
        question: The original question
        responses: Dict of model responses from stage 1
        callback: async function(reviewer, review, all_reviews) called as each completes
    """
    logger.info("=" * 60)
    logger.info("STAGE 2: PEER REVIEWS (STREAMING)")
    logger.info(f"Each of {len(responses)} models reviewing {len(responses)-1} responses")
    logger.info("=" * 60)

    async def review_with_id(reviewer):
        anonymized, mapping = _anonymize_responses(responses, exclude_model=reviewer)
        logger.info(f"[{get_display_name(reviewer)}] Reviewing responses: {list(mapping.keys())}")
        review = await _get_single_review(reviewer, question, anonymized, mapping)
        return reviewer, review

    tasks = [asyncio.create_task(review_with_id(reviewer)) for reviewer in responses.keys()]
    reviews = {}

    for coro in asyncio.as_completed(tasks):
        reviewer, review = await coro
        reviews[reviewer] = review
        await callback(reviewer, review, reviews)

    logger.info("Stage 2 complete.")
    return reviews


# --- Peer Review ---

def _anonymize_responses(responses: dict[str, ModelResponse], exclude_model: str) -> tuple[str, dict[str, str]]:
    """
    Create anonymized response text for peer review.

    Returns:
        Tuple of (formatted_text, mapping) where mapping is {"A": model_name, ...}
    """
    # Filter out the reviewer's own response
    other_responses = {k: v for k, v in responses.items() if k != exclude_model}

    # Create mapping: letter -> model name
    mapping = {}
    formatted_parts = []

    for i, (model, response) in enumerate(other_responses.items()):
        letter = chr(65 + i)  # A, B, C, ...
        mapping[letter] = model
        formatted_parts.append(f"=== RESPONSE {letter} ===\n{response.content}\n")

    return "\n".join(formatted_parts), mapping


async def _get_single_review(
    reviewer: str,
    question: str,
    anonymized_responses: str,
    response_mapping: dict[str, str],
) -> PeerReview:
    """Get peer review from a single model."""

    prompt = f"""You are reviewing responses from other council members on the following question.
Your task is to evaluate each response and rank them.

ORIGINAL QUESTION: {question}

RESPONSES TO REVIEW:
{anonymized_responses}

Please evaluate each response on three criteria (1-10 scale):
- ACCURACY: How factually correct is the response?
- INSIGHT: How deep and valuable are the insights?
- COMPLETENESS: How thoroughly does it address the question?

Provide your evaluation in this exact format:

RANKINGS:
Response A: Accuracy=X, Insight=X, Completeness=X
Response B: Accuracy=X, Insight=X, Completeness=X
[continue for all responses]

ANALYSIS:
[2-3 sentences maximum. Focus only on key differentiators between responses.]"""

    response = await _call_ollama(reviewer, prompt)

    # Parse rankings from response (basic parsing - could be more robust)
    rankings = []
    for letter in response_mapping.keys():
        rankings.append({
            "response": letter,
            "model": response_mapping[letter],  # Store actual model for later
        })

    return PeerReview(
        reviewer=reviewer,
        rankings=rankings,
        analysis=response.content,
    )


async def get_peer_reviews(
    question: str,
    responses: dict[str, ModelResponse],
) -> dict[str, PeerReview]:
    """
    Get peer reviews from all council members.
    Each model reviews the OTHER models' responses (anonymized).

    Args:
        question: The original question
        responses: Dict of model responses from stage 1

    Returns:
        Dict mapping reviewer model names to their reviews
    """
    logger.info("=" * 60)
    logger.info("STAGE 2: PEER REVIEWS")
    logger.info(f"Each of {len(responses)} models reviewing {len(responses)-1} responses")
    logger.info("=" * 60)

    tasks = []
    for reviewer in responses.keys():
        anonymized, mapping = _anonymize_responses(responses, exclude_model=reviewer)
        logger.info(f"[{get_display_name(reviewer)}] Reviewing responses: {list(mapping.keys())}")
        tasks.append(_get_single_review(reviewer, question, anonymized, mapping))

    results = await asyncio.gather(*tasks)

    reviews = {r.reviewer: r for r in results}
    logger.info("Stage 2 complete.")

    return reviews


# --- Consensus Calculation ---

import re
import statistics


def _parse_review_scores(analysis: str) -> dict[str, dict[str, int]]:
    """
    Parse scores from review analysis text.

    Returns dict like: {"A": {"accuracy": 8, "insight": 7, "completeness": 9}, ...}
    """
    scores = {}

    # Pattern: Response A: Accuracy=8, Insight=7, Completeness=9
    # Also handles variations like "Response A - Accuracy: 8"
    pattern = r'Response\s+([A-Z])[\s:=-]+.*?Accuracy[\s:=]+(\d+).*?Insight[\s:=]+(\d+).*?Completeness[\s:=]+(\d+)'

    for match in re.finditer(pattern, analysis, re.IGNORECASE | re.DOTALL):
        letter = match.group(1).upper()
        scores[letter] = {
            "accuracy": int(match.group(2)),
            "insight": int(match.group(3)),
            "completeness": int(match.group(4)),
        }

    return scores


def calculate_consensus(reviews: dict[str, PeerReview], responses: dict[str, ModelResponse]) -> ConsensusScore:
    """
    Calculate consensus score from peer reviews.

    Measures how much reviewers agree on the quality of responses.
    High variance in scores = low consensus (disagreement).
    Low variance = high consensus (agreement).
    """
    if len(reviews) < 2:
        return ConsensusScore(
            score=50.0,
            level="unknown",
            description="Need at least 2 reviews to calculate consensus",
            disagreement_areas=[]
        )

    # Parse all review scores
    all_scores = {}  # {reviewer: {response_letter: {criterion: score}}}
    response_to_model = {}  # Map response letters back to models

    for reviewer, review in reviews.items():
        parsed = _parse_review_scores(review.analysis)
        if parsed:
            all_scores[reviewer] = parsed
            # Build response-to-model mapping from rankings
            for ranking in review.rankings:
                if "response" in ranking and "model" in ranking:
                    response_to_model[ranking["response"]] = ranking["model"]

    if len(all_scores) < 2:
        return ConsensusScore(
            score=50.0,
            level="unknown",
            description="Could not parse enough review scores",
            disagreement_areas=[]
        )

    # Calculate variance for each response on each criterion
    variances = []
    disagreement_details = []

    # Get all response letters that were reviewed
    all_letters = set()
    for reviewer_scores in all_scores.values():
        all_letters.update(reviewer_scores.keys())

    for letter in sorted(all_letters):
        for criterion in ["accuracy", "insight", "completeness"]:
            # Collect scores from all reviewers for this response+criterion
            criterion_scores = []
            for reviewer, reviewer_scores in all_scores.items():
                if letter in reviewer_scores and criterion in reviewer_scores[letter]:
                    criterion_scores.append(reviewer_scores[letter][criterion])

            if len(criterion_scores) >= 2:
                variance = statistics.variance(criterion_scores)
                variances.append(variance)

                # Track significant disagreements (variance > 4 means scores differ by ~2+ points)
                if variance > 4:
                    model_name = response_to_model.get(letter, f"Response {letter}")
                    if model_name in responses:
                        model_name = get_display_name(model_name)
                    disagreement_details.append(
                        f"{criterion.capitalize()} of {model_name} (variance: {variance:.1f})"
                    )

    if not variances:
        return ConsensusScore(
            score=50.0,
            level="unknown",
            description="Insufficient data to calculate consensus",
            disagreement_areas=[]
        )

    # Average variance across all scores
    # Scale: 0 variance = perfect agreement, 20+ variance = major disagreement
    avg_variance = statistics.mean(variances)

    # Convert to 0-100 score (inverse: low variance = high score)
    # Max reasonable variance is ~20 (scores ranging 1-10)
    consensus_score = max(0, min(100, 100 - (avg_variance * 5)))

    # Determine level
    if consensus_score >= 75:
        level = "high"
        description = "Strong agreement among council members"
    elif consensus_score >= 50:
        level = "medium"
        description = "Moderate agreement with some differing perspectives"
    else:
        level = "low"
        description = "Significant disagreement - topic may be complex or subjective"

    # Add summary of disagreements
    if disagreement_details:
        description += f". Key disagreements: {len(disagreement_details)} areas"

    return ConsensusScore(
        score=round(consensus_score, 1),
        level=level,
        description=description,
        disagreement_areas=disagreement_details[:5]  # Top 5 disagreements
    )


# --- Chairman Synthesis ---

# Web search tool definition for the chairman
WEB_SEARCH_TOOL = {
    "name": "web_search",
    "description": "Search the web for current information to supplement council responses. Use this when you need to verify facts, get up-to-date information, or find information the council members may have missed.",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query to look up"
            }
        },
        "required": ["query"]
    }
}


async def _perform_web_search(query: str) -> str:
    """
    Perform a web search using a simple search API.
    Returns formatted search results.
    """
    import urllib.parse
    import urllib.request
    import json

    logger.info(f"[Chairman] Web search: {query}")

    try:
        # Use DuckDuckGo Instant Answer API (free, no key required)
        encoded_query = urllib.parse.quote(query)
        url = f"https://api.duckduckgo.com/?q={encoded_query}&format=json&no_html=1"

        req = urllib.request.Request(url, headers={'User-Agent': 'LLM-Council/1.0'})
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode())

        results = []

        # Abstract (main summary)
        if data.get("Abstract"):
            results.append(f"Summary: {data['Abstract']}")
            if data.get("AbstractSource"):
                results.append(f"Source: {data['AbstractSource']}")

        # Related topics
        if data.get("RelatedTopics"):
            results.append("\nRelated information:")
            for topic in data["RelatedTopics"][:5]:
                if isinstance(topic, dict) and topic.get("Text"):
                    results.append(f"- {topic['Text'][:200]}")

        if not results:
            return f"No results found for: {query}"

        return "\n".join(results)

    except Exception as e:
        logger.warning(f"Web search failed: {e}")
        return f"Web search failed: {e}"


async def get_chairman_synthesis(
    question: str,
    responses: dict[str, ModelResponse],
    reviews: dict[str, PeerReview],
    consensus: ConsensusScore | None = None,
    document_filename: str | None = None,
    enable_independent_thinking: bool = True,
    enable_web_search: bool = True,
) -> tuple[str, ResponseComposition]:
    """
    Get final synthesis from the chairman (Claude Opus) with independent thinking and web search.

    Args:
        question: The original question
        responses: All council member responses
        reviews: All peer reviews
        consensus: Optional consensus score to guide synthesis emphasis
        document_filename: Optional filename if responses were based on an uploaded document
        enable_independent_thinking: Allow chairman to add independent analysis
        enable_web_search: Allow chairman to search the web for additional info

    Returns:
        Tuple of (synthesis text, ResponseComposition)
    """
    logger.info("=" * 60)
    logger.info("STAGE 3: CHAIRMAN SYNTHESIS")
    logger.info(f"Chairman: {get_display_name(CHAIRMAN_MODEL)}")
    logger.info(f"Independent thinking: {'enabled' if enable_independent_thinking else 'disabled'}")
    logger.info(f"Web search: {'enabled' if enable_web_search else 'disabled'}")
    if document_filename:
        logger.info(f"Document analyzed: {document_filename}")
    if consensus:
        logger.info(f"Consensus level: {consensus.level} ({consensus.score})")
    logger.info("=" * 60)

    # Track composition
    web_searches_performed = []
    chairman_insights = []

    # Format responses with model names (no longer anonymous for chairman)
    responses_text = ""
    for model, response in responses.items():
        display_name = get_display_name(model)
        responses_text += f"=== {display_name} ({response.elapsed_seconds:.1f}s) ===\n"
        responses_text += f"{response.content}\n\n"

    # Format reviews
    reviews_text = ""
    for reviewer, review in reviews.items():
        display_name = get_display_name(reviewer)
        reviews_text += f"=== Review by {display_name} ===\n"
        reviews_text += f"{review.analysis}\n\n"

    # Build consensus context for the prompt
    consensus_context = ""
    disagreement_emphasis = ""

    if consensus:
        consensus_context = f"""
CONSENSUS ANALYSIS:
- Agreement Score: {consensus.score}/100 ({consensus.level} consensus)
- {consensus.description}
"""
        if consensus.disagreement_areas:
            consensus_context += "- Specific disagreements: " + "; ".join(consensus.disagreement_areas) + "\n"

        # When consensus is low, emphasize the importance of addressing disagreements
        if consensus.level == "low":
            disagreement_emphasis = """
IMPORTANT: The council shows LOW CONSENSUS on this topic. This indicates the question may be complex, subjective, or have multiple valid perspectives. Pay special attention to:
- Why different council members reached different conclusions
- What underlying assumptions or values might explain the disagreement
- Whether the disagreement reveals important nuances the user should know about
- Present multiple perspectives fairly rather than forcing artificial agreement
"""

    # Add document context if a document was analyzed
    document_context = ""
    if document_filename:
        document_context = f"""
NOTE: The council members analyzed an uploaded document "{document_filename}" to answer this question.
Their responses are based on the content of that document.
"""

    # Build independent thinking instructions
    independent_thinking_instructions = ""
    if enable_independent_thinking:
        independent_thinking_instructions = """
CHAIRMAN'S INDEPENDENT ANALYSIS:
As Chairman, you are not limited to just summarizing the council's responses. You should:
- Add your own expert insights and analysis that go beyond what the council provided
- Identify gaps in the council's responses and fill them with your own knowledge
- Correct any factual errors you notice, even if the council members agreed on them
- Provide additional context or nuances the council may have missed

When you add independent insights, clearly mark them with [CHAIRMAN'S INSIGHT] so users know this came from your analysis rather than the council.
"""

    # Build web search instructions
    web_search_instructions = ""
    if enable_web_search:
        web_search_instructions = """
WEB SEARCH CAPABILITY:
You have access to web search to verify facts or find current information. Use it when:
- Council responses conflict and you need to verify which is correct
- The question involves recent events or rapidly changing information
- You want to supplement the council's knowledge with additional sources

When you use web search, mark information from it with [WEB SOURCE] so users know its origin.
"""

    prompt = f"""You are the Chairman of an expert council. Your council members have deliberated on a question and reviewed each other's responses. Your task is to synthesize the best possible final answer.

ORIGINAL QUESTION:
{question}
{document_context}{consensus_context}
COUNCIL MEMBER RESPONSES:
{responses_text}

PEER REVIEWS:
{reviews_text}
{disagreement_emphasis}{independent_thinking_instructions}{web_search_instructions}
As Chairman, please provide:

1. SYNTHESIS: The best comprehensive answer, combining the strongest elements from all responses while correcting any errors noted in reviews.{' Add your own independent insights where valuable.' if enable_independent_thinking else ''}

2. KEY CONTRIBUTORS: Note which council members provided particularly valuable insights and what they contributed.

3. AREAS OF CONSENSUS: Where did the council agree?

4. AREAS OF DISAGREEMENT: Where did opinions differ, and how did you resolve this?{' Pay particular attention here given the low consensus score.' if consensus and consensus.level == 'low' else ''}

5. CHAIRMAN'S ADDITIONS: List any independent insights you added or web searches you performed to enhance the answer.

Please begin your synthesis:"""

    # Call Claude via Anthropic API
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logger.error("ANTHROPIC_API_KEY not set!")
        default_composition = ResponseComposition(
            council_contribution=100.0, chairman_independent=0.0, web_search_used=0.0,
            web_searches_performed=[], chairman_insights=[]
        )
        return "[Error: ANTHROPIC_API_KEY environment variable not set]", default_composition

    logger.info("Calling Claude Opus for synthesis...")
    start = time.perf_counter()

    try:
        client = Anthropic(api_key=api_key)

        # Build tools list if web search is enabled
        tools = [WEB_SEARCH_TOOL] if enable_web_search else []

        messages = [{"role": "user", "content": prompt}]
        synthesis = ""

        # Handle tool use loop (for web search)
        max_iterations = 5  # Limit web searches
        for iteration in range(max_iterations):
            if tools:
                response = await asyncio.to_thread(
                    client.messages.create,
                    model=CHAIRMAN_MODEL,
                    max_tokens=4096,
                    messages=messages,
                    tools=tools,
                )
            else:
                response = await asyncio.to_thread(
                    client.messages.create,
                    model=CHAIRMAN_MODEL,
                    max_tokens=4096,
                    messages=messages,
                )

            # Check if we need to handle tool use
            if response.stop_reason == "tool_use":
                # Process tool calls
                tool_results = []
                assistant_content = response.content

                for block in response.content:
                    if block.type == "tool_use" and block.name == "web_search":
                        query = block.input.get("query", "")
                        web_searches_performed.append(query)
                        search_result = await _perform_web_search(query)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": search_result
                        })

                # Add assistant response and tool results to messages
                messages.append({"role": "assistant", "content": assistant_content})
                messages.append({"role": "user", "content": tool_results})
            else:
                # No more tool use, extract final synthesis
                for block in response.content:
                    if hasattr(block, 'text'):
                        synthesis = block.text
                        break
                break

        elapsed = time.perf_counter() - start
        logger.info(f"Chairman synthesis complete in {elapsed:.1f}s ({len(synthesis)} chars)")
        logger.info(f"Web searches performed: {len(web_searches_performed)}")

        # Parse chairman insights from synthesis
        import re
        insight_matches = re.findall(r'\[CHAIRMAN\'S INSIGHT\][:\s]*([^\[]+?)(?=\[|$)', synthesis, re.IGNORECASE | re.DOTALL)
        chairman_insights = [m.strip()[:200] for m in insight_matches if m.strip()][:5]

        # Calculate composition percentages
        # Base calculation on presence of markers and searches
        has_web_content = len(web_searches_performed) > 0 or "[WEB SOURCE]" in synthesis.upper()
        has_chairman_insights = len(chairman_insights) > 0 or "[CHAIRMAN'S INSIGHT]" in synthesis.upper()

        if has_web_content and has_chairman_insights:
            council_contribution = 60.0
            chairman_independent = 25.0
            web_search_used = 15.0
        elif has_chairman_insights:
            council_contribution = 70.0
            chairman_independent = 30.0
            web_search_used = 0.0
        elif has_web_content:
            council_contribution = 75.0
            chairman_independent = 10.0
            web_search_used = 15.0
        else:
            council_contribution = 90.0
            chairman_independent = 10.0
            web_search_used = 0.0

        composition = ResponseComposition(
            council_contribution=council_contribution,
            chairman_independent=chairman_independent,
            web_search_used=web_search_used,
            web_searches_performed=web_searches_performed,
            chairman_insights=chairman_insights
        )

        return synthesis, composition

    except Exception as e:
        elapsed = time.perf_counter() - start
        logger.error(f"Chairman synthesis failed after {elapsed:.1f}s: {e}")
        default_composition = ResponseComposition(
            council_contribution=100.0, chairman_independent=0.0, web_search_used=0.0,
            web_searches_performed=[], chairman_insights=[]
        )
        return f"[Error during synthesis: {e}]", default_composition


async def get_chairman_early_synthesis(
    question: str,
    responses: dict[str, ModelResponse],
    document_filename: str | None = None,
    enable_independent_thinking: bool = True,
    enable_web_search: bool = True,
) -> tuple[str, ResponseComposition]:
    """
    Get chairman synthesis without peer reviews (for early termination).

    This is used when the user stops deliberation early and wants
    the chairman to synthesize based only on available responses.

    Args:
        question: The original question
        responses: Available council member responses (may be partial)
        document_filename: Optional filename if responses were based on an uploaded document
        enable_independent_thinking: Allow chairman to add independent analysis
        enable_web_search: Allow chairman to search the web for additional info

    Returns:
        Tuple of (synthesis text, ResponseComposition)
    """
    logger.info("=" * 60)
    logger.info("EARLY SYNTHESIS (Deliberation stopped by user)")
    logger.info(f"Chairman: {get_display_name(CHAIRMAN_MODEL)}")
    logger.info(f"Available responses: {len(responses)}")
    logger.info("=" * 60)

    if not responses:
        default_composition = ResponseComposition(
            council_contribution=0.0, chairman_independent=100.0, web_search_used=0.0,
            web_searches_performed=[], chairman_insights=["No council responses available"]
        )
        return "No council responses were collected before deliberation was stopped.", default_composition

    # Track composition
    web_searches_performed = []
    chairman_insights = []

    # Format available responses
    responses_text = ""
    for model, response in responses.items():
        display_name = get_display_name(model)
        responses_text += f"=== {display_name} ({response.elapsed_seconds:.1f}s) ===\n"
        responses_text += f"{response.content}\n\n"

    # Add document context if a document was analyzed
    document_context = ""
    if document_filename:
        document_context = f"""
NOTE: The council members analyzed an uploaded document "{document_filename}" to answer this question.
"""

    # Build independent thinking instructions
    independent_thinking_instructions = ""
    if enable_independent_thinking:
        independent_thinking_instructions = """
CHAIRMAN'S INDEPENDENT ANALYSIS:
Since the full deliberation was stopped early, your independent analysis is especially important.
- Add your own expert insights to supplement the limited council input
- Fill in any gaps that might exist due to the incomplete deliberation
- Provide a comprehensive answer using your own knowledge

When you add independent insights, clearly mark them with [CHAIRMAN'S INSIGHT].
"""

    # Build web search instructions
    web_search_instructions = ""
    if enable_web_search:
        web_search_instructions = """
WEB SEARCH CAPABILITY:
You have access to web search. Given the limited council input, you may want to search
for additional information to provide a more complete answer.
Mark information from web search with [WEB SOURCE].
"""

    prompt = f"""You are the Chairman of an expert council. The deliberation was stopped early by the user, so you have limited input from the council. Please synthesize the best possible answer using the available responses and your own expertise.

ORIGINAL QUESTION:
{question}
{document_context}
AVAILABLE COUNCIL RESPONSES ({len(responses)} of expected):
{responses_text}

NOTE: Peer reviews were not completed due to early termination.
{independent_thinking_instructions}{web_search_instructions}
As Chairman, please provide:

1. SYNTHESIS: The best answer possible given the available responses. Supplement with your own knowledge where the council input is limited.

2. AVAILABLE CONTRIBUTORS: Note which council members provided input and what they contributed.

3. CHAIRMAN'S ADDITIONS: Since the deliberation was limited, explain what insights you've added independently.

Please begin your synthesis:"""

    # Call Claude via Anthropic API
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logger.error("ANTHROPIC_API_KEY not set!")
        default_composition = ResponseComposition(
            council_contribution=100.0, chairman_independent=0.0, web_search_used=0.0,
            web_searches_performed=[], chairman_insights=[]
        )
        return "[Error: ANTHROPIC_API_KEY environment variable not set]", default_composition

    logger.info("Calling Claude Opus for early synthesis...")
    start = time.perf_counter()

    try:
        client = Anthropic(api_key=api_key)

        # Build tools list if web search is enabled
        tools = [WEB_SEARCH_TOOL] if enable_web_search else []

        messages = [{"role": "user", "content": prompt}]
        synthesis = ""

        # Handle tool use loop (for web search)
        max_iterations = 5
        for iteration in range(max_iterations):
            if tools:
                response = await asyncio.to_thread(
                    client.messages.create,
                    model=CHAIRMAN_MODEL,
                    max_tokens=4096,
                    messages=messages,
                    tools=tools,
                )
            else:
                response = await asyncio.to_thread(
                    client.messages.create,
                    model=CHAIRMAN_MODEL,
                    max_tokens=4096,
                    messages=messages,
                )

            if response.stop_reason == "tool_use":
                tool_results = []
                assistant_content = response.content

                for block in response.content:
                    if block.type == "tool_use" and block.name == "web_search":
                        query = block.input.get("query", "")
                        web_searches_performed.append(query)
                        search_result = await _perform_web_search(query)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": search_result
                        })

                messages.append({"role": "assistant", "content": assistant_content})
                messages.append({"role": "user", "content": tool_results})
            else:
                for block in response.content:
                    if hasattr(block, 'text'):
                        synthesis = block.text
                        break
                break

        elapsed = time.perf_counter() - start
        logger.info(f"Early synthesis complete in {elapsed:.1f}s ({len(synthesis)} chars)")

        # Parse chairman insights
        import re
        insight_matches = re.findall(r'\[CHAIRMAN\'S INSIGHT\][:\s]*([^\[]+?)(?=\[|$)', synthesis, re.IGNORECASE | re.DOTALL)
        chairman_insights = [m.strip()[:200] for m in insight_matches if m.strip()][:5]

        # For early synthesis, chairman contributes more
        has_web_content = len(web_searches_performed) > 0 or "[WEB SOURCE]" in synthesis.upper()
        has_chairman_insights = len(chairman_insights) > 0 or "[CHAIRMAN'S INSIGHT]" in synthesis.upper()

        # Scale council contribution based on how many responses we have
        base_council = min(50.0, len(responses) * 15.0)  # ~15% per response, max 50%

        if has_web_content and has_chairman_insights:
            council_contribution = base_council
            chairman_independent = 70.0 - base_council
            web_search_used = 30.0
        elif has_chairman_insights:
            council_contribution = base_council
            chairman_independent = 100.0 - base_council
            web_search_used = 0.0
        elif has_web_content:
            council_contribution = base_council
            chairman_independent = 50.0 - base_council
            web_search_used = 50.0
        else:
            council_contribution = base_council + 20.0
            chairman_independent = 80.0 - base_council
            web_search_used = 0.0

        composition = ResponseComposition(
            council_contribution=council_contribution,
            chairman_independent=chairman_independent,
            web_search_used=web_search_used,
            web_searches_performed=web_searches_performed,
            chairman_insights=chairman_insights
        )

        return synthesis, composition

    except Exception as e:
        elapsed = time.perf_counter() - start
        logger.error(f"Early synthesis failed after {elapsed:.1f}s: {e}")
        default_composition = ResponseComposition(
            council_contribution=100.0, chairman_independent=0.0, web_search_used=0.0,
            web_searches_performed=[], chairman_insights=[]
        )
        return f"[Error during synthesis: {e}]", default_composition


# --- Main Orchestration ---

async def run_council(question: str, models: list[str] | None = None) -> CouncilResult:
    """
    Run a complete council session.

    Stages:
    1. Get initial responses from all models (parallel)
    2. Get peer reviews (each model reviews others, parallel)
    3. Chairman synthesizes final answer

    Args:
        question: The question for the council to deliberate
        models: Optional list of models (defaults to COUNCIL_MODELS)

    Returns:
        CouncilResult with all responses, reviews, and synthesis
    """
    logger.info("#" * 60)
    logger.info("LLM COUNCIL SESSION STARTING")
    logger.info("#" * 60)

    total_start = time.perf_counter()

    # Stage 1: Initial responses
    responses = await get_initial_responses(question, models)

    # Stage 2: Peer reviews
    reviews = await get_peer_reviews(question, responses)

    # Calculate consensus
    consensus = calculate_consensus(reviews, responses)

    # Stage 3: Chairman synthesis
    synthesis, composition = await get_chairman_synthesis(question, responses, reviews, consensus)

    total_elapsed = time.perf_counter() - total_start

    logger.info("#" * 60)
    logger.info(f"COUNCIL SESSION COMPLETE in {total_elapsed:.1f}s")
    logger.info("#" * 60)

    return CouncilResult(
        question=question,
        responses=responses,
        reviews=reviews,
        synthesis=synthesis,
        total_elapsed=total_elapsed,
        consensus=consensus,
        composition=composition,
    )


# --- CLI Testing ---

async def _test():
    """Quick test of the council."""
    question = "What are the key differences between Python and Rust, and when should you use each?"
    result = await run_council(question)

    print("\n" + "=" * 60)
    print("FINAL SYNTHESIS")
    print("=" * 60)
    print(result.synthesis)


if __name__ == "__main__":
    asyncio.run(_test())
