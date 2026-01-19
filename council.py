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
class CouncilResult:
    """Complete results from a council session."""
    question: str
    responses: dict[str, ModelResponse]
    reviews: dict[str, PeerReview]
    synthesis: str
    total_elapsed: float


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


async def _call_ollama_streaming(model: str, prompt: str, on_token) -> ModelResponse:
    """Call Ollama with token streaming, calling on_token for each chunk."""
    logger.info(f"[{get_display_name(model)}] Starting streaming generation...")
    start = time.perf_counter()
    content = ""

    try:
        def stream_sync():
            nonlocal content
            for chunk in ollama.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                stream=True
            ):
                token = chunk["message"]["content"]
                content += token
                return token, content, False
            return "", content, True

        # Stream in a thread, yielding tokens
        import queue
        token_queue = queue.Queue()

        def run_stream():
            try:
                for chunk in ollama.chat(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    stream=True
                ):
                    token = chunk["message"]["content"]
                    token_queue.put(("token", token))
                token_queue.put(("done", None))
            except Exception as e:
                token_queue.put(("error", str(e)))

        import threading
        thread = threading.Thread(target=run_stream)
        thread.start()

        while True:
            try:
                msg_type, data = token_queue.get(timeout=300)
                if msg_type == "done":
                    break
                elif msg_type == "error":
                    content = f"[Error: {data}]"
                    break
                elif msg_type == "token":
                    content += data
                    await on_token(model, content)
            except queue.Empty:
                break

        thread.join()
        elapsed = time.perf_counter() - start
        logger.info(f"[{get_display_name(model)}] Done in {elapsed:.1f}s ({len(content)} chars)")
        return ModelResponse(model=model, content=content, elapsed_seconds=elapsed)

    except Exception as e:
        elapsed = time.perf_counter() - start
        logger.error(f"[{get_display_name(model)}] Failed after {elapsed:.1f}s: {e}")
        return ModelResponse(model=model, content=f"[Error: {e}]", elapsed_seconds=elapsed)


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


async def stream_initial_responses_live(question: str, models: list[str], on_token_callback, on_complete_callback):
    """
    Stream initial responses with live token updates.

    Args:
        question: The question to ask the council
        models: List of model names
        on_token_callback: async function(partial_responses: dict) called on each token
        on_complete_callback: async function(model, response, all_responses) called when model completes
    """
    logger.info("=" * 60)
    logger.info("STAGE 1: INITIAL RESPONSES (LIVE STREAMING)")
    logger.info(f"Question: {question[:100]}{'...' if len(question) > 100 else ''}")
    logger.info(f"Council members: {len(models)}")
    logger.info("=" * 60)

    prompt = f"""You are a member of an expert council deliberating on the following question.
Provide your best, most thoughtful answer. Be thorough but concise.

QUESTION: {question}

YOUR RESPONSE:"""

    # Track partial and complete responses
    partial_responses = {m: "" for m in models}
    complete_responses = {}
    start_times = {m: time.perf_counter() for m in models}

    # Lock for thread-safe updates
    import threading
    lock = threading.Lock()

    async def on_token(model, content):
        with lock:
            partial_responses[model] = content
        await on_token_callback(partial_responses, complete_responses)

    async def run_model(model):
        response = await _call_ollama_streaming(model, prompt, on_token)
        with lock:
            complete_responses[model] = response
            partial_responses[model] = response.content
        await on_complete_callback(model, response, complete_responses)
        return response

    # Run all models concurrently
    tasks = [asyncio.create_task(run_model(m)) for m in models]
    await asyncio.gather(*tasks)

    logger.info(f"Stage 1 complete. {len(complete_responses)} responses collected.")
    return complete_responses


async def stream_peer_reviews_live(question: str, responses: dict[str, ModelResponse], on_token_callback, on_complete_callback):
    """
    Stream peer reviews with live token updates.

    Args:
        question: The original question
        responses: Dict of model responses from stage 1
        on_token_callback: async function(partial_reviews: dict) called on each token
        on_complete_callback: async function(reviewer, review, all_reviews) called when review completes
    """
    logger.info("=" * 60)
    logger.info("STAGE 2: PEER REVIEWS (LIVE STREAMING)")
    logger.info(f"Each of {len(responses)} models reviewing {len(responses)-1} responses")
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
[Brief explanation of your rankings and what stood out about each response]"""

        async def on_token(model, content):
            with lock:
                partial_reviews[reviewer] = content
            await on_token_callback(partial_reviews, complete_reviews)

        response = await _call_ollama_streaming(reviewer, prompt, on_token)

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

    tasks = [asyncio.create_task(review_model(r)) for r in responses.keys()]
    await asyncio.gather(*tasks)

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
[Brief explanation of your rankings and what stood out about each response]"""

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


# --- Chairman Synthesis ---

async def get_chairman_synthesis(
    question: str,
    responses: dict[str, ModelResponse],
    reviews: dict[str, PeerReview],
) -> str:
    """
    Get final synthesis from the chairman (Claude Opus).

    Args:
        question: The original question
        responses: All council member responses
        reviews: All peer reviews

    Returns:
        Chairman's synthesized answer
    """
    logger.info("=" * 60)
    logger.info("STAGE 3: CHAIRMAN SYNTHESIS")
    logger.info(f"Chairman: {get_display_name(CHAIRMAN_MODEL)}")
    logger.info("=" * 60)

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

    prompt = f"""You are the Chairman of an expert council. Your council members have deliberated on a question and reviewed each other's responses. Your task is to synthesize the best possible final answer.

ORIGINAL QUESTION:
{question}

COUNCIL MEMBER RESPONSES:
{responses_text}

PEER REVIEWS:
{reviews_text}

As Chairman, please provide:

1. SYNTHESIS: The best comprehensive answer, combining the strongest elements from all responses while correcting any errors noted in reviews.

2. KEY CONTRIBUTORS: Note which council members provided particularly valuable insights and what they contributed.

3. AREAS OF CONSENSUS: Where did the council agree?

4. AREAS OF DISAGREEMENT: Where did opinions differ, and how did you resolve this?

Please begin your synthesis:"""

    # Call Claude via Anthropic API
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logger.error("ANTHROPIC_API_KEY not set!")
        return "[Error: ANTHROPIC_API_KEY environment variable not set]"

    logger.info("Calling Claude Opus for synthesis...")
    start = time.perf_counter()

    try:
        client = Anthropic(api_key=api_key)
        response = await asyncio.to_thread(
            client.messages.create,
            model=CHAIRMAN_MODEL,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        elapsed = time.perf_counter() - start
        synthesis = response.content[0].text
        logger.info(f"Chairman synthesis complete in {elapsed:.1f}s ({len(synthesis)} chars)")
        return synthesis

    except Exception as e:
        elapsed = time.perf_counter() - start
        logger.error(f"Chairman synthesis failed after {elapsed:.1f}s: {e}")
        return f"[Error during synthesis: {e}]"


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

    # Stage 3: Chairman synthesis
    synthesis = await get_chairman_synthesis(question, responses, reviews)

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
