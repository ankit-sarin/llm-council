"""
LLM Council - Gradio Interface

Multi-model debate system with peer review and chairman synthesis.
"""

import asyncio
import gradio as gr

from config import (
    AVAILABLE_MODELS,
    CHAIRMAN_MODEL,
    DEFAULT_ENABLED_MODELS,
    MODEL_DISPLAY_NAMES,
    get_display_name,
)
from council import (
    CouncilResult,
    get_initial_responses,
    get_peer_reviews,
    get_chairman_synthesis,
    stream_initial_responses,
    stream_peer_reviews,
)

# --- State ---

verbose_mode: bool = False


# --- Formatting Helpers ---

def format_response_tabs(responses: dict) -> str:
    """Format individual responses as markdown with tabs simulation."""
    if not responses:
        return "*No responses yet*"

    md = ""
    for model, response in responses.items():
        display_name = get_display_name(model)
        elapsed = response.elapsed_seconds
        md += f"### {display_name}\n"
        md += f"*Generated in {elapsed:.1f}s*\n\n"
        md += f"{response.content}\n\n"
        md += "---\n\n"

    return md


def format_reviews(reviews: dict, responses: dict) -> str:
    """Format peer reviews as markdown."""
    if not reviews:
        return "*No reviews yet*"

    md = ""
    for reviewer, review in reviews.items():
        reviewer_name = get_display_name(reviewer)
        md += f"### Review by {reviewer_name}\n\n"

        # Show which responses they reviewed
        reviewed = [get_display_name(m) for m in responses.keys() if m != reviewer]
        md += f"*Reviewed: {', '.join(reviewed)}*\n\n"

        md += f"{review.analysis}\n\n"
        md += "---\n\n"

    return md


def format_synthesis(synthesis: str) -> str:
    """Format chairman synthesis."""
    if not synthesis:
        return "*Awaiting chairman synthesis...*"

    chairman_name = get_display_name(CHAIRMAN_MODEL)
    return f"## {chairman_name}\n\n{synthesis}"


def calculate_agreement_summary(reviews: dict) -> str:
    """Generate agreement/disagreement summary from reviews."""
    if not reviews or len(reviews) < 2:
        return "*Need at least 2 reviews to calculate agreement*"

    # Simple agreement indicator based on number of reviewers
    num_reviewers = len(reviews)
    md = f"**{num_reviewers} council members** participated in peer review.\n\n"
    md += "See individual reviews above for detailed rankings and analysis."

    return md


# --- Main Council Runner ---

async def run_council_streaming(question: str, selected_models: list[str], update_queue):
    """
    Run the council with streaming updates via queue.
    Pushes updates to queue as each model completes.
    """
    if not question.strip():
        await update_queue.put((
            "Please enter a question.",
            "", "", "",
            "No responses yet",
            "No reviews yet",
            "Awaiting synthesis...",
            ""
        ))
        return

    # Map display names back to model IDs
    name_to_id = {v: k for k, v in MODEL_DISPLAY_NAMES.items()}
    active_models = [name_to_id[name] for name in selected_models if name in name_to_id]

    if not active_models:
        await update_queue.put((
            "No council members selected! Please select at least one model.",
            "", "", "",
            "No models selected",
            "No reviews possible",
            "Cannot synthesize without responses",
            ""
        ))
        return

    model_names = [get_display_name(m) for m in active_models]
    pending_models = set(active_models)
    responses = {}
    reviews = {}

    # --- Stage 1: Initial Responses ---
    # Show initial state with all models pending
    pending_status = "\n".join([f"- {get_display_name(m)}: ⏳ *working...*" for m in active_models])

    await update_queue.put((
        f"**Stage 1/3:** Gathering responses ({len(active_models)} models)...",
        "",
        f"**Status:**\n{pending_status}",
        "",
        "*Waiting for first response...*",
        "*Waiting for Stage 1 to complete...*",
        "*Waiting for peer reviews...*",
        ""
    ))

    # Callback for when each response completes
    async def on_response(model, response, all_responses):
        responses[model] = response
        pending_models.discard(model)

        # Build status with completed and pending
        status_lines = []
        for m in active_models:
            if m in responses:
                status_lines.append(f"- {get_display_name(m)}: ✅ {responses[m].elapsed_seconds:.1f}s")
            else:
                status_lines.append(f"- {get_display_name(m)}: ⏳ *working...*")

        timing_status = "\n".join(status_lines)
        completed = len(responses)
        total = len(active_models)

        await update_queue.put((
            f"**Stage 1/3:** Gathering responses ({completed}/{total} complete)...",
            "",
            f"**Status:**\n{timing_status}",
            "",
            format_response_tabs(responses),
            "*Waiting for Stage 1 to complete...*",
            "*Waiting for peer reviews...*",
            ""
        ))

    # Stream responses
    responses = await stream_initial_responses(question, active_models, on_response)

    # Format final timing
    timing_summary = "**Response Times:**\n"
    for model in active_models:
        if model in responses:
            timing_summary += f"- {get_display_name(model)}: ✅ {responses[model].elapsed_seconds:.1f}s\n"

    # --- Stage 2: Peer Reviews ---
    pending_reviewers = set(responses.keys())

    await update_queue.put((
        f"**Stage 2/3:** Peer review in progress ({len(responses)} reviewers)...",
        "",
        timing_summary,
        "",
        format_response_tabs(responses),
        "*Starting peer reviews...*",
        "*Waiting for peer reviews...*",
        ""
    ))

    # Callback for when each review completes
    async def on_review(reviewer, review, all_reviews):
        reviews[reviewer] = review
        pending_reviewers.discard(reviewer)

        completed = len(reviews)
        total = len(responses)

        # Build review status
        review_status_lines = []
        for m in responses.keys():
            if m in reviews:
                review_status_lines.append(f"- {get_display_name(m)}: ✅ review complete")
            else:
                review_status_lines.append(f"- {get_display_name(m)}: ⏳ *reviewing...*")

        review_timing = "\n".join(review_status_lines)

        await update_queue.put((
            f"**Stage 2/3:** Peer review ({completed}/{total} complete)...\n\n{review_timing}",
            "",
            timing_summary,
            "",
            format_response_tabs(responses),
            format_reviews(reviews, responses),
            "*Waiting for all reviews...*",
            calculate_agreement_summary(reviews) if len(reviews) >= 2 else ""
        ))

    # Stream reviews
    reviews = await stream_peer_reviews(question, responses, on_review)

    # --- Stage 3: Chairman Synthesis ---
    await update_queue.put((
        "**Stage 3/3:** Chairman synthesizing final answer...\n\n*Claude Opus 4.5 is analyzing all responses and reviews...*",
        "",
        timing_summary,
        "",
        format_response_tabs(responses),
        format_reviews(reviews, responses),
        "*Chairman is reviewing all responses and rankings...*",
        calculate_agreement_summary(reviews)
    ))

    synthesis = await get_chairman_synthesis(question, responses, reviews)

    # Final results
    final_status = f"**Council session complete!**\n\nActive members: {', '.join(model_names)}"

    await update_queue.put((
        final_status,
        synthesis,
        timing_summary,
        "",
        format_response_tabs(responses),
        format_reviews(reviews, responses),
        format_synthesis(synthesis),
        calculate_agreement_summary(reviews)
    ))


def run_council_generator(question: str, selected_models: list[str], progress=gr.Progress()):
    """
    Generator wrapper for streaming updates to all tabs.
    Yields updates as each model completes.
    """
    import queue
    import threading

    # Thread-safe queue for updates
    result_queue = queue.Queue()

    def run_async():
        async def run_with_queue():
            # Create async queue
            async_queue = asyncio.Queue()

            # Task to transfer from async queue to sync queue
            async def transfer():
                while True:
                    item = await async_queue.get()
                    if item is None:
                        break
                    result_queue.put(("update", item))
                result_queue.put(("done", None))

            # Run council and transfer concurrently
            transfer_task = asyncio.create_task(transfer())
            await run_council_streaming(question, selected_models, async_queue)
            await async_queue.put(None)  # Signal completion
            await transfer_task

        asyncio.run(run_with_queue())

    # Start async runner in thread
    thread = threading.Thread(target=run_async)
    thread.start()

    # Yield updates as they come
    while True:
        try:
            msg_type, data = result_queue.get(timeout=600)  # 10 min timeout
            if msg_type == "done":
                break
            yield data
        except queue.Empty:
            break

    thread.join()


# --- Settings Callbacks ---

def toggle_verbose(value: bool) -> str:
    """Toggle verbose mode."""
    global verbose_mode
    verbose_mode = value
    return f"Verbose mode: {'ON' if value else 'OFF'}"


# --- Build the Interface ---

def create_app():
    """Create the Gradio application."""

    with gr.Blocks(title="LLM Council") as app:

        gr.Markdown(
            """
            # LLM Council
            ### Multi-model deliberation with peer review and chairman synthesis
            """,
            elem_classes="council-header"
        )

        with gr.Tabs():

            # === TAB 1: Ask the Council ===
            with gr.Tab("Ask the Council"):
                with gr.Row():
                    with gr.Column(scale=2):
                        question_input = gr.Textbox(
                            label="Your Question",
                            placeholder="Enter a question for the council to deliberate...",
                            lines=3,
                        )

                        # Model selection checkboxes
                        model_choices = [get_display_name(m) for m in AVAILABLE_MODELS]
                        default_selected = [get_display_name(m) for m in DEFAULT_ENABLED_MODELS]

                        model_select = gr.CheckboxGroup(
                            choices=model_choices,
                            value=default_selected,
                            label="Select Council Members",
                            info="Choose which models will deliberate on your question",
                        )

                        submit_btn = gr.Button("Submit to Council", variant="primary", size="lg")

                    with gr.Column(scale=1):
                        status_display = gr.Markdown(
                            value="Ready to convene the council.",
                            elem_classes="stage-indicator"
                        )
                        timing_display = gr.Markdown(value="")

                gr.Markdown("### Chairman's Final Answer")
                final_answer = gr.Markdown(
                    value="*Submit a question to see the council's synthesized answer.*"
                )

                # Hidden outputs for other tabs (updated together)
                hidden_spacer = gr.Markdown(value="", visible=False)

            # === TAB 2: Council Deliberation ===
            with gr.Tab("Council Deliberation"):
                gr.Markdown("### Detailed view of the deliberation process")

                with gr.Accordion("Stage 1: Individual Responses", open=True):
                    responses_display = gr.Markdown(value="*No responses yet*")

                with gr.Accordion("Stage 2: Peer Reviews", open=True):
                    reviews_display = gr.Markdown(value="*No reviews yet*")
                    gr.Markdown("#### Agreement Summary")
                    agreement_display = gr.Markdown(value="*Submit a question to see agreement analysis*")

                with gr.Accordion("Stage 3: Chairman's Analysis", open=True):
                    synthesis_display = gr.Markdown(value="*Awaiting synthesis...*")

            # === TAB 3: Settings ===
            with gr.Tab("Settings"):
                gr.Markdown("### Council Configuration")

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### Chairman")
                        gr.Markdown(f"**{get_display_name(CHAIRMAN_MODEL)}**")
                        gr.Markdown("*The chairman synthesizes the final answer from all council deliberations.*")

                    with gr.Column():
                        gr.Markdown("#### Display Options")

                        verbose_toggle = gr.Checkbox(
                            label="Verbose Mode",
                            value=False,
                            info="Show additional detail in responses"
                        )
                        verbose_status = gr.Markdown(value="Verbose mode: OFF")

                gr.Markdown("---")
                gr.Markdown(
                    """
                    #### Model Information
                    | Model | Size | Description |
                    |-------|------|-------------|
                    | Llama 3.3 70B | ~40GB | Meta's flagship model |
                    | Qwen 2.5 32B | ~20GB | Alibaba's multilingual model |
                    | Gemma 2 27B | ~17GB | Google's efficient model |
                    | DeepSeek R1 32B | ~20GB | Reasoning specialist |
                    | Mistral 7B | ~4GB | Fast European model |
                    | Claude Opus 4.5 | API | Chairman (Anthropic) |
                    """
                )

                verbose_toggle.change(
                    fn=toggle_verbose,
                    inputs=[verbose_toggle],
                    outputs=[verbose_status]
                )

        # === Main Submit Action ===
        # Using generator for streaming updates to all tabs
        submit_btn.click(
            fn=run_council_generator,
            inputs=[question_input, model_select],
            outputs=[
                status_display,
                final_answer,
                timing_display,
                hidden_spacer,
                responses_display,
                reviews_display,
                synthesis_display,
                agreement_display,
            ],
        )

    return app


# --- Entry Point ---

if __name__ == "__main__":
    app = create_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=False,
        show_error=True,
        theme=gr.themes.Soft(),
        css="""
        .council-header { text-align: center; margin-bottom: 20px; }
        .stage-indicator { font-size: 1.1em; padding: 10px; border-radius: 8px; background: #f0f0f0; }
        """,
    )
