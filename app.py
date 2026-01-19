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
    stream_initial_responses_live,
    stream_peer_reviews_live,
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


def format_responses_side_by_side(partial_responses: dict, complete_responses: dict, models: list) -> str:
    """Format responses in a side-by-side grid layout with live streaming text."""
    import html as html_module

    if not partial_responses and not complete_responses:
        return "<em>Waiting for responses...</em>"

    if not models:
        return "<em>No models selected</em>"

    # Build HTML grid for side-by-side display
    num_models = len(models)
    cols = min(num_models, 3)  # Max 3 columns

    html = f'<div style="display: grid; grid-template-columns: repeat({cols}, 1fr); gap: 16px;">'

    for model in models:
        display_name = html_module.escape(get_display_name(model))
        is_complete = model in complete_responses

        # Get content
        if is_complete:
            content = complete_responses[model].content
            elapsed = complete_responses[model].elapsed_seconds
            status = f"✅ {elapsed:.1f}s"
            border_color = "#4CAF50"
        else:
            content = partial_responses.get(model, "")
            status = "⏳ generating..."
            border_color = "#FF9800"

        # Escape HTML in content and truncate if too long
        content = html_module.escape(content) if content else ""
        display_content = content if len(content) < 2000 else "..." + content[-1800:]

        html += f'''
        <div style="border: 2px solid {border_color}; border-radius: 8px; padding: 12px; background: #fafafa; min-height: 200px; max-height: 400px; overflow-y: auto;">
            <div style="font-weight: bold; color: #333; border-bottom: 1px solid #ddd; padding-bottom: 8px; margin-bottom: 8px;">
                {display_name} <span style="font-weight: normal; color: {'#4CAF50' if is_complete else '#FF9800'};">{status}</span>
            </div>
            <div style="font-size: 0.9em; white-space: pre-wrap; word-wrap: break-word;">{display_content or "<em>Starting...</em>"}</div>
        </div>
        '''

    html += '</div>'
    return html


def format_reviews_side_by_side(partial_reviews: dict, complete_reviews: dict, models: list) -> str:
    """Format reviews in a side-by-side grid layout with live streaming text."""
    import html as html_module

    if not partial_reviews and not complete_reviews:
        return "<em>Waiting for reviews...</em>"

    if not models:
        return "<em>No reviewers</em>"

    num_models = len(models)
    cols = min(num_models, 3)

    html = f'<div style="display: grid; grid-template-columns: repeat({cols}, 1fr); gap: 16px;">'

    for model in models:
        display_name = html_module.escape(get_display_name(model))
        is_complete = model in complete_reviews

        if is_complete:
            content = complete_reviews[model].analysis
            status = "✅ complete"
            border_color = "#4CAF50"
        else:
            content = partial_reviews.get(model, "")
            status = "⏳ reviewing..."
            border_color = "#FF9800"

        # Escape HTML in content and truncate if too long
        content = html_module.escape(content) if content else ""
        display_content = content if len(content) < 1500 else "..." + content[-1300:]

        html += f'''
        <div style="border: 2px solid {border_color}; border-radius: 8px; padding: 12px; background: #fff8f0; min-height: 150px; max-height: 350px; overflow-y: auto;">
            <div style="font-weight: bold; color: #333; border-bottom: 1px solid #ddd; padding-bottom: 8px; margin-bottom: 8px;">
                {display_name} <span style="font-weight: normal; color: {'#4CAF50' if is_complete else '#FF9800'};">{status}</span>
            </div>
            <div style="font-size: 0.9em; white-space: pre-wrap; word-wrap: break-word;">{display_content or "<em>Starting review...</em>"}</div>
        </div>
        '''

    html += '</div>'
    return html


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
    """Format chairman synthesis for Tab 2 (full details)."""
    if not synthesis:
        return "*Awaiting chairman synthesis...*"

    chairman_name = get_display_name(CHAIRMAN_MODEL)
    return f"## {chairman_name}\n\n{synthesis}"


def format_synthesis_brief(synthesis: str) -> str:
    """Format chairman synthesis for Tab 1 (only the main answer, no meta-analysis)."""
    if not synthesis:
        return "*Awaiting chairman synthesis...*"

    # Extract just the SYNTHESIS section, skip KEY CONTRIBUTORS, CONSENSUS, DISAGREEMENT
    lines = synthesis.split('\n')
    brief_lines = []
    in_synthesis = False
    skip_sections = ['KEY CONTRIBUTORS', 'AREAS OF CONSENSUS', 'AREAS OF DISAGREEMENT',
                     '2.', '3.', '4.']

    for line in lines:
        # Check if we hit a section to skip
        should_skip = any(skip in line.upper() for skip in ['KEY CONTRIBUTOR', 'AREAS OF CONSENSUS', 'AREAS OF DISAGREEMENT'])
        if should_skip or (line.strip().startswith(('2.', '3.', '4.')) and any(x in line.upper() for x in ['KEY', 'AREA', 'CONSENSUS', 'DISAGREEMENT'])):
            break
        brief_lines.append(line)

    return '\n'.join(brief_lines).strip()


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
    Run the council with live token streaming updates via queue.
    Shows text as models generate it in side-by-side boxes.
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
    complete_responses = {}
    complete_reviews = {}

    # Throttle updates to avoid overwhelming the UI
    import time as time_module
    last_update = [0]
    update_interval = 0.3  # Update UI every 300ms max

    # --- Stage 1: Initial Responses with live streaming ---
    await update_queue.put((
        f"**Stage 1/3:** Generating responses ({len(active_models)} models)...",
        "",
        "",
        "",
        format_responses_side_by_side({m: "" for m in active_models}, {}, active_models),
        "### ⏸️ Waiting for Stage 1...\n\n*Peer reviews will begin after all responses are complete.*",
        "### ⏸️ Waiting for responses and reviews...\n\n*Chairman will synthesize after peer reviews.*",
        ""
    ))

    # Token callback - update UI with partial responses
    async def on_token_stage1(partial_responses, done_responses):
        now = time_module.time()
        if now - last_update[0] < update_interval:
            return  # Throttle updates
        last_update[0] = now

        completed = len(done_responses)
        total = len(active_models)
        status = f"**Stage 1/3:** Generating responses ({completed}/{total} complete)..."

        await update_queue.put((
            status,
            "",
            "",
            "",
            format_responses_side_by_side(partial_responses, done_responses, active_models),
            "### ⏸️ Waiting for Stage 1...\n\n*Peer reviews will begin after all responses are complete.*",
            "### ⏸️ Waiting for responses and reviews...",
            ""
        ))

    # Complete callback
    async def on_complete_stage1(model, response, all_done):
        complete_responses[model] = response
        completed = len(all_done)
        total = len(active_models)

        await update_queue.put((
            f"**Stage 1/3:** Generating responses ({completed}/{total} complete)...",
            "",
            "",
            "",
            format_responses_side_by_side({m: complete_responses.get(m, type('', (), {'content': ''})()).content if m in complete_responses else "" for m in active_models}, complete_responses, active_models),
            "### ⏸️ Waiting for Stage 1...",
            "### ⏸️ Waiting for responses and reviews...",
            ""
        ))

    # Run Stage 1 with live streaming
    responses = await stream_initial_responses_live(question, active_models, on_token_stage1, on_complete_stage1)
    complete_responses = responses

    # Format timing summary
    timing_summary = "**Response Times:**\n"
    for model in active_models:
        if model in responses:
            timing_summary += f"- {get_display_name(model)}: ✅ {responses[model].elapsed_seconds:.1f}s\n"

    # --- Stage 2: Peer Reviews with live streaming ---
    reviewer_models = list(responses.keys())

    await update_queue.put((
        f"**Stage 2/3:** Peer review in progress ({len(responses)} reviewers)...",
        "",
        timing_summary,
        "",
        format_responses_side_by_side({}, responses, active_models),
        format_reviews_side_by_side({m: "" for m in reviewer_models}, {}, reviewer_models),
        "### ⏸️ Waiting for peer reviews...\n\n*Chairman synthesis will begin after all reviews are done.*",
        ""
    ))

    # Token callback for Stage 2
    async def on_token_stage2(partial_reviews, done_reviews):
        now = time_module.time()
        if now - last_update[0] < update_interval:
            return
        last_update[0] = now

        completed = len(done_reviews)
        total = len(reviewer_models)

        await update_queue.put((
            f"**Stage 2/3:** Peer review ({completed}/{total} complete)...",
            "",
            timing_summary,
            "",
            format_responses_side_by_side({}, responses, active_models),
            format_reviews_side_by_side(partial_reviews, done_reviews, reviewer_models),
            "### ⏸️ Waiting for peer reviews...",
            calculate_agreement_summary(done_reviews) if len(done_reviews) >= 2 else ""
        ))

    # Complete callback for Stage 2
    async def on_complete_stage2(reviewer, review, all_done):
        complete_reviews[reviewer] = review
        completed = len(all_done)
        total = len(reviewer_models)

        await update_queue.put((
            f"**Stage 2/3:** Peer review ({completed}/{total} complete)...",
            "",
            timing_summary,
            "",
            format_responses_side_by_side({}, responses, active_models),
            format_reviews_side_by_side({m: complete_reviews.get(m, type('', (), {'analysis': ''})()).analysis if m in complete_reviews else "" for m in reviewer_models}, complete_reviews, reviewer_models),
            "### ⏸️ Waiting for peer reviews...",
            calculate_agreement_summary(all_done) if len(all_done) >= 2 else ""
        ))

    # Run Stage 2 with live streaming
    reviews = await stream_peer_reviews_live(question, responses, on_token_stage2, on_complete_stage2)
    complete_reviews = reviews

    # --- Stage 3: Chairman Synthesis ---
    await update_queue.put((
        "**Stage 3/3:** Chairman synthesizing final answer...\n\n*Claude Opus 4.5 is analyzing all responses and reviews...*",
        "",
        timing_summary,
        "",
        format_responses_side_by_side({}, responses, active_models),
        format_reviews_side_by_side({}, reviews, reviewer_models),
        "### 🔄 Chairman is synthesizing...\n\n*Analyzing all responses and peer reviews to create the best answer.*",
        calculate_agreement_summary(reviews)
    ))

    synthesis = await get_chairman_synthesis(question, responses, reviews)

    # Final results
    final_status = f"**Council session complete!**\n\nActive members: {', '.join(model_names)}"

    await update_queue.put((
        final_status,
        format_synthesis_brief(synthesis),  # Tab 1: brief synthesis only
        timing_summary,
        "",
        format_responses_side_by_side({}, responses, active_models),
        format_reviews_side_by_side({}, reviews, reviewer_models),
        format_synthesis(synthesis),  # Tab 2: full synthesis with all sections
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
                            elem_classes="question-input"
                        )

                        # Model selection - vertical checkboxes with clear labels
                        gr.Markdown("#### Select Council Members")
                        model_checkboxes = []
                        for model_id in AVAILABLE_MODELS:
                            display_name = get_display_name(model_id)
                            is_default = model_id in DEFAULT_ENABLED_MODELS
                            cb = gr.Checkbox(
                                label=display_name,
                                value=is_default,
                                elem_classes="model-checkbox"
                            )
                            model_checkboxes.append(cb)

                        submit_btn = gr.Button("Submit to Council", variant="primary", size="lg")

                    with gr.Column(scale=1):
                        status_display = gr.Markdown(
                            value="Ready to convene the council.",
                            elem_classes="stage-indicator"
                        )
                        timing_display = gr.Markdown(value="")

                gr.Markdown("### Chairman's Final Answer", elem_classes="final-answer-header")
                final_answer = gr.Markdown(
                    value="*Submit a question to see the council's synthesized answer.*",
                    elem_classes="final-answer"
                )

                # Hidden outputs for other tabs (updated together)
                hidden_spacer = gr.Markdown(value="", visible=False)

            # === TAB 2: Council Deliberation ===
            with gr.Tab("Council Deliberation"):
                gr.Markdown("### Detailed view of the deliberation process")

                with gr.Accordion("Stage 1: Individual Responses", open=True, elem_classes="stage-1"):
                    responses_display = gr.HTML(value="<em>No responses yet</em>")

                with gr.Accordion("Stage 2: Peer Reviews", open=True, elem_classes="stage-2"):
                    reviews_display = gr.HTML(value="<em>No reviews yet</em>")
                    gr.Markdown("#### Agreement Summary")
                    agreement_display = gr.Markdown(value="*Submit a question to see agreement analysis*")

                with gr.Accordion("Stage 3: Chairman's Analysis", open=True, elem_classes="stage-3"):
                    synthesis_display = gr.Markdown(value="*Awaiting synthesis...*", elem_classes="stage-3-content")

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
        # Wrapper to collect checkbox values and call generator
        def collect_and_run(question, *checkbox_values):
            # Build list of selected model display names
            selected = []
            for i, is_checked in enumerate(checkbox_values):
                if is_checked:
                    model_id = AVAILABLE_MODELS[i]
                    selected.append(get_display_name(model_id))
            # Return generator results
            yield from run_council_generator(question, selected)

        submit_btn.click(
            fn=collect_and_run,
            inputs=[question_input] + model_checkboxes,
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

        /* Model checkboxes styling */
        .model-checkbox {
            padding: 8px 12px;
            margin: 4px 0;
            border: 1px solid #e0e0e0;
            border-radius: 6px;
            background: #fafafa;
        }
        .model-checkbox:hover {
            background: #f0f0f0;
        }

        /* Stage 1 - Blue theme */
        .stage-1 {
            border-left: 4px solid #2196F3 !important;
            background: linear-gradient(to right, #e3f2fd, transparent) !important;
        }
        .stage-1-content {
            border-left: 2px solid #2196F3;
            padding-left: 12px;
            margin-left: 8px;
        }

        /* Stage 2 - Orange theme */
        .stage-2 {
            border-left: 4px solid #FF9800 !important;
            background: linear-gradient(to right, #fff3e0, transparent) !important;
        }
        .stage-2-content {
            border-left: 2px solid #FF9800;
            padding-left: 12px;
            margin-left: 8px;
        }

        /* Stage 3 - Green theme */
        .stage-3 {
            border-left: 4px solid #4CAF50 !important;
            background: linear-gradient(to right, #e8f5e9, transparent) !important;
        }
        .stage-3-content {
            border-left: 2px solid #4CAF50;
            padding-left: 12px;
            margin-left: 8px;
        }

        /* Chairman's Final Answer - Purple/Gold theme */
        .final-answer-header {
            color: #6A1B9A;
        }
        .final-answer {
            border: 2px solid #9C27B0;
            border-radius: 12px;
            padding: 20px;
            background: linear-gradient(135deg, #f3e5f5 0%, #fff8e1 100%);
            box-shadow: 0 4px 6px rgba(156, 39, 176, 0.1);
        }

        /* Question Input - Purple/Gold theme */
        .question-input {
            border: 2px solid #9C27B0 !important;
            border-radius: 12px !important;
            background: linear-gradient(135deg, #f3e5f5 0%, #fff8e1 100%) !important;
            box-shadow: 0 4px 6px rgba(156, 39, 176, 0.1) !important;
        }
        .question-input textarea {
            background: transparent !important;
        }
        .question-input label {
            color: #6A1B9A !important;
            font-weight: bold !important;
        }
        """,
    )
