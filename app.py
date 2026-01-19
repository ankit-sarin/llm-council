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
    MAX_CONCURRENT_VRAM_GB,
    get_display_name,
)
from council import (
    CouncilResult,
    ConsensusScore,
    ModelResponse,
    PeerReview,
    get_initial_responses,
    get_peer_reviews,
    get_chairman_synthesis,
    stream_initial_responses,
    stream_peer_reviews,
    stream_initial_responses_live,
    stream_peer_reviews_live,
    calculate_consensus,
)

import json
import os
from datetime import datetime
from pathlib import Path

# --- Session Storage ---

SESSIONS_DIR = Path(__file__).parent / "sessions"
SESSIONS_DIR.mkdir(exist_ok=True)


def save_session(
    question: str,
    responses: dict[str, "ModelResponse"],
    reviews: dict[str, "PeerReview"],
    synthesis: str,
    consensus: ConsensusScore,
    models: list[str],
    document_filename: str | None = None,
    document_char_count: int | None = None,
    document_preview: str | None = None
) -> dict:
    """
    Save a council session to JSON file.

    Args:
        question: The user's question
        responses: Dict of model responses
        reviews: Dict of peer reviews
        synthesis: Chairman's synthesis
        consensus: Consensus score object
        models: List of model IDs used
        document_filename: Optional filename if a document was uploaded
        document_char_count: Optional character count of the document
        document_preview: Optional first 1000 chars of document for reference
    """
    timestamp = datetime.now()
    session_id = timestamp.strftime("%Y%m%d_%H%M%S")

    session_data = {
        "id": session_id,
        "timestamp": timestamp.isoformat(),
        "question": question,
        "models": [get_display_name(m) for m in models],
        "responses": {
            get_display_name(model): {
                "content": resp.content,
                "elapsed_seconds": resp.elapsed_seconds
            }
            for model, resp in responses.items()
        },
        "reviews": {
            get_display_name(reviewer): {
                "rankings": review.rankings,
                "analysis": review.analysis
            }
            for reviewer, review in reviews.items()
        },
        "consensus": {
            "score": consensus.score,
            "level": consensus.level,
            "description": consensus.description,
            "disagreement_areas": consensus.disagreement_areas
        },
        "synthesis": synthesis
    }

    # Add document info if a file was uploaded
    if document_filename:
        session_data["document"] = {
            "filename": document_filename,
            "char_count": document_char_count,
            "preview": document_preview  # First 1000 chars for reference
        }

    filepath = SESSIONS_DIR / f"session_{session_id}.json"
    with open(filepath, "w") as f:
        json.dump(session_data, f, indent=2)

    return session_data


def list_sessions() -> list[dict]:
    """List all saved sessions (newest first)."""
    sessions = []
    for filepath in sorted(SESSIONS_DIR.glob("session_*.json"), reverse=True):
        try:
            with open(filepath) as f:
                data = json.load(f)
                sessions.append({
                    "id": data["id"],
                    "timestamp": data["timestamp"],
                    "question": data["question"][:100] + ("..." if len(data["question"]) > 100 else ""),
                    "consensus_score": data["consensus"]["score"],
                    "consensus_level": data["consensus"]["level"],
                    "models": data["models"],
                    "filepath": str(filepath)
                })
        except Exception:
            continue
    return sessions


def load_session(session_id: str) -> dict | None:
    """Load a specific session by ID."""
    filepath = SESSIONS_DIR / f"session_{session_id}.json"
    if filepath.exists():
        with open(filepath) as f:
            return json.load(f)
    return None


def export_session_to_markdown(session_id: str) -> str:
    """Export a session to Markdown format."""
    session = load_session(session_id)
    if not session:
        return ""

    # Add document info if present
    document_info = ""
    if session.get("document"):
        doc = session["document"]
        document_info = f"""
**Document Analyzed:** {doc['filename']} ({doc['char_count']:,} characters)
"""

    md = f"""# Council Session: {session['id']}

**Date:** {session['timestamp']}

**Question:** {session['question']}

**Models:** {', '.join(session['models'])}{document_info}

**Consensus:** {session['consensus']['level'].upper()} ({session['consensus']['score']}/100)
{session['consensus']['description']}

---

## Individual Responses

"""
    for model, resp in session['responses'].items():
        md += f"### {model}\n*Generated in {resp['elapsed_seconds']:.1f}s*\n\n{resp['content']}\n\n---\n\n"

    md += "## Peer Reviews\n\n"
    for reviewer, review in session['reviews'].items():
        md += f"### Review by {reviewer}\n\n{review['analysis']}\n\n---\n\n"

    md += f"""## Chairman's Synthesis

{session['synthesis']}

---

*Generated by LLM Council*
"""
    return md


# --- File Upload Configuration ---
# These constants control file upload limits

MAX_FILE_SIZE_MB = 10  # Maximum file size in megabytes
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024  # Convert to bytes
MAX_TEXT_CHARS = 150000  # Maximum characters to extract from a file
LONG_DOC_WARNING_CHARS = 100000  # Show warning if document exceeds this
GEMMA_EXCLUSION_CHARS = 30000  # Exclude Gemma 2 (8K context) if document exceeds this


# --- File Text Extraction ---
# This section handles extracting text from uploaded files (.txt, .md, .pdf)

def extract_text_from_file(file_path: str) -> tuple[str, str | None]:
    """
    Extract text content from an uploaded file.

    Supports three file types:
    - .txt: Plain text files, read directly
    - .md: Markdown files, read directly (preserves formatting)
    - .pdf: PDF files, uses pdfplumber for better table handling

    Args:
        file_path: Path to the uploaded file

    Returns:
        Tuple of (extracted_text, error_message)
        - If successful: (text, None)
        - If failed: ("", error_message)
    """
    import os

    # Check if file exists
    if not file_path or not os.path.exists(file_path):
        return "", "File not found."

    # Get file extension (lowercase for comparison)
    file_ext = os.path.splitext(file_path)[1].lower()

    # Check if file type is supported
    supported_types = ['.txt', '.md', '.pdf', '.docx']
    if file_ext not in supported_types:
        return "", f"Please upload a .txt, .md, .pdf, or .docx file. Got: {file_ext}"

    # Check file size
    file_size = os.path.getsize(file_path)
    if file_size > MAX_FILE_SIZE_BYTES:
        size_mb = file_size / (1024 * 1024)
        return "", f"File exceeds {MAX_FILE_SIZE_MB}MB limit. Your file: {size_mb:.1f}MB. Please use a smaller file."

    try:
        # Handle plain text files (.txt and .md)
        # These are simple - just read the contents directly
        if file_ext in ['.txt', '.md']:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()

        # Handle PDF files using pdfplumber
        # pdfplumber is better than PyMuPDF for tables
        elif file_ext == '.pdf':
            import pdfplumber

            text_parts = []  # Collect text from each page

            # Open the PDF and iterate through pages
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    # Extract text from this page
                    page_text = page.extract_text()

                    if page_text:
                        text_parts.append(page_text)

                    # Also try to extract tables and format them
                    # This helps preserve table structure
                    tables = page.extract_tables()
                    for table in tables:
                        if table:
                            # Convert table to text format
                            # Each row becomes a line with | separators
                            for row in table:
                                # Filter out None values and join with |
                                row_text = " | ".join(str(cell) if cell else "" for cell in row)
                                text_parts.append(row_text)

            # Join all pages with double newlines
            text = "\n\n".join(text_parts)

            # Check if we got any text (might be scanned/image PDF)
            if not text.strip():
                return "", "Could not extract text from this PDF. It may be scanned or image-based."

        # Handle Word documents (.docx) using python-docx
        elif file_ext == '.docx':
            from docx import Document

            text_parts = []  # Collect text from each paragraph

            # Open the Word document
            doc = Document(file_path)

            # Extract text from paragraphs
            for para in doc.paragraphs:
                if para.text.strip():
                    text_parts.append(para.text)

            # Also extract text from tables
            for table in doc.tables:
                for row in table.rows:
                    row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
                    if row_text:
                        text_parts.append(row_text)

            # Join all parts with newlines
            text = "\n\n".join(text_parts)

            if not text.strip():
                return "", "Could not extract text from this Word document."

        # Check if extraction produced any text
        if not text.strip():
            return "", "No text could be extracted from this file."

        # Truncate if text exceeds maximum
        if len(text) > MAX_TEXT_CHARS:
            text = text[:MAX_TEXT_CHARS]
            # Note: We'll show a warning in the UI, not an error

        return text, None

    except UnicodeDecodeError:
        return "", "Could not read file. It may use an unsupported character encoding."
    except Exception as e:
        return "", f"Error reading file: {str(e)}"


def get_file_info(file_path: str) -> dict:
    """
    Get information about an uploaded file for display in the UI.

    Args:
        file_path: Path to the uploaded file

    Returns:
        Dictionary with file information:
        - filename: Original filename
        - size_bytes: File size in bytes
        - size_display: Human-readable size (e.g., "1.5 MB")
        - text: Extracted text content
        - char_count: Number of characters extracted
        - preview: First 500 characters for preview
        - is_long: True if document exceeds warning threshold
        - is_truncated: True if text was truncated
        - error: Error message if extraction failed
    """
    import os

    if not file_path:
        return {"error": "No file provided"}

    # Get basic file info
    filename = os.path.basename(file_path)
    file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0

    # Format file size for display
    if file_size < 1024:
        size_display = f"{file_size} bytes"
    elif file_size < 1024 * 1024:
        size_display = f"{file_size / 1024:.1f} KB"
    else:
        size_display = f"{file_size / (1024 * 1024):.1f} MB"

    # Extract text
    text, error = extract_text_from_file(file_path)

    if error:
        return {
            "filename": filename,
            "size_bytes": file_size,
            "size_display": size_display,
            "error": error
        }

    # Calculate character count and check thresholds
    char_count = len(text)
    is_long = char_count > LONG_DOC_WARNING_CHARS
    is_truncated = char_count >= MAX_TEXT_CHARS
    exclude_gemma = char_count > GEMMA_EXCLUSION_CHARS  # Gemma 2 has 8K context limit

    # Create preview (first 500 chars)
    preview = text[:500] + ("..." if len(text) > 500 else "")

    return {
        "filename": filename,
        "size_bytes": file_size,
        "size_display": size_display,
        "text": text,
        "char_count": char_count,
        "preview": preview,
        "is_long": is_long,
        "is_truncated": is_truncated,
        "exclude_gemma": exclude_gemma,  # True if doc exceeds 30K chars
        "error": None
    }


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


def format_consensus_meter(consensus: ConsensusScore | None) -> str:
    """Generate a visual consensus meter with score and interpretation."""
    if consensus is None:
        return "*Calculating consensus...*"

    if consensus.level == "unknown":
        return f"*{consensus.description}*"

    # Color based on level
    colors = {
        "high": ("#4CAF50", "#e8f5e9"),  # Green
        "medium": ("#FF9800", "#fff3e0"),  # Orange
        "low": ("#f44336", "#ffebee"),  # Red
    }
    fg_color, bg_color = colors.get(consensus.level, ("#9e9e9e", "#f5f5f5"))

    # Build the meter bar
    fill_pct = consensus.score
    level_emoji = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(consensus.level, "⚪")

    html = f'''
<div style="background: {bg_color}; border: 2px solid {fg_color}; border-radius: 12px; padding: 16px; margin: 8px 0;">
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
        <span style="font-weight: bold; color: #333;">Consensus Score</span>
        <span style="font-size: 1.2em; font-weight: bold; color: {fg_color};">{level_emoji} {consensus.score:.0f}/100</span>
    </div>

    <div style="background: #e0e0e0; border-radius: 8px; height: 12px; overflow: hidden; margin-bottom: 12px;">
        <div style="background: {fg_color}; height: 100%; width: {fill_pct}%; transition: width 0.3s ease;"></div>
    </div>

    <div style="font-size: 0.95em; color: #555;">
        <strong>{consensus.level.upper()} CONSENSUS:</strong> {consensus.description}
    </div>
'''

    # Show disagreement areas if any
    if consensus.disagreement_areas:
        html += '''
    <div style="margin-top: 12px; padding-top: 12px; border-top: 1px solid #ddd;">
        <strong style="color: #333;">Key Areas of Disagreement:</strong>
        <ul style="margin: 8px 0 0 0; padding-left: 20px; color: #666;">
'''
        for area in consensus.disagreement_areas:
            html += f'            <li>{area}</li>\n'
        html += '        </ul>\n    </div>\n'

    html += '</div>'
    return html


def calculate_agreement_summary(reviews: dict, responses: dict = None) -> str:
    """Generate agreement/disagreement summary from reviews using consensus calculation."""
    if not reviews or len(reviews) < 2:
        return "*Need at least 2 reviews to calculate consensus*"

    # Calculate actual consensus if we have responses
    if responses:
        consensus = calculate_consensus(reviews, responses)
        return format_consensus_meter(consensus)

    # Fallback if no responses provided
    num_reviewers = len(reviews)
    return f"**{num_reviewers} council members** participated in peer review."


# --- Main Council Runner ---

async def run_council_streaming(
    question: str,
    selected_models: list[str],
    update_queue,
    document_text: str | None = None,
    document_filename: str | None = None
):
    """
    Run the council with live token streaming updates via queue.
    Shows text as models generate it in side-by-side boxes.

    Args:
        question: The user's question
        selected_models: List of model display names to use
        update_queue: Async queue for streaming UI updates
        document_text: Optional text extracted from an uploaded document
        document_filename: Optional filename of the uploaded document
    """
    if not question.strip():
        await update_queue.put((
            "Please enter a question.",
            "", "", "",
            "No responses yet",
            "No reviews yet",
            "Awaiting synthesis...",
            "",
            ""  # consensus_display_tab1
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
            "",
            ""  # consensus_display_tab1
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
        "",
        ""  # consensus_display_tab1
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
            "",
            ""  # consensus_display_tab1
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
            "",
            ""  # consensus_display_tab1
        ))

    # Run Stage 1 with live streaming
    # Pass document_text if a file was uploaded
    responses = await stream_initial_responses_live(
        question, active_models, on_token_stage1, on_complete_stage1,
        document_text=document_text
    )
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
        "",
        ""  # consensus_display_tab1
    ))

    # Token callback for Stage 2
    async def on_token_stage2(partial_reviews, done_reviews):
        now = time_module.time()
        if now - last_update[0] < update_interval:
            return
        last_update[0] = now

        completed = len(done_reviews)
        total = len(reviewer_models)
        consensus_html = calculate_agreement_summary(done_reviews, responses) if len(done_reviews) >= 2 else ""

        await update_queue.put((
            f"**Stage 2/3:** Peer review ({completed}/{total} complete)...",
            "",
            timing_summary,
            "",
            format_responses_side_by_side({}, responses, active_models),
            format_reviews_side_by_side(partial_reviews, done_reviews, reviewer_models),
            "### ⏸️ Waiting for peer reviews...",
            consensus_html,
            consensus_html  # consensus_display_tab1
        ))

    # Complete callback for Stage 2
    async def on_complete_stage2(reviewer, review, all_done):
        complete_reviews[reviewer] = review
        completed = len(all_done)
        total = len(reviewer_models)
        consensus_html = calculate_agreement_summary(all_done, responses) if len(all_done) >= 2 else ""

        await update_queue.put((
            f"**Stage 2/3:** Peer review ({completed}/{total} complete)...",
            "",
            timing_summary,
            "",
            format_responses_side_by_side({}, responses, active_models),
            format_reviews_side_by_side({m: complete_reviews.get(m, type('', (), {'analysis': ''})()).analysis if m in complete_reviews else "" for m in reviewer_models}, complete_reviews, reviewer_models),
            "### ⏸️ Waiting for peer reviews...",
            consensus_html,
            consensus_html  # consensus_display_tab1
        ))

    # Run Stage 2 with live streaming
    reviews = await stream_peer_reviews_live(question, responses, on_token_stage2, on_complete_stage2)
    complete_reviews = reviews

    # --- Stage 3: Chairman Synthesis ---
    # Calculate consensus from reviews
    consensus = calculate_consensus(reviews, responses)
    consensus_display = format_consensus_meter(consensus)

    await update_queue.put((
        "**Stage 3/3:** Chairman synthesizing final answer...\n\n*Claude Opus 4.5 is analyzing all responses and reviews...*",
        "",
        timing_summary,
        "",
        format_responses_side_by_side({}, responses, active_models),
        format_reviews_side_by_side({}, reviews, reviewer_models),
        "### 🔄 Chairman is synthesizing...\n\n*Analyzing all responses and peer reviews to create the best answer.*",
        consensus_display,
        consensus_display  # consensus_display_tab1
    ))

    # Pass consensus and document info to chairman
    synthesis = await get_chairman_synthesis(
        question, responses, reviews, consensus,
        document_filename=document_filename
    )

    # Save session to JSON (including document info if present)
    session_data = save_session(
        question=question,
        responses=responses,
        reviews=reviews,
        synthesis=synthesis,
        consensus=consensus,
        models=active_models,
        document_filename=document_filename,
        document_char_count=len(document_text) if document_text else None,
        document_preview=document_text[:1000] if document_text else None
    )

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
        consensus_display,  # Show consensus meter with score
        consensus_display  # consensus_display_tab1
    ))


def run_council_generator(
    question: str,
    selected_models: list[str],
    document_text: str | None = None,
    document_filename: str | None = None,
    progress=gr.Progress()
):
    """
    Generator wrapper for streaming updates to all tabs.
    Yields updates as each model completes.

    Args:
        question: The user's question
        selected_models: List of model display names to use
        document_text: Optional extracted text from uploaded file
        document_filename: Optional filename of uploaded document
        progress: Gradio progress indicator (unused currently)
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
            await run_council_streaming(
                question, selected_models, async_queue,
                document_text=document_text,
                document_filename=document_filename
            )
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
                        # --- File Upload Section ---
                        # Optional: users can upload a document for the council to analyze
                        gr.Markdown("#### Upload a Document (Optional)")

                        # File upload component - accepts .txt, .md, .pdf, .docx
                        file_upload = gr.File(
                            label="Upload a file for the council to analyze",
                            file_types=[".txt", ".md", ".pdf", ".docx"],
                            file_count="single",  # Only one file at a time
                        )

                        # Display area for file information
                        # Shows filename, size, preview, and any warnings
                        file_info_display = gr.HTML(
                            value="<em>No file uploaded. The council will answer your question directly.</em>"
                        )

                        # Clear file button - allows user to remove uploaded file
                        clear_file_btn = gr.Button("🗑️ Clear File", size="sm", visible=False)

                        # Info note about limitations
                        gr.Markdown(
                            "*Note: Graphs and images cannot be analyzed. Tables may have formatting issues in PDFs and Word docs.*",
                            elem_classes="file-note"
                        )

                        gr.Markdown("---")  # Visual separator

                        # --- Question Input Section ---
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

                # Consensus meter for Tab 1
                gr.Markdown("### Council Consensus")
                consensus_display_tab1 = gr.HTML(value="<em>Submit a question to see consensus analysis</em>")

                # Hidden outputs for other tabs (updated together)
                hidden_spacer = gr.Markdown(value="", visible=False)

            # === TAB 2: Council Deliberation ===
            with gr.Tab("Council Deliberation"):
                gr.Markdown("### Detailed view of the deliberation process")

                with gr.Accordion("Stage 1: Individual Responses", open=True, elem_classes="stage-1"):
                    responses_display = gr.HTML(value="<em>No responses yet</em>")

                with gr.Accordion("Stage 2: Peer Reviews", open=True, elem_classes="stage-2"):
                    reviews_display = gr.HTML(value="<em>No reviews yet</em>")
                    gr.Markdown("#### Consensus Meter")
                    agreement_display = gr.HTML(value="<em>Submit a question to see consensus analysis</em>")

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

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### GPU Memory Management")
                        gr.Markdown("*Models run in VRAM-aware batches automatically. Use this to clear memory if needed.*")
                        clear_gpu_btn = gr.Button("🔄 Clear GPU Memory (Restart Ollama)", variant="secondary")
                        gpu_status = gr.Markdown(value="")

                    with gr.Column():
                        gr.Markdown("#### VRAM Configuration")
                        gr.Markdown(f"*Max concurrent VRAM: {MAX_CONCURRENT_VRAM_GB}GB*")
                        gr.Markdown("*Edit `config.py` to adjust VRAM limits.*")

                gr.Markdown("---")
                gr.Markdown(
                    """
                    #### Model Information
                    | Model | VRAM | Description |
                    |-------|------|-------------|
                    | Llama 3.3 70B | ~40GB | Meta's flagship model |
                    | Qwen 2.5 32B | ~20GB | Alibaba's multilingual model |
                    | Gemma 2 27B | ~17GB | Google's efficient model |
                    | DeepSeek R1 32B | ~20GB | Reasoning specialist |
                    | Mistral 7B | ~4GB | Fast European model |
                    | Claude Opus 4.5 | API | Chairman (Anthropic) |
                    """
                )

                def clear_gpu_memory():
                    """Restart Ollama to clear GPU memory."""
                    import subprocess
                    try:
                        # Try systemctl first (common on Linux)
                        result = subprocess.run(
                            ["sudo", "systemctl", "restart", "ollama"],
                            capture_output=True,
                            text=True,
                            timeout=30
                        )
                        if result.returncode == 0:
                            return "✅ Ollama restarted successfully. GPU memory cleared."

                        # Fallback: try killing and restarting ollama directly
                        subprocess.run(["pkill", "-f", "ollama"], capture_output=True, timeout=10)
                        import time
                        time.sleep(2)
                        subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        time.sleep(3)
                        return "✅ Ollama restarted. GPU memory should be cleared."

                    except subprocess.TimeoutExpired:
                        return "⚠️ Restart timed out. Try manually: `sudo systemctl restart ollama`"
                    except Exception as e:
                        return f"❌ Failed to restart Ollama: {e}\nTry manually: `sudo systemctl restart ollama`"

                clear_gpu_btn.click(
                    fn=clear_gpu_memory,
                    outputs=[gpu_status]
                )

                verbose_toggle.change(
                    fn=toggle_verbose,
                    inputs=[verbose_toggle],
                    outputs=[verbose_status]
                )

            # === TAB 4: History ===
            with gr.Tab("History"):
                gr.Markdown("### Past Council Sessions")
                gr.Markdown("Browse and export previous deliberations.")

                with gr.Row():
                    refresh_btn = gr.Button("🔄 Refresh", size="sm")
                    export_md_btn = gr.Button("📄 Export to Markdown", size="sm")

                session_dropdown = gr.Dropdown(
                    label="Select Session",
                    choices=[],
                    interactive=True,
                    elem_classes="session-dropdown"
                )

                with gr.Row():
                    session_info = gr.HTML(value="<em>Select a session to view details</em>")

                with gr.Accordion("Session Details", open=True):
                    session_question = gr.Markdown(value="")
                    session_consensus = gr.HTML(value="")
                    session_synthesis = gr.Markdown(value="")

                markdown_output = gr.Textbox(
                    label="Markdown Export (copy to clipboard)",
                    lines=10,
                    visible=False
                )

                # History tab callbacks
                def refresh_sessions():
                    sessions = list_sessions()
                    if not sessions:
                        return gr.update(choices=[], value=None), "<em>No sessions found</em>"
                    choices = [
                        (f"{s['timestamp'][:10]} | {s['consensus_level'].upper()} ({s['consensus_score']:.0f}) | {s['question'][:50]}...", s['id'])
                        for s in sessions
                    ]
                    return gr.update(choices=choices, value=None), f"<em>Found {len(sessions)} sessions</em>"

                def load_session_details(session_id):
                    if not session_id:
                        return "", "", ""
                    session = load_session(session_id)
                    if not session:
                        return "Session not found", "", ""

                    question_md = f"**Question:** {session['question']}\n\n**Models:** {', '.join(session['models'])}\n\n**Date:** {session['timestamp']}"

                    consensus = session['consensus']
                    colors = {"high": ("#4CAF50", "#e8f5e9"), "medium": ("#FF9800", "#fff3e0"), "low": ("#f44336", "#ffebee")}
                    fg, bg = colors.get(consensus['level'], ("#9e9e9e", "#f5f5f5"))
                    emoji = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(consensus['level'], "⚪")

                    consensus_html = f'''
                    <div style="background: {bg}; border: 2px solid {fg}; border-radius: 8px; padding: 12px; margin: 8px 0;">
                        <strong>{emoji} {consensus['level'].upper()} CONSENSUS</strong> ({consensus['score']}/100)<br/>
                        {consensus['description']}
                    </div>
                    '''

                    synthesis_md = f"### Chairman's Synthesis\n\n{session['synthesis']}"

                    return question_md, consensus_html, synthesis_md

                def export_markdown(session_id):
                    if not session_id:
                        return gr.update(visible=False, value="")
                    md = export_session_to_markdown(session_id)
                    return gr.update(visible=True, value=md)

                refresh_btn.click(
                    fn=refresh_sessions,
                    outputs=[session_dropdown, session_info]
                )

                session_dropdown.change(
                    fn=load_session_details,
                    inputs=[session_dropdown],
                    outputs=[session_question, session_consensus, session_synthesis]
                )

                export_md_btn.click(
                    fn=export_markdown,
                    inputs=[session_dropdown],
                    outputs=[markdown_output]
                )

                # Load sessions on startup
                app.load(
                    fn=refresh_sessions,
                    outputs=[session_dropdown, session_info]
                )

        # === File Upload Handlers ===
        # These functions handle file upload, display info, and clearing

        # State to store the extracted document text
        # gr.State persists data between interactions
        document_state = gr.State(value=None)

        def handle_file_upload(file):
            """
            Called when a user uploads a file.
            Extracts text and returns display info.

            Args:
                file: The uploaded file object from Gradio

            Returns:
                Tuple of (info_html, clear_button_visible, document_dict)
            """
            if file is None:
                # No file uploaded - reset to default state
                return (
                    "<em>No file uploaded. The council will answer your question directly.</em>",
                    gr.update(visible=False),  # Hide clear button
                    None  # Clear document state
                )

            # Get file info and extract text
            info = get_file_info(file.name)

            if info.get("error"):
                # Show error message in red
                error_html = f'''
                <div style="background: #ffebee; border: 2px solid #f44336; border-radius: 8px; padding: 12px;">
                    <strong style="color: #c62828;">❌ Error:</strong> {info["error"]}
                </div>
                '''
                return (
                    error_html,
                    gr.update(visible=True),  # Show clear button so user can try again
                    None  # No document to store
                )

            # Build success display with file info
            # Show warnings if document is long or truncated
            warnings = []
            if info.get("is_truncated"):
                warnings.append(f"⚠️ Document truncated to {MAX_TEXT_CHARS:,} characters")
            if info.get("is_long"):
                warnings.append("⚠️ Long document - some models may truncate")

            # Show Gemma exclusion note if document exceeds 30K chars
            gemma_note = ""
            if info.get("exclude_gemma"):
                gemma_note = '''
                <div style="background: #e3f2fd; border: 1px solid #2196F3; border-radius: 4px; padding: 8px; margin-top: 8px;">
                    ℹ️ <strong>Note:</strong> Gemma 2 excluded for this query due to document length (8K context limit)
                </div>
                '''

            warning_html = ""
            if warnings:
                warning_html = f'''
                <div style="background: #fff3e0; border: 1px solid #FF9800; border-radius: 4px; padding: 8px; margin-top: 8px;">
                    {" | ".join(warnings)}
                </div>
                '''

            # Format the info display
            info_html = f'''
            <div style="background: #e8f5e9; border: 2px solid #4CAF50; border-radius: 8px; padding: 12px;">
                <div style="margin-bottom: 8px;">
                    <strong>📄 {info["filename"]}</strong> ({info["size_display"]})
                    <span style="color: #666; margin-left: 8px;">{info["char_count"]:,} characters extracted</span>
                </div>
                <div style="background: #f5f5f5; border-radius: 4px; padding: 8px; font-family: monospace; font-size: 0.85em; max-height: 150px; overflow-y: auto; white-space: pre-wrap;">
{info["preview"]}
                </div>
                {warning_html}
                {gemma_note}
            </div>
            '''

            # Store document info in state for use during council session
            document_dict = {
                "filename": info["filename"],
                "char_count": info["char_count"],
                "text": info["text"],
                "preview": info["text"][:1000],  # First 1000 chars for session saving
                "exclude_gemma": info.get("exclude_gemma", False)  # Exclude Gemma if >30K chars
            }

            return (
                info_html,
                gr.update(visible=True),  # Show clear button
                document_dict  # Store in state
            )

        def clear_file():
            """
            Called when user clicks the Clear File button.
            Resets file upload and state.
            """
            return (
                None,  # Clear file upload component
                "<em>No file uploaded. The council will answer your question directly.</em>",
                gr.update(visible=False),  # Hide clear button
                None  # Clear document state
            )

        # Connect file upload events
        file_upload.change(
            fn=handle_file_upload,
            inputs=[file_upload],
            outputs=[file_info_display, clear_file_btn, document_state]
        )

        clear_file_btn.click(
            fn=clear_file,
            outputs=[file_upload, file_info_display, clear_file_btn, document_state]
        )

        # === Main Submit Action ===
        # Wrapper to collect checkbox values, file content, and call generator
        def collect_and_run(question, document, *checkbox_values):
            """
            Collects all inputs and runs the council session.

            Args:
                question: The user's question
                document: Dict with document info (or None if no file)
                *checkbox_values: Which models are selected
            """
            # Check if we need to exclude Gemma 2 due to document length
            exclude_gemma = document.get("exclude_gemma", False) if document else False

            # Build list of selected model display names
            selected = []
            for i, is_checked in enumerate(checkbox_values):
                if is_checked:
                    model_id = AVAILABLE_MODELS[i]
                    # Skip Gemma 2 if document exceeds 30K chars (8K context limit)
                    if exclude_gemma and "gemma2" in model_id.lower():
                        continue
                    selected.append(get_display_name(model_id))

            # Extract document text if present
            document_text = document.get("text") if document else None
            document_filename = document.get("filename") if document else None

            # Return generator results
            yield from run_council_generator(question, selected, document_text, document_filename)

        submit_btn.click(
            fn=collect_and_run,
            inputs=[question_input, document_state] + model_checkboxes,
            outputs=[
                status_display,
                final_answer,
                timing_display,
                hidden_spacer,
                responses_display,
                reviews_display,
                synthesis_display,
                agreement_display,
                consensus_display_tab1,
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
