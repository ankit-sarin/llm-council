# Module 5: Anonymous peer review and consensus

### Teaching Arc
- **Metaphor:** **Blind grant review.** Three researchers each submit a proposal. The reviewers see all three proposals — but with the names removed (Proposal A, B, C). Each reviewer scores each *other* proposal on three criteria. Then a panel chair looks at how aligned the reviewers were. If they all gave similar scores, there's strong consensus. If their scores are all over the place, the topic is genuinely contested.
- **Opening hook:** "After Stage 1 finishes, each model has written its answer. Then something strange happens — every model gets to read everyone *else's* answers, but with the names stripped off, and is asked to grade them. The math on those grades becomes the **consensus score** you see at the top of the final answer."
- **Key insight:** The council fights bias with two tricks: (1) **anonymization** — reviewers see "Response A/B/C" instead of "the Llama answer," so they can't favor their own team; (2) **variance-as-disagreement** — if reviewers disagree wildly on a score, that's a genuine signal that the topic is contested. Low variance = high consensus.
- **"Why should I care?":** Most AI apps give you a confident answer with no idea whether it's actually trustworthy. This module shows the engineering pattern for *quantifying* AI confidence by cross-checking multiple models. You can request this exact pattern from AI assistants when you want a calibrated answer instead of a confident-but-maybe-wrong one.

### Code Snippets (pre-extracted)

**Snippet A — Anonymizing responses for the reviewer** (`council.py` lines 593-612):

```python
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
```

**Snippet B — The review prompt each reviewer sees** (`council.py` lines 487-508):

```python
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
```

**Snippet C — The regex that parses scores from free-form review text** (`council.py` lines 707-727):

```python
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
```

**Snippet D — The consensus math: variance across reviewers** (`council.py` lines 826-870):

```python
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
                disagreement_details.append(
                    f"{criterion.capitalize()} of {model_name} (variance: {variance:.1f})"
                )

# Average variance across all scores
avg_variance = statistics.mean(variances)

# Convert to 0-100 score (inverse: low variance = high score)
# Max reasonable variance is ~20 (scores ranging 1-10)
consensus_score = max(0, min(100, 100 - (avg_variance * 5)))
```

### Screens (5)
1. **The blind grant review** — set up the metaphor. Two-panel visual: WITH names (biased) vs. WITHOUT names (fair).
2. **Hiding the names** — Code↔English of Snippet A. English emphasizes "drop the reviewer's own answer (no self-grading), assign letter labels A/B/C, keep a private mapping so we can map back later."
3. **What we ask the reviewer to do** — Code↔English of Snippet B (the review prompt). English explains the 3 criteria (accuracy/insight/completeness, 1-10 each) and *why* we ask for a strict format (so we can parse it).
4. **The actual conversation, animated** — **Group Chat Animation**: 3 models reviewing each other. Sample messages:
   - Engine: "Here are responses A, B, C. Score each on Accuracy, Insight, Completeness."
   - Qwen: "Response A: Accuracy=8, Insight=7, Completeness=9. Response B: Accuracy=6, Insight=8, Completeness=7."
   - Gemma: "Response A: Accuracy=8, Insight=7, Completeness=8. Response B: Accuracy=7, Insight=8, Completeness=8."
   - Mistral: "Response A: Accuracy=2, Insight=3, Completeness=4."
   - Engine: "Two reviewers agree closely on A (~8). One reviewer scored A way lower (2-4). High variance on A's accuracy — flagging disagreement."
5. **Turning scores into a 0–100 consensus score + quiz** — Code↔English of Snippet D (the variance math). English: "For each thing we scored, collect all the scores reviewers gave it. Measure how spread out they are. Average the spread. Convert to a friendly 0–100 number (low spread = high consensus)." Add a callout that variance is the standard way statisticians measure disagreement.

### Interactive Elements
- [x] **Code↔English translation** — Snippet A (anonymization).
- [x] **Code↔English translation #2** — Snippet D (the variance → consensus math).
- [x] **Group Chat Animation** — at least 6-8 messages, as described in Screen 4. Actors: Engine, Qwen, Gemma, Mistral. Use chat-window with id `chat-module5`.
- [x] **Quiz** — 4 scenario questions:
  - Q1: "Why are responses anonymized before review?" (Correct: "So reviewers can't favor their own team or recognize a model they know. Removing names forces them to judge content, not source. This is the same reason scientific peer review is double-blind." Wrong: "To save bandwidth." / "Because the models can't read names.")
  - Q2: "All three reviewers gave Response A's Accuracy a 9. They gave Response B's Accuracy 3, 7, and 10. Which has higher consensus on Accuracy and why?" (Correct: "Response A — same score from all reviewers means variance = 0 = perfect consensus on that criterion. Response B's scores are spread from 3 to 10, which is high variance, meaning the reviewers genuinely disagree about whether B is accurate." Tests the variance intuition.)
  - Q3: "The final consensus score is 32/100. The chairman should..." (Correct: "...emphasize the disagreement. Low consensus means the topic is genuinely contested, and the chairman is told to present multiple perspectives instead of forcing artificial agreement. There's a special instruction in the chairman prompt for this." Tests cross-module connection to Module 6.)
  - Q4: "You'd like to add a 4th criterion called 'Clarity' to the reviews. List the *two* places you'd need to change for the consensus math to actually use it." (Correct: "(1) the review prompt — add Clarity to the criteria list and the rankings format; (2) the score parser regex `_parse_review_scores` — extend the pattern to capture Clarity. The variance math iterates `['accuracy', 'insight', 'completeness']` — you'd need to add Clarity there too. So strictly: three places." Tests their ability to trace data through the pipeline.)
- [x] **Callout** — "Aha! moment": "**Variance is how engineers measure disagreement.** It's the squared average distance from the mean. Two reviewers giving 5 and 5 → variance 0. Two reviewers giving 1 and 10 → variance 40.5. Anytime you see a 'confidence' or 'agreement' number in software, there's a variance (or its cousin, standard deviation) under the hood."
- [x] **Callout** — "Why the prompt is so strict about format": "Asking the LLM to output in *exactly* `Accuracy=X, Insight=X, Completeness=X` format makes the response *parseable*. Free-text praise is useless to a regex. This is a basic but critical pattern: when you want structured output from an LLM, *tell it exactly what shape you want*."
- [x] **Glossary tooltips** — anonymization, peer review, regex, parse, variance, statistics, criterion/criteria, prompt engineering, structured output, blind review, mean (average), confidence.

### Reference Files to Read
- `references/interactive-elements.md` → Code↔English Translation Blocks, Group Chat Animation, Multiple-Choice Quizzes, Callout Boxes, Glossary Tooltips
- `references/content-philosophy.md` → all of it
- `references/gotchas.md` → all of it — especially the data-steps single-quote warning if you use any flow animation
- `references/design-system.md` → actor colors (--color-actor-1..4 for the chat avatars)

### Connections
- **Previous module:** Module 4 (VRAM batching) explained how the council manages compute. This module is what they *do* with that compute in Stage 2.
- **Next module:** Module 6 — The chairman's gavel. We've seen anonymized scoring and a consensus number. Module 6 is what Claude does with all of that: synthesize a final answer, optionally with web search.
- **Tone/style notes:**
  - Accent: teal `#2A7B9B`.
  - **The blind-grant-review metaphor lives only here.** Don't reuse "peer review" / "review" as a metaphor in M6 — there, the chairman *isn't* a peer.
  - It's OK to lean into the *fairness* angle — this is a nice place to talk about why bias matters in AI evaluation.
  - Use the **chat animation as the centerpiece**. This module's strongest moment is watching the models score each other.
