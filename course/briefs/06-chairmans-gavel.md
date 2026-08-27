# Module 6: The chairman's gavel — synthesis with web search

### Teaching Arc
- **Metaphor:** **A Supreme Court chief justice writing the majority opinion.** The justices below (the council models) have all weighed in; their clerks (the peer reviews) have flagged strengths and weaknesses; the chief justice (Claude) reads all of it and writes the final opinion. The chief is allowed to: agree with parts, override factual mistakes, *add their own reasoning* the lower court missed, and even consult **outside sources** (a law clerk fetching precedents = a **web search**). Whatever they add, they mark clearly.
- **Opening hook:** "When you read the final answer at the top of the page, you might see little tags like **[CHAIRMAN'S INSIGHT]** or **[WEB SOURCE]** sprinkled through it. Those aren't decorations — they're an honesty signal: 'this part didn't come from the council, it came from me, or from a web search.'"
- **Key insight:** The chairman isn't a passive summarizer — it's an active editor with three super-powers: (1) **synthesize** the council's responses, (2) **add independent insights** with `[CHAIRMAN'S INSIGHT]` tags, (3) **call a web search tool** with `[WEB SOURCE]` tags. The output is tracked as a **composition** — what % came from the council vs. the chairman vs. the web. That breakdown is the meter you see in the UI.
- **"Why should I care?":** Every modern AI app eventually wants two things: (a) a *more capable* model on top of cheaper/faster ones, (b) the model to be able to *use tools* (web search, calculator, your database). This module shows the cleanest end-to-end example of *tool use* and *attribution* in one short pipeline. You'll be able to ask for "tool use with attribution tags" by name when you direct AI coding tools.

### Code Snippets (pre-extracted)

**Snippet A — The web search tool, declared as a JSON schema** (`council.py` lines 901-915):

```python
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
```

**Snippet B — The chairman's marching orders** (`council.py` lines 1050-1059 — independent thinking) and (`council.py` lines 1064-1072 — web search):

```python
# Independent thinking instructions
"""
CHAIRMAN'S INDEPENDENT ANALYSIS:
As Chairman, you are not limited to just summarizing the council's responses. You should:
- Add your own expert insights and analysis that go beyond what the council provided
- Identify gaps in the council's responses and fill them with your own knowledge
- Correct any factual errors you notice, even if the council members agreed on them
- Provide additional context or nuances the council may have missed

When you add independent insights, clearly mark them with [CHAIRMAN'S INSIGHT] so users know this came from your analysis rather than the council.
"""

# Web search instructions
"""
WEB SEARCH CAPABILITY:
You have access to web search to verify facts or find current information. Use it when:
- Council responses conflict and you need to verify which is correct
- The question involves recent events or rapidly changing information
- You want to supplement the council's knowledge with additional sources

When you use web search, mark information from it with [WEB SOURCE] so users know its origin.
"""
```

**Snippet C — Special handling when consensus is low** (`council.py` lines 1030-1037):

```python
# When consensus is low, emphasize the importance of addressing disagreements
if consensus.level == "low":
    disagreement_emphasis = """
IMPORTANT: The council shows LOW CONSENSUS on this topic. This indicates the question may be complex, subjective, or have multiple valid perspectives. Pay special attention to:
- Why different council members reached different conclusions
- What underlying assumptions or values might explain the disagreement
- Whether the disagreement reveals important nuances the user should know about
- Present multiple perspectives fairly rather than forcing artificial agreement
"""
```

**Snippet D — The 5-part synthesis structure** (`council.py` lines 1085-1097):

```python
prompt = f"""You are the Chairman of an expert council. Your council members have deliberated on a question and reviewed each other's responses. Your task is to synthesize the best possible final answer.

ORIGINAL QUESTION:
{question}
...

As Chairman, please provide:

1. SYNTHESIS: The best comprehensive answer, combining the strongest elements from all responses while correcting any errors noted in reviews.

2. KEY CONTRIBUTORS: Note which council members provided particularly valuable insights and what they contributed.

3. AREAS OF CONSENSUS: Where did the council agree?

4. AREAS OF DISAGREEMENT: Where did opinions differ, and how did you resolve this?

5. CHAIRMAN'S ADDITIONS: List any independent insights you added or web searches you performed to enhance the answer.

Please begin your synthesis:"""
```

**Snippet E — How "composition" is tracked as a dataclass** (`council.py` lines 59-66):

```python
@dataclass
class ResponseComposition:
    """Tracks the source composition of the chairman's synthesis."""
    council_contribution: float  # 0-100, percentage from council model responses
    chairman_independent: float  # 0-100, percentage from chairman's own analysis
    web_search_used: float  # 0-100, percentage from web search
    web_searches_performed: list[str]  # List of search queries performed
    chairman_insights: list[str]  # Key insights added by chairman independently
```

### Screens (6)
1. **The chief justice writes the opinion** — set up the metaphor. A small Pattern Cards row: Council Members → Peer Reviews → Consensus Score → Chairman. Establish that Claude is *not* one of the council — it's the synthesizer.
2. **Why a different model on top?** — short text + callout. Three reasons: Claude is generally stronger; we get diversity from the council and *quality* from the chairman; if Claude were on the council it would dominate and we'd lose diversity.
3. **The marching orders** — Code↔English of Snippet B (both blocks side by side). Emphasize the *attribution rule*: "if you add it, tag it." This is the honesty mechanism.
4. **What a tool definition looks like** — Code↔English of Snippet A (the web_search tool schema). Explain *tool use* / *function calling*: we hand the model a JSON description of a function; the model decides when to call it; we run the call and feed the result back; the model continues. Show the loop:
   - User → Chairman with prompt + tool list
   - Chairman: "I want to call `web_search(query='...')`"
   - Engine runs the search, returns results
   - Chairman: continues writing, marks `[WEB SOURCE]`
   Use a **flow animation** for this loop.
5. **Adapting to low consensus** — Code↔English of Snippet C. English explains the conditional: when the consensus score from Module 5 is "low," the prompt grows a special section telling Claude to *not* force agreement. Connect back to Module 5's variance math.
6. **Composition tracking + quiz** — Code↔English of Snippet E. Explain the meter the user sees: 3 numbers that sum to 100. Then quiz.

### Interactive Elements
- [x] **Code↔English translation** — Snippet A (web_search tool schema).
- [x] **Code↔English translation #2** — Snippet B (independent thinking + web search instructions).
- [x] **Code↔English translation #3** — Snippet E (ResponseComposition dataclass).
- [x] **Data Flow Animation** — the tool-use loop. Actors: User → Chairman (Claude) → Web Search Tool → Search Results → Chairman. Steps (mind the single-quote rule — avoid apostrophes in labels):
  1. `{"highlight":"flow-actor-1","label":"Engine sends prompt with web_search tool defined"}`
  2. `{"highlight":"flow-actor-2","label":"Chairman starts writing the synthesis","packet":true,"from":"actor-1","to":"actor-2"}`
  3. `{"highlight":"flow-actor-2","label":"Chairman decides to verify a fact"}`
  4. `{"highlight":"flow-actor-3","label":"Chairman calls web_search","packet":true,"from":"actor-2","to":"actor-3"}`
  5. `{"highlight":"flow-actor-3","label":"Search returns results"}`
  6. `{"highlight":"flow-actor-2","label":"Chairman continues with WEB SOURCE tag","packet":true,"from":"actor-3","to":"actor-2"}`
  7. `{"highlight":"flow-actor-2","label":"Final synthesis is returned"}`
- [x] **Quiz** — 4 scenario questions:
  - Q1: "You read a final answer and there are NO `[CHAIRMAN'S INSIGHT]` or `[WEB SOURCE]` tags. What's the most likely conclusion?" (Correct: "The council's responses were sufficient — Claude didn't feel the need to add independent analysis or look anything up. That's a healthy outcome, not a bug." Wrong: "Web search is broken." / "Claude is lazy.")
  - Q2: "Why give the chairman web search but not the council members?" (Correct: "(a) The local models on the council are running on-device and don't have a search tool; (b) the *point* of the council is to capture each model's internal knowledge — adding search would dilute that signal; (c) Anthropic's tool-use API supports it cleanly. Search at the synthesis step verifies and supplements; search during the council would homogenize." Tests architecture reasoning.)
  - Q3: "Your friend is building a customer-support bot and asks: 'How do I make my AI tell me when it's making things up vs. quoting docs?' Based on this module, what's the cleanest pattern to suggest?" (Correct: "Have it cite sources with tags like `[FROM DOCS]` and `[GENERAL KNOWLEDGE]`. This is exactly the chairman's `[CHAIRMAN'S INSIGHT]` / `[WEB SOURCE]` pattern — explicit attribution lets users calibrate trust." Tests transfer to a new problem.)
  - Q4: "The council reports a consensus score of 28/100. The chairman's prompt grows a new section. What's its purpose, and where in the code is that decided?" (Correct: "The new section is the `disagreement_emphasis` block in `council.py`. It tells Claude to *not* force artificial agreement and to present multiple perspectives. It's triggered by `if consensus.level == 'low'` — and 'low' is determined by the variance math from Module 5. So a high-variance Stage 2 *changes the prompt* in Stage 3." Tests the cross-module data flow.)
- [x] **Callout** — "Aha! moment": "**Tool use** (also called function calling) is the single biggest unlock in modern LLM APIs. The model becomes a *decision-maker* that orchestrates external systems instead of a text-generator that only knows its training data. Any time you build an AI app and the model needs to look something up, run a calculation, or call your database, this is the pattern."
- [x] **Callout** — "Why explicit attribution matters": "When AI outputs blend multiple sources, it becomes impossible for the reader to know what's verified vs. invented. **Forcing the model to tag its own sources** is one of the easiest, most underrated techniques for trustworthy AI."
- [x] **Glossary tooltips** — synthesis, tool use, function calling, JSON schema, attribution, citation, hallucination, dataclass, percentage composition, prompt injection (just in case), low/high consensus, instruction, system prompt.
- [x] **(Optional) Final closing card** — a short send-off telling the reader what they now know how to do: read this codebase, request the council pattern from an AI assistant, debug streaming, batch heavy models, score consensus, demand attribution. Two sentences max.

### Reference Files to Read
- `references/interactive-elements.md` → Code↔English Translation Blocks, Message Flow / Data Flow Animation (with single-quote warning), Multiple-Choice Quizzes, Pattern/Feature Cards, Callout Boxes, Glossary Tooltips
- `references/content-philosophy.md` → all of it
- `references/gotchas.md` → all of it (especially the data-steps single-quote rule)
- `references/design-system.md` → tokens

### Connections
- **Previous module:** Module 5 — peer review and consensus. Module 6 picks up exactly where 5 ended (the consensus score) and shows how it feeds into the final synthesis.
- **Next module:** None — this is the closer.
- **Tone/style notes:**
  - Accent: teal `#2A7B9B`.
  - **The chief-justice metaphor lives only here.**
  - **Avoid apostrophes inside `data-steps` JSON strings** (use plain phrasing or `&apos;` if you must). The flow animation will silently break otherwise.
  - End the module on a *practical* note: "you can now ask AI for this exact pattern in your own apps." Tie back to the vibe-coder mission.
