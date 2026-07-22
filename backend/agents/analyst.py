import re
import time
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config import settings
from core.state import AgentState
from core.logger import get_logger
from core.usage import UsageCallbackHandler
from core import events

logger = get_logger(__name__)

# Phrases an LLM uses to say "no gaps here" — these must not be counted as gaps,
# otherwise a clean analysis triggers a pointless second research pass.
#
# Anchored end-to-end on purpose: the whole item has to be the non-answer.
# A bare "no ..." prefix would swallow real gaps like "No regional breakdown"
# or "No 2026 figures", which are exactly what the retry pass exists to chase.
_NON_GAP_PATTERN = re.compile(
    r"^\W*(?:"
    r"(?:none|n/?a|nothing|not\s+applicable)"
    r"(?:\s+(?:significant|major|identified|found|noted|apparent|obvious|missing|notable|of\s+note))?"
    r"|no\s+(?:significant|major|obvious|notable|apparent|clear)?\s*"
    r"(?:gaps?|issues?|contradictions?|concerns?|omissions?|limitations?)"
    r")\W*$",
    re.IGNORECASE,
)


def _extract_gaps(analysis_text: str) -> list[str]:
    """Pull real gap items out of the analyst's GAPS IDENTIFIED section.

    Counts bullet/numbered *items* rather than lines, so a single gap that
    wraps across two lines is one gap, and drops "None identified" style
    non-answers.
    """
    if "GAPS IDENTIFIED:" not in analysis_text:
        return []

    section = analysis_text.split("GAPS IDENTIFIED:")[-1].strip()

    gaps: list[str] = []
    for raw_line in section.split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        # Only bullet or numbered items start a new gap; anything else is a
        # continuation of the previous one.
        if re.match(r"^([-*•]|\d+[.)])\s+", line):
            item = re.sub(r"^([-*•]|\d+[.)])\s+", "", line).strip()
            if item and not _NON_GAP_PATTERN.match(item):
                gaps.append(item)
        elif gaps:
            gaps[-1] = f"{gaps[-1]} {line}"

    return gaps


def get_analyst_llm():
    return ChatGroq(
        model=settings.llm_model,
        api_key=settings.groq_api_key,
        temperature=0.1,
        callbacks=[UsageCallbackHandler("analyst")],
    )


ANALYST_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an Analyst Agent. Your ONLY job is to analyze research findings.

Rules:
- Work ONLY with the provided research findings
- Extract patterns, trends, and key insights
- Quantify claims where possible
- Identify gaps or contradictions in the research
- Do NOT add external information
- Flag low-confidence analysis with [LOW-CONFIDENCE]

Format your output as:
KEY INSIGHTS:
1. [insight]
2. [insight]
...

DATA POINTS:
- [specific numbers/facts from research]

GAPS IDENTIFIED:
- [what is missing or unclear]
""",
    ),
    ("human", "Analyze these research findings for the query: {query}\n\nResearch:\n{research}"),
])


def analyst_node(state: AgentState) -> AgentState:
    """Analyst agent — extracts insights from research."""
    research = state.get("research_output", "")
    start = time.perf_counter()
    logger.info(f"Analyst started | research_length={len(research) if research else 0}")
    events.emit("agent_start", agent="analyst")

    if not research:
        logger.warning("Analyst received empty research — flagging for review")
        return {
            **state,
            "analysis_output": "No research available to analyze.",
            "confidence": 0.1,
            "needs_human_review": True,
            "review_reason": "Analyst received empty research output.",
            "next_agent": "writer",
        }

    llm = get_analyst_llm()
    chain = ANALYST_PROMPT | llm | StrOutputParser()

    try:
        result = chain.invoke({
            "query": state["query"],
            "research": research,
        })

        confidence = state.get("confidence", 0.8)
        if "[LOW-CONFIDENCE]" in result:
            confidence = min(confidence, 0.5)

        # Detect research gaps for potential retry
        gaps = _extract_gaps(result)
        significant_gaps = len(gaps) >= 2

        retry_count = state.get("retry_count", 0)
        max_retries = settings.max_research_retries

        # If significant gaps found and we haven't retried yet, go back to researcher
        if significant_gaps and retry_count < max_retries:
            logger.info(
                f"Analyst found {len(gaps)} gaps — routing back to researcher "
                f"(retry {retry_count + 1}/{max_retries})"
            )
            next_agent = "researcher"
        else:
            next_agent = "writer"

        elapsed = round((time.perf_counter() - start) * 1000, 2)
        logger.info(
            f"Analyst completed | confidence={confidence} "
            f"gaps={len(gaps)} next={next_agent} duration_ms={elapsed}"
        )

        return {
            **state,
            "analysis_output": result,
            "confidence": confidence,
            "research_gaps": significant_gaps,
            # Handed to the researcher so the retry targets these specifically.
            # Cleared when moving on, so the writer never sees stale gaps.
            "research_gaps_detail": gaps if next_agent == "researcher" else [],
            "retry_count": retry_count + (1 if next_agent == "researcher" else 0),
            "iterations": state.get("iterations", 0) + 1,
            "next_agent": next_agent,
        }
    except Exception as e:
        elapsed = round((time.perf_counter() - start) * 1000, 2)
        logger.error(f"Analyst failed | error={e} duration_ms={elapsed}")
        return {
            **state,
            "analysis_output": f"Analysis failed: {str(e)}",
            "confidence": 0.2,
            "needs_human_review": True,
            "review_reason": f"Analyst agent error: {str(e)}",
            "next_agent": "writer",
        }