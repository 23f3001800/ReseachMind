import time
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config import settings
from core.state import AgentState
from core.logger import get_logger

logger = get_logger(__name__)


def get_analyst_llm():
    return ChatGroq(
        model=settings.llm_model,
        api_key=settings.groq_api_key,
        temperature=0.1,
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
        has_gaps = "GAPS IDENTIFIED:" in result
        gap_lines = []
        if has_gaps:
            gap_section = result.split("GAPS IDENTIFIED:")[-1].strip()
            gap_lines = [l.strip() for l in gap_section.split("\n") if l.strip() and len(l.strip()) > 5]
        significant_gaps = len(gap_lines) >= 2

        retry_count = state.get("retry_count", 0)
        max_retries = 1

        # If significant gaps found and we haven't retried yet, go back to researcher
        if significant_gaps and retry_count < max_retries:
            logger.info(
                f"Analyst found {len(gap_lines)} gaps — routing back to researcher "
                f"(retry {retry_count + 1}/{max_retries})"
            )
            next_agent = "researcher"
        else:
            next_agent = "writer"

        elapsed = round((time.perf_counter() - start) * 1000, 2)
        logger.info(
            f"Analyst completed | confidence={confidence} "
            f"gaps={len(gap_lines)} next={next_agent} duration_ms={elapsed}"
        )

        return {
            **state,
            "analysis_output": result,
            "confidence": confidence,
            "research_gaps": significant_gaps,
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