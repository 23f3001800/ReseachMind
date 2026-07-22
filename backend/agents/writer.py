import time
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from config import settings
from core.state import AgentState
from schemas.models import WriterReport
from core.logger import get_logger
from core.usage import UsageCallbackHandler
from core import events

logger = get_logger(__name__)


def get_writer_llm():
    return ChatGroq(
        model=settings.llm_model,
        api_key=settings.groq_api_key,
        temperature=0.2,
        callbacks=[UsageCallbackHandler("writer")],
    )


WRITER_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a Writer Agent. Your job is to produce a clean, structured final report.

Rules:
- Synthesize research AND analysis into a coherent report
- Be concise but complete
- Use plain language
- Do not add information not present in research/analysis
- Provide a clear title, executive summary, key findings, analysis insights, and a conclusion
""",
    ),
    (
        "human",
        """Write a comprehensive report for:
Query: {query}

Research findings:
{research}

Analysis:
{analysis}
""",
    ),
])


def writer_node(state: AgentState) -> AgentState:
    """Writer agent — produces final structured report via LLM structured output."""
    research = state.get("research_output", "No research available.")
    analysis = state.get("analysis_output", "No analysis available.")
    start = time.perf_counter()
    logger.info("Writer started | generating structured report")
    events.emit("agent_start", agent="writer")

    llm = get_writer_llm()
    structured_llm = llm.with_structured_output(WriterReport)
    chain = WRITER_PROMPT | structured_llm

    try:
        report: WriterReport = chain.invoke({
            "query": state["query"],
            "research": research,
            "analysis": analysis,
        })

        # Guardrail: flag if confidence is too low
        confidence = state.get("confidence", 0.8)
        needs_review = confidence < settings.confidence_threshold

        elapsed = round((time.perf_counter() - start) * 1000, 2)
        logger.info(
            f"Writer completed | title='{report.title[:60]}' "
            f"confidence={confidence} needs_review={needs_review} duration_ms={elapsed}"
        )

        return {
            **state,
            "final_report": report.model_dump(),
            "confidence": confidence,
            "needs_human_review": needs_review,
            "review_reason": (
                f"Confidence {confidence:.2f} below threshold {settings.confidence_threshold}"
                if needs_review else None
            ),
            "iterations": state.get("iterations", 0) + 1,
            "next_agent": "END",
        }
    except Exception as e:
        elapsed = round((time.perf_counter() - start) * 1000, 2)
        logger.error(f"Writer failed | error={e} duration_ms={elapsed}")
        # Fallback: produce a degraded report dict on failure
        return {
            **state,
            "final_report": {
                "title": "Report Generation Failed",
                "summary": str(e),
                "research_findings": [],
                "analysis": [],
                "conclusion": f"Report generation failed: {str(e)}",
            },
            "confidence": 0.0,
            "needs_human_review": True,
            "review_reason": f"Writer agent error: {str(e)}",
        }