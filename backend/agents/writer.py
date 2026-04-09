from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config import settings
from core.state import AgentState


def get_writer_llm():
    return ChatGroq(
        model=settings.llm_model,
        api_key=settings.groq_api_key,
        temperature=0.2,
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
- Structure output strictly as shown below

Output format:
TITLE: [Report title]

SUMMARY:
[2-3 sentence executive summary]

KEY FINDINGS:
1. [finding]
2. [finding]
3. [finding]

ANALYSIS:
1. [insight]
2. [insight]

CONCLUSION:
[Final conclusion paragraph]
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
    """Writer agent — produces final structured report."""
    research = state.get("research_output", "No research available.")
    analysis = state.get("analysis_output", "No analysis available.")

    llm = get_writer_llm()
    chain = WRITER_PROMPT | llm | StrOutputParser()

    try:
        result = chain.invoke({
            "query": state["query"],
            "research": research,
            "analysis": analysis,
        })

        # Guardrail: flag if confidence is too low
        confidence = state.get("confidence", 0.8)
        needs_review = confidence < settings.confidence_threshold

        return {
            **state,
            "final_report": result,
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
        return {
            **state,
            "final_report": f"Report generation failed: {str(e)}",
            "confidence": 0.0,
            "needs_human_review": True,
            "review_reason": f"Writer agent error: {str(e)}",
        }