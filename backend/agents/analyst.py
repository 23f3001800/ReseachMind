from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config import settings
from core.state import AgentState


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

    if not research:
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

        return {
            **state,
            "analysis_output": result,
            "confidence": confidence,
            "iterations": state.get("iterations", 0) + 1,
            "next_agent": "writer",
        }
    except Exception as e:
        return {
            **state,
            "analysis_output": f"Analysis failed: {str(e)}",
            "confidence": 0.2,
            "needs_human_review": True,
            "review_reason": f"Analyst agent error: {str(e)}",
            "next_agent": "writer",
        }