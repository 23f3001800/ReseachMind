from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools import DuckDuckGoSearchRun
from config import settings
from core.state import AgentState
import re


def get_researcher_llm():
    return ChatGroq(
        model=settings.llm_model,
        api_key=settings.groq_api_key,
        temperature=0.1,
    )


RESEARCHER_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a Research Agent. Your ONLY job is to gather factual information.

Rules:
- Provide factual, sourced information only
- List specific findings as numbered points
- Include source references where possible
- Do NOT write conclusions or recommendations
- Flag uncertainty explicitly with [UNCERTAIN]
- If you cannot find reliable info, say so clearly

Format your output as:
FINDINGS:
1. [finding]
2. [finding]
...

SOURCES:
- [source or search query used]
""",
    ),
    ("human", "Research this topic thoroughly: {query}\n\nPrevious context: {context}"),
])


def researcher_node(state: AgentState) -> AgentState:
    """Researcher agent — gathers factual information."""
    query = state["query"]
    context = ""

    # Use search tool if Tavily not configured
    try:
        search = DuckDuckGoSearchRun()
        search_results = search.run(query)
        context = f"Web search results:\n{search_results}"
    except Exception:
        context = "Web search unavailable. Using internal knowledge only."

    llm = get_researcher_llm()
    chain = RESEARCHER_PROMPT | llm | StrOutputParser()

    try:
        result = chain.invoke({"query": query, "context": context})
        confidence = 0.8 if "[UNCERTAIN]" not in result else 0.5

        # Extract sources from output
        sources = []
        if "SOURCES:" in result:
            source_section = result.split("SOURCES:")[-1]
            sources = [
                line.strip("- ").strip()
                for line in source_section.strip().split("\n")
                if line.strip()
            ]

        return {
            **state,
            "research_output": result,
            "sources": sources,
            "confidence": confidence,
            "iterations": state.get("iterations", 0) + 1,
            "next_agent": "analyst",
        }
    except Exception as e:
        return {
            **state,
            "research_output": f"Research failed: {str(e)}",
            "confidence": 0.2,
            "needs_human_review": True,
            "review_reason": f"Researcher agent error: {str(e)}",
            "next_agent": "writer",
        }