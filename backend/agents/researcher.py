import time
from langchain_groq import ChatGroq
from langchain_core.messages import ToolMessage
from config import settings
from core.state import AgentState
from core.logger import get_logger
from agents.tools import get_search_tools

logger = get_logger(__name__)



def get_researcher_llm():
    return ChatGroq(
        model=settings.llm_model,
        api_key=settings.groq_api_key,
        temperature=0.1,
    )


RESEARCHER_SYSTEM = """You are a Research Agent. Your ONLY job is to gather factual information.

You have access to a web_search tool. Use it to find current, factual information.
You may call the tool multiple times with different queries to be thorough.

Rules:
- Provide factual, sourced information only
- List specific findings as numbered points
- Include source references where possible
- Do NOT write conclusions or recommendations
- Flag uncertainty explicitly with [UNCERTAIN]
- If you cannot find reliable info, say so clearly

After gathering information via search, compile your final output as:
FINDINGS:
1. [finding]
2. [finding]
...

SOURCES:
- [source or search query used]
"""


def researcher_node(state: AgentState) -> AgentState:
    """Researcher agent — gathers factual information via tool-calling ReAct loop."""
    query = state["query"]
    start = time.perf_counter()
    logger.info(f"Researcher started | query='{query[:80]}'")

    # Check for RAG context from uploaded documents
    rag_context = ""
    try:
        from core.vectorstore import get_context_for_query, has_documents
        if has_documents():
            rag_context = get_context_for_query(query, k=3)
            if rag_context:
                logger.info(f"RAG context injected | chars={len(rag_context)}")
    except ImportError:
        pass  # Vector store dependencies not installed

    llm = get_researcher_llm()
    search_tools = get_search_tools()
    llm_with_tools = llm.bind_tools(search_tools)

    # Build user message with optional RAG context
    user_msg = f"Research this topic thoroughly: {query}"
    if rag_context:
        user_msg = (
            f"The user has uploaded documents. Here is relevant context from those documents:\n\n"
            f"{rag_context}\n\n"
            f"---\n\n"
            f"Now research this topic thoroughly, combining the uploaded document context "
            f"with web search results: {query}"
        )

    messages = [
        {"role": "system", "content": RESEARCHER_SYSTEM},
        {"role": "human", "content": user_msg},
    ]

    # Tool-calling map for execution
    tool_map = {t.name: t for t in search_tools}

    try:
        # ReAct loop: let the LLM call tools until it produces a final text response
        max_tool_rounds = 5
        for round_num in range(max_tool_rounds):
            response = llm_with_tools.invoke(messages)
            messages.append(response)

            # If no tool calls, the LLM is done — break out
            if not response.tool_calls:
                logger.info(f"Researcher LLM finished | rounds={round_num + 1}")
                break

            # Execute each tool call and append results
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                logger.info(f"Tool call | tool={tool_name} args={tool_args}")

                tool_fn = tool_map.get(tool_name)
                if tool_fn:
                    tool_result = tool_fn.invoke(tool_args)
                else:
                    tool_result = f"Unknown tool: {tool_name}"

                messages.append(
                    ToolMessage(content=str(tool_result), tool_call_id=tool_call["id"])
                )
        else:
            logger.warning(f"Researcher hit max tool rounds ({max_tool_rounds})")

        # Extract the final text response
        result = response.content if hasattr(response, "content") else str(response)
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

        elapsed = round((time.perf_counter() - start) * 1000, 2)
        logger.info(
            f"Researcher completed | confidence={confidence} "
            f"sources={len(sources)} duration_ms={elapsed}"
        )

        return {
            **state,
            "research_output": result,
            "sources": sources,
            "confidence": confidence,
            "iterations": state.get("iterations", 0) + 1,
            "next_agent": "analyst",
        }
    except Exception as e:
        elapsed = round((time.perf_counter() - start) * 1000, 2)
        logger.error(f"Researcher failed | error={e} duration_ms={elapsed}")
        return {
            **state,
            "research_output": f"Research failed: {str(e)}",
            "confidence": 0.2,
            "needs_human_review": True,
            "review_reason": f"Researcher agent error: {str(e)}",
            "next_agent": "writer",
        }