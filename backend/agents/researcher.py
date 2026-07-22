import time
from langchain_groq import ChatGroq
from langchain_core.messages import ToolMessage
from config import settings
from core.state import AgentState
from core.logger import get_logger
from core.usage import UsageCallbackHandler
from core import events
from agents.tools import get_search_tools, get_collected_sources

logger = get_logger(__name__)


def _is_tool_call_failure(exc: Exception) -> bool:
    """True for provider-side malformed-tool-call errors, which are retryable."""
    text = str(exc).lower()
    return "tool_use_failed" in text or "was not in request.tools" in text


def _synthesise_without_tools(llm, messages: list):
    """Ask the model to write up findings with tool calling disabled.

    Used when the provider rejects a malformed tool call. Any search results
    already gathered are in the message history, so this keeps them rather than
    discarding the whole pass.
    """
    wrap_up = list(messages) + [{
        "role": "human",
        "content": (
            "Stop searching and write up your findings now, using only the search "
            "results already gathered above. Follow the FINDINGS format. Mark any "
            "claim not supported by those results with [UNVERIFIED]."
        ),
    }]
    return llm.invoke(wrap_up)


def get_researcher_llm():
    return ChatGroq(
        model=settings.llm_model,
        api_key=settings.groq_api_key,
        temperature=0.1,
        callbacks=[UsageCallbackHandler("researcher")],
    )


RESEARCHER_SYSTEM = """You are a Research Agent. Your ONLY job is to gather factual information.

You have access to a web_search tool. Use it to find current, factual information.
You may call the tool multiple times with different queries to be thorough.

Rules:
- Provide factual information drawn from the search results you retrieved
- List specific findings as numbered points
- Attribute each finding to the page it came from, using the real URL from the
  search results — never invent, guess, or paraphrase a URL
- Do NOT write conclusions or recommendations
- Flag uncertainty explicitly with [UNCERTAIN]
- If a claim comes from your own knowledge rather than a search result, mark it
  [UNVERIFIED] rather than attributing it to a source
- If you cannot find reliable info, say so clearly

After gathering information via search, compile your final output as:
FINDINGS:
1. [finding] — [url it came from]
2. [finding] — [url it came from]
...
"""


def researcher_node(state: AgentState) -> AgentState:
    """Researcher agent — gathers factual information via tool-calling ReAct loop."""
    query = state["query"]
    start = time.perf_counter()
    logger.info(f"Researcher started | query='{query[:80]}'")
    events.emit("agent_start", agent="researcher", pass_number=state.get("retry_count", 0) + 1)

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

    # Second pass: target the specific gaps the analyst found, rather than
    # repeating the identical first-pass research.
    gaps = state.get("research_gaps_detail") or []
    if gaps:
        gap_list = "\n".join(f"- {gap}" for gap in gaps)
        previous = state.get("research_output") or ""
        user_msg = (
            f"A previous research pass on this topic left specific gaps. Your job now is to "
            f"CLOSE THOSE GAPS — do not simply repeat the earlier findings.\n\n"
            f"Gaps to close:\n{gap_list}\n\n"
            f"Already established (do not re-report unless you can add detail):\n"
            f"{previous[:2000]}\n\n"
            f"---\n\n"
            f"Run targeted searches for the gaps above and report what you find "
            f"for the original query: {query}"
        )
        logger.info(f"Researcher retry | targeting {len(gaps)} gap(s)")

    messages = [
        {"role": "system", "content": RESEARCHER_SYSTEM},
        {"role": "human", "content": user_msg},
    ]

    # Tool-calling map for execution
    tool_map = {t.name: t for t in search_tools}

    try:
        # ReAct loop: let the LLM call tools until it produces a final text response
        max_tool_rounds = settings.max_tool_rounds
        response = None
        for round_num in range(max_tool_rounds):
            try:
                response = llm_with_tools.invoke(messages)
            except Exception as e:
                # Groq intermittently emits a malformed function call, which the
                # API rejects with tool_use_failed. Losing the whole research pass
                # over one bad round throws away every source already retrieved,
                # so fall back to a plain (tool-free) synthesis of what we have.
                if not _is_tool_call_failure(e):
                    raise
                logger.warning(
                    f"Malformed tool call from provider on round {round_num + 1} — "
                    f"synthesising from {len(get_collected_sources())} source(s) already retrieved"
                )
                response = _synthesise_without_tools(llm, messages)
                break

            messages.append(response)

            # If no tool calls, the LLM is done — break out
            if not response.tool_calls:
                logger.info(f"Researcher LLM finished | rounds={round_num + 1}")
                break

            # Execute each tool call and append results
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                search_query = str(tool_args.get("query", ""))
                logger.info(f"Tool call | tool={tool_name} args={tool_args}")

                # Report the search before running it — this is the part of the
                # run the user waits longest for, and seeing the actual query is
                # what makes the agent legible rather than a spinner.
                events.emit(
                    "tool_call",
                    agent="researcher",
                    tool=tool_name,
                    query=search_query,
                )

                before = len(get_collected_sources())
                tool_fn = tool_map.get(tool_name)
                if tool_fn:
                    tool_result = tool_fn.invoke(tool_args)
                else:
                    tool_result = f"Unknown tool: {tool_name}"

                events.emit(
                    "tool_result",
                    agent="researcher",
                    tool=tool_name,
                    query=search_query,
                    new_sources=len(get_collected_sources()) - before,
                )

                messages.append(
                    ToolMessage(content=str(tool_result), tool_call_id=tool_call["id"])
                )
        else:
            logger.warning(f"Researcher hit max tool rounds ({max_tool_rounds})")

        # Extract the final text response
        result = response.content if hasattr(response, "content") else str(response)
        confidence = 0.8 if "[UNCERTAIN]" not in result else 0.5

        # Sources come from search results the system actually retrieved —
        # never from text the model wrote. No search, no citation.
        sources = get_collected_sources()

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