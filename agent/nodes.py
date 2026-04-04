# agent/nodes.py
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from .tools import search_web, search_rag
import os
import dotenv

dotenv.load_dotenv()


def get_llm(temperature=0.2):
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=temperature
    )

# ── Node 1: Planner ──────────────────────────────────────────
def planner_node(state: dict) -> dict:
    """Generate 3 targeted search queries from the topic."""
    llm = get_llm()
    response = llm.invoke(f"""
You are a research planner. Given this research topic, generate exactly 3 
specific search queries that will gather comprehensive information.

Topic: {state['topic']}
Depth: {state['depth']}

Return ONLY a JSON array of 3 strings. Example:
["query one", "query two", "query three"]""")

    try:
        queries = json.loads(response.content.strip())
    except Exception:
        # Fallback if JSON parsing fails
        queries = [state["topic"], f"{state['topic']} 2026", f"{state['topic']} impact"]

    return {
        "search_queries": queries,
        "steps_taken": state.get("steps_taken", []) + ["planned 3 search queries"]
    }

# ── Node 2: Web Search (runs 3 queries) ─────────────────────
def search_node(state: dict) -> dict:
    """Execute all search queries and collect results."""
    all_results = []
    citations = []

    for query in state["search_queries"]:
        results = search_web(query)
        for r in results:
            if r.get("content"):
                all_results.append({
                    "query": query,
                    "content": r["content"][:1500],  # truncate
                    "url": r.get("url", ""),
                    "title": r.get("title", "")
                })
                if r.get("url"):
                    citations.append(r["url"])

    return {
        "search_results": all_results,
        "citations": list(set(citations)),  # deduplicate
        "steps_taken": state["steps_taken"] + [f"searched {len(all_results)} sources"]
    }

# ── Node 3: Extract key facts ────────────────────────────────
def extract_node(state: dict) -> dict:
    """Extract the most important facts from raw search results."""
    llm = get_llm()

    # Compile search content (stay within token limit)
    content = "\n\n---\n\n".join([
        f"Source: {r['title']}\n{r['content']}"
        for r in state["search_results"][:6]   # top 6 results
    ])

    response = llm.invoke(f"""
Extract the 8-10 most important, specific facts from these search results.
Topic: {state['topic']}

Search results:
{content}

Return a JSON array of fact strings. Be specific — include numbers, dates, names.
Example: ["LLMs reduced code review time by 35% at Microsoft in 2025", ...]

Return ONLY valid JSON array.""")

    try:
        facts = json.loads(response.content.strip())
    except Exception:
        facts = [response.content[:500]]

    return {
        "extracted_facts": facts,
        "steps_taken": state["steps_taken"] + [f"extracted {len(facts)} key facts"]
    }

# ── Node 4: RAG lookup ───────────────────────────────────────
def rag_node(state: dict, retriever=None) -> dict:
    """Optionally enrich with knowledge base context."""
    if retriever is None:
        return {"rag_context": "", "steps_taken": state["steps_taken"] + ["no knowledge base loaded"]}

    context = search_rag(retriever, state["topic"])
    return {
        "rag_context": context,
        "steps_taken": state["steps_taken"] + ["enriched with knowledge base"]
    }

# ── Node 5: Synthesise ───────────────────────────────────────
def synthesise_node(state: dict) -> dict:
    """Write the full structured report."""
    llm = get_llm(temperature=0.2)  # slight creativity for writing

    facts_text = "\n".join([f"• {f}" for f in state["extracted_facts"]])
    rag_section = f"\n\nAdditional context from knowledge base:\n{state['rag_context']}" \
                  if state.get("rag_context") else ""

    response = llm.invoke(f"""
Write a comprehensive research report on: {state['topic']}

Key facts gathered:
{facts_text}
{rag_section}

Structure the report with these exact sections:
## Executive Summary
## Key Findings
## Analysis
## Implications
## Conclusion

Use markdown formatting. Be specific — cite the facts. 
Aim for 400-600 words total. Professional tone.""")

    return {
        "final_report": response.content,
        "steps_taken": state["steps_taken"] + ["synthesised final report"]
    }