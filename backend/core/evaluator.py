"""Evaluation framework — LLM-as-judge scoring for report quality.

Evaluates agent-generated reports on 4 dimensions:
  1. Factual Accuracy — are findings verifiable and correct?
  2. Analytical Depth — are insights meaningful and well-reasoned?
  3. Completeness — does the report cover the query thoroughly?
  4. Clarity — is the writing clear, well-structured, and professional?

Each dimension is scored 1-5 with an explanation. Returns an overall
weighted average score plus the per-dimension breakdown.
"""

from typing import Optional
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from config import settings
from core.logger import get_logger

logger = get_logger(__name__)


class DimensionScore(BaseModel):
    """Score for a single evaluation dimension."""
    score: int = Field(ge=1, le=5, description="Score from 1 (poor) to 5 (excellent)")
    explanation: str = Field(description="Brief justification for the score")


class EvaluationResult(BaseModel):
    """Full evaluation result from LLM-as-judge."""
    factual_accuracy: DimensionScore
    analytical_depth: DimensionScore
    completeness: DimensionScore
    clarity: DimensionScore
    overall_score: float = Field(ge=1.0, le=5.0, description="Weighted average score")
    summary: str = Field(description="One-paragraph evaluation summary")


EVAL_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert evaluator of research reports. You must evaluate
the following report based on 4 dimensions, scoring each 1-5:

1. Factual Accuracy (1-5): Are the findings factually correct and verifiable?
2. Analytical Depth (1-5): Are the insights meaningful, nuanced, and well-reasoned?
3. Completeness (1-5): Does the report thoroughly address the research query?
4. Clarity (1-5): Is the writing clear, well-structured, and professional?

Be strict and fair. Most reports should score 2-4. Reserve 5 for truly exceptional work
and 1 for completely failing on a dimension."""),
    ("human", """**Research Query:** {query}

**Report to Evaluate:**
Title: {title}
Summary: {summary}

Key Findings:
{findings}

Analysis:
{analysis}

Conclusion: {conclusion}

Sources: {sources}

Evaluate this report on the 4 dimensions above."""),
])


def evaluate_report(
    query: str,
    report: dict,
    sources: list = None,
) -> Optional[EvaluationResult]:
    """Run LLM-as-judge evaluation on a generated report.

    Args:
        query: The original research query.
        report: The report dict (from WriterReport.model_dump()).
        sources: List of source strings.

    Returns:
        EvaluationResult with per-dimension scores and overall score,
        or None if evaluation fails.
    """
    if not report:
        logger.warning("No report to evaluate")
        return None

    try:
        llm = ChatGroq(
            model=settings.llm_model,
            api_key=settings.groq_api_key,
            temperature=0.0,  # Deterministic for evaluation
        )

        structured_llm = llm.with_structured_output(EvaluationResult)
        chain = EVAL_PROMPT | structured_llm

        findings_str = "\n".join(f"- {f}" for f in report.get("research_findings", []))
        analysis_str = "\n".join(f"- {a}" for a in report.get("analysis", []))
        sources_str = ", ".join(sources or report.get("sources", []))

        result = chain.invoke({
            "query": query,
            "title": report.get("title", "Untitled"),
            "summary": report.get("summary", ""),
            "findings": findings_str or "None provided",
            "analysis": analysis_str or "None provided",
            "conclusion": report.get("conclusion", ""),
            "sources": sources_str or "None",
        })

        # Calculate weighted average (accuracy and depth weighted higher)
        weights = {
            "factual_accuracy": 0.30,
            "analytical_depth": 0.25,
            "completeness": 0.25,
            "clarity": 0.20,
        }
        weighted_sum = (
            result.factual_accuracy.score * weights["factual_accuracy"]
            + result.analytical_depth.score * weights["analytical_depth"]
            + result.completeness.score * weights["completeness"]
            + result.clarity.score * weights["clarity"]
        )
        result.overall_score = round(weighted_sum, 2)

        logger.info(
            f"Evaluation completed | overall={result.overall_score} "
            f"accuracy={result.factual_accuracy.score} "
            f"depth={result.analytical_depth.score} "
            f"completeness={result.completeness.score} "
            f"clarity={result.clarity.score}"
        )

        return result

    except Exception as e:
        logger.error(f"Evaluation failed | error={e}")
        return None
