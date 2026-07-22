"""Golden-set evaluation harness.

Runs a fixed set of queries through the full pipeline, scores each report with
the LLM judge, and fails if quality regresses below a threshold. The judge and
the pipeline already existed; this turns them into a regression signal.

Deliberately NOT part of the normal test suite: it makes real LLM and search
calls, costs money, and takes minutes. Run it before a release or on a schedule.

    python -m eval.run_eval                  # full set
    python -m eval.run_eval --limit 2        # smoke check
    python -m eval.run_eval --json out.json  # machine-readable, for trending

Exit code is non-zero if the mean score is below `min_overall_score` or any
report cites nothing, so CI can gate on it.
"""

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.evaluator import evaluate_report  # noqa: E402
from core.supervisor import run_agent  # noqa: E402

GOLDEN_SET = Path(__file__).parent / "golden_set.json"


async def evaluate_one(query: str, index: int, total: int) -> dict:
    print(f"\n[{index}/{total}] {query}")
    started = time.perf_counter()

    try:
        result = await run_agent(query=query, thread_id=f"eval-{index}")
    except Exception as e:
        print(f"    PIPELINE FAILED: {type(e).__name__}: {str(e)[:120]}")
        return {"query": query, "error": str(e)[:300], "overall_score": 0.0, "sources": 0}

    report = result.get("final_report") or {}
    sources = result.get("sources") or []
    usage = result.get("token_usage") or {}
    elapsed = round(time.perf_counter() - started, 1)

    if not report or report.get("title") == "Report Generation Failed":
        print(f"    NO REPORT ({elapsed}s)")
        return {"query": query, "error": "no report", "overall_score": 0.0, "sources": 0}

    evaluation = await asyncio.to_thread(
        evaluate_report, query=query, report=report, sources=sources
    )
    if evaluation is None:
        print(f"    JUDGE FAILED ({elapsed}s)")
        return {"query": query, "error": "judge failed", "overall_score": 0.0,
                "sources": len(sources)}

    print(
        f"    score {evaluation.overall_score:.2f}/5 · {len(sources)} sources · "
        f"{usage.get('total_tokens', 0)} tokens · {elapsed}s"
    )
    return {
        "query": query,
        "title": report.get("title", ""),
        "overall_score": evaluation.overall_score,
        "factual_accuracy": evaluation.factual_accuracy.score,
        "analytical_depth": evaluation.analytical_depth.score,
        "completeness": evaluation.completeness.score,
        "clarity": evaluation.clarity.score,
        "sources": len(sources),
        "tokens": usage.get("total_tokens", 0),
        "cost_usd": usage.get("estimated_cost_usd", 0.0),
        "latency_s": elapsed,
    }


async def main() -> int:
    parser = argparse.ArgumentParser(description="Run the golden-set evaluation.")
    parser.add_argument("--limit", type=int, help="only run the first N queries")
    parser.add_argument("--json", dest="json_out", help="write results to this path")
    args = parser.parse_args()

    spec = json.loads(GOLDEN_SET.read_text(encoding="utf-8"))
    queries = spec["queries"][: args.limit] if args.limit else spec["queries"]
    min_score = float(spec.get("min_overall_score", 3.0))
    min_sources = int(spec.get("min_sources_per_report", 1))

    print(f"Golden set: {len(queries)} queries · threshold {min_score}/5")

    # Sequential on purpose: concurrent runs would trip the provider rate limit
    # and the failures would look like quality regressions.
    results = []
    for i, query in enumerate(queries, 1):
        results.append(await evaluate_one(query, i, len(queries)))

    scored = [r["overall_score"] for r in results]
    mean = statistics.mean(scored) if scored else 0.0
    total_cost = sum(r.get("cost_usd", 0.0) for r in results)
    sourceless = [r for r in results if r.get("sources", 0) < min_sources]

    print("\n" + "=" * 64)
    print(f"mean score     : {mean:.2f}/5   (threshold {min_score})")
    print(f"per query      : {[round(s, 2) for s in scored]}")
    print(f"total cost     : ${total_cost:.4f}")
    print(f"uncited reports: {len(sourceless)}")
    print("=" * 64)

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(
                {"mean_score": mean, "threshold": min_score,
                 "total_cost_usd": total_cost, "results": results},
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"wrote {args.json_out}")

    failures = []
    if mean < min_score:
        failures.append(f"mean score {mean:.2f} below threshold {min_score}")
    if sourceless:
        # A confident report with no citations is the failure mode this whole
        # project exists to prevent, so it fails the run outright.
        failures.append(f"{len(sourceless)} report(s) cited nothing")

    if failures:
        print("FAIL: " + "; ".join(failures))
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    if not os.getenv("GROQ_API_KEY") and not Path(".env").exists():
        print("GROQ_API_KEY not configured", file=sys.stderr)
        sys.exit(2)
    sys.exit(asyncio.run(main()))
