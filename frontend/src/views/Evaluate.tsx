import { useRef, useState } from "react";
import { api, ApiError } from "@/api/client";
import type { EvaluateResponse } from "@/types";
import { SourceList } from "@/components/ReportView";
import { Alert, Button, Card, EmptyState, Meter, SectionTitle, Stat } from "@/components/ui";
import "./Evaluate.css";

const DIMENSIONS = [
  { key: "factual_accuracy", label: "Factual accuracy", weight: "30%" },
  { key: "analytical_depth", label: "Analytical depth", weight: "25%" },
  { key: "completeness", label: "Completeness", weight: "25%" },
  { key: "clarity", label: "Clarity", weight: "20%" },
] as const;

export function Evaluate({ threadId }: { threadId: string }) {
  const [query, setQuery] = useState("");
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<EvaluateResponse | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const run = async () => {
    const q = query.trim();
    if (q.length < 3) return;
    setRunning(true);
    setError(null);
    setData(null);
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    try {
      setData(await api.evaluate(q, threadId, ctrl.signal));
    } catch (e) {
      if ((e as Error).name === "AbortError") setError("Evaluation cancelled.");
      else setError(e instanceof ApiError ? e.message : (e as Error).message);
    } finally {
      setRunning(false);
      abortRef.current = null;
    }
  };

  const overall = data?.evaluation.overall_score ?? 0;
  const tone = overall >= 4 ? "success" : overall >= 3 ? "warning" : "danger";

  return (
    <div className="evaluate">
      <SectionTitle hint="Runs the full pipeline, then scores the report with an LLM judge across four weighted dimensions. Takes roughly twice as long as a normal run.">
        Evaluate
      </SectionTitle>

      <Card>
        <div className="evaluate__form">
          <input
            className="input"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && !running && run()}
            placeholder="Query to run and grade…"
            disabled={running}
          />
          {running ? (
            <Button variant="danger" onClick={() => abortRef.current?.abort()}>
              Cancel
            </Button>
          ) : (
            <Button variant="primary" onClick={run} disabled={query.trim().length < 3}>
              Run &amp; grade
            </Button>
          )}
        </div>
        {running && (
          <p className="evaluate__running">
            Running the pipeline and grading the result — this can take a couple of minutes.
          </p>
        )}
      </Card>

      {error && (
        <Alert tone="danger" title="Evaluation failed">
          {error}
        </Alert>
      )}

      {data && (
        <>
          <div className="evaluate__overall">
            <Stat
              label="Overall score"
              value={`${overall.toFixed(2)} / 5`}
              tone={tone}
            />
            <Card className="evaluate__summary">
              <h3 className="evaluate__h3">Judge summary</h3>
              <p>{data.evaluation.summary}</p>
            </Card>
          </div>

          <Card>
            <SectionTitle>Dimension breakdown</SectionTitle>
            <div className="dims">
              {DIMENSIONS.map((d) => {
                const dim = data.evaluation[d.key];
                return (
                  <div key={d.key} className="dim">
                    <div className="dim__head">
                      <span className="dim__label">{d.label}</span>
                      <span className="dim__weight mono">weight {d.weight}</span>
                      <span className="dim__score mono">{dim.score}/5</span>
                    </div>
                    <Meter value={dim.score} max={5} label={`${d.label} score`} />
                    <p className="dim__why">{dim.explanation}</p>
                  </div>
                );
              })}
            </div>
          </Card>

          <Card>
            <SectionTitle>The graded report</SectionTitle>
            <h3 className="evaluate__report-title">{data.report.title}</h3>
            <p className="evaluate__report-summary">{data.report.summary}</p>
            <h3 className="evaluate__h3">Sources</h3>
            <SourceList sources={data.sources ?? []} />
          </Card>
        </>
      )}

      {!data && !error && !running && (
        <EmptyState icon="⚖️" title="No evaluation yet">
          Grade a report to see how the pipeline scores on accuracy, depth, completeness
          and clarity.
        </EmptyState>
      )}
    </div>
  );
}
