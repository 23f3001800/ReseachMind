import { useCallback, useEffect, useRef, useState } from "react";
import { api, ApiError, streamChat, type Activity } from "@/api/client";
import type { RunResult } from "@/types";
import { Pipeline, type NodeState, type PipelineNode } from "@/components/Pipeline";
import { ActivityFeed } from "@/components/ActivityFeed";
import { ReportView } from "@/components/ReportView";
import { Alert, Button, Card, EmptyState } from "@/components/ui";
import "./Research.css";

const AGENTS = [
  { name: "researcher", label: "Researcher", hint: "Searches the web via tool calls" },
  { name: "analyst", label: "Analyst", hint: "Extracts insights, finds gaps" },
  { name: "writer", label: "Writer", hint: "Composes the structured report" },
] as const;

const EXAMPLES = [
  "What are the latest advances in multi-agent AI systems?",
  "Compare vector databases for production RAG in 2026",
  "What are the main security risks of autonomous AI agents?",
];

interface Props {
  threadId: string;
  streaming: boolean;
}

export function Research({ threadId, streaming }: Props) {
  const [query, setQuery] = useState("");
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<RunResult | null>(null);

  const [passes, setPasses] = useState<Record<string, number>>({});
  const [lastDone, setLastDone] = useState<string | null>(null);
  const [activeAgent, setActiveAgent] = useState<string | null>(null);
  const [failedAt, setFailedAt] = useState<string | null>(null);
  const [details, setDetails] = useState<Record<string, string>>({});
  const [activity, setActivity] = useState<Activity[]>([]);
  const [elapsed, setElapsed] = useState(0);

  const abortRef = useRef<AbortController | null>(null);
  const startedAt = useRef(0);

  // Drive the elapsed clock only while a run is in flight.
  useEffect(() => {
    if (!running) return;
    const id = setInterval(() => setElapsed(Date.now() - startedAt.current), 100);
    return () => clearInterval(id);
  }, [running]);

  // Abort any in-flight request if the view unmounts.
  useEffect(() => () => abortRef.current?.abort(), []);

  const resetRun = () => {
    setError(null);
    setResult(null);
    setPasses({});
    setLastDone(null);
    setActiveAgent(null);
    setFailedAt(null);
    setDetails({});
    setActivity([]);
    setElapsed(0);
    startedAt.current = Date.now();
  };

  const nodeState = useCallback(
    (name: string, index: number): NodeState => {
      if (failedAt === name) return "error";
      // The backend now reports agent_start, so which stage is running is known
      // rather than inferred from whichever node finished last.
      if (running && activeAgent === name) return "active";
      if (passes[name]) return "done";
      if (!running) return "idle";
      // No agent_start seen yet (non-streaming mode): fall back to position.
      if (activeAgent === null) {
        const doneIdx = lastDone ? AGENTS.findIndex((a) => a.name === lastDone) : -1;
        return index === doneIdx + 1 ? "active" : "idle";
      }
      return "idle";
    },
    [failedAt, passes, lastDone, running, activeAgent],
  );

  const nodes: PipelineNode[] = AGENTS.map((a, i) => ({
    name: a.name,
    label: a.label,
    hint: a.hint,
    state: nodeState(a.name, i),
    detail: details[a.name],
    passes: passes[a.name] ?? 0,
  }));

  const run = async () => {
    const q = query.trim();
    if (q.length < 3) {
      setError("Enter a query of at least 3 characters.");
      return;
    }

    resetRun();
    setRunning(true);
    const ctrl = new AbortController();
    abortRef.current = ctrl;

    try {
      if (streaming) {
        await streamChat(
          q,
          threadId,
          {
            onAgentStart: (agent) => setActiveAgent(agent),
            onActivity: (a) => setActivity((prev) => [...prev, a]),
            onAgentEnd: (agent, data) => {
              setPasses((p) => ({ ...p, [agent]: (p[agent] ?? 0) + 1 }));
              setLastDone(agent);
              setActiveAgent(null);
              const conf = typeof data.confidence === "number" ? data.confidence : null;
              const found =
                typeof data.sources_found === "number" ? data.sources_found : null;
              const bits: string[] = [];
              if (conf !== null) bits.push(`confidence ${conf.toFixed(2)}`);
              if (found) bits.push(`${found} source${found === 1 ? "" : "s"}`);
              if (bits.length) setDetails((d) => ({ ...d, [agent]: bits.join(" · ") }));
            },
            onComplete: (r) => {
              if (r.report) setResult(r);
              else setError("The pipeline finished without producing a report.");
            },
            onError: (msg) => setError(msg),
          },
          ctrl.signal,
        );
      } else {
        const res = await api.chat(q, threadId, ctrl.signal);
        setResult({
          report: res.report,
          sources: res.report.sources ?? [],
          confidence: res.report.confidence ?? 0,
          needs_human_review: res.report.needs_human_review ?? false,
          iterations: res.iterations,
          latency_ms: res.latency_ms,
          token_usage: res.token_usage ?? null,
        });
        setPasses({ researcher: 1, analyst: 1, writer: 1 });
        setLastDone("writer");
      }
    } catch (e) {
      if ((e as Error).name === "AbortError") {
        setError("Run cancelled.");
      } else if (e instanceof ApiError) {
        setError(
          e.status === 429
            ? "Rate limit reached. Wait a minute and try again."
            : e.message,
        );
        setFailedAt(lastDone ? nextOf(lastDone) : "researcher");
      } else {
        setError((e as Error).message);
      }
    } finally {
      setRunning(false);
      abortRef.current = null;
    }
  };

  const cancel = () => abortRef.current?.abort();

  const retried = (passes.researcher ?? 0) > 1;
  const showPipeline = running || Boolean(result) || Boolean(failedAt);

  return (
    <div className="research">
      <Card className="composer">
        <label htmlFor="query" className="composer__label">
          Research query
        </label>
        <textarea
          id="query"
          className="composer__input"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Ask anything that needs current, sourced information…"
          rows={3}
          disabled={running}
          onKeyDown={(e) => {
            if ((e.metaKey || e.ctrlKey) && e.key === "Enter") run();
          }}
        />

        <div className="composer__foot">
          <div className="composer__examples">
            {!query &&
              !running &&
              EXAMPLES.map((ex) => (
                <button key={ex} className="chip" onClick={() => setQuery(ex)} type="button">
                  {ex}
                </button>
              ))}
          </div>

          <div className="composer__actions">
            <kbd className="composer__kbd">⌘↵</kbd>
            {running ? (
              <Button variant="danger" onClick={cancel}>
                Cancel
              </Button>
            ) : (
              <Button variant="primary" onClick={run} disabled={query.trim().length < 3}>
                Run agents
              </Button>
            )}
          </div>
        </div>
      </Card>

      {showPipeline && (
        <Pipeline
          nodes={nodes}
          retried={retried}
          elapsedMs={result ? result.latency_ms : elapsed}
        />
      )}

      <ActivityFeed items={activity} running={running} />

      <div aria-live="polite" className="visually-hidden">
        {running ? "Pipeline running" : result ? "Report ready" : ""}
      </div>

      {error && (
        <Alert tone="danger" title="Run failed">
          {error}
        </Alert>
      )}

      {result && <ReportView result={result} />}

      {!showPipeline && !error && (
        <EmptyState icon="🔬" title="No research yet">
          Enter a query above. Three agents will search, analyse and write a report —
          citing only pages they actually retrieved.
        </EmptyState>
      )}
    </div>
  );
}

function nextOf(agent: string): string {
  const i = AGENTS.findIndex((a) => a.name === agent);
  return AGENTS[Math.min(i + 1, AGENTS.length - 1)]!.name;
}
