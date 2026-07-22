import { useMemo } from "react";
import type { RunResult, Source } from "@/types";
import { Alert, Badge, Button, Card, Stat } from "./ui";
import "./ReportView.css";

function hostOf(url: string): string {
  try {
    return new URL(url).hostname.replace(/^www\./, "");
  } catch {
    return url;
  }
}

function confidenceTone(c: number) {
  if (c >= 0.75) return "success" as const;
  if (c >= 0.5) return "warning" as const;
  return "danger" as const;
}

export function toMarkdown(r: RunResult): string {
  const lines: string[] = [
    `# ${r.report.title}`,
    "",
    "## Summary",
    r.report.summary,
    "",
    "## Key findings",
    ...r.report.research_findings.filter(Boolean).map((f) => `- ${f}`),
    "",
    "## Analysis",
    ...r.report.analysis.filter(Boolean).map((a) => `- ${a}`),
    "",
    "## Conclusion",
    r.report.conclusion,
    "",
  ];
  if (r.sources.length) {
    lines.push("## Sources");
    for (const s of r.sources) lines.push(`- [${s.title || s.url}](${s.url})`);
    lines.push("");
  }
  lines.push(
    `---`,
    `Confidence ${r.confidence.toFixed(2)} · ${r.iterations} iterations · ${Math.round(
      r.latency_ms,
    )}ms` + (r.token_usage ? ` · ${r.token_usage.total_tokens} tokens` : ""),
  );
  return lines.join("\n");
}

export function SourceList({ sources }: { sources: Source[] }) {
  if (!sources.length) {
    return (
      <Alert tone="warning" title="No sources retrieved">
        The agent produced this report without a successful web search, so nothing in it is
        citable. Treat the findings as unverified.
      </Alert>
    );
  }

  return (
    <ol className="sources">
      {sources.map((s, i) => (
        <li key={s.url + i} className="source">
          <span className="source__index mono">{i + 1}</span>
          <div className="source__body">
            <a href={s.url} target="_blank" rel="noopener noreferrer" className="source__title">
              {s.title || s.url}
            </a>
            <div className="source__meta">
              <span className="source__host mono">{hostOf(s.url)}</span>
              {s.provider && <Badge tone="neutral">{s.provider}</Badge>}
            </div>
          </div>
        </li>
      ))}
    </ol>
  );
}

export function ReportView({ result }: { result: RunResult }) {
  const { report, sources, token_usage } = result;

  const markdown = useMemo(() => toMarkdown(result), [result]);

  const download = () => {
    const blob = new Blob([markdown], { type: "text/markdown" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${report.title.replace(/[^\w\s-]/g, "").slice(0, 60) || "report"}.md`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const copy = () => navigator.clipboard?.writeText(markdown);

  return (
    <div className="report">
      {result.needs_human_review && (
        <Alert
          tone="warning"
          title="Human review recommended"
        >
          Confidence came in at {result.confidence.toFixed(2)}, below the configured
          threshold. Verify the findings against the sources before relying on this.
        </Alert>
      )}

      <div className="report__stats">
        <Stat
          label="Confidence"
          value={result.confidence.toFixed(2)}
          tone={confidenceTone(result.confidence)}
        />
        <Stat label="Sources" value={sources.length} tone={sources.length ? "neutral" : "danger"} />
        <Stat label="Iterations" value={result.iterations} />
        <Stat label="Latency" value={`${(result.latency_ms / 1000).toFixed(1)}s`} />
        {token_usage && token_usage.total_tokens > 0 && (
          <Stat
            label="Tokens"
            value={token_usage.total_tokens.toLocaleString()}
            hint={`≈ $${token_usage.estimated_cost_usd.toFixed(6)}`}
          />
        )}
      </div>

      <Card className="report__body">
        <header className="report__header">
          <h2 className="report__title">{report.title}</h2>
          <div className="report__actions">
            <Button size="sm" variant="ghost" onClick={copy}>
              Copy
            </Button>
            <Button size="sm" variant="secondary" onClick={download}>
              Download .md
            </Button>
          </div>
        </header>

        <p className="report__summary">{report.summary}</p>

        <div className="report__columns">
          <section>
            <h3 className="report__h3">Key findings</h3>
            {report.research_findings.filter(Boolean).length ? (
              <ul className="report__list">
                {report.research_findings.filter(Boolean).map((f, i) => (
                  <li key={i}>{f}</li>
                ))}
              </ul>
            ) : (
              <p className="report__none">No findings returned.</p>
            )}
          </section>

          <section>
            <h3 className="report__h3">Analysis</h3>
            {report.analysis.filter(Boolean).length ? (
              <ul className="report__list">
                {report.analysis.filter(Boolean).map((a, i) => (
                  <li key={i}>{a}</li>
                ))}
              </ul>
            ) : (
              <p className="report__none">No analysis returned.</p>
            )}
          </section>
        </div>

        <section className="report__conclusion">
          <h3 className="report__h3">Conclusion</h3>
          <p>{report.conclusion}</p>
        </section>

        <section className="report__sources">
          <h3 className="report__h3">
            Sources
            <span className="report__sources-note">
              retrieved pages only — never model-generated
            </span>
          </h3>
          <SourceList sources={sources} />
        </section>
      </Card>
    </div>
  );
}
