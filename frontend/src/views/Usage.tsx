import { useCallback, useEffect, useState } from "react";
import { api } from "@/api/client";
import type { UsageSummary } from "@/types";
import { Alert, Button, Card, EmptyState, SectionTitle, Skeleton, Stat } from "@/components/ui";
import "./Usage.css";

export function Usage() {
  const [data, setData] = useState<UsageSummary | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      setData(await api.usage());
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const reset = async () => {
    try {
      await api.resetUsage();
      await load();
    } catch (e) {
      setError((e as Error).message);
    }
  };

  return (
    <div className="usage">
      <SectionTitle
        hint="Token counts come from the provider's own response metadata — measured per call, not estimated."
        action={
          <div className="usage__actions">
            <Button size="sm" variant="ghost" onClick={() => void load()} loading={loading}>
              Refresh
            </Button>
            <Button size="sm" variant="danger" onClick={reset}>
              Reset
            </Button>
          </div>
        }
      >
        Usage &amp; cost
      </SectionTitle>

      {error && (
        <Alert tone="danger" title="Could not load usage">
          {error}
        </Alert>
      )}

      {loading && !data ? (
        <div className="usage__stats">
          {Array.from({ length: 4 }).map((_, i) => (
            <Card key={i}>
              <Skeleton h={12} w="55%" />
              <div style={{ height: 8 }} />
              <Skeleton h={22} w="70%" />
            </Card>
          ))}
        </div>
      ) : data ? (
        <>
          <div className="usage__stats">
            <Stat label="Requests" value={data.total_requests.toLocaleString()} />
            <Stat label="Total tokens" value={data.total_tokens.toLocaleString()} />
            <Stat
              label="Avg / request"
              value={data.avg_tokens_per_request.toLocaleString()}
            />
            <Stat
              label="Est. cost"
              value={`$${data.total_cost_usd.toFixed(4)}`}
              tone="accent"
            />
          </div>

          <Card>
            <SectionTitle>Recent requests</SectionTitle>
            {data.recent_requests.length === 0 ? (
              <EmptyState title="No requests recorded yet">
                Run a query and the token and cost figures land here.
              </EmptyState>
            ) : (
              <div className="scroll-x">
                <table className="table">
                  <thead>
                    <tr>
                      <th>Thread</th>
                      <th className="num">Tokens</th>
                      <th className="num">Latency</th>
                      <th className="num">Cost</th>
                      <th>Per agent</th>
                    </tr>
                  </thead>
                  <tbody>
                    {[...data.recent_requests].reverse().map((r, i) => (
                      <tr key={i}>
                        <td className="mono">{r.thread_id || "—"}</td>
                        <td className="num mono">{r.total_tokens.toLocaleString()}</td>
                        <td className="num mono">{(r.latency_ms / 1000).toFixed(1)}s</td>
                        <td className="num mono">${r.estimated_cost_usd.toFixed(6)}</td>
                        <td className="usage__breakdown">
                          {Object.entries(r.agent_breakdown ?? {}).map(([agent, u]) => (
                            <span key={agent} className="usage__chip">
                              {agent} <b className="mono">{u.total_tokens.toLocaleString()}</b>
                            </span>
                          ))}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </Card>
        </>
      ) : null}
    </div>
  );
}
