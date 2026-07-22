import { useCallback, useEffect, useState } from "react";
import { api } from "@/api/client";
import type { HistoryResponse } from "@/types";
import { Alert, Button, Card, EmptyState, SectionTitle } from "@/components/ui";
import "./History.css";

export function History({ threadId }: { threadId: string }) {
  const [data, setData] = useState<HistoryResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      setData(await api.history(threadId));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setLoading(false);
    }
  }, [threadId]);

  useEffect(() => {
    void load();
  }, [load]);

  const clear = async () => {
    try {
      await api.clearHistory(threadId);
      await load();
    } catch (e) {
      setError((e as Error).message);
    }
  };

  return (
    <div className="history">
      <SectionTitle
        hint={`Exchanges saved against thread “${threadId}”. Both streaming and non-streaming runs are recorded.`}
        action={
          <div className="history__actions">
            <Button size="sm" variant="ghost" onClick={() => void load()} loading={loading}>
              Refresh
            </Button>
            <Button size="sm" variant="danger" onClick={clear} disabled={!data?.count}>
              Clear thread
            </Button>
          </div>
        }
      >
        History
      </SectionTitle>

      {error && (
        <Alert tone="danger" title="Could not load history">
          {error}
        </Alert>
      )}

      <Alert tone="neutral">
        History lives in the backend’s local SQLite file, which does not survive a restart
        or scale-to-zero. Treat it as session-scoped until shared storage is wired up.
      </Alert>

      {data && data.count === 0 ? (
        <EmptyState icon="🗒️" title="No history for this thread">
          Completed runs are saved here automatically.
        </EmptyState>
      ) : (
        <div className="history__list">
          {data?.exchanges.map((ex, i) => (
            <Card key={i} className="exchange">
              <div className="exchange__q">
                <span className="exchange__n mono">{i + 1}</span>
                <span>{ex.query}</span>
              </div>
              <p className="exchange__a">{ex.report}</p>
            </Card>
          ))}
        </div>
      )}
    </div>
  );
}
