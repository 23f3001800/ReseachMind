import type {
  ChatResponse,
  DocumentInfo,
  EvaluateResponse,
  GraphInfo,
  Health,
  HistoryResponse,
  RunResult,
  SearchHit,
  StreamEvent,
  UsageSummary,
} from "@/types";

/**
 * All calls go to a same-origin "/api" prefix.
 *
 * In the container, nginx proxies that path to the backend and attaches the
 * X-API-Key header itself, so the credential never reaches the browser and
 * there is no cross-origin request to configure. In dev, Vite's proxy does the
 * same thing. Either way the app never knows the API key exists.
 */
const BASE = "/api";

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function parseError(res: Response): Promise<string> {
  try {
    const body = await res.json();
    if (typeof body?.detail === "string") return body.detail;
    if (Array.isArray(body?.detail)) {
      // FastAPI validation errors
      return body.detail
        .map((d: { loc?: string[]; msg?: string }) =>
          d.msg ? `${d.loc?.slice(1).join(".") ?? ""} ${d.msg}`.trim() : "",
        )
        .filter(Boolean)
        .join("; ");
    }
    return JSON.stringify(body).slice(0, 300);
  } catch {
    return `${res.status} ${res.statusText}`;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${BASE}${path}`, {
      ...init,
      headers: {
        ...(init?.body && !(init.body instanceof FormData)
          ? { "Content-Type": "application/json" }
          : {}),
        ...init?.headers,
      },
    });
  } catch (e) {
    if ((e as Error).name === "AbortError") throw e;
    throw new ApiError("Could not reach the API. Is the backend running?", 0);
  }

  if (!res.ok) throw new ApiError(await parseError(res), res.status);
  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

export const api = {
  health: () => request<Health>("/health"),
  graph: () => request<GraphInfo>("/agent/graph"),

  chat: (message: string, threadId: string, signal?: AbortSignal) =>
    request<ChatResponse>("/agent/chat", {
      method: "POST",
      body: JSON.stringify({ message, thread_id: threadId }),
      signal,
    }),

  evaluate: (message: string, threadId: string, signal?: AbortSignal) =>
    request<EvaluateResponse>("/agent/evaluate", {
      method: "POST",
      body: JSON.stringify({ message, thread_id: threadId }),
      signal,
    }),

  history: (threadId: string) =>
    request<HistoryResponse>(`/agent/history/${encodeURIComponent(threadId)}`),

  clearHistory: (threadId: string) =>
    request<{ message: string }>(`/agent/history/${encodeURIComponent(threadId)}`, {
      method: "DELETE",
    }),

  documents: () =>
    request<{ documents: DocumentInfo[]; count: number }>("/agent/documents"),

  upload: (file: File, signal?: AbortSignal) => {
    const form = new FormData();
    form.append("file", file);
    return request<{
      filename: string;
      text_length: number;
      num_chunks: number;
      chunk_preview: string;
    }>("/agent/upload", { method: "POST", body: form, signal });
  },

  search: (query: string, k = 5) =>
    request<{ query: string; results: SearchHit[]; count: number }>("/agent/search", {
      method: "POST",
      body: JSON.stringify({ query, k }),
    }),

  usage: () => request<UsageSummary>("/agent/usage"),
  resetUsage: () => request<{ message: string }>("/agent/usage", { method: "DELETE" }),
};

export interface StreamHandlers {
  onAgentStart?: (agent: string, data: Record<string, unknown>) => void;
  onAgentEnd?: (agent: string, data: Record<string, unknown>) => void;
  /** Fired as each search runs, before its result comes back. */
  onActivity?: (activity: Activity) => void;
  onComplete?: (result: RunResult) => void;
  onError?: (message: string) => void;
}

export interface Activity {
  kind: "agent_start" | "tool_call" | "tool_result" | "agent_end";
  agent: string;
  label: string;
  query?: string;
  newSources?: number;
  at: number;
}

/**
 * Consume the SSE pipeline stream.
 *
 * EventSource can't be used here because the endpoint is a POST, so this reads
 * the body stream directly. The buffer is split on the SSE record separator
 * ("\n\n") rather than per-chunk, since a network chunk boundary can land in
 * the middle of a JSON payload.
 */
export async function streamChat(
  message: string,
  threadId: string,
  handlers: StreamHandlers,
  signal?: AbortSignal,
): Promise<void> {
  let res: Response;
  try {
    res = await fetch(`${BASE}/agent/chat/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, thread_id: threadId }),
      signal,
    });
  } catch (e) {
    if ((e as Error).name === "AbortError") throw e;
    throw new ApiError("Could not reach the API. Is the backend running?", 0);
  }

  if (!res.ok) throw new ApiError(await parseError(res), res.status);
  if (!res.body) throw new ApiError("Streaming is not supported by this browser.", 0);

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  const handleRecord = (raw: string) => {
    const line = raw
      .split("\n")
      .find((l) => l.startsWith("data: "));
    if (!line) return;

    let evt: StreamEvent;
    try {
      evt = JSON.parse(line.slice(6));
    } catch {
      return; // a malformed record shouldn't kill the stream
    }

    const data = (evt.data ?? {}) as Record<string, unknown>;

    if (evt.event === "agent_start" && evt.agent) {
      handlers.onAgentStart?.(evt.agent, data);
      handlers.onActivity?.({
        kind: "agent_start",
        agent: evt.agent,
        label: evt.content ?? `${evt.agent} started`,
        at: Date.now(),
      });
    } else if (evt.event === "tool_call") {
      handlers.onActivity?.({
        kind: "tool_call",
        agent: evt.agent ?? "researcher",
        label: evt.content ?? "Searching",
        query: typeof data.query === "string" ? data.query : undefined,
        at: Date.now(),
      });
    } else if (evt.event === "tool_result") {
      handlers.onActivity?.({
        kind: "tool_result",
        agent: evt.agent ?? "researcher",
        label: evt.content ?? "Search finished",
        query: typeof data.query === "string" ? data.query : undefined,
        newSources: typeof data.new_sources === "number" ? data.new_sources : undefined,
        at: Date.now(),
      });
    } else if (evt.event === "agent_end" && evt.agent) {
      handlers.onAgentEnd?.(evt.agent, data);
      handlers.onActivity?.({
        kind: "agent_end",
        agent: evt.agent,
        label: `${evt.agent} finished`,
        at: Date.now(),
      });
    } else if (evt.event === "complete") {
      handlers.onComplete?.({
        report: data.report as RunResult["report"],
        sources: (data.sources as RunResult["sources"]) ?? [],
        confidence: (data.confidence as number) ?? 0,
        needs_human_review: (data.needs_human_review as boolean) ?? false,
        iterations: (data.iterations as number) ?? 0,
        latency_ms: (data.latency_ms as number) ?? 0,
        token_usage: (data.token_usage as RunResult["token_usage"]) ?? null,
      });
    } else if (evt.event === "error") {
      handlers.onError?.(evt.content ?? "Unknown pipeline error");
    }
  };

  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      let sep: number;
      while ((sep = buffer.indexOf("\n\n")) !== -1) {
        const record = buffer.slice(0, sep);
        buffer = buffer.slice(sep + 2);
        if (record.trim()) handleRecord(record);
      }
    }
    // Flush any trailing record that arrived without a final separator.
    if (buffer.trim()) handleRecord(buffer);
  } finally {
    reader.cancel().catch(() => {});
  }
}
