/** Mirrors backend/schemas/models.py. Keep in sync with the Pydantic models. */

export interface Source {
  url: string;
  title: string;
  provider: string;
}

export interface Report {
  title: string;
  summary: string;
  research_findings: string[];
  analysis: string[];
  conclusion: string;
  sources?: Source[];
  confidence?: number;
  needs_human_review?: boolean;
}

export interface TokenUsage {
  total_input_tokens: number;
  total_output_tokens: number;
  total_tokens: number;
  estimated_cost_usd: number;
  agent_breakdown: Record<string, AgentUsage>;
  latency_ms?: number;
  thread_id?: string;
}

export interface AgentUsage {
  input_tokens: number;
  output_tokens: number;
  total_tokens: number;
  model: string;
}

/** Normalised result shape shared by the streaming and non-streaming paths. */
export interface RunResult {
  report: Report;
  sources: Source[];
  confidence: number;
  needs_human_review: boolean;
  iterations: number;
  latency_ms: number;
  token_usage?: TokenUsage | null;
}

export interface ChatResponse {
  thread_id: string;
  report: Report;
  latency_ms: number;
  iterations: number;
  token_usage?: TokenUsage | null;
}

export type AgentName = "researcher" | "analyst" | "writer";

export interface StreamEvent {
  // Mirrors the Literal on backend/schemas/models.py StreamEvent — keep in sync.
  event:
    | "agent_start"
    | "agent_end"
    | "tool_call"
    | "tool_result"
    | "error"
    | "complete";
  agent?: string | null;
  content?: string | null;
  data?: Record<string, unknown> | null;
}

export interface Health {
  status: string;
  service: string;
  rag_available: boolean;
  auth_required: boolean;
}

export interface GraphInfo {
  agents: string[];
  flow: string;
  routing: string;
  memory: string;
  guardrails: string[];
}

export interface DocumentInfo {
  filename: string;
  text_length: number;
  num_chunks: number;
}

export interface SearchHit {
  content: string;
  source: string;
  chunk_index: number;
  score: number;
}

export interface DimensionScore {
  score: number;
  explanation: string;
}

export interface Evaluation {
  factual_accuracy: DimensionScore;
  analytical_depth: DimensionScore;
  completeness: DimensionScore;
  clarity: DimensionScore;
  overall_score: number;
  summary: string;
}

export interface EvaluateResponse {
  query: string;
  report: Report;
  sources: Source[];
  evaluation: Evaluation;
}

export interface UsageSummary {
  total_requests: number;
  total_input_tokens: number;
  total_output_tokens: number;
  total_tokens: number;
  total_cost_usd: number;
  avg_tokens_per_request: number;
  recent_requests: Array<{
    thread_id: string;
    latency_ms: number;
    total_tokens: number;
    estimated_cost_usd: number;
    agent_breakdown: Record<string, AgentUsage>;
  }>;
}

export interface HistoryResponse {
  thread_id: string;
  exchanges: Array<{ query: string; report: string }>;
  count: number;
}
