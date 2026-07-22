import { useCallback, useEffect, useRef, useState } from "react";
import { api, ApiError } from "@/api/client";
import type { DocumentInfo, SearchHit } from "@/types";
import { Alert, Badge, Button, Card, EmptyState, SectionTitle } from "@/components/ui";
import "./Documents.css";

const ACCEPT = ".pdf,.txt,.md";

export function Documents({ ragAvailable }: { ragAvailable: boolean | null }) {
  const [docs, setDocs] = useState<DocumentInfo[]>([]);
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);

  const [q, setQ] = useState("");
  const [hits, setHits] = useState<SearchHit[] | null>(null);
  const [searching, setSearching] = useState(false);

  const inputRef = useRef<HTMLInputElement>(null);

  const load = useCallback(async () => {
    if (!ragAvailable) return;
    setLoading(true);
    try {
      const res = await api.documents();
      setDocs(res.documents);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setLoading(false);
    }
  }, [ragAvailable]);

  useEffect(() => {
    void load();
  }, [load]);

  const upload = async (file: File) => {
    setUploading(true);
    setError(null);
    setNotice(null);
    try {
      const res = await api.upload(file);
      setNotice(`Indexed “${res.filename}” — ${res.num_chunks} chunks from ${res.text_length.toLocaleString()} characters.`);
      await load();
    } catch (e) {
      setError(e instanceof ApiError ? e.message : (e as Error).message);
    } finally {
      setUploading(false);
      if (inputRef.current) inputRef.current.value = "";
    }
  };

  const search = async () => {
    if (!q.trim()) return;
    setSearching(true);
    setError(null);
    try {
      const res = await api.search(q.trim(), 5);
      setHits(res.results);
    } catch (e) {
      setError((e as Error).message);
      setHits(null);
    } finally {
      setSearching(false);
    }
  };

  if (ragAvailable === false) {
    return (
      <div className="docs">
        <SectionTitle hint="Upload PDF, TXT or Markdown files. The researcher queries them alongside web search.">
          Documents
        </SectionTitle>
        <Alert tone="neutral" title="RAG is not enabled on this backend">
          The optional vector-search dependencies (faiss, sentence-transformers) aren’t
          installed. They need roughly 1&nbsp;GB of RAM, so they’re left out of the slim
          deployment image. Install <code className="mono">backend/requirements-rag.txt</code>{" "}
          and redeploy to turn this on — <code className="mono">/health</code> will then
          report <code className="mono">rag_available: true</code>.
        </Alert>
      </div>
    );
  }

  return (
    <div className="docs">
      <SectionTitle hint="Upload PDF, TXT or Markdown files. The researcher queries them alongside web search.">
        Documents
      </SectionTitle>

      {error && (
        <Alert tone="danger" title="Something went wrong">
          {error}
        </Alert>
      )}
      {notice && <Alert tone="success">{notice}</Alert>}

      <div
        className={`dropzone ${dragging ? "dropzone--active" : ""}`}
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          const f = e.dataTransfer.files?.[0];
          if (f) void upload(f);
        }}
      >
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPT}
          className="visually-hidden"
          id="file"
          onChange={(e) => {
            const f = e.target.files?.[0];
            if (f) void upload(f);
          }}
        />
        <div className="dropzone__icon" aria-hidden="true">
          📄
        </div>
        <div className="dropzone__title">
          {uploading ? "Indexing…" : "Drop a document here"}
        </div>
        <div className="dropzone__hint">PDF, TXT or Markdown · max 10 MB</div>
        <Button
          variant="secondary"
          size="sm"
          loading={uploading}
          onClick={() => inputRef.current?.click()}
        >
          Choose file
        </Button>
      </div>

      <Card>
        <SectionTitle
          action={
            <Button size="sm" variant="ghost" onClick={() => void load()} loading={loading}>
              Refresh
            </Button>
          }
        >
          Indexed documents
        </SectionTitle>

        {docs.length === 0 ? (
          <EmptyState title="Nothing indexed yet">
            Uploaded documents appear here and are searched automatically during research.
          </EmptyState>
        ) : (
          <ul className="doclist">
            {docs.map((d) => (
              <li key={d.filename} className="doclist__item">
                <span className="doclist__name">{d.filename}</span>
                <span className="doclist__meta">
                  <Badge tone="neutral">{d.num_chunks} chunks</Badge>
                  <span className="mono">{d.text_length.toLocaleString()} chars</span>
                </span>
              </li>
            ))}
          </ul>
        )}
      </Card>

      <Card>
        <SectionTitle hint="Check what the retriever returns before relying on it during a run.">
          Test retrieval
        </SectionTitle>
        <div className="docs__search">
          <input
            className="input"
            value={q}
            onChange={(e) => setQ(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && search()}
            placeholder="Query the vector index…"
            disabled={docs.length === 0}
          />
          <Button
            variant="secondary"
            onClick={search}
            loading={searching}
            disabled={docs.length === 0 || !q.trim()}
          >
            Search
          </Button>
        </div>

        {hits && (
          <div className="hits">
            {hits.length === 0 ? (
              <EmptyState title="No matches" />
            ) : (
              hits.map((h, i) => (
                <details key={i} className="hit">
                  <summary className="hit__summary">
                    <span className="hit__source">{h.source}</span>
                    <span className="hit__score mono">score {h.score}</span>
                  </summary>
                  <p className="hit__content">{h.content}</p>
                </details>
              ))
            )}
          </div>
        )}
      </Card>
    </div>
  );
}
