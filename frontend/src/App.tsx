import { useCallback, useEffect, useState } from "react";
import { api } from "@/api/client";
import type { GraphInfo, Health } from "@/types";
import { useLocalStorage, useTheme } from "@/hooks/useLocalStorage";
import { Research } from "@/views/Research";
import { Documents } from "@/views/Documents";
import { Evaluate } from "@/views/Evaluate";
import { Usage } from "@/views/Usage";
import { History } from "@/views/History";
import { Badge } from "@/components/ui";
import "./App.css";

type Tab = "research" | "documents" | "evaluate" | "usage" | "history";

const TABS: Array<{ id: Tab; label: string; icon: string }> = [
  { id: "research", label: "Research", icon: "🔬" },
  { id: "documents", label: "Documents", icon: "📄" },
  { id: "evaluate", label: "Evaluate", icon: "⚖️" },
  { id: "usage", label: "Usage", icon: "📊" },
  { id: "history", label: "History", icon: "🗒️" },
];

function newThreadId() {
  const rand =
    globalThis.crypto?.randomUUID?.().slice(0, 8) ??
    Math.random().toString(16).slice(2, 10);
  return `session-${rand}`;
}

export default function App() {
  const [tab, setTab] = useState<Tab>("research");
  const [threadId, setThreadId] = useLocalStorage("ra.thread", newThreadId());
  const [streaming, setStreaming] = useLocalStorage("ra.streaming", true);
  const [navOpen, setNavOpen] = useState(false);
  const { mode, cycle } = useTheme();

  const [health, setHealth] = useState<Health | null>(null);
  const [graph, setGraph] = useState<GraphInfo | null>(null);
  const [offline, setOffline] = useState(false);

  const ping = useCallback(async () => {
    try {
      const h = await api.health();
      setHealth(h);
      setOffline(false);
      try {
        setGraph(await api.graph());
      } catch {
        /* graph is behind auth; not fatal for the shell */
      }
    } catch {
      setOffline(true);
      setHealth(null);
    }
  }, []);

  useEffect(() => {
    void ping();
  }, [ping]);

  // Close the mobile drawer whenever the route changes.
  useEffect(() => setNavOpen(false), [tab]);

  const themeLabel = mode === "system" ? "Auto" : mode === "dark" ? "Dark" : "Light";
  const themeIcon = mode === "system" ? "◐" : mode === "dark" ? "☾" : "☀";

  return (
    <div className="app">
      <a href="#main" className="skip-link">
        Skip to content
      </a>

      <aside className={`sidebar ${navOpen ? "sidebar--open" : ""}`}>
        <div className="brand">
          <div className="brand__mark" aria-hidden="true">
            ⬡
          </div>
          <div>
            <div className="brand__name">researchMind</div>
            <div className="brand__sub">Researcher → Analyst → Writer</div>
          </div>
        </div>

        <nav className="nav" aria-label="Sections">
          {TABS.map((t) => (
            <button
              key={t.id}
              className={`nav__item ${tab === t.id ? "nav__item--active" : ""}`}
              onClick={() => setTab(t.id)}
              aria-current={tab === t.id ? "page" : undefined}
            >
              <span className="nav__icon" aria-hidden="true">
                {t.icon}
              </span>
              {t.label}
              {t.id === "documents" && health?.rag_available === false && (
                <span className="nav__dot" title="RAG disabled on this backend" />
              )}
            </button>
          ))}
        </nav>

        <div className="panel">
          <div className="panel__title">Session</div>
          <label className="field">
            <span className="field__label">Thread ID</span>
            <div className="field__row">
              <input
                className="field__input mono"
                value={threadId}
                onChange={(e) => setThreadId(e.target.value)}
                spellCheck={false}
              />
              <button
                className="field__btn"
                onClick={() => setThreadId(newThreadId())}
                title="Start a new thread"
                type="button"
              >
                ↻
              </button>
            </div>
          </label>

          <label className="toggle">
            <input
              type="checkbox"
              checked={streaming}
              onChange={(e) => setStreaming(e.target.checked)}
            />
            <span className="toggle__track" aria-hidden="true">
              <span className="toggle__thumb" />
            </span>
            <span className="toggle__label">
              Live streaming
              <span className="toggle__hint">Show agents as they finish</span>
            </span>
          </label>
        </div>

        <div className="sidebar__foot">
          <button className="statusbar" onClick={() => void ping()} type="button">
            <span
              className={`statusbar__led statusbar__led--${
                offline ? "off" : "on"
              }`}
              aria-hidden="true"
            />
            <span className="statusbar__text">
              {offline ? "API unreachable" : "API online"}
            </span>
          </button>

          <div className="sidebar__meta">
            {health && (
              <>
                <Badge tone={health.auth_required ? "success" : "warning"}>
                  {health.auth_required ? "authenticated" : "open"}
                </Badge>
                <Badge tone={health.rag_available ? "accent" : "neutral"}>
                  RAG {health.rag_available ? "on" : "off"}
                </Badge>
              </>
            )}
          </div>

          <button className="theme-btn" onClick={cycle} type="button">
            <span aria-hidden="true">{themeIcon}</span> {themeLabel}
          </button>
        </div>
      </aside>

      {navOpen && (
        <button
          className="scrim"
          onClick={() => setNavOpen(false)}
          aria-label="Close navigation"
        />
      )}

      <div className="main-wrap">
        <header className="topbar">
          <button
            className="topbar__menu"
            onClick={() => setNavOpen((v) => !v)}
            aria-label="Toggle navigation"
            aria-expanded={navOpen}
            type="button"
          >
            ☰
          </button>
          <h1 className="topbar__title">
            {TABS.find((t) => t.id === tab)?.label}
          </h1>
          {graph && tab === "research" && (
            <span className="topbar__flow mono">{graph.flow}</span>
          )}
        </header>

        <main id="main" className="main">
          {offline && (
            <div className="offline">
              <strong>Can’t reach the API.</strong> The backend may be starting up — it
              scales to zero when idle, so the first request after a pause can take up to a
              minute. <button onClick={() => void ping()}>Retry</button>
            </div>
          )}

          {tab === "research" && <Research threadId={threadId} streaming={streaming} />}
          {tab === "documents" && <Documents ragAvailable={health?.rag_available ?? null} />}
          {tab === "evaluate" && <Evaluate threadId={threadId} />}
          {tab === "usage" && <Usage />}
          {tab === "history" && <History threadId={threadId} />}
        </main>
      </div>
    </div>
  );
}
