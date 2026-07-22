import { useEffect, useRef } from "react";
import type { Activity } from "@/api/client";
import "./ActivityFeed.css";

const ICONS: Record<Activity["kind"], string> = {
  agent_start: "▸",
  tool_call: "⌕",
  tool_result: "✓",
  agent_end: "●",
};

/**
 * Live feed of what the agents are doing mid-run.
 *
 * The pipeline strip shows *which* stage is active; this shows *what it is
 * actually doing* — the search queries the researcher chose. That is the part
 * that makes an agent legible rather than a spinner, and it is the only window
 * into the ~40s the researcher spends before its node completes.
 */
export function ActivityFeed({ items, running }: { items: Activity[]; running: boolean }) {
  const endRef = useRef<HTMLDivElement>(null);
  const listRef = useRef<HTMLOListElement>(null);

  // Follow the tail, but only when the user hasn't scrolled up to read.
  useEffect(() => {
    const list = listRef.current;
    if (!list) return;
    const nearBottom = list.scrollHeight - list.scrollTop - list.clientHeight < 48;
    if (nearBottom) endRef.current?.scrollIntoView({ block: "nearest" });
  }, [items]);

  if (!items.length) return null;

  return (
    <div className="activity">
      <div className="activity__head">
        <span className="activity__title">Activity</span>
        {running && <span className="activity__live">live</span>}
      </div>

      <ol className="activity__list" ref={listRef} aria-live="polite" aria-relevant="additions">
        {items.map((a, i) => (
          <li key={i} className={`act act--${a.kind}`}>
            <span className="act__icon" aria-hidden="true">
              {ICONS[a.kind]}
            </span>
            <span className="act__label">
              {a.kind === "tool_call" && a.query ? (
                <>
                  Searching <span className="act__query">{a.query}</span>
                </>
              ) : (
                a.label
              )}
            </span>
            {a.newSources !== undefined && a.newSources > 0 && (
              <span className="act__count">+{a.newSources}</span>
            )}
          </li>
        ))}
        <div ref={endRef} />
      </ol>
    </div>
  );
}
