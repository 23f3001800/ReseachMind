import "./Pipeline.css";

export type NodeState = "idle" | "active" | "done" | "error";

export interface PipelineNode {
  name: string;
  label: string;
  hint: string;
  state: NodeState;
  detail?: string;
  passes: number;
}

/**
 * Live view of the Researcher → Analyst → Writer graph.
 *
 * The backend emits one event per node *completion*, so "active" is inferred:
 * whichever node follows the last completed one. The retry edge is drawn when
 * the researcher runs more than once.
 */
export function Pipeline({
  nodes,
  retried,
  elapsedMs,
}: {
  nodes: PipelineNode[];
  retried: boolean;
  elapsedMs: number;
}) {
  return (
    <div className="pipeline">
      <div className="pipeline__head">
        <span className="pipeline__title">Agent pipeline</span>
        <span className="pipeline__elapsed mono">{(elapsedMs / 1000).toFixed(1)}s</span>
      </div>

      <ol className="pipeline__track" aria-label="Agent pipeline progress">
        {nodes.map((n, i) => (
          <li key={n.name} className="pipeline__item">
            <div className={`pnode pnode--${n.state}`}>
              <div className="pnode__dot" aria-hidden="true">
                {n.state === "done" ? (
                  <svg viewBox="0 0 16 16" width="12" height="12" fill="none">
                    <path
                      d="M3.5 8.5l3 3 6-7"
                      stroke="currentColor"
                      strokeWidth="2.2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    />
                  </svg>
                ) : n.state === "error" ? (
                  <svg viewBox="0 0 16 16" width="12" height="12" fill="none">
                    <path
                      d="M4 4l8 8M12 4l-8 8"
                      stroke="currentColor"
                      strokeWidth="2.2"
                      strokeLinecap="round"
                    />
                  </svg>
                ) : (
                  <span className="pnode__index">{i + 1}</span>
                )}
              </div>

              <div className="pnode__body">
                <div className="pnode__name">
                  {n.label}
                  {n.passes > 1 && (
                    <span className="pnode__passes" title={`Ran ${n.passes} times`}>
                      ×{n.passes}
                    </span>
                  )}
                </div>
                <div className="pnode__hint">
                  {n.state === "active" ? (
                    <span className="pnode__working">
                      Working<span className="dots" aria-hidden="true" />
                    </span>
                  ) : (
                    (n.detail ?? n.hint)
                  )}
                </div>
              </div>
            </div>

            {i < nodes.length - 1 && (
              <div
                className={`pipeline__edge ${
                  nodes[i + 1]!.state !== "idle" ? "pipeline__edge--lit" : ""
                }`}
                aria-hidden="true"
              />
            )}
          </li>
        ))}
      </ol>

      {retried && (
        <div className="pipeline__retry">
          <svg viewBox="0 0 16 16" width="13" height="13" fill="none" aria-hidden="true">
            <path
              d="M13 8a5 5 0 11-1.6-3.7M13 2v3h-3"
              stroke="currentColor"
              strokeWidth="1.6"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
          Analyst found gaps — researcher ran a second, gap-targeted pass
        </div>
      )}
    </div>
  );
}
