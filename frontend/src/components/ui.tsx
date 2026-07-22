import type { ReactNode, ButtonHTMLAttributes } from "react";
import "./ui.css";

/* ── Button ──────────────────────────────────────────────── */
type ButtonProps = ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: "primary" | "secondary" | "ghost" | "danger";
  size?: "sm" | "md";
  loading?: boolean;
  icon?: ReactNode;
};

export function Button({
  variant = "secondary",
  size = "md",
  loading = false,
  icon,
  children,
  className = "",
  disabled,
  ...rest
}: ButtonProps) {
  return (
    <button
      className={`btn btn--${variant} btn--${size} ${className}`}
      disabled={disabled || loading}
      {...rest}
    >
      {loading ? <span className="btn__spinner" aria-hidden="true" /> : icon}
      {children}
    </button>
  );
}

/* ── Card ────────────────────────────────────────────────── */
export function Card({
  children,
  className = "",
  padded = true,
}: {
  children: ReactNode;
  className?: string;
  padded?: boolean;
}) {
  return (
    <div className={`card ${padded ? "card--padded" : ""} ${className}`}>{children}</div>
  );
}

/* ── Badge ───────────────────────────────────────────────── */
export function Badge({
  children,
  tone = "neutral",
  title,
}: {
  children: ReactNode;
  tone?: "neutral" | "accent" | "success" | "warning" | "danger";
  title?: string;
}) {
  return (
    <span className={`badge badge--${tone}`} title={title}>
      {children}
    </span>
  );
}

/* ── Stat ────────────────────────────────────────────────── */
export function Stat({
  label,
  value,
  hint,
  tone = "neutral",
}: {
  label: string;
  value: ReactNode;
  hint?: string;
  tone?: "neutral" | "accent" | "success" | "warning" | "danger";
}) {
  return (
    <div className="stat" title={hint}>
      <div className="stat__label">{label}</div>
      <div className={`stat__value stat__value--${tone}`}>{value}</div>
      {hint && <div className="stat__hint">{hint}</div>}
    </div>
  );
}

/* ── Alert ───────────────────────────────────────────────── */
export function Alert({
  tone = "neutral",
  title,
  children,
  action,
}: {
  tone?: "neutral" | "accent" | "success" | "warning" | "danger";
  title?: string;
  children?: ReactNode;
  action?: ReactNode;
}) {
  return (
    <div className={`alert alert--${tone}`} role={tone === "danger" ? "alert" : undefined}>
      <div className="alert__body">
        {title && <div className="alert__title">{title}</div>}
        {children && <div className="alert__text">{children}</div>}
      </div>
      {action && <div className="alert__action">{action}</div>}
    </div>
  );
}

/* ── Empty state ─────────────────────────────────────────── */
export function EmptyState({
  icon,
  title,
  children,
}: {
  icon?: ReactNode;
  title: string;
  children?: ReactNode;
}) {
  return (
    <div className="empty">
      {icon && <div className="empty__icon" aria-hidden="true">{icon}</div>}
      <div className="empty__title">{title}</div>
      {children && <div className="empty__text">{children}</div>}
    </div>
  );
}

/* ── Skeleton ────────────────────────────────────────────── */
export function Skeleton({ w = "100%", h = 14 }: { w?: string | number; h?: number }) {
  return <span className="skeleton" style={{ width: w, height: h }} aria-hidden="true" />;
}

/* ── Section heading ─────────────────────────────────────── */
export function SectionTitle({
  children,
  hint,
  action,
}: {
  children: ReactNode;
  hint?: string;
  action?: ReactNode;
}) {
  return (
    <div className="section-title">
      <div>
        <h2>{children}</h2>
        {hint && <p className="section-title__hint">{hint}</p>}
      </div>
      {action}
    </div>
  );
}

/* ── Meter (score bar) ───────────────────────────────────── */
export function Meter({
  value,
  max = 5,
  label,
}: {
  value: number;
  max?: number;
  label: string;
}) {
  const pct = Math.max(0, Math.min(100, (value / max) * 100));
  return (
    <div
      className="meter"
      role="meter"
      aria-valuenow={value}
      aria-valuemin={0}
      aria-valuemax={max}
      aria-label={label}
    >
      <div className="meter__fill" style={{ width: `${pct}%` }} />
    </div>
  );
}
