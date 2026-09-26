import type { ReactNode } from "react";

export function Arrow({ diagonal = false }: { diagonal?: boolean }) {
  return (
    <svg
      width="16"
      height="16"
      viewBox="0 0 24 24"
      fill="none"
      aria-hidden="true"
    >
      <path
        d={diagonal ? "M5 19 19 5M5 5h14v14" : "M4 12h16m-6-6 6 6-6 6"}
        stroke="currentColor"
        strokeWidth="1.5"
      />
    </svg>
  );
}
/** Pause bars while playing, a play triangle while paused. */
export function PlayPause({ playing }: { playing: boolean }) {
  return (
    <svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true">
      {playing ? (
        <path d="M3 1.5v9M9 1.5v9" stroke="currentColor" strokeWidth="1.6" />
      ) : (
        <path d="M3 1.2v9.6L10.6 6z" fill="currentColor" />
      )}
    </svg>
  );
}
export function Brand() {
  return (
    <span className="brand">
      <svg
        width="29"
        height="30"
        viewBox="0 0 29 30"
        fill="none"
        aria-hidden="true"
      >
        <path
          d="m3 26 8-22h7l8 22M7 18h15M14.5 5v5m0 4v5m0 4v5"
          stroke="currentColor"
          strokeWidth="1.8"
        />
      </svg>
      <span className="brand-word">RoadWatch</span>
    </span>
  );
}
/** Inner-page frame: a large title and an optional lead, then the page's sections. */
export function Page({
  title,
  lead,
  children,
}: {
  title: string;
  lead?: ReactNode;
  children: ReactNode;
}) {
  return (
    <main id="main-content" className="page">
      <header className="page-head site-container">
        <h1>{title}</h1>
        {lead && <p className="page-lead">{lead}</p>}
      </header>
      <div className="page-body site-container">{children}</div>
    </main>
  );
}
/** A titled block of a page; `changed` states what the finding changed in the solution. */
export function Section({
  title,
  changed,
  children,
}: {
  title: string;
  changed?: string;
  children: ReactNode;
}) {
  return (
    <section className="page-section">
      <div className="section-head">
        <h2>{title}</h2>
        {changed && (
          <p className="section-note">
            <strong>What it changed:</strong> {changed}
          </p>
        )}
      </div>
      {children}
    </section>
  );
}
/** One-of-many picker (sample video and the like). */
export function Choices({
  label,
  options,
  value,
  onChange,
}: {
  label: string;
  options: string[];
  value: number;
  onChange: (index: number) => void;
}) {
  return (
    <div className="choices" role="group" aria-label={label}>
      {options.map((option, i) => (
        <button
          key={option}
          type="button"
          aria-pressed={i === value}
          onClick={() => onChange(i)}
        >
          {option}
        </button>
      ))}
    </div>
  );
}
export function Card({
  children,
  className = "",
}: {
  children: ReactNode;
  className?: string;
}) {
  return <div className={`panel ${className}`}>{children}</div>;
}
export function Missing({ what, how }: { what: string; how: string }) {
  return (
    <Card className="missing-state">
      <p>{what} is not available yet.</p>
      <details>
        <summary>How to generate it</summary>
        <code>{how}</code>
      </details>
    </Card>
  );
}
export function Loading() {
  return (
    <div className="loading-state" role="status">
      <span className="loading-line" aria-hidden="true" />
      Loading…
    </div>
  );
}
