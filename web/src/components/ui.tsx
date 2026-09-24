// Small layout pieces shared by every page.
import type { ReactNode } from "react";

export function Page({ title, lead, children }: { title: string; lead?: ReactNode; children: ReactNode }) {
  return (
    <main className="mx-auto w-full max-w-6xl px-4 py-8 sm:py-12">
      <h1 className="text-2xl font-semibold tracking-tight sm:text-3xl">{title}</h1>
      {lead && <p className="mt-2 max-w-3xl text-zinc-400">{lead}</p>}
      <div className="mt-8 space-y-12">{children}</div>
    </main>
  );
}

/** A titled block; `changed` is the one sentence on what this finding changed in our solution. */
export function Section({ title, changed, children }: { title: string; changed?: string; children: ReactNode }) {
  return (
    <section>
      <h2 className="text-lg font-semibold">{title}</h2>
      {changed && <p className="mt-1 text-sm text-sky-300/90">What this changed: {changed}</p>}
      <div className="mt-4">{children}</div>
    </section>
  );
}

export function Card({ children, className = "" }: { children: ReactNode; className?: string }) {
  return <div className={`rounded-xl border border-white/10 bg-white/[0.03] p-4 ${className}`}>{children}</div>;
}

/** Shown while a generated data file does not exist yet: says which script produces it. */
export function Missing({ what, how }: { what: string; how: string }) {
  return (
    <Card className="border-dashed text-sm text-zinc-400">
      {what} is not generated yet. <span className="text-zinc-500">({how})</span>
    </Card>
  );
}

export function Loading() {
  return <div className="h-24 animate-pulse rounded-xl bg-white/5" />;
}

export function Stat({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <Card>
      <div className="text-3xl font-semibold tabular-nums">{value}</div>
      <div className="mt-1 text-sm text-zinc-400">{label}</div>
      {hint && <div className="mt-1 text-xs text-zinc-500">{hint}</div>}
    </Card>
  );
}
