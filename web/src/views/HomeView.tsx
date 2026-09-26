"use client";
// Hero clip and the three key numbers; every number comes from a generated file or shows "—".
import { Stat } from "@/components/ui";
import { dataUrl, useJson, type Metrics, type ResultsIndex } from "@/lib/data";

export default function HomeView() {
  const results = useJson<ResultsIndex>("results/index.json");
  const metrics = useJson<Metrics>("metrics.json");
  const hero = results.state === "ok" ? results.data.videos[0] : undefined;
  const enabled = results.state === "ok" ? results.data.enabled_classes : undefined;
  const runtime = results.state === "ok" ? results.data.runtime_x_duration : undefined;

  return (
    <>
      {hero && (
        <video
          src={`${dataUrl(hero.video)}#t=0,20`}
          poster={hero.poster ? dataUrl(hero.poster) : undefined}
          autoPlay
          muted
          loop
          playsInline
          className="mt-10 aspect-video w-full rounded-2xl border border-white/10 bg-black"
        />
      )}
      <div className="mt-10 grid gap-4 sm:grid-cols-3">
        <Stat label="event classes emitted" value={enabled ? `${enabled.length} / 14` : "—"} hint="only classes that passed our checks" />
        <Stat
          label="dev-set Score A (mean F1)"
          value={metrics.state === "ok" ? metrics.data.score_a.toFixed(2) : "—"}
          hint={metrics.state === "ok" ? `${metrics.data.gt_events} labelled events, ${metrics.data.videos} videos` : undefined}
        />
        <Stat label="runtime × video length on a T4" value={runtime ? `${runtime.toFixed(2)}×` : "—"} hint="official harness, worst sample · budget: 3×" />
      </div>
    </>
  );
}
