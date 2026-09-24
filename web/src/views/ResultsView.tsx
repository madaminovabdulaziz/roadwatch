"use client";
import { useState } from "react";
import SampleResult from "@/components/SampleResult";
import { Card, Loading, Missing, Section } from "@/components/ui";
import { className, classColor } from "@/lib/classes";
import { dataUrl, useJson, type Metrics, type ResultsIndex } from "@/lib/data";

interface GalleryItem {
  label: string;
  clip: string;
  poster?: string;
  video: string;
  start: number;
}
interface FailureCase {
  title: string;
  why: string;
  clip?: string;
}

export default function ResultsView() {
  const index = useJson<ResultsIndex>("results/index.json");
  const metrics = useJson<Metrics>("metrics.json");
  const gallery = useJson<GalleryItem[]>("results/gallery.json");
  const failures = useJson<FailureCase[]>("results/failures.json");
  const [pick, setPick] = useState(0);

  return (
    <>
      <Section title="Annotated samples">
        {index.state === "loading" && <Loading />}
        {index.state !== "loading" && index.state !== "ok" && <Missing what="The annotated samples" how="scripts/render_samples.py" />}
        {index.state === "ok" && (
          <>
            <div className="mb-4 flex flex-wrap gap-2">
              {index.data.videos.map((v, i) => (
                <button
                  key={v.id}
                  type="button"
                  onClick={() => setPick(i)}
                  className={`rounded-lg px-3 py-1.5 text-sm ${i === pick ? "bg-sky-500 text-white" : "border border-white/15 hover:bg-white/10"}`}
                >
                  {v.name}
                </button>
              ))}
            </div>
            {index.data.videos[pick] && <SampleResult key={index.data.videos[pick].id} video={index.data.videos[pick]} />}
          </>
        )}
      </Section>

      <Section title="Dev-set scores per class">
        {metrics.state === "ok" ? <MetricsTable m={metrics.data} /> : metrics.state !== "loading" && <Missing what="The dev-set evaluation" how="scripts/eval_dev.py" />}
      </Section>

      {gallery.state === "ok" && gallery.data.length > 0 && (
        <Section title="One clip per detected class">
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {gallery.data.map((g) => (
              <Card key={g.clip}>
                <video src={dataUrl(g.clip)} poster={g.poster ? dataUrl(g.poster) : undefined} controls playsInline preload="none" className="aspect-video w-full rounded-lg bg-black" />
                <div className="mt-2 text-sm">
                  <span className="mr-2 inline-block h-2.5 w-2.5 rounded-sm" style={{ background: classColor(g.label) }} />
                  {className(g.label)} <span className="text-zinc-500">· {g.video} at {g.start.toFixed(1)} s</span>
                </div>
              </Card>
            ))}
          </div>
        </Section>
      )}

      {failures.state === "ok" && failures.data.length > 0 && (
        <Section title="Where it fails">
          <div className="grid gap-4 sm:grid-cols-2">
            {failures.data.map((f) => (
              <Card key={f.title}>
                {f.clip && <video src={dataUrl(f.clip)} controls playsInline preload="none" className="mb-2 aspect-video w-full rounded-lg bg-black" />}
                <div className="font-medium">{f.title}</div>
                <p className="mt-1 text-sm text-zinc-400">{f.why}</p>
              </Card>
            ))}
          </div>
        </Section>
      )}
    </>
  );
}

function MetricsTable({ m }: { m: Metrics }) {
  const classes = Object.keys(m.per_class);
  if (!classes.length) return <p className="text-sm text-zinc-500">No labelled or predicted events yet.</p>;
  return (
    <div className="overflow-x-auto">
      <p className="mb-2 text-sm text-zinc-400">
        Score A (mean over classes of mean F1 at tIoU 0.3/0.5/0.7) = <b>{m.score_a.toFixed(3)}</b> on {m.videos} labelled
        videos ({m.gt_events} events). Official evaluate.py, our own labels.
      </p>
      <table className="w-full text-left text-sm tabular-nums">
        <thead className="text-xs uppercase text-zinc-500">
          <tr>
            <th className="py-2 pr-4 font-medium">Class</th>
            <th className="py-2 pr-4 font-medium">F1@0.3</th>
            <th className="py-2 pr-4 font-medium">F1@0.5</th>
            <th className="py-2 pr-4 font-medium">F1@0.7</th>
            <th className="py-2 pr-4 font-medium">Mean</th>
            <th className="py-2 font-medium">TP / FP / FN @0.5</th>
          </tr>
        </thead>
        <tbody>
          {classes.map((c) => {
            const r = m.per_class[c];
            return (
              <tr key={c} className="border-t border-white/5">
                <td className="py-1.5 pr-4">
                  <span className="mr-2 inline-block h-2.5 w-2.5 rounded-sm" style={{ background: classColor(c) }} />
                  {className(c)}
                </td>
                <td className="py-1.5 pr-4">{r["f1@0.3"].toFixed(2)}</td>
                <td className="py-1.5 pr-4">{r["f1@0.5"].toFixed(2)}</td>
                <td className="py-1.5 pr-4">{r["f1@0.7"].toFixed(2)}</td>
                <td className="py-1.5 pr-4 font-medium">{r.f1_mean.toFixed(2)}</td>
                <td className="py-1.5">{r["tp_fp_fn@0.5"].join(" / ")}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
