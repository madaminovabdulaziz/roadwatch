"use client";
import { useState } from "react";
import SampleResult from "@/components/SampleResult";
import { Choices, Loading, Missing, Section } from "@/components/ui";
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
        {index.state !== "loading" && index.state !== "ok" && (
          <Missing
            what="The annotated samples"
            how="scripts/render_samples.py"
          />
        )}
        {index.state === "ok" && (
          <>
            <Choices
              label="Sample video"
              options={index.data.videos.map((v) => v.name)}
              value={pick}
              onChange={setPick}
            />
            {index.data.videos[pick] && (
              <SampleResult
                key={index.data.videos[pick].id}
                video={index.data.videos[pick]}
              />
            )}
          </>
        )}
      </Section>

      <Section title="Dev-set scores per class">
        {metrics.state === "ok" ? (
          <MetricsTable m={metrics.data} />
        ) : (
          metrics.state !== "loading" && (
            <Missing what="The dev-set evaluation" how="scripts/eval_dev.py" />
          )
        )}
      </Section>

      {gallery.state === "ok" && gallery.data.length > 0 && (
        <Section title="One clip per detected class">
          <div className="clip-grid">
            {gallery.data.map((g) => (
              <figure key={g.clip}>
                <video
                  src={dataUrl(g.clip)}
                  poster={g.poster ? dataUrl(g.poster) : undefined}
                  controls
                  playsInline
                  preload="none"
                />
                <figcaption>
                  <span
                    className="class-swatch"
                    style={{ background: classColor(g.label) }}
                  />
                  {className(g.label)}{" "}
                  <span className="meta">
                    {g.video} at {g.start.toFixed(1)} s
                  </span>
                </figcaption>
              </figure>
            ))}
          </div>
        </Section>
      )}

      {failures.state === "ok" && failures.data.length > 0 && (
        <Section title="Where it fails">
          <div className="clip-grid clip-grid-two">
            {failures.data.map((f) => (
              <figure key={f.title}>
                {f.clip && (
                  // No poster for these clips: load the metadata so the first frame shows.
                  <video
                    src={`${dataUrl(f.clip)}#t=0.1`}
                    controls
                    playsInline
                    preload="metadata"
                  />
                )}
                <figcaption>
                  <strong>{f.title}</strong>
                  <p>{f.why}</p>
                </figcaption>
              </figure>
            ))}
          </div>
        </Section>
      )}
    </>
  );
}

function MetricsTable({ m }: { m: Metrics }) {
  const classes = Object.keys(m.per_class);
  if (!classes.length)
    return <p className="meta">No labelled or predicted events yet.</p>;
  return (
    <div className="table-wrap">
      <p className="table-note">
        Score A (mean over classes of mean F1 at tIoU 0.3/0.5/0.7) ={" "}
        <b>{m.score_a.toFixed(3)}</b> on {m.videos} labelled videos (
        {m.gt_events} events). Official evaluate.py, our own labels.
      </p>
      <table className="data-table">
        <thead>
          <tr>
            <th>Class</th>
            <th>F1@0.3</th>
            <th>F1@0.5</th>
            <th>F1@0.7</th>
            <th>Mean</th>
            <th>TP / FP / FN @0.5</th>
          </tr>
        </thead>
        <tbody>
          {classes.map((c) => {
            const r = m.per_class[c];
            return (
              <tr key={c}>
                <td>
                  <span
                    className="class-swatch"
                    style={{ background: classColor(c) }}
                  />
                  {className(c)}
                </td>
                <td>{r["f1@0.3"].toFixed(2)}</td>
                <td>{r["f1@0.5"].toFixed(2)}</td>
                <td>{r["f1@0.7"].toFixed(2)}</td>
                <td className="strong">{r.f1_mean.toFixed(2)}</td>
                <td>{r["tp_fp_fn@0.5"].join(" / ")}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
