"use client";
// Aggregates the events of every rendered sample: per class, per minute, and where they happen.
import { useEffect, useState } from "react";
import Chart from "@/components/Chart";
import { Loading, Missing, Section } from "@/components/ui";
import { CLASSES, className, classColor, type EventTuple } from "@/lib/classes";
import { fetchJson, useJson, type ResultsIndex } from "@/lib/data";

export default function DashboardView() {
  const index = useJson<ResultsIndex>("results/index.json");
  const [events, setEvents] = useState<
    { video: string; e: EventTuple }[] | null
  >(null);
  const [heatOk, setHeatOk] = useState(true);

  useEffect(() => {
    if (index.state !== "ok") return;
    Promise.all(
      index.data.videos.map(async (v) =>
        ((await fetchJson<EventTuple[]>(v.events)) ?? []).map((e) => ({
          video: v.name,
          e,
        })),
      ),
    )
      .then((all) => setEvents(all.flat()))
      .catch(() => setEvents([]));
  }, [index]);

  if (index.state === "loading") return <Loading />;
  if (index.state !== "ok")
    return (
      <Missing what="The rendered samples" how="scripts/render_samples.py" />
    );
  if (events === null) return <Loading />;

  const labels = CLASSES.filter((c) => events.some((x) => x.e[2] === c));
  const maxMinute = Math.max(0, ...events.map((x) => Math.floor(x.e[0] / 60)));
  const minutes = Array.from({ length: maxMinute + 1 }, (_, i) => i);

  return (
    <>
      <dl className="figures">
        <div>
          <dt>events in {index.data.videos.length} videos</dt>
          <dd>{events.length}</dd>
        </div>
        <div>
          <dt>different classes</dt>
          <dd>{labels.length}</dd>
        </div>
        <div>
          <dt>of video reviewed</dt>
          <dd>
            {(
              index.data.videos.reduce((s, v) => s + v.duration, 0) / 60
            ).toFixed(1)}{" "}
            min
          </dd>
        </div>
      </dl>

      {events.length === 0 ? (
        <p className="meta">No events detected yet.</p>
      ) : (
        <>
          <Section title="Events per class">
            <Chart
              data={[
                {
                  x: labels.map(className),
                  y: labels.map(
                    (c) => events.filter((x) => x.e[2] === c).length,
                  ),
                  type: "bar",
                  marker: { color: labels.map(classColor) },
                },
              ]}
            />
          </Section>
          <Section title="Events per minute of video">
            <Chart
              data={labels.map((c) => ({
                x: minutes,
                y: minutes.map(
                  (m) =>
                    events.filter(
                      (x) => x.e[2] === c && Math.floor(x.e[0] / 60) === m,
                    ).length,
                ),
                type: "bar",
                name: className(c),
                marker: { color: classColor(c) },
              }))}
              layout={{
                barmode: "stack",
                xaxis: { title: { text: "minute" } },
              }}
            />
          </Section>
        </>
      )}

      {heatOk && (
        <Section title="Where events happen">
          <figure className="figure">
            {/* eslint-disable-next-line @next/next/no-img-element -- static export, plain img */}
            <img
              src="/data/dashboard/event_heat.jpg"
              alt="Paths of the road users behind each event on the reference frame"
              loading="lazy"
              onError={() => setHeatOk(false)}
            />
            <figcaption>
              The paths of the road users behind each event, while the rule held
              for them, on the reference frame.
            </figcaption>
          </figure>
        </Section>
      )}
    </>
  );
}
