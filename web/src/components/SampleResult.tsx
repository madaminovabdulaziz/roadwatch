"use client";
// One precomputed sample (scripts/render_samples.py): loads its events and risk JSON, then plays it.
import VideoResult from "@/components/VideoResult";
import { Loading, Missing } from "@/components/ui";
import type { EventTuple, RiskPoint } from "@/lib/classes";
import { dataUrl, useJson, type ResultVideo } from "@/lib/data";

export default function SampleResult({ video }: { video: ResultVideo }) {
  const events = useJson<EventTuple[]>(video.events);
  const risk = useJson<RiskPoint[]>(video.risk);
  if (events.state === "loading" || risk.state === "loading")
    return <Loading />;
  if (events.state !== "ok")
    return (
      <Missing
        what={`Events of ${video.name}`}
        how="scripts/render_samples.py"
      />
    );
  return (
    <VideoResult
      src={dataUrl(video.video)}
      poster={video.poster ? dataUrl(video.poster) : undefined}
      events={events.data}
      risk={risk.state === "ok" ? risk.data : []}
      duration={video.duration}
    />
  );
}
