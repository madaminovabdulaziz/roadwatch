"use client";
// Annotated playback + clickable timeline + risk curve + events table, all synced to the video.
import { useCallback, useRef, useState } from "react";
import EventTimeline from "@/components/EventTimeline";
import EventsTable from "@/components/EventsTable";
import RiskCurve from "@/components/RiskCurve";
import type { EventTuple, RiskPoint } from "@/lib/classes";

export default function VideoResult({
  src,
  poster,
  events,
  risk,
  duration,
}: {
  src: string | null;
  poster?: string;
  events: EventTuple[];
  risk: RiskPoint[];
  duration: number;
}) {
  const video = useRef<HTMLVideoElement>(null);
  const [now, setNow] = useState(0);
  const seek = useCallback((t: number) => {
    setNow(t);
    if (video.current) {
      video.current.currentTime = t;
      void video.current.play().catch(() => undefined); // autoplay can be refused; seeking still works
    }
  }, []);

  return (
    <div className="space-y-4">
      {src ? (
        <video
          ref={video}
          src={src}
          poster={poster}
          controls
          playsInline
          preload="metadata"
          className="aspect-video w-full rounded-xl bg-black"
          onTimeUpdate={(e) => setNow(e.currentTarget.currentTime)}
        />
      ) : (
        <div className="grid aspect-video w-full place-items-center rounded-xl bg-white/5 text-sm text-zinc-500">
          Annotated video not available
        </div>
      )}
      <EventTimeline events={events} duration={duration} currentTime={now} onSeek={seek} />
      <RiskCurve risk={risk} duration={duration} currentTime={now} />
      <EventsTable events={events} onSeek={seek} />
    </div>
  );
}
