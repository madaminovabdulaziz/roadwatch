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
    <div className="video-workspace">
      {src ? (
        <video
          ref={video}
          src={src}
          poster={poster}
          controls
          playsInline
          preload="metadata"
          onTimeUpdate={(e) => setNow(e.currentTarget.currentTime)}
        />
      ) : (
        <div className="video-missing">Annotated video not available</div>
      )}
      <section className="analysis-panel">
        <div className="analysis-heading">
          <h3>Event timeline</h3>
          <span>
            {events.length} events in {Math.round(duration)} s. Select one to
            jump to it.
          </span>
        </div>
        <EventTimeline
          events={events}
          duration={duration}
          currentTime={now}
          onSeek={seek}
        />
      </section>
      <section className="analysis-panel">
        <div className="analysis-heading">
          <h3>Accident risk</h3>
          <span>Probability that an accident starts within 5 s</span>
        </div>
        <RiskCurve risk={risk} duration={duration} currentTime={now} />
      </section>
      <section className="analysis-panel">
        <div className="analysis-heading">
          <h3>All events</h3>
        </div>
        <EventsTable events={events} onSeek={seek} />
      </section>
    </div>
  );
}
