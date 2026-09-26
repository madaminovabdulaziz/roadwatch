"use client";
// One row per class, one bar per event; clicking a bar (or anywhere on the track) seeks the video.
import { useMemo } from "react";
import {
  className,
  classColor,
  formatTime,
  type EventTuple,
} from "@/lib/classes";

export default function EventTimeline({
  events,
  duration,
  currentTime,
  onSeek,
}: {
  events: EventTuple[];
  duration: number;
  currentTime: number;
  onSeek: (t: number) => void;
}) {
  const rows = useMemo(
    () => [...new Set(events.map((e) => e[2]))].sort(),
    [events],
  );
  const pct = (t: number) =>
    `${Math.min(100, Math.max(0, (t / Math.max(duration, 1e-6)) * 100))}%`;
  const seekFromClick = (ev: React.MouseEvent<HTMLDivElement>) => {
    const r = ev.currentTarget.getBoundingClientRect();
    onSeek(((ev.clientX - r.left) / r.width) * duration);
  };

  if (!events.length)
    return <p className="meta">No events detected in this video.</p>;
  return (
    <div className="timeline" aria-label="event timeline">
      {rows.map((label) => (
        <div key={label} className="timeline-row">
          <div className="timeline-label">{className(label)}</div>
          <div className="timeline-track" onClick={seekFromClick}>
            {events
              .filter((e) => e[2] === label)
              .map(([s, e], i) => (
                <button
                  key={i}
                  type="button"
                  title={`${className(label)} ${formatTime(s)}–${formatTime(e)}`}
                  className="timeline-event"
                  style={{
                    left: pct(s),
                    width: `calc(${pct(e)} - ${pct(s)})`,
                    background: classColor(label),
                  }}
                  onClick={(ev) => {
                    ev.stopPropagation();
                    onSeek(s);
                  }}
                />
              ))}
            <div
              className="timeline-cursor"
              style={{ left: pct(currentTime) }}
            />
          </div>
        </div>
      ))}
    </div>
  );
}
