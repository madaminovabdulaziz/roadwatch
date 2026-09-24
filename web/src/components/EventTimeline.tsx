"use client";
// One row per class, one bar per event; clicking a bar (or anywhere on the track) seeks the video.
import { useMemo } from "react";
import { className, classColor, formatTime, type EventTuple } from "@/lib/classes";

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
  const rows = useMemo(() => [...new Set(events.map((e) => e[2]))].sort(), [events]);
  const pct = (t: number) => `${Math.min(100, Math.max(0, (t / Math.max(duration, 1e-6)) * 100))}%`;
  const seekFromClick = (ev: React.MouseEvent<HTMLDivElement>) => {
    const r = ev.currentTarget.getBoundingClientRect();
    onSeek(((ev.clientX - r.left) / r.width) * duration);
  };

  if (!events.length) return <p className="text-sm text-zinc-500">No events detected in this video.</p>;
  return (
    <div className="space-y-1.5" aria-label="event timeline">
      {rows.map((label) => (
        <div key={label} className="flex items-center gap-2">
          <div className="w-28 shrink-0 truncate text-xs text-zinc-400 sm:w-36">{className(label)}</div>
          <div className="relative h-5 flex-1 cursor-pointer rounded bg-white/5" onClick={seekFromClick}>
            {events
              .filter((e) => e[2] === label)
              .map(([s, e], i) => (
                <button
                  key={i}
                  type="button"
                  title={`${className(label)} ${formatTime(s)}–${formatTime(e)}`}
                  className="absolute top-0 h-5 min-w-[3px] rounded-sm opacity-90 hover:opacity-100"
                  style={{ left: pct(s), width: `calc(${pct(e)} - ${pct(s)})`, background: classColor(label) }}
                  onClick={(ev) => {
                    ev.stopPropagation();
                    onSeek(s);
                  }}
                />
              ))}
            <div className="pointer-events-none absolute top-[-2px] h-6 w-px bg-white" style={{ left: pct(currentTime) }} />
          </div>
        </div>
      ))}
    </div>
  );
}
