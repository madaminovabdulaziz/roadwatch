"use client";
import { className, classColor, formatTime, type EventTuple } from "@/lib/classes";

export default function EventsTable({ events, onSeek }: { events: EventTuple[]; onSeek?: (t: number) => void }) {
  if (!events.length) return null;
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-left text-sm">
        <thead className="text-xs uppercase text-zinc-500">
          <tr>
            <th className="py-2 pr-4 font-medium">Class</th>
            <th className="py-2 pr-4 font-medium">Start</th>
            <th className="py-2 pr-4 font-medium">End</th>
            <th className="py-2 font-medium">Duration</th>
          </tr>
        </thead>
        <tbody>
          {events.map(([s, e, label], i) => (
            <tr
              key={i}
              className={`border-t border-white/5 ${onSeek ? "cursor-pointer hover:bg-white/5" : ""}`}
              onClick={() => onSeek?.(s)}
            >
              <td className="py-1.5 pr-4">
                <span className="mr-2 inline-block h-2.5 w-2.5 rounded-sm" style={{ background: classColor(label) }} />
                {className(label)}
              </td>
              <td className="py-1.5 pr-4 tabular-nums">{formatTime(s)}</td>
              <td className="py-1.5 pr-4 tabular-nums">{formatTime(e)}</td>
              <td className="py-1.5 tabular-nums">{(e - s).toFixed(1)} s</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
