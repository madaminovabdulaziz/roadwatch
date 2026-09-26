"use client";
import {
  className,
  classColor,
  formatTime,
  type EventTuple,
} from "@/lib/classes";

export default function EventsTable({
  events,
  onSeek,
}: {
  events: EventTuple[];
  onSeek?: (t: number) => void;
}) {
  if (!events.length) return null;
  return (
    <div className="table-wrap">
      <table className="data-table">
        <thead>
          <tr>
            <th>Class</th>
            <th>Start</th>
            <th>End</th>
            <th>Duration</th>
          </tr>
        </thead>
        <tbody>
          {events.map(([s, e, label], i) => (
            <tr
              key={i}
              className={onSeek ? "is-clickable" : undefined}
              onClick={() => onSeek?.(s)}
            >
              <td>
                <span
                  className="class-swatch"
                  style={{ background: classColor(label) }}
                />
                {onSeek ? (
                  <button
                    type="button"
                    className="row-link"
                    aria-label={`Seek to ${className(label)} at ${formatTime(s)}`}
                    onClick={(event) => {
                      event.stopPropagation();
                      onSeek(s);
                    }}
                  >
                    {className(label)}
                  </button>
                ) : (
                  className(label)
                )}
              </td>
              <td>{formatTime(s)}</td>
              <td>{formatTime(e)}</td>
              <td>{(e - s).toFixed(1)} s</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
