"use client";
// EDA charts from web/public/data/eda/ (scripts/eda.py). Each "changed" sentence states a decision that
// is actually in the code (docs/SPEC.md §12), nothing aspirational.
import { useState } from "react";
import Chart from "@/components/Chart";
import { Choices, Loading, Missing, Section } from "@/components/ui";
import { OBJECT_COLORS } from "@/lib/classes";
import { dataUrl, useJson, type EdaVideo } from "@/lib/data";

interface Counts {
  t: number[];
  series: Record<string, number[]>;
}
interface Density {
  minute: number[];
  vehicles: number[];
  persons: number[];
}
interface Speeds {
  bin_edges_kmh: number[];
  lanes: Record<string, { counts: number[]; median_kmh: number }>;
}
type Signals = Record<string, [number, number, string][]>;

const STATE_COLORS: Record<string, string> = {
  red: "#ef4444",
  yellow: "#eab308",
  green: "#22c55e",
  unknown: "#52525b",
};

export default function EdaView() {
  const summary = useJson<{ videos: EdaVideo[] }>("eda/summary.json");
  const [pick, setPick] = useState(0);
  if (summary.state === "loading") return <Loading />;
  if (summary.state !== "ok")
    return <Missing what="The EDA data" how="python scripts/eda.py samples/" />;
  const videos = summary.data.videos;
  const stem = videos[pick]?.name.replace(/\.[^.]+$/, "");

  return (
    <>
      <Section
        title="The videos"
        changed="the samples are 4K 4:2:2 10-bit H.264 at 29.97 fps, which the T4 cannot decode in hardware, so Part A decodes only reference frames and never assumes 25 fps."
      >
        <VideoTable videos={videos} />
      </Section>

      <Section
        title="Lighting over time"
        changed="one sample is at dusk, so lamp colours are judged per lamp against its own on/off levels instead of fixed thresholds."
      >
        <Chart
          data={videos
            .filter((v) => v.brightness)
            .map((v) => ({
              x: v.brightness!.t,
              y: v.brightness!.mean_luma,
              type: "scatter",
              mode: "lines",
              name: v.name,
            }))}
          layout={{
            xaxis: { title: { text: "time (s)" } },
            yaxis: { title: { text: "mean luma (0–255)" }, range: [0, 255] },
          }}
        />
      </Section>

      <Choices
        label="Video"
        options={videos.map((v) => v.name)}
        value={pick}
        onChange={setPick}
      />
      {stem && <PerVideo key={stem} stem={stem} />}

      <Section title="Where things move">
        <Images
          files={["heatmap.jpg", "trajectories.jpg"]}
          captions={[
            "Tracks passing each spot (log scale)",
            "Tracks, colour = direction of travel",
          ]}
        />
      </Section>

      <Section
        title="Where people walk"
        changed="people also cut across the carriageway between and beside the zebras, so jaywalking is judged by where a person's feet are on the road, not by a hand-made list of spots."
      >
        <Images
          files={["heatmap_people.jpg"]}
          captions={["Pedestrian tracks passing each spot (log scale)"]}
        />
      </Section>

      <Section
        title="Learned traffic direction"
        changed="lane directions in the scene file are proposed from this field (scripts/learn_lane_flow.py) instead of being guessed by hand."
      >
        <Images
          files={["lane_flow.jpg"]}
          captions={["Mean motion per grid cell over the scene layers"]}
        />
      </Section>
    </>
  );
}

function VideoTable({ videos }: { videos: EdaVideo[] }) {
  return (
    <div className="table-wrap">
      <table className="data-table">
        <thead>
          <tr>
            {[
              "Video",
              "Resolution",
              "FPS",
              "Duration",
              "Codec",
              "Pixel format",
              "Bit rate",
            ].map((h) => (
              <th key={h}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {videos.map((v) => (
            <tr key={v.name}>
              <td>{v.name}</td>
              <td>
                {v.width}×{v.height}
              </td>
              <td>{v.fps}</td>
              <td>{(v.duration / 60).toFixed(1)} min</td>
              <td>{v.codec ?? "—"}</td>
              <td>{v.pix_fmt ?? "—"}</td>
              <td>{v.bit_rate_mbps ? `${v.bit_rate_mbps} Mbps` : "—"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function PerVideo({ stem }: { stem: string }) {
  const counts = useJson<Counts>(`eda/${stem}/counts.json`);
  const density = useJson<Density>(`eda/${stem}/density.json`);
  const speeds = useJson<Speeds>(`eda/${stem}/speeds.json`);
  const signals = useJson<Signals>(`eda/${stem}/signals.json`);
  return (
    <>
      <Section title="Objects in view over time">
        {counts.state === "ok" ? (
          <Chart
            data={Object.entries(counts.data.series).map(([cls, y]) => ({
              x: counts.data.t,
              y,
              type: "scatter",
              mode: "lines",
              stackgroup: "one",
              name: cls,
              line: {
                width: 0.5,
                color: OBJECT_COLORS[cls] ?? OBJECT_COLORS.other,
              },
            }))}
            layout={{
              xaxis: { title: { text: "time (s)" } },
              yaxis: { title: { text: "objects visible" } },
            }}
          />
        ) : (
          counts.state !== "loading" && (
            <Missing what="Object counts" how="scripts/eda.py" />
          )
        )}
      </Section>
      <Section title="Traffic density per minute">
        {density.state === "ok" && (
          <Chart
            data={[
              {
                x: density.data.minute,
                y: density.data.vehicles,
                type: "bar",
                name: "vehicles",
                marker: { color: OBJECT_COLORS.car },
              },
              {
                x: density.data.minute,
                y: density.data.persons,
                type: "bar",
                name: "pedestrians",
                marker: { color: OBJECT_COLORS.person },
              },
            ]}
            layout={{
              barmode: "group",
              xaxis: { title: { text: "minute" } },
              yaxis: { title: { text: "new tracks per minute" } },
            }}
          />
        )}
      </Section>
      {speeds.state === "ok" && Object.keys(speeds.data.lanes).length > 0 && (
        <Section title="Speed per lane">
          <Chart
            data={Object.entries(speeds.data.lanes).map(([lane, s]) => ({
              x: speeds.data.bin_edges_kmh.slice(0, -1),
              y: s.counts,
              type: "bar",
              name: `${lane || "outside lanes"} (median ${s.median_kmh} km/h)`,
            }))}
            layout={{
              barmode: "overlay",
              xaxis: { title: { text: "km/h (bin start)" } },
              yaxis: { title: { text: "samples" } },
            }}
          />
        </Section>
      )}
      {signals.state === "ok" && Object.keys(signals.data).length > 0 && (
        <Section title="Traffic-light phases">
          <Chart
            height={60 + 50 * Object.keys(signals.data).length}
            data={Object.entries(signals.data).flatMap(([sid, segs]) =>
              segs.map(([t0, t1, state]) => ({
                x: [t1 - t0],
                base: [t0],
                y: [sid],
                type: "bar" as const,
                orientation: "h" as const,
                marker: { color: STATE_COLORS[state] ?? STATE_COLORS.unknown },
                showlegend: false,
                hovertemplate: `${sid} ${state}: ${t0.toFixed(1)}–${t1.toFixed(1)} s<extra></extra>`,
              })),
            )}
            layout={{
              barmode: "overlay",
              xaxis: { title: { text: "time (s)" } },
            }}
          />
        </Section>
      )}
    </>
  );
}

function Images({ files, captions }: { files: string[]; captions: string[] }) {
  return (
    <div className={files.length > 1 ? "figure-grid" : undefined}>
      {files.map((f, i) => (
        <figure key={f} className="figure">
          {/* eslint-disable-next-line @next/next/no-img-element -- static export, plain img */}
          <img src={dataUrl(`eda/${f}`)} alt={captions[i]} loading="lazy" />
          <figcaption>{captions[i]}</figcaption>
        </figure>
      ))}
    </div>
  );
}
