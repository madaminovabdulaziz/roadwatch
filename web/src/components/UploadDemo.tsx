"use client";
// Live demo (WEBSITE_SPEC "Live demo" + "Demo API"): upload -> poll progress -> annotated result.
// Every failure ends in a readable message with a retry; the example buttons show precomputed results
// and work even when the backend is asleep or not configured.
import { useEffect, useRef, useState } from "react";
import SampleResult from "@/components/SampleResult";
import VideoResult from "@/components/VideoResult";
import { Card } from "@/components/ui";
import type { EventTuple, RiskPoint } from "@/lib/classes";
import { useJson, type ResultsIndex, type ResultVideo } from "@/lib/data";

const API = (process.env.NEXT_PUBLIC_DEMO_API ?? "").replace(/\/$/, "");
const MAX_MB = 100;
const MAX_SEC = 120;
const POLL_MS = 1500;
const GIVE_UP_MS = 15 * 60 * 1000;

interface JobResult {
  events: EventTuple[];
  risk: RiskPoint[];
  video_url: string;
  camera_match: boolean;
  duration: number;
}
interface JobStatus {
  status: "queued" | "running" | "done" | "error";
  progress?: number;
  stage?: string;
  message?: string;
  result?: JobResult;
}

type View =
  | { kind: "idle" }
  | { kind: "uploading" }
  | { kind: "job"; id: string; status: JobStatus }
  | { kind: "done"; result: JobResult }
  | { kind: "error"; message: string }
  | { kind: "example"; video: ResultVideo };

function videoDuration(file: File): Promise<number> {
  return new Promise((resolve) => {
    const v = document.createElement("video");
    v.preload = "metadata";
    v.onloadedmetadata = () => {
      URL.revokeObjectURL(v.src);
      resolve(v.duration);
    };
    v.onerror = () => resolve(NaN); // unknown: let the server decide
    v.src = URL.createObjectURL(file);
  });
}

export default function UploadDemo() {
  const [view, setView] = useState<View>({ kind: "idle" });
  const input = useRef<HTMLInputElement>(null);
  const examples = useJson<ResultsIndex>("results/index.json");

  // Poll the job until it finishes, fails, or takes unreasonably long.
  const jobId = view.kind === "job" ? view.id : null;
  useEffect(() => {
    if (!jobId) return;
    const started = Date.now();
    let alive = true;
    const tick = async () => {
      try {
        const res = await fetch(`${API}/api/jobs/${jobId}`, { cache: "no-store" });
        if (!res.ok) throw new Error(`the server answered ${res.status}`);
        const status = (await res.json()) as JobStatus;
        if (!alive) return;
        if (status.status === "done" && status.result) setView({ kind: "done", result: status.result });
        else if (status.status === "error") setView({ kind: "error", message: status.message ?? "processing failed" });
        else if (Date.now() - started > GIVE_UP_MS) setView({ kind: "error", message: "processing took too long" });
        else {
          setView({ kind: "job", id: jobId, status });
          timer = setTimeout(tick, POLL_MS);
        }
      } catch (e) {
        if (alive) setView({ kind: "error", message: `lost contact with the demo server (${String(e)})` });
      }
    };
    let timer = setTimeout(tick, POLL_MS);
    return () => {
      alive = false;
      clearTimeout(timer);
    };
  }, [jobId]);

  async function upload(file: File) {
    if (!/\.mp4$/i.test(file.name)) return setView({ kind: "error", message: "please choose an .mp4 file" });
    if (file.size > MAX_MB * 1024 * 1024) return setView({ kind: "error", message: `the file is larger than ${MAX_MB} MB` });
    const dur = await videoDuration(file);
    if (dur > MAX_SEC) return setView({ kind: "error", message: `the video is ${Math.round(dur)} s; the limit is ${MAX_SEC} s` });
    if (!API) return setView({ kind: "error", message: "the demo server is not configured on this deployment; try an example below" });
    setView({ kind: "uploading" });
    try {
      const body = new FormData();
      body.append("file", file);
      const res = await fetch(`${API}/api/jobs`, { method: "POST", body });
      const json = (await res.json().catch(() => ({}))) as { job_id?: string; message?: string };
      if (!res.ok || !json.job_id) throw new Error(json.message ?? `the server answered ${res.status}`);
      setView({ kind: "job", id: json.job_id, status: { status: "queued", progress: 0 } });
    } catch (e) {
      setView({ kind: "error", message: `upload failed: ${e instanceof Error ? e.message : String(e)}` });
    }
  }

  const busy = view.kind === "uploading" || view.kind === "job";
  return (
    <div className="space-y-6">
      <Card>
        <div className="flex flex-col gap-3 sm:flex-row sm:items-center">
          <button
            type="button"
            disabled={busy}
            onClick={() => input.current?.click()}
            className="rounded-lg bg-sky-500 px-4 py-2 font-medium text-white hover:bg-sky-400 disabled:opacity-50"
          >
            Upload a video
          </button>
          <p className="text-sm text-zinc-400">
            .mp4, up to {MAX_SEC / 60} minutes and {MAX_MB} MB. Scene rules need the same camera as the samples;
            other videos get the camera-independent classes only.
          </p>
          <input
            ref={input}
            type="file"
            accept="video/mp4,.mp4"
            className="hidden"
            onChange={(e) => {
              const f = e.target.files?.[0];
              e.target.value = "";
              if (f) void upload(f);
            }}
          />
        </div>
        {examples.state === "ok" && examples.data.videos.length > 0 && (
          <div className="mt-4 flex flex-wrap gap-2 text-sm">
            <span className="text-zinc-500">Or see an example:</span>
            {examples.data.videos.slice(0, 2).map((v) => (
              <button
                key={v.id}
                type="button"
                disabled={busy}
                onClick={() => setView({ kind: "example", video: v })}
                className="rounded-lg border border-white/15 px-3 py-1 hover:bg-white/10 disabled:opacity-50"
              >
                {v.name}
              </button>
            ))}
          </div>
        )}
      </Card>

      {view.kind === "uploading" && <Progress label="Uploading" value={null} />}
      {view.kind === "job" && (
        <Progress label={view.status.stage ? `Processing: ${view.status.stage}` : "Waiting in the queue"} value={view.status.progress ?? 0} />
      )}
      {view.kind === "error" && (
        <Card className="border-red-500/40 text-sm">
          <p className="text-red-300">Sorry: {view.message}.</p>
          <button type="button" className="mt-2 text-sky-300 underline" onClick={() => setView({ kind: "idle" })}>
            Try again
          </button>
        </Card>
      )}
      {view.kind === "done" && (
        <div className="space-y-3">
          {!view.result.camera_match && (
            <Card className="border-amber-500/40 text-sm text-amber-200">
              Different camera: scene-specific rules (red light, stop line, lanes) were disabled for this video.
            </Card>
          )}
          <VideoResult
            src={`${API}${view.result.video_url}`}
            events={view.result.events}
            risk={view.result.risk}
            duration={view.result.duration}
          />
          <a
            className="inline-block text-sm text-sky-300 underline"
            download="roadwatch_result.json"
            href={`data:application/json,${encodeURIComponent(JSON.stringify({ events: view.result.events, risk: view.result.risk }))}`}
          >
            Download the result as JSON
          </a>
        </div>
      )}
      {view.kind === "example" && <SampleResult video={view.video} />}
    </div>
  );
}

function Progress({ label, value }: { label: string; value: number | null }) {
  return (
    <Card>
      <div className="mb-2 text-sm text-zinc-300">{label}…</div>
      <div className="h-2 overflow-hidden rounded bg-white/10">
        <div
          className={`h-full bg-sky-500 transition-all ${value === null ? "w-1/3 animate-pulse" : ""}`}
          style={value === null ? undefined : { width: `${Math.max(3, Math.min(100, value))}%` }}
        />
      </div>
    </Card>
  );
}
