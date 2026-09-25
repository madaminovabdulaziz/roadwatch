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
// Keep in sync with demo/config.yaml (max_upload_mb, max_process_sec).
const MAX_MB = 2048;
const ANALYSED_SEC = 20;
const POLL_MS = 1500;
const POLL_RETRIES = 5; // consecutive failed polls before giving up (Space restarts, flaky networks)
const STALL_MS = 5 * 60 * 1000; // no change in status for this long: the job is lost

interface JobResult {
  events: EventTuple[];
  risk: RiskPoint[];
  video_url: string;
  camera_match: boolean;
  duration: number;
  source_duration?: number;
}
interface JobStatus {
  status: "queued" | "running" | "done" | "error";
  progress?: number;
  stage?: string;
  ahead?: number;
  eta_sec?: number;
  message?: string;
  result?: JobResult;
}

type View =
  | { kind: "idle" }
  | { kind: "uploading"; fraction: number | null }
  | { kind: "job"; id: string; status: JobStatus }
  | { kind: "done"; result: JobResult }
  | { kind: "error"; message: string }
  | { kind: "example"; video: ResultVideo };

class HttpError extends Error {
  constructor(readonly status: number, message: string) {
    super(message);
  }
}

// fetch() reports no upload progress, and a raw camera file takes minutes to send.
function send(file: File, onProgress: (fraction: number) => void): Promise<{ job_id?: string; message?: string }> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${API}/api/jobs`);
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) onProgress(e.loaded / e.total);
    };
    xhr.onload = () => {
      let json: { job_id?: string; message?: string } = {};
      try {
        json = JSON.parse(xhr.responseText);
      } catch {
        // a proxy error page: keep the status code
      }
      if (xhr.status >= 200 && xhr.status < 300 && json.job_id) resolve(json);
      else reject(new HttpError(xhr.status, json.message ?? `the server answered ${xhr.status}`));
    };
    xhr.onerror = () => reject(new Error("the connection to the demo server failed"));
    const body = new FormData();
    body.append("file", file);
    xhr.send(body);
  });
}

function humanDuration(sec: number): string {
  return sec < 90 ? `${Math.max(5, Math.round(sec / 5) * 5)} s` : `${Math.round(sec / 60)} min`;
}

function progressLabel(status: JobStatus): string {
  if (status.status === "queued") {
    return status.ahead ? `Waiting in the queue (${status.ahead} ahead of you)` : "Waiting in the queue";
  }
  const eta = status.eta_sec !== undefined ? `, about ${humanDuration(status.eta_sec)} left` : "";
  return `Processing: ${status.stage || "starting"}${eta}`;
}

export default function UploadDemo() {
  const [view, setView] = useState<View>({ kind: "idle" });
  const input = useRef<HTMLInputElement>(null);
  const examples = useJson<ResultsIndex>("results/index.json");

  // Poll the job until it finishes or fails; ride out a few failed polls, give up only on a stall.
  const jobId = view.kind === "job" ? view.id : null;
  useEffect(() => {
    if (!jobId) return;
    let alive = true;
    let failures = 0;
    let lastChange = Date.now();
    let last = "";
    const tick = async () => {
      try {
        const res = await fetch(`${API}/api/jobs/${jobId}`, { cache: "no-store" });
        if (res.status === 404) throw new HttpError(404, "the job expired or the demo server restarted");
        if (!res.ok) throw new Error(`the server answered ${res.status}`);
        const status = (await res.json()) as JobStatus;
        if (!alive) return;
        failures = 0;
        const key = `${status.status}|${status.stage}|${status.progress}|${status.ahead}`;
        if (key !== last) [last, lastChange] = [key, Date.now()];
        if (status.status === "done" && status.result) setView({ kind: "done", result: status.result });
        else if (status.status === "error") setView({ kind: "error", message: status.message ?? "processing failed" });
        else if (Date.now() - lastChange > STALL_MS) setView({ kind: "error", message: "the job stopped making progress" });
        else {
          setView({ kind: "job", id: jobId, status });
          timer = setTimeout(tick, POLL_MS);
        }
      } catch (e) {
        if (!alive) return;
        failures += 1;
        if (e instanceof HttpError || failures > POLL_RETRIES) {
          setView({ kind: "error", message: `lost contact with the demo server (${e instanceof Error ? e.message : String(e)})` });
        } else {
          timer = setTimeout(tick, POLL_MS * 2 ** failures);
        }
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
    if (file.size > MAX_MB * 1024 * 1024) {
      return setView({ kind: "error", message: `the file is larger than ${MAX_MB / 1024} GB; please trim it first` });
    }
    if (!API) return setView({ kind: "error", message: "the demo server is not configured on this deployment; try an example below" });
    setView({ kind: "uploading", fraction: null });
    try {
      const json = await send(file, (fraction) => setView({ kind: "uploading", fraction }));
      setView({ kind: "job", id: json.job_id as string, status: { status: "queued", progress: 0 } });
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
            Any .mp4 up to {MAX_MB / 1024} GB: raw camera files are fine as they are. We analyse the first {ANALYSED_SEC} s
            on a CPU server, which takes a few minutes. Scene rules need the same camera as the samples; other
            videos get the camera-independent classes only.
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

      {view.kind === "uploading" && (
        <Progress
          label={view.fraction === null ? "Uploading" : `Uploading (${Math.round(view.fraction * 100)}%)`}
          value={view.fraction === null ? null : view.fraction * 100}
        />
      )}
      {view.kind === "job" && (
        <Progress label={progressLabel(view.status)} value={view.status.status === "queued" ? null : (view.status.progress ?? 0)} />
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
          {(view.result.source_duration ?? 0) > view.result.duration + 0.5 && (
            <Card className="text-sm text-zinc-300">
              Analysed the first {Math.round(view.result.duration)} s of your {humanDuration(view.result.source_duration ?? 0)} video.
            </Card>
          )}
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
