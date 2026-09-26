"use client";
// Runtime loading of generated data under public/data/. A missing file is a normal state ("not generated
// yet"), never an error page, so the site works at every stage of the project.
import { useEffect, useState } from "react";

export type Loaded<T> =
  | { state: "loading" }
  | { state: "missing" }
  | { state: "error"; message: string }
  | { state: "ok"; data: T };

export function dataUrl(path: string): string {
  return path.startsWith("http") || path.startsWith("/") ? path : `/data/${path}`;
}

export async function fetchJson<T>(path: string): Promise<T | null> {
  const res = await fetch(dataUrl(path), { cache: "no-store" });
  if (res.status === 404) return null;
  if (!res.ok) throw new Error(`${path}: HTTP ${res.status}`);
  return (await res.json()) as T;
}

export function useJson<T>(path: string | null): Loaded<T> {
  const [loaded, setLoaded] = useState<Loaded<T>>({ state: "loading" });
  useEffect(() => {
    if (!path) return;
    let alive = true;
    setLoaded({ state: "loading" });
    fetchJson<T>(path)
      .then((data) => alive && setLoaded(data === null ? { state: "missing" } : { state: "ok", data }))
      .catch((e: unknown) => alive && setLoaded({ state: "error", message: String(e) }));
    return () => {
      alive = false;
    };
  }, [path]);
  return loaded;
}

// Data contracts written by the Python scripts (keep in sync with them).
export interface Metrics {
  source: string;
  videos: number;
  gt_events: number;
  pred_events: number;
  score_a: number;
  per_class: Record<string, { "f1@0.3": number; "f1@0.5": number; "f1@0.7": number; f1_mean: number; "tp_fp_fn@0.5": number[] }>;
}

export interface ResultVideo {
  id: string;
  name: string;
  duration: number;
  video: string; // annotated H.264 MP4
  preview?: string; // its first 20 s (home page loop)
  poster?: string;
  events: string; // JSON: EventTuple[]
  risk: string; // JSON: RiskPoint[]
}

export interface ResultsIndex {
  videos: ResultVideo[];
  enabled_classes?: string[];
  runtime_x_duration?: number; // Part A + B wall time / video duration on a T4 (scripts/bench.py)
}

export interface EdaVideo {
  name: string;
  width: number;
  height: number;
  fps: number;
  n_frames: number;
  duration: number;
  codec?: string;
  pix_fmt?: string;
  bit_rate_mbps?: number | null;
  brightness: { t: number[]; mean_luma: number[] } | null;
}
