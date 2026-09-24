// Event classes and their colours. The palette is configs/palette.json (copied by scripts/sync-palette.mjs),
// the same file roadwatch/render.py reads, so a class has one colour in the video, timeline and charts.
import palette from "@/generated/palette.json";

export type EventTuple = [number, number, string]; // [start_sec, end_sec, label], evaluate.py format
export type RiskPoint = [number, number]; // [t_sec, score]

export const CLASS_COLORS: Record<string, string> = palette.classes;
export const OBJECT_COLORS: Record<string, string> = palette.objects;
export const CLASSES = Object.keys(palette.classes);

const NAMES: Record<string, string> = {
  accident: "Accident",
  near_miss: "Near miss",
  red_light: "Red-light running",
  wrong_way: "Wrong way",
  illegal_u_turn: "Illegal U-turn",
  stopped_vehicle: "Stopped vehicle",
  jaywalking: "Jaywalking",
  failure_to_yield: "Failure to yield",
  illegal_turn: "Illegal turn",
  solid_line_crossing: "Solid-line crossing",
  stop_line: "Stop-line violation",
  congestion: "Congestion",
  road_obstacle: "Road obstacle",
  fire_smoke: "Fire / smoke",
};

export function className(label: string): string {
  return NAMES[label] ?? label;
}

export function classColor(label: string): string {
  return CLASS_COLORS[label] ?? "#a3a3a3";
}

export function formatTime(sec: number): string {
  const m = Math.floor(sec / 60);
  const s = sec - m * 60;
  return `${m}:${s.toFixed(1).padStart(4, "0")}`;
}
