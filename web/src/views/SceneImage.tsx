"use client";
// The calibrated scene drawn on the reference frame (scripts/render_scene.py, copied to public/data/).
import { useState } from "react";
import { Missing } from "@/components/ui";

export default function SceneImage() {
  const [failed, setFailed] = useState(false);
  if (failed) return <Missing what="The scene overlay" how="python scripts/render_scene.py --out web/public/data/scene_overlay.jpg" />;
  return (
    // eslint-disable-next-line @next/next/no-img-element -- static export, plain img
    <img
      src="/data/scene_overlay.jpg"
      alt="Calibrated scene layers: lanes, stop lines, crosswalks, signals and homography points on the reference frame"
      className="w-full rounded-xl border border-white/10"
      loading="lazy"
      onError={() => setFailed(true)}
    />
  );
}
