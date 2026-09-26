"use client";
import { useEffect, useRef, useState } from "react";
import { PlayPause } from "@/components/ui";
import { dataUrl } from "@/lib/data";

/**
 * Home hero: the raw camera footage full-bleed behind one headline.
 * The poster paints first; the loop (scripts/hero_clip.sh) starts only when motion is welcome and the
 * connection is not in Save-Data mode, and phones get the 720p encode.
 */
export default function HeroStage() {
  const video = useRef<HTMLVideoElement>(null);
  const [playing, setPlaying] = useState(false);
  const [enabled, setEnabled] = useState(false);

  useEffect(() => {
    const player = video.current;
    if (!player) return;
    const motion = window.matchMedia("(prefers-reduced-motion: reduce)");
    const saveData = (
      navigator as Navigator & { connection?: { saveData?: boolean } }
    ).connection?.saveData;
    if (motion.matches || saveData) return;
    player.src = dataUrl(
      window.innerWidth <= 900 ? "hero/hero-720.mp4" : "hero/hero-1080.mp4",
    );
    setEnabled(true);
    void player.play().catch(() => undefined);
    const stopForMotion = () => {
      if (motion.matches) player.pause();
    };
    motion.addEventListener("change", stopForMotion);
    return () => motion.removeEventListener("change", stopForMotion);
  }, []);

  return (
    <section className="hero-stage" aria-labelledby="hero-title">
      <video
        ref={video}
        className={`hero-media ${playing ? "is-playing" : ""}`}
        poster={dataUrl("hero/hero-poster.jpg")}
        muted
        loop
        playsInline
        preload="none"
        aria-hidden="true"
        onPlay={() => setPlaying(true)}
        onPause={() => setPlaying(false)}
      />
      <div className="hero-scrim" aria-hidden="true" />
      <h1 id="hero-title" className="hero-headline">
        Intelligence
        <br /> for every road.
      </h1>
      {enabled && (
        <button
          type="button"
          className="media-toggle"
          aria-label={
            playing ? "Pause background video" : "Play background video"
          }
          onClick={() => {
            const player = video.current;
            if (!player) return;
            if (player.paused) void player.play().catch(() => undefined);
            else player.pause();
          }}
        >
          <PlayPause playing={playing} />
        </button>
      )}
    </section>
  );
}
