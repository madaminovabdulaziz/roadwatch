"use client";
import Link from "next/link";
import { useEffect, useRef, useState } from "react";
import HeroStage from "@/components/HeroStage";
import { Arrow, PlayPause } from "@/components/ui";
import TeamView from "@/views/TeamView";
import {
  dataUrl,
  useJson,
  type Metrics,
  type ResultVideo,
  type ResultsIndex,
} from "@/lib/data";

const STEPS: [string, string][] = [
  [
    "Perceive",
    "A pretrained detector finds every vehicle and pedestrian; tracking follows each one across frames.",
  ],
  [
    "Interpret",
    "Calibrated lanes, stop lines, crossings and traffic lights turn those tracks, in metres, into explainable events.",
  ],
  [
    "Review",
    "Annotated video, timed events and an accident-risk curve put every observation on one timeline.",
  ],
];

export default function HomeView() {
  const results = useJson<ResultsIndex>("results/index.json");
  const metrics = useJson<Metrics>("metrics.json");
  const sample = results.state === "ok" ? results.data.videos[0] : undefined;
  const enabled =
    results.state === "ok" ? results.data.enabled_classes : undefined;
  const runtime =
    results.state === "ok" ? results.data.runtime_x_duration : undefined;
  const score = metrics.state === "ok" ? metrics.data : undefined;

  return (
    <>
      <HeroStage />
      <MachineView sample={sample} loading={results.state === "loading"} />

      <section className="home-section" aria-labelledby="proof-title">
        <div className="site-container">
          <h2 id="proof-title" className="visually-hidden">
            Key numbers
          </h2>
          <p className="proof-line">
            <strong>{enabled ? `${enabled.length} of 14` : "—"}</strong> event
            classes covered.{" "}
            <strong>{score ? score.score_a.toFixed(2) : "—"}</strong> mean F1 on
            our labelled samples.{" "}
            <strong>{runtime ? `${runtime.toFixed(1)}×` : "—"}</strong> the
            video's length on one T4 GPU.
          </p>
          <p className="proof-note">
            {score
              ? `Score A of the official evaluate.py over ${score.gt_events} labelled events in ${score.videos} videos; runtime from the official run, within the 3× budget.`
              : "Scores from the official evaluate.py; runtime from the official run."}{" "}
            <Link href="/results/" className="text-link">
              Inspect the evaluation <Arrow diagonal />
            </Link>
          </p>
        </div>
      </section>

      <section className="home-section" aria-labelledby="steps-title">
        <div className="site-container">
          <div className="home-head">
            <h2 id="steps-title">One camera, a chain of evidence.</h2>
            <Link href="/approach/" className="text-link">
              Inside the approach <Arrow diagonal />
            </Link>
          </div>
          <ol className="steps">
            {STEPS.map(([title, body]) => (
              <li key={title}>
                <h3>{title}</h3>
                <p>{body}</p>
              </li>
            ))}
          </ol>
        </div>
      </section>

      <section className="home-section" aria-labelledby="explore-title">
        <div className="site-container">
          <div className="home-head">
            <h2 id="explore-title">Open to inspection.</h2>
          </div>
          <div className="explore-grid">
            <Link href="/eda/" className="explore-card">
              <img
                src="/data/eda/trajectories.jpg"
                alt="Observed road-user trajectories coloured by travel direction"
                loading="lazy"
              />
              <h3>
                The scene, explored <Arrow diagonal />
              </h3>
              <p>Movement, lighting, density and speed across the samples.</p>
            </Link>
            <Link href="/dashboard/" className="explore-card">
              <img
                src="/data/dashboard/event_heat.jpg"
                alt="Detected event paths on the road camera reference frame"
                loading="lazy"
              />
              <h3>
                The operator view <Arrow diagonal />
              </h3>
              <p>
                Events per class, per minute, and where on the road they happen.
              </p>
            </Link>
          </div>
        </div>
      </section>

      <section id="team" className="home-section" aria-labelledby="team-title">
        <div className="site-container">
          <div className="home-head">
            <h2 id="team-title">Built by three.</h2>
            <Link href="/team/" className="text-link">
              Meet the team <Arrow diagonal />
            </Link>
          </div>
          <TeamView />
        </div>
      </section>

      <section className="closing-section" aria-labelledby="closing-title">
        <div className="site-container">
          <h2 id="closing-title">See it on your own footage.</h2>
          <Link href="/demo/" className="button">
            Launch the live demo <Arrow />
          </Link>
        </div>
      </section>
    </>
  );
}

/** The annotated render of the same camera, right under the raw footage: what the model adds. */
function MachineView({
  sample,
  loading,
}: {
  sample?: ResultVideo;
  loading: boolean;
}) {
  const video = useRef<HTMLVideoElement>(null);
  const [playing, setPlaying] = useState(false);
  // Plays while on screen, unless the visitor prefers reduced motion.
  useEffect(() => {
    const player = video.current;
    if (!sample || !player) return;
    const motion = window.matchMedia("(prefers-reduced-motion: reduce)");
    const seen = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting && !motion.matches)
          void player.play().catch(() => undefined);
        else if (!entry.isIntersecting) player.pause();
      },
      { threshold: 0.4 },
    );
    seen.observe(player);
    return () => seen.disconnect();
  }, [sample]);

  return (
    <section className="machine-view" aria-labelledby="machine-title">
      <div className="site-container">
        <div className="machine-intro">
          <h2 id="machine-title">
            The same road,
            <br /> as RoadWatch sees it.
          </h2>
          <div className="machine-copy">
            <p>
              Every road user detected, tracked and measured in metres. Traffic
              events and accident risk, second by second.
            </p>
            <div className="machine-actions">
              <Link href="/demo/" className="button">
                Try the live demo <Arrow />
              </Link>
              <Link href="/approach/" className="text-link">
                Read the approach <Arrow diagonal />
              </Link>
            </div>
          </div>
        </div>
        <figure className="machine-frame">
          {sample ? (
            <video
              ref={video}
              src={`${dataUrl(sample.video)}#t=0,20`}
              poster={sample.poster ? dataUrl(sample.poster) : undefined}
              muted
              loop
              playsInline
              preload="metadata"
              onPlay={() => setPlaying(true)}
              onPause={() => setPlaying(false)}
              aria-label={`Annotated RoadWatch output on sample ${sample.id}`}
            />
          ) : (
            <div className="machine-placeholder">
              {loading ? "Loading camera sample…" : "Camera sample unavailable"}
            </div>
          )}
          {sample && (
            <figcaption>
              <button
                type="button"
                className="media-toggle"
                aria-label={
                  playing ? "Pause sample video" : "Play sample video"
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
              <span>Sample {sample.id}, first 20 seconds</span>
              <Link href="/results/" className="text-link">
                Full results <Arrow diagonal />
              </Link>
            </figcaption>
          )}
        </figure>
      </div>
    </section>
  );
}
