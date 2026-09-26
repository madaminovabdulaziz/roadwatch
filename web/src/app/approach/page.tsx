import type { Metadata } from "next";
import { Page, Section } from "@/components/ui";
import SceneImage from "@/views/SceneImage";

export const metadata: Metadata = { title: "Approach" };

const STEPS = [
  [
    "Decode",
    "PyAV, reference frames only (every 3rd), straight to detector size",
  ],
  ["Detect", "YOLO11m (COCO), FP16 on the GPU"],
  ["Track", "ByteTrack, one tracker per object group"],
  [
    "Metres",
    "homography from the lane markings' vanishing points; speed, heading",
  ],
  [
    "Scene rules",
    "one rule per class on tracks + lanes, lines, crossings, lights",
  ],
  ["Post-process", "merge, minimum duration, boundary refinement, union"],
  ["Events", "[start, end, label] per video"],
];

function PipelineDiagram() {
  const w = 150;
  const gap = 18;
  const total = STEPS.length * w + (STEPS.length - 1) * gap;
  return (
    <div className="diagram">
      <svg
        viewBox={`0 0 ${total} 170`}
        role="img"
        aria-label="pipeline diagram"
      >
        {STEPS.map(([name, detail], i) => {
          const x = i * (w + gap);
          return (
            <g key={name}>
              <rect
                x={x}
                y={10}
                width={w}
                height={70}
                fill="rgba(239,238,232,0.03)"
                stroke="rgba(239,238,232,0.2)"
              />
              <text
                x={x + w / 2}
                y={36}
                textAnchor="middle"
                fill="#efeee8"
                fontSize={14}
                fontWeight={600}
              >
                {name}
              </text>
              <foreignObject x={x + 6} y={42} width={w - 12} height={36}>
                <div
                  style={{
                    fontSize: 9.5,
                    color: "#a3a79e",
                    textAlign: "center",
                    lineHeight: 1.25,
                  }}
                >
                  {detail}
                </div>
              </foreignObject>
              {i < STEPS.length - 1 && (
                <path
                  d={`M${x + w} 45 h${gap}`}
                  stroke="#6d726b"
                  strokeWidth={1.5}
                  markerEnd="url(#arrow)"
                />
              )}
            </g>
          );
        })}
        <path
          d={`M${w / 2} 80 v40 h${2 * (w + gap)}`}
          stroke="#ef8855"
          fill="none"
          strokeWidth={1.5}
          markerEnd="url(#arrow)"
        />
        <rect
          x={2 * (w + gap) + 10}
          y={100}
          width={2 * w + gap}
          height={50}
          fill="rgba(239,136,85,0.06)"
          stroke="rgba(239,136,85,0.6)"
        />
        <text
          x={2 * (w + gap) + 10 + (2 * w + gap) / 2}
          y={122}
          textAnchor="middle"
          fill="#efeee8"
          fontSize={13}
          fontWeight={600}
        >
          Part B: online risk
        </text>
        <text
          x={2 * (w + gap) + 10 + (2 * w + gap) / 2}
          y={140}
          textAnchor="middle"
          fill="#a3a79e"
          fontSize={10}
        >
          past frames only → braking conflicts → P(accident in 5 s)
        </text>
        <defs>
          <marker
            id="arrow"
            viewBox="0 0 10 10"
            refX="9"
            refY="5"
            markerWidth="6"
            markerHeight="6"
            orient="auto"
          >
            <path d="M0 0 L10 5 L0 10 z" fill="#6d726b" />
          </marker>
        </defs>
      </svg>
    </div>
  );
}

export default function ApproachPage() {
  return (
    <Page
      title="How it works"
      lead="A pretrained detector finds road users; everything after that is measured geometry and explicit rules, so every event can be explained and every threshold is a number in one config file."
    >
      <Section title="Pipeline">
        <PipelineDiagram />
      </Section>

      <Section title="What is learned and what is rule-based">
        <div className="two-col">
          <div>
            <h3>Learned</h3>
            <p>
              Object detection only: YOLO11m trained on COCO (cars, buses,
              trucks, motorcycles, bicycles, people, animals, bags). There is no
              labelled data for this camera to train event models on, and a
              detector trained on millions of images generalises far better than
              anything we could fit on a few videos.
            </p>
          </div>
          <div>
            <h3>Rule-based</h3>
            <p>
              Tracking, the conversion to metres, and one rule per event class
              following the annotation rules (when an event starts and ends).
              Rules are testable with synthetic tracks, run in milliseconds, and
              a class is only switched on after it passes its checks, because
              predicting an absent class costs score.
            </p>
          </div>
        </div>
      </Section>

      <Section title="Scene calibration">
        <SceneImage />
      </Section>

      <Section title="Models, data and licences">
        <ul className="plain-list">
          <li>
            YOLO11m, COCO-pretrained (Ultralytics, AGPL-3.0), run without the
            ultralytics package from our own TorchScript export.
          </li>
          <li>ByteTrack from supervision 0.30.5 (MIT).</li>
          <li>
            COCO dataset (annotations CC BY 4.0) via the detector weights; our
            own labels of the sample videos for tuning.
          </li>
          <li>
            PyTorch, torchvision, OpenCV, PyAV/FFmpeg, NumPy, SciPy, pandas.
          </li>
        </ul>
      </Section>
    </Page>
  );
}
