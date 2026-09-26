import type { Metadata } from "next";
import { Arrow, Page } from "@/components/ui";

export const metadata: Metadata = { title: "Links" };

const REPO = "https://github.com/madaminovabdulaziz/roadwatch";

const LINKS: [string, string, string][] = [
  [
    "Source code",
    REPO,
    "the full repository: package, scripts, tests, this website",
  ],
  [
    "Weights",
    `${REPO}/releases/tag/weights-v1`,
    "YOLO11m TorchScript export, fetched by weights/download.sh",
  ],
  [
    "predictions_samples.json",
    "/data/predictions_samples.json",
    "our output on the sample videos, reproducible with one command",
  ],
  [
    "README",
    `${REPO}#readme`,
    "setup, the two commands, models, datasets and licences",
  ],
];

export default function LinksPage() {
  return (
    <Page title="Links">
      <ul className="link-list">
        {LINKS.map(([title, href, what]) => (
          <li key={title}>
            <a href={href}>
              <span className="link-title">{title}</span>
              <span className="link-what">{what}</span>
              <Arrow diagonal />
            </a>
          </li>
        ))}
      </ul>
    </Page>
  );
}
