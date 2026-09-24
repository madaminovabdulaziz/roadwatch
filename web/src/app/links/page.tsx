import type { Metadata } from "next";
import { Card, Page } from "@/components/ui";

export const metadata: Metadata = { title: "Links" };

const REPO = "https://github.com/madaminovabdulaziz/roadwatch";

const LINKS: [string, string, string][] = [
  ["Source code", REPO, "the full repository: package, scripts, tests, this website"],
  ["Weights", `${REPO}/releases/tag/weights-v1`, "YOLO11m TorchScript export, fetched by weights/download.sh"],
  ["predictions_samples.json", "/data/predictions_samples.json", "our output on the sample videos, reproducible with one command"],
  ["README", `${REPO}#readme`, "setup, the two commands, models, datasets and licences"],
];

export default function LinksPage() {
  return (
    <Page title="Links">
      <div className="grid gap-4 sm:grid-cols-2">
        {LINKS.map(([title, href, what]) => (
          <a key={title} href={href} className="block">
            <Card className="h-full hover:bg-white/[0.06]">
              <div className="font-medium text-sky-300">{title}</div>
              <div className="mt-1 text-sm text-zinc-400">{what}</div>
            </Card>
          </a>
        ))}
      </div>
    </Page>
  );
}
