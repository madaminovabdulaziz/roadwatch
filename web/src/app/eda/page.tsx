import type { Metadata } from "next";
import { Page } from "@/components/ui";
import EdaView from "@/views/EdaView";

export const metadata: Metadata = { title: "EDA" };

export default function EdaPage() {
  return (
    <Page
      title="Exploring the sample videos"
      lead="What the camera sees, measured from our own detections and tracks (scripts/eda.py). Every chart says what it changed in the solution."
    >
      <EdaView />
    </Page>
  );
}
