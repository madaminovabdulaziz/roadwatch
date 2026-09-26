import type { Metadata } from "next";
import UploadDemo from "@/components/UploadDemo";
import { Page } from "@/components/ui";

export const metadata: Metadata = { title: "Live demo" };

export default function DemoPage() {
  return (
    <Page
      title="Live demo"
      lead="From footage to findings. Upload a video or explore a prepared sample, then follow each event through annotated playback and synchronized analysis."
    >
      <UploadDemo />
    </Page>
  );
}
