import type { Metadata } from "next";
import UploadDemo from "@/components/UploadDemo";
import { Page } from "@/components/ui";

export const metadata: Metadata = { title: "Live demo" };

export default function DemoPage() {
  return (
    <Page
      title="Live demo"
      lead="Upload a short clip and get the annotated video, the event timeline and the risk curve. Processing runs on a CPU server, so a 2-minute clip takes a few minutes."
    >
      <UploadDemo />
    </Page>
  );
}
