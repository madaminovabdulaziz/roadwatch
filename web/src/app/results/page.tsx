import type { Metadata } from "next";
import { Page } from "@/components/ui";
import ResultsView from "@/views/ResultsView";

export const metadata: Metadata = { title: "Results" };

export default function ResultsPage() {
  return (
    <Page
      title="Results on the sample videos"
      lead="Click an event on the timeline to jump to it. The orange curve is the Part B risk: the probability that an accident starts within 5 seconds, computed only from frames already seen."
    >
      <ResultsView />
    </Page>
  );
}
