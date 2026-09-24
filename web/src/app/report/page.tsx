import type { Metadata } from "next";
import { Page } from "@/components/ui";
import ReportView from "@/views/ReportView";

export const metadata: Metadata = { title: "Report" };

export default function ReportPage() {
  return (
    <Page title="Technical report" lead="What we built, what worked, what did not, and what we would do next.">
      <ReportView />
    </Page>
  );
}
