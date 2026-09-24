import type { Metadata } from "next";
import { Page } from "@/components/ui";
import DashboardView from "@/views/DashboardView";

export const metadata: Metadata = { title: "Dashboard" };

export default function DashboardPage() {
  return (
    <Page title="Operator dashboard" lead="All events detected in the sample videos, the way a traffic operator would review a shift.">
      <DashboardView />
    </Page>
  );
}
