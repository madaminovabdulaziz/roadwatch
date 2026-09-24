import type { Metadata } from "next";
import { Page } from "@/components/ui";
import TeamView from "@/views/TeamView";

export const metadata: Metadata = { title: "Team" };

export default function TeamPage() {
  return (
    <Page title="Team">
      <TeamView />
    </Page>
  );
}
