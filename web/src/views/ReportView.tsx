"use client";
// The one-page report (scripts/write_report.py) lives in public/data/report.json so it is written with the numbers
// from the generated files and can change without a rebuild.
import { Card, Loading, Missing } from "@/components/ui";
import { useJson } from "@/lib/data";

interface Report {
  sections: { title: string; paragraphs: string[] }[];
}

export default function ReportView() {
  const report = useJson<Report>("report.json");
  if (report.state === "loading") return <Loading />;
  if (report.state !== "ok") return <Missing what="The report" how="python scripts/write_report.py" />;
  return (
    <div className="space-y-6">
      {report.data.sections.map((s) => (
        <Card key={s.title}>
          <h2 className="font-semibold">{s.title}</h2>
          {s.paragraphs.map((p, i) => (
            <p key={i} className="mt-2 text-sm leading-relaxed text-zinc-300">
              {p}
            </p>
          ))}
        </Card>
      ))}
    </div>
  );
}
