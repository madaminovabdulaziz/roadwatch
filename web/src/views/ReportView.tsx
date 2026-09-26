"use client";
// The one-page report (scripts/write_report.py) lives in public/data/report.json so it is written with the numbers
// from the generated files and can change without a rebuild.
import { Loading, Missing } from "@/components/ui";
import { useJson } from "@/lib/data";

interface Report {
  sections: { title: string; paragraphs: string[] }[];
}

export default function ReportView() {
  const report = useJson<Report>("report.json");
  if (report.state === "loading") return <Loading />;
  if (report.state !== "ok")
    return <Missing what="The report" how="python scripts/write_report.py" />;
  return (
    <div className="report">
      {report.data.sections.map((s) => (
        <article key={s.title}>
          <h2>{s.title}</h2>
          {s.paragraphs.map((p, i) => (
            <p key={i}>{p}</p>
          ))}
        </article>
      ))}
    </div>
  );
}
