"use client";
// Team cards from public/data/team.json (edited by hand; photos go to public/data/team/).
import { Card, Loading, Missing } from "@/components/ui";
import { dataUrl, useJson } from "@/lib/data";

interface Member {
  name: string;
  role: string;
  built: string[];
  photo?: string;
  github?: string;
  linkedin?: string;
  portfolio?: string;
  projects?: { title: string; url?: string }[];
}

export default function TeamView() {
  const team = useJson<{ members: Member[] }>("team.json");
  if (team.state === "loading") return <Loading />;
  if (team.state !== "ok") return <Missing what="The team page data" how="edit web/public/data/team.json" />;
  return (
    <div className="grid gap-4 md:grid-cols-3">
      {team.data.members.map((m) => (
        <Card key={m.role}>
          {m.photo ? (
            // eslint-disable-next-line @next/next/no-img-element -- static export, plain img
            <img src={dataUrl(m.photo)} alt={m.name} className="mb-3 aspect-square w-24 rounded-full object-cover" />
          ) : (
            <div className="mb-3 grid aspect-square w-24 place-items-center rounded-full bg-white/10 text-2xl">
              {m.name ? m.name[0] : "?"}
            </div>
          )}
          <div className="font-semibold">{m.name || "Name to be added"}</div>
          <div className="text-sm text-sky-300">{m.role}</div>
          {m.built.length > 0 && (
            <ul className="mt-3 list-disc space-y-1 pl-5 text-sm text-zinc-300">
              {m.built.map((b) => (
                <li key={b}>{b}</li>
              ))}
            </ul>
          )}
          <div className="mt-3 flex flex-wrap gap-3 text-sm">
            {m.github && <a className="text-sky-300 underline" href={m.github}>GitHub</a>}
            {m.linkedin && <a className="text-sky-300 underline" href={m.linkedin}>LinkedIn</a>}
            {m.portfolio && <a className="text-sky-300 underline" href={m.portfolio}>Portfolio</a>}
          </div>
          {m.projects && m.projects.length > 0 && (
            <div className="mt-3 text-sm">
              <div className="text-zinc-500">Past projects</div>
              {m.projects.map((p) =>
                p.url ? (
                  <a key={p.title} href={p.url} className="block text-zinc-300 underline">{p.title}</a>
                ) : (
                  <div key={p.title} className="text-zinc-300">{p.title}</div>
                ),
              )}
            </div>
          )}
        </Card>
      ))}
    </div>
  );
}
