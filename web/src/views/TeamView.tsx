"use client";
import { Arrow, Loading, Missing } from "@/components/ui";
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
  affiliations?: {
    name: string;
    label?: string;
    logo?: string;
    logoLayout?: "padded-square";
    url?: string;
  }[];
}
export default function TeamView() {
  const team = useJson<{ members: Member[] }>("team.json");
  if (team.state === "loading") return <Loading />;
  if (team.state !== "ok")
    return (
      <Missing what="The team page data" how="edit web/public/data/team.json" />
    );
  return (
    <div className="team-grid">
      {team.data.members.map((m, index) => (
        <article className="team-card" key={`${m.role}-${index}`}>
          <div className="portrait-stage">
            {m.photo ? (
              <img
                src={dataUrl(m.photo)}
                alt={m.name || m.role}
                className="team-portrait"
                loading="lazy"
              />
            ) : (
              <span className="portrait-initial" aria-hidden="true">
                {(m.name || m.role).charAt(0)}
              </span>
            )}
          </div>
          <h3>{m.name || m.role}</h3>
          {m.name && <p className="team-role">{m.role}</p>}
          {m.built.length > 0 && (
            <ul className="member-contributions">
              {m.built.map((b) => (
                <li key={b}>{b}</li>
              ))}
            </ul>
          )}
          {!!m.affiliations?.length && (
            <ul
              className="affiliations"
              aria-label={`${m.name || m.role}: background and affiliations`}
            >
              {m.affiliations.map((a) => {
                const logo = a.logo ? (
                  <span
                    className={`affiliation-logo ${a.logoLayout === "padded-square" ? "affiliation-logo-padded" : ""}`}
                  >
                    <img src={dataUrl(a.logo)} alt={a.name} loading="lazy" />
                  </span>
                ) : (
                  <span className="affiliation-name">{a.name}</span>
                );
                return (
                  <li
                    key={a.name}
                    title={a.label ? `${a.label}: ${a.name}` : a.name}
                  >
                    {a.url ? <a href={a.url}>{logo}</a> : logo}
                  </li>
                );
              })}
            </ul>
          )}
          {(m.github || m.linkedin || m.portfolio) && (
            <div className="member-links">
              {m.github && (
                <a href={m.github}>
                  GitHub <Arrow diagonal />
                </a>
              )}
              {m.linkedin && (
                <a href={m.linkedin}>
                  LinkedIn <Arrow diagonal />
                </a>
              )}
              {m.portfolio && (
                <a href={m.portfolio}>
                  Portfolio <Arrow diagonal />
                </a>
              )}
            </div>
          )}
          {!!m.projects?.length && (
            <div className="member-projects">
              {m.projects.map((p) =>
                p.url ? (
                  <a key={p.title} href={p.url}>
                    {p.title} <Arrow diagonal />
                  </a>
                ) : (
                  <p key={p.title}>{p.title}</p>
                ),
              )}
            </div>
          )}
        </article>
      ))}
    </div>
  );
}
