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
/** GitHub mark (simple-icons path, CC0). */
function GitHubIcon() {
  return (
    <svg className="social-icon" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
      <path d="M12 .3a12 12 0 0 0-3.8 23.4c.6.1.8-.3.8-.6v-2c-3.3.7-4-1.6-4-1.6-.6-1.4-1.4-1.8-1.4-1.8-1-.7.1-.7.1-.7 1.2.1 1.8 1.2 1.8 1.2 1 1.8 2.8 1.3 3.5 1 0-.8.4-1.3.7-1.6-2.7-.3-5.5-1.3-5.5-6 0-1.2.5-2.3 1.3-3.1-.2-.4-.6-1.6 0-3.2 0 0 1-.3 3.4 1.2a11.5 11.5 0 0 1 6 0c2.3-1.5 3.3-1.2 3.3-1.2.6 1.6.2 2.8 0 3.2.9.8 1.3 1.9 1.3 3.2 0 4.6-2.8 5.6-5.5 5.9.5.4.9 1 .9 2.2v3.3c0 .3.1.7.8.6A12 12 0 0 0 12 .3" />
    </svg>
  );
}

/** LinkedIn mark (simple-icons path, CC0). */
function LinkedInIcon() {
  return (
    <svg className="social-icon" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
      <path d="M20.45 20.45h-3.56v-5.57c0-1.33-.02-3.04-1.85-3.04-1.85 0-2.14 1.45-2.14 2.94v5.67H9.35V9h3.42v1.56h.05c.48-.9 1.64-1.85 3.37-1.85 3.6 0 4.27 2.37 4.27 5.46v6.28zM5.34 7.43a2.06 2.06 0 1 1 0-4.13 2.06 2.06 0 0 1 0 4.13zM7.12 20.45H3.56V9h3.56v11.45zM22.22 0H1.77C.79 0 0 .77 0 1.73v20.54C0 23.23.79 24 1.77 24h20.45c.98 0 1.78-.77 1.78-1.73V1.73C24 .77 23.2 0 22.22 0z" />
    </svg>
  );
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
                <a href={m.github} className="social-link" target="_blank" rel="noopener noreferrer">
                  <GitHubIcon /> GitHub
                </a>
              )}
              {m.linkedin && (
                <a href={m.linkedin} className="social-link" target="_blank" rel="noopener noreferrer">
                  <LinkedInIcon /> LinkedIn
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
