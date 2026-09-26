"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useRef, useState } from "react";
import { Brand } from "@/components/ui";

const PAGES: [string, string][] = [
  ["/demo/", "Live demo"],
  ["/results/", "Results"],
  ["/eda/", "EDA"],
  ["/approach/", "Approach"],
  ["/dashboard/", "Dashboard"],
  ["/report/", "Report"],
  ["/team/", "Team"],
  ["/links/", "Links"],
];
// The bar keeps five; the menu panel lists every page.
const BAR: [string, string][] = [
  ["/demo/", "Demo"],
  ["/results/", "Results"],
  ["/approach/", "Approach"],
  ["/eda/", "EDA"],
  ["/team/", "Team"],
];

const same = (a: string, b: string) =>
  a.replace(/\/$/, "") === b.replace(/\/$/, "");

export default function Nav() {
  const path = usePathname();
  const home = same(path, "/");
  const [open, setOpen] = useState(false);
  const [solid, setSolid] = useState(!home);
  const panel = useRef<HTMLDivElement>(null);
  const toggle = useRef<HTMLButtonElement>(null);

  // Over the home hero the bar is transparent; it takes the page colour once the video is scrolled past.
  useEffect(() => {
    if (!home) {
      setSolid(true);
      return;
    }
    const update = () => setSolid(window.scrollY > window.innerHeight * 0.6);
    update();
    window.addEventListener("scroll", update, { passive: true });
    return () => window.removeEventListener("scroll", update);
  }, [home]);

  // Menu panel: Escape closes, Tab stays inside, the page behind does not scroll.
  useEffect(() => {
    if (!open) return;
    const root = document.documentElement;
    root.style.overflow = "hidden";
    panel.current?.querySelector<HTMLElement>("a")?.focus();
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setOpen(false);
        toggle.current?.focus();
        return;
      }
      if (event.key !== "Tab" || !panel.current) return;
      const items = [
        toggle.current,
        ...panel.current.querySelectorAll<HTMLElement>("a"),
      ].filter((el): el is HTMLElement => !!el);
      const first = items[0];
      const last = items[items.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => {
      root.style.overflow = "";
      window.removeEventListener("keydown", onKey);
    };
  }, [open]);

  const close = () => setOpen(false);
  return (
    <header
      className={`site-header ${home ? "is-over-hero" : ""} ${solid ? "is-solid" : ""} ${open ? "is-open" : ""}`}
    >
      <div className="nav-inner">
        <Link href="/" aria-label="RoadWatch home" onClick={close}>
          <Brand />
        </Link>
        <nav className="desktop-nav" aria-label="Main navigation">
          {BAR.map(([href, label]) => (
            <Link
              key={href}
              href={href}
              aria-current={same(path, href) ? "page" : undefined}
              className="nav-link"
            >
              {label}
            </Link>
          ))}
        </nav>
        <button
          ref={toggle}
          type="button"
          aria-expanded={open}
          aria-controls="site-menu"
          className="menu-toggle"
          onClick={() => setOpen(!open)}
        >
          <span className="menu-toggle-label">{open ? "Close" : "Menu"}</span>
          <span className="menu-toggle-bars" aria-hidden="true">
            <span />
            <span />
          </span>
        </button>
      </div>
      <div
        id="site-menu"
        ref={panel}
        className="menu-panel"
        hidden={!open}
        role="dialog"
        aria-modal="true"
        aria-label="Site menu"
      >
        <nav aria-label="All pages">
          {PAGES.map(([href, label]) => (
            <Link
              key={href}
              href={href}
              onClick={close}
              aria-current={same(path, href) ? "page" : undefined}
              className="menu-link"
            >
              {label}
            </Link>
          ))}
        </nav>
      </div>
    </header>
  );
}
