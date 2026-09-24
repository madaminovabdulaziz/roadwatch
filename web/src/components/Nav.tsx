"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState } from "react";

export const PAGES: [string, string][] = [
  ["/demo/", "Live demo"],
  ["/results/", "Results"],
  ["/eda/", "EDA"],
  ["/approach/", "Approach"],
  ["/dashboard/", "Dashboard"],
  ["/report/", "Report"],
  ["/team/", "Team"],
  ["/links/", "Links"],
];

export default function Nav() {
  const path = usePathname();
  const [open, setOpen] = useState(false);
  const item = (href: string, label: string) => (
    <Link
      key={href}
      href={href}
      onClick={() => setOpen(false)}
      className={`rounded-md px-3 py-2 text-sm ${path === href ? "bg-white/10 text-white" : "text-zinc-400 hover:text-white"}`}
    >
      {label}
    </Link>
  );
  return (
    <header className="sticky top-0 z-20 border-b border-white/10 bg-zinc-950/85 backdrop-blur">
      <div className="mx-auto flex max-w-6xl items-center justify-between px-4 py-2">
        <Link href="/" className="font-semibold tracking-tight">
          Road<span className="text-sky-400">Watch</span>
        </Link>
        <nav className="hidden gap-1 lg:flex">{PAGES.map(([h, l]) => item(h, l))}</nav>
        <button
          type="button"
          aria-label="menu"
          aria-expanded={open}
          className="rounded-md px-3 py-2 text-sm text-zinc-300 lg:hidden"
          onClick={() => setOpen(!open)}
        >
          {open ? "Close" : "Menu"}
        </button>
      </div>
      {open && <nav className="flex flex-col px-4 pb-3 lg:hidden">{PAGES.map(([h, l]) => item(h, l))}</nav>}
    </header>
  );
}
