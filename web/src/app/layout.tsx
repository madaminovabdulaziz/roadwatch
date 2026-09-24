import type { Metadata, Viewport } from "next";
import type { ReactNode } from "react";
import Nav from "@/components/Nav";
import "./globals.css";

export const metadata: Metadata = {
  title: { default: "RoadWatch", template: "%s · RoadWatch" },
  description: "Traffic event detection and accident anticipation for a fixed road camera (WIUT Hackathon 2026).",
};

export const viewport: Viewport = { width: "device-width", initialScale: 1, themeColor: "#09090b" };

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen antialiased">
        <Nav />
        {children}
        <footer className="mx-auto max-w-6xl px-4 py-10 text-xs text-zinc-600">
          RoadWatch · WIUT Hackathon 2026 · open weights only, runs offline on one T4
        </footer>
      </body>
    </html>
  );
}
