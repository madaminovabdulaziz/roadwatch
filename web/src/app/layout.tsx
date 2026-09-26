import type { Metadata, Viewport } from "next";
import localFont from "next/font/local";
import type { ReactNode } from "react";
import Nav from "@/components/Nav";
import Link from "next/link";
import { Brand } from "@/components/ui";
import "./globals.css";

// Self-hosted (no font service at build or run time), the same faces as the CADIS landing.
const sans = localFont({
  src: "./fonts/inter-latin.woff2",
  weight: "400 600",
  variable: "--font-sans",
  display: "swap",
});
const mono = localFont({
  src: [
    { path: "./fonts/ibm-plex-mono-400-latin.woff2", weight: "400" },
    { path: "./fonts/ibm-plex-mono-500-latin.woff2", weight: "500" },
  ],
  variable: "--font-mono",
  display: "swap",
});

export const metadata: Metadata = {
  title: { default: "RoadWatch", template: "%s · RoadWatch" },
  description:
    "Traffic event detection and accident anticipation for a fixed road camera (WIUT Hackathon 2026).",
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  themeColor: "#111312",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html
      lang="en"
      data-scroll-behavior="smooth"
      className={`${sans.variable} ${mono.variable}`}
    >
      <body className="min-h-screen antialiased">
        <a className="skip-link" href="#main-content">
          Skip to content
        </a>
        <Nav />
        {children}
        <footer className="site-footer">
          <div className="site-container">
            <Link href="/" aria-label="RoadWatch home">
              <Brand />
            </Link>
            <nav className="footer-links" aria-label="Footer">
              <Link href="/report/">Report</Link>
              <Link href="/team/">Team</Link>
              <Link href="/links/">Source and weights</Link>
            </nav>
            <p>
              WIUT Hackathon 2026. Open weights, offline inference, one T4 GPU.
            </p>
          </div>
        </footer>
      </body>
    </html>
  );
}
