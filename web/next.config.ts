import type { NextConfig } from "next";

// Static export (docs/WEBSITE_SPEC.md): `npm run build` writes out/, which any static host (Vercel)
// serves. Data under public/data/ is fetched at runtime, so regenerating it needs no code change.
const config: NextConfig = {
  output: "export",
  trailingSlash: true,
  images: { unoptimized: true },
  reactStrictMode: true,
};

export default config;
