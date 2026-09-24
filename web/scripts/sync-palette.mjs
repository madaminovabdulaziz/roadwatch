// Copy the shared class palette (configs/palette.json, also read by roadwatch/render.py) into the web
// sources, so the video, timeline and charts use the same colours. Runs before `dev` and `build`.
import { copyFileSync, mkdirSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const src = resolve(here, "../../configs/palette.json");
const dst = resolve(here, "../src/generated/palette.json");
mkdirSync(dirname(dst), { recursive: true });
copyFileSync(src, dst);
console.log(`palette: ${src} -> ${dst}`);
