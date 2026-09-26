// Copy the shared class palette (configs/palette.json, also read by roadwatch/render.py) into the web
// sources, so the video, timeline and charts use the same colours. Runs before `dev` and `build`. The copy
// is committed: a host that builds web/ alone (no files outside it) keeps it as it is.
import { copyFileSync, existsSync, mkdirSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const src = resolve(here, "../../configs/palette.json");
const dst = resolve(here, "../src/generated/palette.json");
if (existsSync(src)) {
  mkdirSync(dirname(dst), { recursive: true });
  copyFileSync(src, dst);
  console.log(`palette: ${src} -> ${dst}`);
} else if (existsSync(dst)) {
  console.log(`palette: ${src} not visible here; keeping the committed ${dst}`);
} else {
  throw new Error(`palette: neither ${src} nor ${dst} exists`);
}
