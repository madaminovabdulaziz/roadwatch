# Website

Next.js static export (App Router, TypeScript, Tailwind, Plotly basic bundle). Pages and rules:
`docs/WEBSITE_SPEC.md`. Every number on the site comes from a file under `public/data/` written by a
script; a missing file shows which script makes it instead of breaking the page.

```bash
cd web
npm ci
npm run dev        # http://localhost:3000 (copies configs/palette.json first)
npm run build      # static site in web/out/
```

## Deploy (Vercel)

Import the repository, set **Root Directory** to `web`, keep the Next.js preset (build `npm run build`,
output `out`). Environment variable `NEXT_PUBLIC_DEMO_API` = the demo backend URL (Hugging Face Space,
RUNBOOK P1.7), without a trailing slash. Without it the upload shows a clear message and the example
buttons still work.

## Data (public/data/)

| File | Written by | Used on |
|---|---|---|
| `metrics.json` | `scripts/eval_dev.py` | Home, Results |
| `eda/summary.json`, `eda/<video>/*.json`, `eda/*.jpg` | `scripts/eda.py samples/` | EDA |
| `results/index.json` + `results/<id>/{annotated.mp4, poster.jpg, events.json, risk.json}` | `scripts/render_samples.py` (P2.3) | Home, Results, Demo examples, Dashboard |
| `results/gallery.json`, `results/failures.json`, `dashboard/event_heat.png` | P2.3 / P2.4 | Results, Dashboard |
| `scene_overlay.jpg` | `scripts/render_scene.py --out web/public/data/scene_overlay.jpg` | Approach |
| `team.json` (+ photos in `team/`) | by hand | Team |
| `report.json` | RUNBOOK P4.1 | Report |
| `predictions_samples.json` | copy of the harness output | Links |

`results/index.json`: `{"videos": [{"id", "name", "duration", "video", "poster", "events", "risk"}],
"enabled_classes": [...], "runtime_x_duration": 1.7}`; paths are relative to `public/data/`. Events are
`[[start, end, label], ...]` and risk `[[t, score], ...]`, the evaluate.py formats.
Class colours: `configs/palette.json` (shared with `roadwatch/render.py`).
