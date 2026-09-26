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
`demo/README.md`), without a trailing slash. Without it the upload shows a clear message and the example
buttons still work.

## Data (public/data/)

`bash scripts/build_site_data.sh` regenerates all of it, in order, from `predictions_samples.json` (run it
after the final Kaggle run; the full-length renders take about 30 minutes on a laptop CPU).

| File | Written by | Used on |
|---|---|---|
| `metrics.json` | `scripts/eval_dev.py --pred predictions_samples.json` (the submission run vs the dev labels) | Home, Results |
| `eda/summary.json`, `eda/<video>/*.json`, `eda/*.jpg` | `scripts/eda.py samples/` | EDA |
| `results/index.json` + `results/<id>/{annotated.mp4, poster.jpg, events.json, risk.json}` | `scripts/render_samples.py` | Home, Results, Demo examples, Dashboard |
| `results/gallery.json` | `scripts/render_samples.py` (per class, the event that best matches a dev label) | Results |
| `results/failures.json` + `results/failures/*.mp4` | `scripts/render_failures.py` | Results |
| `dashboard/event_heat.jpg` | `scripts/event_heat.py` (the tracks behind each event, from the rules on the caches) | Dashboard |
| `scene_overlay.jpg` | `scripts/render_scene.py --out web/public/data/scene_overlay.jpg` | Approach |
| `team.json` (+ photos in `team/`) | by hand | Team |
| `report.json` | `scripts/write_report.py` (prose in the script, numbers from the files above) | Report |
| `predictions_samples.json` | copy of the harness output | Links |

`results/index.json`: `{"videos": [{"id", "name", "duration", "video", "poster", "events", "risk"}],
"enabled_classes": [...], "runtime_x_duration": 1.7}`; paths are relative to `public/data/`. Events are
`[[start, end, label], ...]` and risk `[[t, score], ...]`, the evaluate.py formats.
Class colours: `configs/palette.json` (shared with `roadwatch/render.py`).
