# Design

The site's look, as built (`src/app/globals.css`). Reference the user pinned: the CADIS Aero landing
(`~/dev/CADIS-landing`), Palantir, Astro Mechanica.

## Principles
- **The camera leads.** The home hero is the raw footage, full-bleed, one headline bottom-left:
  "Intelligence for every road." Right under it, the same camera through the model.
- **No boxes.** No borders or cards around content; space and type separate things. Hairlines only
  where rows need them (tables, lists, the footer's top edge).
- **One colour scheme.** Monochrome UI on the night-road ground. Colour belongs to the data: the event
  classes (`configs/palette.json`, shared with the renders). The one UI accent, orange, means accident
  risk (risk curve, Part B in the approach diagram) and nothing else.
- **Minimal words.** No eyebrow labels, no section numbers, no decorative dots or crosses.

## Tokens (`:root`)
| Token | Value | Use |
|---|---|---|
| `--ground` | `#111312` | page background; the hero scrim grades into it |
| `--raised` | `#181b19` | panels (notices, missing data), portrait backdrop |
| `--ink` | `#efeee8` | text, primary button, selected choice |
| `--muted` | `#a3a79e` | secondary text (≥ 7:1 on ground) |
| `--faint` | `#6d726b` | large text only (the words around the key numbers) |
| `--line` | ink at 10% | table and list hairlines |
| `--accent` | `#ef8855` | accident risk only |
| `--max` / `--gutter` | 1680px / clamp(20px, 4vw, 56px) | one column edge for header, hero headline and every page |

## Type
Inter (self-hosted, `src/app/fonts/`, via `next/font/local`) for everything; IBM Plex Mono only for code.
- Hero headline: 600, clamp(2.5rem, 6.6vw, 6rem), tracking -0.035em.
- Page title: 600, clamp(2.75rem, 6vw, 5.5rem). Home section titles: clamp(2rem, 4.4vw, 4rem).
  Inner section titles: clamp(1.5rem, 2.6vw, 2.25rem). Card titles 20–22px.
- Body 16px, lead 17–19px, meta 13–14px. Nothing smaller than 12px.
- Tracked uppercase (0.22em nav, 0.12em table heads) only in navigation and table heads.

## Components
- **Header:** logo lockup (mark + ROADWATCH tracked 0.34em); five links (Demo, Results, Approach, EDA,
  Team) plus MENU, which opens a full-screen list of every page. Transparent over the home hero, solid
  elsewhere and after scrolling.
- **Buttons:** primary = ink block, square, 52px; secondary = text link with an arrow.
- **Choices** (sample pickers): square outlined buttons, selected = ink fill.
- **Tables:** no fills, hairline rows, tabular numbers.
- **Charts** (Plotly, `Chart.tsx`): transparent, faint grid, legend above the plot, Inter via `var(--sans)`.
- **Team:** portrait on a quiet gradient, name, role, logos on white tiles in greyscale (colour on hover).

## Motion
One entrance (header and headline rise, 0.8–1s, ease-out), the menu's staggered rise, hover changes of
colour or a 3px arrow nudge. Everything is off under `prefers-reduced-motion`; the hero video does not
autoplay then, or on Save-Data.
