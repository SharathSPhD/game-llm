# Kinetic AI Project Site

An Astro-based project site for Kinetic AI: Language Modeling as Equilibrium Computation.

## Build

```bash
npm ci       # install dependencies (do not use npm install)
npm run build   # build the site to dist/
npm run dev    # start dev server (localhost:3000)
npm run preview  # preview production build
```

## Data Flow: Findings → Results

Validated findings are the source of truth:

1. **Write findings** to `research/memory/findings.md` (peer-reviewed, confidence intervals, Holm–Bonferroni correction).
2. **Update** `src/data/results.json` with structured findings (ID, title, status, summary).
3. **Paper figures** read hardcoded constants from `paper/figures/make_paper_figures.py` (extracted from the same findings).
4. **Results page** (`src/pages/results.astro`) renders from `src/data/results.json` dynamically.

This ensures the paper, site, and code are **always in sync**—single source of truth.

## Structure

```
site/
  src/
    layouts/Base.astro         # shared page layout
    pages/
      index.astro              # overview
      method.astro             # architecture & method
      results.astro            # validated findings (rendered from results.json)
    data/results.json          # findings data
  package.json
  astro.config.mjs
  tsconfig.json
```

## Deployment

Site is served on GitHub Pages at `https://SharathSPhD.github.io/game-llm` (see `astro.config.mjs`).
