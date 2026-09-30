# Social simulation audit

A static, English-language research article, designed to match the author's
auditory categorization page: white background, narrow prose, blue/orange
figures, and controls embedded next to the evidence they change.

Published at https://danceofthepkz.github.io/intervention-response-audit/.
GitHub Pages serves the `docs/` folder on `main`; updates to that folder publish
automatically. `.nojekyll` keeps the files as a plain static site.

## Preview

Open `index.html` directly, or from the repository root run:

```sh
python -m http.server 8765 --directory docs
```

Then open http://localhost:8765. No build service, API key, or remote JavaScript
dependency is needed. This folder can be served as a static website.

## Reading sequence

The page is a continuous explanation with three section headings, not a dashboard.
It first explains how the simulation works, then follows one recorded message pair.

1. Change the wording and see the teacher's recorded response.
2. See why the same 27 input features force the replacement model to give the
   same answer. Move that answer to try matching both teacher probabilities.
3. Add message information and compare the reported mean response recovery.

All 60 examples, repeated calls, KL calculations, and source tables remain
available in optional disclosures. The main text explains the experiment before
introducing results. Alternative representations are not an ordered ladder.

## Data and validation

Using the repository dependencies, regenerate the web data with:

```sh
python docs/build_data.py
```

This writes `data.json`, its local-file-compatible `data.js` wrapper, and
`validation.json`. The export uses an explicit allowlist of fields; it does not
include API response IDs, credentials, internal notes, or raw API metadata.
Source paths and SHA-256 hashes are recorded in the data bundle.

The exporter verifies state/prompt joins, three calls per arm, unchanged prompt
context, all 60 primary contrasts, feature construction, the exact per-state
KL identity, and the frozen aggregate values. R0 predictions come from the
archived cross-fit. The initial example is nearest the median signed response.

`representation-summary.json` is explicitly transcribed from the author-supplied
manuscript. R1–R3 model files and per-state predictions were not available in the
clean release. This page does not invent those predictions or claim to refit
them. It reports the unseen-wording result qualitatively; no numeric subset
scores are synthesized.

The page does not simulate new diffusion trajectories. It displays LLM-stated
reshare probabilities, not human outcomes. Blue denotes authority and orange
denotes hedge in the paired view; other charts explicitly label their colors.

## Design references

- https://distill.pub/2020/communicating-with-interactive-articles/
- https://playground.tensorflow.org/
- https://danceofthepkz.github.io/behavioral-modeling-thesis/auditory_categorization_learning/
