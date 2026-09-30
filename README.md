# intervention-response-audit

This repository contains the code and frozen synthetic artifacts needed to
reproduce the quantitative results for **When Aggregate Fidelity Misses
Intervention Response: A Pre-Fitting Screen for LLM Social Simulation**.

## Main result

Across 60 paired simulator states, authority-framed messages increased the
teacher-stated reshare probability by `0.0866` relative to matched
hedge-framed messages. A 27-feature text-blind representation maps both sides
of every pair to the same input, so its intervention response must be zero.
The corresponding representation-imposed aliasing floor is `0.01121` nats.

## Contents

- `intervention_response_audit/`: simulation, collection, integrity, and analysis code.
- `tests/`: automated tests.
- `data/`: the two synthetic source files needed to rebuild the frozen sample
  and rerun cross-fitting.
- `prereg/v2/`: frozen states, prompts, schedules, and acceptance records.
- `results/v2/`: frozen model responses and derived numerical results.
- `paper/figures/make_figures.py`: reproducible figure generator.
- `V2_STUDY_PLAN.md`: the frozen experimental protocol required by the Stage-0
  rebuild test.

Research notes, manuscript drafts, credentials, temporary files, unrelated raw
logs, and unfinished follow-up studies are not included.

## Setup

Python 3.11 or 3.12 is recommended.

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## Verify the repository

No API calls or API key are required.

```bash
.venv/bin/python -m pytest -q
```

The expected result is `110 passed`.

## Reproduce the reported quantities

```bash
.venv/bin/python -m intervention_response_audit.v2.reported_quantities \
  --output reproduced/reported_quantities.json
```

The output should recover:

- authority-minus-hedge response: `0.0866`;
- scenario-equal ordinary-row KL regret: `0.03622` nats;
- paired-anchor total KL regret: `0.03806` nats;
- irreducible aliasing floor: `0.01121` nats.

The frozen reference output is
`results/v2/reported_quantities/reported_quantities.json`.

## Reproduce the figures

```bash
MPLCONFIGDIR=/tmp/intervention-response-audit-matplotlib-cache \
  .venv/bin/python paper/figures/make_figures.py
```

This writes three vector PDFs to `output/pdf/`.

## Interactive article

Read [**Social simulation audit**](https://danceofthepkz.github.io/intervention-response-audit/),
an interactive explanation using recorded examples from this study.

Open [`docs/index.html`](docs/index.html) to explore the paired messages,
representation collision, and loss floor using the frozen data. The page runs
locally without API calls. See [`docs/README.md`](docs/README.md) for preview
and data-regeneration instructions.

## Scope

All agents, messages, networks, and outcomes are synthetic. The package
reproduces measurements of a pinned LLM simulator; it does not estimate a
causal effect on people.
