# Intervention Response Audit v2 Study Plan

## A repeatability-aware, feature-clamped audit of an LLM transition rule

**Status:** protocol ready for Stage-0 freeze; no v2 live calls are authorized by
this document alone. The state manifests, prompt hashes, analysis code, tests,
and live budget must be frozen in a subsequent pre-live commit before the pilot.

**Primary question:** Under the frozen baseline LLM configuration and a
stratified sample of decision states induced by the v1 baseline rollout, does an
authority framing change the environment's reshare transition probability
relative to a hedge framing?

**Role in the paper:** v2 is a confirmatory repair of the ambiguity exposed by
v1. It is not a new factorial sweep, a general test of all framing effects, or an
agent-training experiment.

---

## 1. Study logic

The v1 audit established three facts that determine the v2 design:

1. Local semantic-twin differences can coexist with a failed global audit.
2. A single stochastic call per twin side cannot isolate a framing effect from
   call-level variation.
3. A cross-distribution scaffold statistic is not needed to estimate
   repeatability; byte-identical repeated prompts can measure it directly.

v2 therefore audits one transition variable (`reshare`), under one deployment
configuration, on frozen decision states. It changes only the message framing,
repeats identical prompts when the endpoint is stochastic, and treats the
decision state—not an individual API call—as the unit of analysis.

The study separates two questions:

- **Confirmatory framing question:** Does authority versus hedge produce a
  reproducible signed change in reshare probability?
- **Secondary predictability question:** How much same-test loss remains for the
  pre-specified structured predictor on normal rows and on the counterfactual v2
  anchor states?

These quantities are reported together but are not combined into a single
"richness" score or universal PASS/FAIL judgment.

---

## 2. Scope and claims

### 2.1 Confirmatory scope

v1 identified authority–hedge as the largest descriptive framing contrast. v2
treats that exploratory pattern as a directional hypothesis evaluated on new
frozen states that were not used in the v1 twin analysis.

The target population is deliberately narrow:

> Decision states represented by the 80 frozen v1 baseline scenarios and
> induced by the baseline rollout configuration, conditional on the 15 frozen
> topic × credibility authority–hedge template pairs.

The primary estimand is a fixed-template effect. It does not generalize to
arbitrary linguistic realizations of authority or hedge framing. A separate
template-cluster sensitivity interval probes how strongly inference depends on
the 15 particular wording pairs, but it does not turn them into a representative
sample of all possible framings.

The study does not target every possible campus state, every framing contrast,
another model family, or human behavior.

### 2.2 Permitted conclusions

If the positive direction replicates, the paper may conclude that authority
relative to hedge changes the tested LLM reshare transition under the frozen
configuration and state distribution.

If the sign reverses, the paper may conclude that the tested framing distinction
changes reshare judgments but that the v1 directional pattern was unstable.

If the interval is narrow around zero, the paper may bound the authority–hedge
effect under the tested conditions. A non-significant but wide interval is
inconclusive and must not be described as evidence that framing is unimportant.

No result supports a general claim that "framing matters," "framing does not
matter," the environment is human-valid, or the entire LLM environment is
compressible or irreducible.

---

## 3. Frozen LLM configuration

v2 uses the original baseline configuration because it is the project's frozen
reference, the least expensive evaluated configuration, and the configuration
for which the original authority–hedge pattern was observed.

| Parameter | Frozen value |
| --- | --- |
| Model snapshot | `gpt-5.4-mini-2026-03-17` |
| Reasoning effort | `none` |
| Temperature | `0` |
| Output cap | 300 tokens |
| Response format | strict JSON object |
| Instruction scaffold | scaffold 0 |
| Primary output | `reshare` |
| Secondary output | `believe` |
| Collected but non-analytic output | `attend` |

Before the pilot, a smoke check must verify that the exact snapshot remains
available and accepts every frozen parameter. If the snapshot is unavailable or
rejects temperature 0, the study stops. No automatic model alias, temperature
fallback, or reasoning fallback is permitted.

---

## 4. Units, state allocation, and sampling

### 4.1 Units

- **Sampling and inferential unit:** one v1 scenario.
- **Experimental state:** one non-seed decision row selected from that scenario.
- **Treatment conditions:** authority and hedge versions of the same frozen
  prompt.
- **Replicates:** repeated API calls with byte-identical prompt content within a
  state × condition cell.

At most one experimental state is selected from each scenario. State and
scenario are therefore 1:1, and no scenario-cluster correction is needed beyond
resampling states.

### 4.2 Allocation of the 80 scenarios

Before any v2 call, all scenarios are assigned once and frozen:

- 60 confirmatory scenarios: 12 from each of the five topics;
- 10 pilot scenarios: 2 from each topic;
- 10 unused reserve scenarios: 2 from each topic.

Pilot scenarios can never enter the confirmatory analysis. Reserve scenarios
may replace an unusable state only before the first v2 API call and only through
the pre-specified same-stratum replacement rule. After the first pilot call, no
confirmatory scenario or state may be replaced.

### 4.3 Confirmatory-state margins

The 60 confirmatory states are selected using only treatment-preceding
structured variables. Oracle probabilities, reason codes, original framing, and
v1 outcomes are forbidden selection inputs.

The state manifest must satisfy these exact margins:

| Variable | Target counts |
| --- | --- |
| Topic | 12 each for five topics |
| Credibility | 23 low (`0.25`), 16 mid (`0.55`), 21 high (`0.82`) |
| Exposure count | 32 with 1 exposure, 20 with 2, 8 with 3 or more |
| Diffusion step | 15 at step 1, 16 at step 2, 29 at step 3 or later |

Every one of the 15 topic × credibility template cells must contain at least two
confirmatory states. This makes all canonical pairs observable while retaining
one state per scenario.

Selection uses a deterministic constrained optimizer with random seed
`20260716` only to break ties. Constraints are: exactly 60 rows, at most one row
per scenario, the exact margins above, and a valid archived full prompt for every
selected row.

The identity key for a state is
`(scenario_id, agent_id, step, query_index)`. The manifest stores this key, all
structured features, the original full-prompt hash, topic, credibility tier,
exposure bin, step bin, and allocation role.

### 4.4 Pilot-state selection

From the 20 scenarios left after confirmatory allocation, a deterministic
seeded selection assigns two scenarios per topic to the pilot and two to reserve.
Within each pilot scenario, one valid state is chosen without inspecting any
oracle output and with margins as close as possible to the confirmatory exposure
and step distribution.

---

## 5. Framing intervention and prompt integrity

### 5.1 Intervention

The intervention is **evidential framing**, not a semantics-preserving
paraphrase:

- authority: an official or authority-linked source confirms the topic object;
- hedge: the same topic object is described as unverified or potentially rumor.

The directional prediction is that authority raises reshare probability relative
to hedge.

### 5.2 Variant source

v2 reuses the existing `FRAMED_TEMPLATE_BANK`, which was implemented and frozen
before the v1 live study. The bank already contains authority and hedge versions
for all 5 topics × 3 credibility tiers. No LLM generates v2 variants, and no
free-form per-state rewriting is permitted.

This yields 15 canonical authority–hedge template pairs rather than 120
independently authored messages. The narrower lexical coverage is accepted in
exchange for substantially stronger content control.

### 5.3 Allowed and forbidden differences

Across A/H versions, the following may change:

- source status;
- evidential certainty;
- hedging and rumor language.

The following must remain fixed:

- topic object and factual event category;
- people, location, time, action request, and quantitative details when present;
- persona, relationship descriptions, exposure sources, source comments,
  instruction scaffold, and required JSON schema.

Each A/H prompt is constructed by replacing only the quoted message span in the
archived full prompt. An automated diff must confirm that all bytes outside that
span are identical.

### 5.4 Validation and freeze

Before the pilot:

1. A reviewer blind to v1 probability outcomes checks all 15 canonical template
   pairs for constant factual payload and successful framing manipulation.
2. Every pilot and confirmatory A/H full prompt is rendered.
3. All prompts receive SHA-256 hashes.
4. The state manifest, prompt corpus, validation checklist, and hashes are
   committed and time-stamped.

If a canonical pair fails validation, both sides must be repaired from a shared
canonical fact statement before any v2 call. The repair and its rationale are
frozen. After the first v2 call, wording cannot change.

---

## 6. Practical-effect threshold

The operational threshold is fixed at

\[
\delta = 0.03
\]

in reshare-probability units.

This value is calibrated from the v1 baseline decision distribution rather than
chosen after v2 results. The 80 baseline scenarios queried a mean of 34.5375
unique non-seed agents. Under a first-order fixed-reachability approximation, a
uniform shift of

\[
1/34.5375 = 0.028954
\]

corresponds to one additional expected resharer per scenario. The threshold is
rounded upward to 0.03.

This is a conservative transition-level trigger because a resharer is not merely
an endpoint count: resharing creates a new exposure source and can affect
multiple later decisions in the branching process. The study does not assume or
claim a guaranteed downstream multiplier.

This is an operational scale, not a causal forecast of final cascade size. A
full counterfactual cascade calibration is deliberately not binding because v1
does not contain LLM outputs for branches that were not reached, and the current
project has no trained transition surrogate that could populate those branches
without introducing a new model-dependent assumption.

---

## 7. Repeatability pilot and adaptive replicate count

### 7.1 Pilot calls

For each of 10 pilot states:

- render one authority and one hedge prompt;
- call each byte-identical prompt 10 times;
- randomize and interleave the 20 calls within the state;
- randomize state execution order;
- log prompt hash, scheduled order, timestamp, response ID, token/cache metadata,
  raw response, parsed probabilities, latency, and cost.

The pilot therefore contains exactly 200 valid scientific calls, excluding
engineering smoke calls and parse-failure retries.

### 7.2 Response-reuse check

Input-token caching is permitted, but every scheduled call must produce a new API
usage event and a unique response ID. A repeated response ID, absence of a new
usage event, or evidence of client-side response reuse is an integrity failure
and stops the study pending a documented protocol revision.

Identical numeric outputs with unique response IDs are treated as evidence of
operational determinism, not automatically as response memoization.

### 7.3 Deterministic branch

If all 20 state × condition cells have exactly zero within-cell variance for the
primary `reshare` output across their 10 calls, and the response-reuse check
passes, confirmatory calls use `K = 1` per side.

In this branch, no call-level permutation test is performed. Each state-level
contrast is treated as an exact measurement of the frozen endpoint, and
uncertainty is estimated across the 60 sampled states.

### 7.4 Stochastic branch and K rule

Otherwise, let

\[
v_j=s^2_{j,A}+s^2_{j,H}
\]

be the sum of the two pilot within-prompt sample variances for state `j`, and let
`Q90(v)` be its 90th percentile across pilot states, calculated with the
conservative `higher` quantile convention. For a confirmatory study of
`J = 60` states, the conservative projected call-noise contribution to the
standard error of the overall mean is

\[
SE_{call}(K)=\sqrt{\frac{Q90(v)}{60K}}.
\]

Choose the smallest `K` in `{3, 5, 10}` satisfying

\[
SE_{call}(K) \le \delta/4 = 0.0075.
\]

The rule sizes K for the primary population-average contrast, not for precise
estimation of every state-specific effect. State-level heterogeneity is
descriptive.

If `K = 10` does not satisfy the rule, confirmatory collection does not begin.
Proceeding would require an explicit, time-stamped protocol amendment and new
budget authorization.

### 7.5 Post-pilot freeze

The pilot report records the response-reuse checks, all cell variances, `Q90(v)`,
the mechanically selected K, projected call count, and cost. K is then committed
before any confirmatory call. Pilot probabilities are never pooled with
confirmatory probabilities.

---

## 8. Confirmatory collection

### 8.1 Call count

The confirmatory study contains

\[
60\text{ states}\times2\text{ conditions}\times K
\]

valid calls:

- 120 calls if the endpoint is operationally deterministic;
- 360 calls for `K = 3`;
- 600 calls for `K = 5`;
- 1,200 calls for `K = 10`.

Including the 200-call pilot, the maximum scientific call count is 1,400.

### 8.2 Scheduling

For each state, the `2K` A/H call tasks are randomly ordered with seed
`20260717`. The global scheduler interleaves states and conditions so that all
authority calls cannot precede all hedge calls. Concurrency may change wall-clock
time but not the frozen task order recorded in the manifest.

No adaptive stopping based on observed probabilities is allowed.

### 8.3 Validity and retry policy

Each scheduled call must produce the frozen JSON schema and valid probabilities.
One identical-prompt retry is permitted after a parse or transport failure; the
failed attempt is preserved, and the first valid response fills the scheduled
replicate slot.

Across the maximum 1,400-call design, no more than 14 retry attempts are
permitted. Reaching a fifteenth retry triggers the integrity stop below.

The collection pauses before analysis if:

- a model or parameter differs from the frozen configuration;
- any sent prompt hash differs from the manifest;
- a state × condition cell lacks K valid calls after its permitted retries;
- more than 1% of scheduled calls require repair or retry;
- duplicate response IDs or duplicate composite call keys occur;
- the live cost or call guard is reached.

No failed or inconvenient state is silently replaced after collection starts.

---

## 9. Primary estimand and inference

For state `j`, define the replicate means

\[
\bar p_{A,j}=\frac{1}{K}\sum_{r=1}^{K}p_{A,jr},
\qquad
\bar p_{H,j}=\frac{1}{K}\sum_{r=1}^{K}p_{H,jr},
\]

and the signed state contrast

\[
d_j=\bar p_{A,j}-\bar p_{H,j}.
\]

The primary estimand is the equal-topic standardized mean

\[
\Delta_{AH}
=\frac{1}{5}\sum_{t=1}^{5}
  \left(\frac{1}{12}\sum_{j\in t}d_j\right).
\]

Because every topic contributes exactly 12 states, this equals the unweighted
mean over all 60 states.

### 9.1 Directional prediction and two-sided inference

The directional prediction is

\[
\Delta_{AH}>0.
\]

Inference is nevertheless two-sided. The primary 95% interval conditions on the
15 frozen templates. It is computed using 20,000 stratified bootstrap resamples,
resampling states with replacement **within each topic × credibility template
cell** while preserving that cell's observed state count. Random seed:
`20260718`.

A pre-specified template-cluster sensitivity interval is reported beside the
primary interval. Within each topic, the three credibility/template clusters
are sampled with replacement three times, with selection probabilities
proportional to their confirmatory state counts; the selected cluster means are
averaged, and the five topic means are equally weighted. This procedure is
repeated 20,000 times with seed `20260721`. It treats the 15 wording pairs as the
resampling units and is explicitly a sensitivity analysis, not the primary
fixed-template inference.

In the stochastic branch, a companion two-sided permutation test evaluates the
null that A/H call outputs are exchangeable within state. For each of 100,000
Monte Carlo permutations, the `2K` outputs in each state are randomly relabeled
into K authority and K hedge observations, and the standardized mean is
recomputed. Permutation seed: `20260719`. The p-value is

\[
p=\frac{1+\#\{|\Delta^{\pi}|\ge|\Delta^{obs}|\}}{100001}.
\]

The permutation test is omitted in the deterministic `K = 1` branch; the
pre-specified bootstrap interval remains the primary population-level inference.

If the stochastic branch is triggered by extremely small rather than
substantive within-prompt variation, the permutation null may be nearly
degenerate. Its p-value is still reported, but interpretation remains anchored
to the two intervals and the practical threshold.

### 9.2 Pre-specified interpretation

The interval receives two labels so that statistical direction and practical
magnitude cannot be conflated.

**Directional label:**

1. **Directional replication:** the 95% interval lies entirely above 0.
2. **Sign reversal:** the interval lies entirely below 0.
3. **Direction unresolved:** the interval includes 0.

**Practical-magnitude label:**

1. **Operationally meaningful positive effect:** the interval lies entirely
   above `+0.03`.
2. **Operationally meaningful negative effect:** the interval lies entirely
   below `-0.03`.
3. **Practical equivalence:** the entire interval lies inside
   `[-0.03, +0.03]`.
4. **Practical magnitude unresolved:** none of the three conditions above holds.

These labels are intentionally compatible. For example, an interval of
`[0.01, 0.02]` is a directional replication but is also practically equivalent
under the pre-specified threshold.

The labels are assigned from the primary fixed-template interval. The template-
cluster interval is always shown beside it. If the two intervals differ on sign
or practical-threshold status, the paper must state that the conclusion is
sensitive to the particular frozen wording pairs and may not generalize across
template realizations.

In the stochastic branch, the permutation p-value is reported alongside this
interval classification. It does not replace the effect-size and practical-
threshold interpretation.

---

## 10. Secondary framing analyses

All analyses below are secondary and do not alter the primary interpretation.

### 10.1 Belief

Repeat the signed contrast and interval calculation for `believe`. No multiplicity-
adjusted confirmatory claim is made.

### 10.2 Raw absolute contrast

Report

\[
\frac{1}{60}\sum_j|\bar p_{A,j}-\bar p_{H,j}|
\]

as a descriptive magnitude only. It is not tested against zero because it is
mechanically positive under finite replicate noise.

### 10.3 Text-blind regret

For each state, let

\[
m_j=(\bar p_{A,j}+\bar p_{H,j})/2.
\]

For all entropy and KL calculations, probabilities are clipped to
`[1e-6, 1-1e-6]`. The signed probability-point contrast is not clipped beyond
the API schema's required `[0,1]` range.

The raw text-blind regret is

\[
R_{text}^{raw}
=\frac{1}{60}\sum_j
\left[
H(m_j)-\frac{H(\bar p_{A,j})+H(\bar p_{H,j})}{2}
\right].
\]

In the stochastic branch, the same within-state permutations used for the null
test generate a finite-K noise baseline. Report both

\[
R_{text}^{raw}
\quad\text{and}\quad
E_{\pi}[R_{text}^{perm}],
\]

plus the descriptive difference

\[
R_{text}^{adj}=R_{text}^{raw}-E_{\pi}[R_{text}^{perm}].
\]

`R_text_adj` is called permutation-null-adjusted, not an exactly unbiased
estimator under every alternative.

In the deterministic branch, no call-noise correction is applied.

### 10.4 Heterogeneity

Topic, credibility, exposure-count, and diffusion-step summaries are descriptive
only. Also report the mean signed contrast and number of states for each of the
15 frozen template pairs. No subgroup p-values, winner selection, or subgroup-
specific claims are permitted.

---

## 11. Secondary structured-predictor analysis

### 11.1 Data separation

Only the 80 v1 **normal** scenarios enter the structured-predictor analysis.
Scaffold rows, v1 twin rows, pilot rows, and v2 confirmatory responses are never
used to fit the predictor.

### 11.2 Predictor and fitting

Use the same pre-specified 27 structured features and L2-logistic model as v1,
but remove pseudo-label sampling. Fit soft probabilities exactly by expanding
each row into:

- a positive observation with weight `p`;
- a negative observation with weight `1-p`.

This optimizes the soft-label cross-entropy without adding Bernoulli Monte Carlo
noise.

Generate honest predictions through five-fold scenario-level cross-fitting.
Within every outer training fold, choose regularization from the frozen v1 grid
`{0.001, 0.003, 0.01, 0.03, 0.1, 0.3}` using grouped inner cross-validation by
scenario. No row-wise inner split is permitted. Standardization is fit inside
each training fold; the logistic solver is `liblinear` with at most 1,000
iterations.

Outer folds are assigned by a deterministic constrained allocation with seed
`20260720`: 16 scenarios per fold, topic counts differing by at most one, and
credibility counts as balanced as the corpus permits. Inner grouped folds use
the same fixed ordering and never split a scenario.

Every normal row, including each selected v2 anchor state, receives a prediction
from a model that did not train on its scenario.

### 11.3 Full normal-distribution regret

For out-of-fold normal-row predictions `q_i`, report

\[
R_{global,full}
=\frac{1}{N}\sum_i
KL[\operatorname{Bern}(p_i)\Vert\operatorname{Bern}(q_i)].
\]

This is a secondary reanalysis of v1 normal data and receives no new PASS/FAIL
threshold.

### 11.4 Shared counterfactual-anchor decomposition

For each v2 confirmatory state, the structured predictor gives one framing-blind
out-of-fold prediction `q_j`. The predictor was trained on v1's original normal-
message distribution, whereas the repeated mean below comes from the two
counterfactual template-bank framings. This is therefore a **counterfactual-
anchor** diagnostic, not a natural-distribution anchor regret. Using the repeated
A/H means, report

\[
R_{struct,cf-anchor}
=\frac{1}{60}\sum_j
KL[\operatorname{Bern}(m_j)\Vert\operatorname{Bern}(q_j)].
\]

Also report the total average predictor regret across the two framing sides:

\[
R_{total,cf-anchor}
=\frac{1}{120}\sum_j
\left(
KL[\operatorname{Bern}(\bar p_{A,j})\Vert\operatorname{Bern}(q_j)]
+KL[\operatorname{Bern}(\bar p_{H,j})\Vert\operatorname{Bern}(q_j)]
\right).
\]

For Bernoulli cross-entropy, the anchor quantities satisfy the diagnostic
decomposition

\[
R_{total,cf-anchor}=R_{text}^{raw}+R_{struct,cf-anchor}
\]

up to floating-point error. This identity is an integrity check and a way to put
local text contrast and structured mismatch on the same anchored states. It is
not a new joint decision rule. Neither counterfactual-anchor quantity is compared
to `R_global,full` as though the two were measured on the same distribution.

---

## 12. Blinding, provenance, and analysis freeze

Before live collection:

1. Implement and test state allocation, prompt rendering, hash verification,
   randomized scheduling, pilot K selection, primary analysis, permutation,
   bootstrap, and structured cross-fitting on mock data.
2. Generate a blank report from mocks to verify that every result category is
   handled without manual prose changes.
3. Freeze the Stage-0 protocol, state allocations, templates, full prompt corpus,
   code, seeds, tests, model parameters, and budget guards in Git.
4. Record the commit hash in a time-stamped external snapshot when available;
   otherwise describe the record as prospectively Git-frozen rather than a
   formal public preregistration.
5. Run the pilot and automatically generate its report.
6. Commit the mechanically selected K and confirmatory call manifest before any
   confirmatory call.
7. Run confirmatory collection once, execute the frozen analysis, and archive
   raw logs, Parquet data, reports, and exact software environment.

No threshold, state, prompt, model, target, hypothesis direction, exclusion rule,
or analysis seed may change after seeing confirmatory outputs.

---

## 13. Budget and stop rules

- Maximum pilot calls: 200 valid calls.
- Maximum confirmatory calls: 1,200 valid calls.
- Maximum scientific calls: 1,400 valid calls, plus at most one retry for a
  failed scheduled call and no more than 14 retries overall.
- Engineering smoke calls: at most 20 and excluded from analysis.
- Hard live-spend cap: `$5`; official snapshot availability and current pricing
  must be reverified before the smoke call.

Under the archived baseline pricing and observed normal-run traffic, the mean,
99th-percentile, and maximum per-call costs were approximately `$0.000349`,
`$0.000419`, and `$0.000480`. Even 1,434 attempts—the maximum 1,400 valid calls,
14 retries, and 20 smoke calls—would cost about `$0.69` if every attempt matched
the historical maximum. The `$5` guard therefore provides more than a sevenfold
buffer under the frozen pricing; it remains a stop guard, not a spending target.

The study stops before confirmatory collection if:

- the exact model configuration is unavailable;
- prompt or state integrity fails;
- response reuse cannot be ruled out operationally;
- the stochastic pilot requires more precision than `K = 10` provides;
- projected calls or cost exceed the frozen guard;
- variant validation fails and cannot be resolved before any v2 call.

A stop is reported as an engineering or identifiability stop, not a scientific
null result.

---

## 14. Required artifacts

### Before pilot

- `v2_stage0_protocol.md` or frozen copy of this plan;
- `v2_state_allocation.parquet` and readable CSV manifest;
- `v2_prompt_corpus.jsonl` with A/H prompt hashes;
- template-validation checklist;
- mock analysis report;
- test report;
- pre-pilot commit and external timestamp if available.

### After pilot, before confirmatory calls

- pilot raw log and Parquet;
- response-reuse diagnostic;
- pilot variance/K-selection report;
- frozen confirmatory task manifest and schedule;
- post-pilot/pre-confirmatory commit.

### After confirmatory collection

- raw JSONL with full provenance;
- materialized Parquet;
- integrity report;
- primary and secondary results report;
- structured cross-fitting report;
- reproducibility checklist and final outcome commit.

---

## 15. Paper-facing result template

The result paragraph must report both the directional and practical-magnitude
labels rather than choosing a narrative after seeing the result.

### Positive replication

> Across 60 stratified, scenario-unique decision states, authority framing
> increased the frozen LLM transition rule's mean reshare probability relative to
> hedge framing by `Δ` points (fixed-template 95% interval `[L,U]`; template-
> cluster sensitivity interval `[L_t,U_t]`). The direction replicated the v1
> exploratory contrast. `[If L > 0.03: The lower bound also exceeded the
> pre-specified operational threshold. If U <= 0.03: The interval remained
> inside the practical-equivalence region despite resolving the positive
> direction. Otherwise: The practical magnitude remained unresolved.]`

### Sign reversal

> The authority–hedge contrast differed from zero but reversed the v1 direction:
> authority changed reshare probability by `Δ` points relative to hedge (95%
> interval `[L,U]`). This supports sensitivity to the tested framing distinction
> while showing that its directional effect was not stable across the evaluated
> state samples.

### Practical equivalence

> The 95% interval for the authority–hedge contrast lay within the pre-specified
> ±0.03 equivalence region. Under the frozen model configuration and sampled v1
> baseline state distribution, the study ruled out an authority–hedge shift large
> enough to meet the operational transition threshold.

### Inconclusive result

> The estimated authority–hedge contrast was `Δ`, but its 95% interval included
> both zero and effects exceeding the operational threshold. The confirmatory
> study therefore did not resolve the direction or practical magnitude of the v1
> pattern.

---

## 16. Final verification checklist

- [ ] One primary configuration only; no model sweep.
- [ ] `reshare` is the sole primary target.
- [ ] Pilot and confirmatory scenarios are disjoint.
- [ ] One state per scenario.
- [ ] Confirmatory state margins match Section 4.3 exactly.
- [ ] State selection does not inspect oracle probabilities or v1 outcomes.
- [ ] A/H templates and all prompts are frozen before pilot calls.
- [ ] Only the quoted message span differs across A/H prompts.
- [ ] Exact snapshot and temperature are verified; no fallback is enabled.
- [ ] Pilot response IDs and usage events rule out client/response reuse.
- [ ] K is selected mechanically from the frozen rule.
- [ ] The deterministic and stochastic branches are handled as pre-specified.
- [ ] Signed reshare contrast is primary; absolute/Jensen metrics are secondary.
- [ ] Inference is two-sided despite the positive directional prediction.
- [ ] The primary interval conditions on the 15 fixed templates, and the
      template-cluster sensitivity interval is reported separately.
- [ ] Sign reversal, equivalence, and inconclusive outcomes have distinct language.
- [ ] No non-significant wide interval is called evidence of no effect.
- [ ] Permutation-null correction is applied to nonnegative secondary metrics in
      the stochastic branch.
- [ ] Structured predictor uses normal rows only and scenario-level cross-fitting.
- [ ] Anchor predictions are out of fold for every selected scenario.
- [ ] No downstream cascade claim is inferred from the 0.03 first-order threshold.
- [ ] All stopping, retry, cost, and invalid-output rules are enforced before
      results are read.

---

## 17. Pre-live feasibility verification completed on 2026-07-16

The following checks were performed against the current repository and archived
v1 data. They verify feasibility, not successful execution of v2.

### 17.1 Verified repository facts

- `phase1_full_v1.parquet` contains 4,350 normal decision rows from exactly 80
  scenarios and 16 scenarios for each of five topics.
- Scenario-level credibility counts are 30 low, 22 mid, and 28 high.
- The archived normal JSONL stores full prompts, raw responses, model parameters,
  token/cache metadata, and unique API response IDs.
- The baseline live metadata match the frozen configuration in Section 3.
- `FRAMED_TEMPLATE_BANK` already contains all 15 topic × credibility authority
  and hedge template pairs; no new generative authoring system is required.
- The repository contains no trained general transition surrogate capable of
  providing LLM probabilities on counterfactual cascade branches. This supports
  using the transparent first-order 0.03 calibration rather than presenting a
  model-dependent cascade rerun as ground truth.

### 17.2 Verified allocation feasibility

A binary constrained-selection check found an exact feasible set of 60 rows with:

- 60 distinct scenarios;
- 12 scenarios per topic;
- credibility counts `23/16/21`;
- exposure-bin counts `32/20/8`;
- step-bin counts `15/16/29`.

Adding the requirement of at least two states in every topic × credibility
template cell remained feasible. One verified solution had cell counts between
2 and 6.

The 20 scenarios left by that feasible allocation contain exactly four scenarios
per topic, so the planned 2-pilot/2-reserve topic split is feasible.

This feasibility check did not freeze the final state identities. The production
manifest must be generated by the frozen selection code and committed before the
pilot.

### 17.3 Statistical identity check

The anchor decomposition in Section 11.4 was numerically verified on synthetic
Bernoulli probabilities:

\[
\tfrac12 KL(p_A\Vert q)+\tfrac12 KL(p_H\Vert q)
=JS(p_A,p_H)+KL((p_A+p_H)/2\Vert q).
\]

The equality held to floating-point precision.

### 17.4 Non-binding K feasibility check

The existing baseline identical-prompt diagnostic is too small to replace the
new pilot, but it provides a conservative risk check. Using its 90th-percentile
reshare variance, doubling that value to stand in for two noisy A/H sides, and
applying the frozen K formula gives approximate projected call-noise standard
errors:

- `K = 3`: 0.01310;
- `K = 5`: 0.01014;
- `K = 10`: 0.00717.

Only `K = 10` clears the 0.0075 planning target under this rough calculation.
This suggests that the maximum planned K is plausible, while confirming that the
new 10-repeat A/H pilot is necessary. The pilot—not this diagnostic—selects K.

### 17.5 Non-binding total-precision projection and J decision

For the v1 authority–hedge twins, 193 matched reshare rows had a signed
authority-minus-hedge mean of 0.11570 and a row-level standard deviation of
0.13076. These rows came from only three twin groups and contain single-call
noise, so they are not an independent-sample variance estimate. They are used
only as a conservative planning diagnostic.

Treating 0.13076 as a rough state-effect standard deviation gives, for `J = 60`,

\[
SE_{state}\approx 0.13076/\sqrt{60}=0.01688.
\]

Combining this with the non-binding call-noise projections gives approximate 95%
half-widths:

| States | K | Projected total SE | Projected 95% half-width |
| ---: | ---: | ---: | ---: |
| 60 | 3 | 0.02136 | 0.04188 |
| 60 | 5 | 0.01969 | 0.03860 |
| 60 | 10 | 0.01834 | 0.03595 |
| 70 | 10 | 0.01698 | 0.03328 |

This projection implies that v2 is well positioned to confirm a contrast near
the large v1 point estimate, but it may not establish practical equivalence at
±0.03 if the true effect is near zero. Expanding from 60 to 70 states would reduce
the projected K=10 half-width by only about 0.00267 and would eliminate the
scenario reserve without solving that equivalence limitation.

**Pre-data decision:** retain 60 confirmatory states and 10 reserve scenarios.
Accept that a practically unresolved result is possible. The study is primarily
powered as a directional replication of the large v1 contrast, not as a tight
equivalence study. This decision cannot be revisited after pilot or confirmatory
probabilities are observed.

### 17.6 Verified cost envelope

Archived baseline normal calls used a mean of 199.83 input tokens and 44.18
completion tokens; observed maxima were 411 and 56. At the frozen baseline prices,
1,400 calls would cost about `$0.49` at the observed mean and `$0.67` if every call
matched the historical maximum. Including the full retry and smoke allowance
raises the latter estimate only to about `$0.69`. The `$5` cap is therefore ample
under historical traffic, subject to the required live pricing check.

### 17.7 Remaining pre-live requirements

The design is internally coherent, but v2 is not yet executable or frozen. The
following remain mandatory:

- implement state allocation and generate the final manifests;
- implement prompt rendering/hash checks and complete template validation;
- implement the pilot, K rule, randomized scheduler, bootstrap, permutation, and
  cross-fitting analyses;
- add mock and integrity tests;
- verify exact live-model availability and current pricing;
- create the Stage-0 time-stamped freeze before any pilot call.
