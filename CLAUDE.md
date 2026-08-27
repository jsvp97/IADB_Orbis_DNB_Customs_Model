# MNE Ownership, Market Power, and Trade Policy — Project Brief

**Owner:** Sebastián Velásquez Palacios (IDB, PTI — with Christian Volpe Martincus)
**Status (2026-08-21):** all five layers built. Uniqueness proved for four of them and
established for wages under a checked condition (§31). Stylized facts: five of six,
Fact 5 fails. Baseline changed 2026-08-20: head-office services are a capability, not a
factor input. ~~GE uniqueness verified numerically~~ — superseded, see §31.


# 0. READ THIS FIRST — CURRENT STATE (last updated 2026-08-26)

**If you are a fresh session: read this section, then §29–§31, then stop. Sections 1–28
are the historical record of how the project got here. They are kept deliberately —
struck results are more useful than deleted ones — but WHERE THEY CONFLICT WITH §29–§31,
THE LATER SECTIONS WIN.** The 2026-08-20/21 baseline change touched the cost equation,
so anything in §1–§25 that describes a factory paying its parent's wage, or a
multinational productivity edge being fitted, is out of date.

## 0.1 What the model is, in six lines

A static general-equilibrium model of multinational production with **large firms**
(markups rise with a group's market share), **firm-level ownership** `theta[g,n]`, and
**endogenous entry**. Delivered cost of plant `a` (parent in `h`, producing in `l`,
selling to `n`, sector `k`):

```
a[a,n] = (1/phi_a) * w[l]^(1-nu_k) * PIO[l,k]^nu_k * gamma[h,l] * d[l,n]
F[a,n] = f_k * w[l] * fdist[l,n]                       (fixed cost per market served)
```

**Head-office services are a non-rival CAPABILITY, not a factor input.** `alpha_k` is a
complexity index living inside `phi_a` (local-firm penalty `exp(-hq_gap*alpha_k)`,
parent capability gradient `xi(1+adv_slope*alpha_k)`). The old version where a plant
pays its parent's wage on the `alpha_k` share is `world_economy(...; hq_cost = true)` —
supported, reported alongside, NOT the baseline. Why: §29.

## 0.2 Uniqueness — what is derived and what is checked

| layer | status |
|---|---|
| one market | **derived, unconditional.** A decreasing curve crosses a level once. |
| input prices | **derived, unconditional.** Contraction, modulus `max nu < 1`. |
| spending and incomes | **derived, unconditional.** Linear, `rho(B) < 1`. |
| entry | **derived, conditional** on no group above half a market. Closed-form threshold; checked at the solution (0.297 Cournot, 0.365 Bertrand). |
| **wages** | **NOT derived unconditionally.** Theorem 6 (§31): the solver's own map is a Birkhoff contraction wherever its Jacobian is entrywise positive. That positivity is now **computer-certified on the CONTINUUM of wage boxes around the ν = 0 solution** (§32, outward-rounded interval arithmetic — no between-points gap), and beyond those boxes it is checked on grids out to spread 2.0, never proved. It is FALSE in the `hq_cost = true` variant. |

**Do not write "the model has a unique equilibrium" without the qualifier.** The honest
sentence is in §31.5, upgraded by §32: uniqueness near the solution is now certified on
the continuum (interval arithmetic, ν = 0); the region beyond is still grid-checked.

## 0.3 The stylized facts — five of six, and magnitudes overshoot

| # | model | data | label |
|---|---|---|---|
| 1 | 0.689 | 0.46–0.74 | level FITTED; foreign/domestic split generated |
| 2 | +0.42 | +0.43 | FITTED (`hq_gap` targets it); domestic half wrong-signed, not robust |
| 3 | 0.273 / 0.344 | 0.130 / 0.250 | generated, overshoots ~2x |
| 4 | x1.94 | x1.12 | generated prediction, overshoots ~1.7x |
| 5 | −97% core; **+45.5% in the λ config** (§37) | positive (+0.087) | core FAILS (kept visible); **λ configuration POSITIVE on both margins** |
| 6 | −0.515 / +0.701 | −0.16 / +0.05 | right signs and ordering, magnitudes 3–14x too large |

Two are targets, three are out-of-sample predictions with the sign right and the
magnitude too big, one fails. **Fact 5 belongs in the abstract, not a footnote.**

## 0.4 Files, and how to run them

```
mne_model.jl            THE model + driver. 2,900 lines, one file, no packages.
                        julia mne_model.jl            -> run_model_full.txt  (~9 min)
                        modes: core | entry | ge | facts | quick | bertrand
wage_uniqueness.jl      Theorem 4 + the counterexample -> run_wage_uniqueness.txt
simple_model.jl         Theorem 5 + the simplification -> run_simple_model.txt
experiments.jl          the one-off grids behind §29–§30 (spillover, nu, adv_slope)
interval_certificate.jl the CONTINUUM certificate (§32): outward-rounded interval
                        arithmetic + centred forms -> run_interval_certificate.txt
docs/                   model_guide.pdf (the paper-shaped document), model_summary.pdf
                        (5 pp), derivations.md, entry_options.md, papers/ (the three
                        reference PDFs)
superseded/             everything historical (2026-08-27 cleanup): the entry
                        prototypes, the pre-consolidation layer files (src/, the
                        Python mirrors, test/), uniqueness_probe.jl, the stale run
                        files (run_entry_*, run_calib_*, run_uniqueness_probe), and
                        the old top-level model PDF. Nothing was deleted.
```

**Bookkeeping note (2026-08-26).** The session logged below as "2026-08-21
(handoff pass)" actually ran on 2026-08-25 by every file timestamp — the content
is right, the date label is not. (`mne_model.jl`'s scorecard text was corrected to
THREE internally calibrated parameters and every run file was regenerated
2026-08-27; runs and code now agree everywhere.)

**Run them ONE AT A TIME.** In parallel they thrash: the driver alone takes 9.3 min but
took 24.9 min when two others were competing. Committed outputs are the `run_*.txt`
files; regenerate them whenever the code changes so they never disagree.

Documents: `docs/model_guide.pdf` (40 pp, the full model in plain language — the reader
entry point) and `docs/model_summary.pdf` (5 pp companion). Both rebuild with
`pdflatex` run three times. **Check `grep -c "^!" model_guide.log` is 0 afterwards.**

## 0.5 What to do next, ranked

1. ~~**Fact 5 decisive test: add destination × product × year FE.**~~ **DONE
   2026-08-26, and the fact SURVIVES at half size** (§33): intensive 0.166 → 0.087
   (s.e. 0.017), extensive 0.117 → 0.061, PPML on the level +0.229, all 1%. Half the
   published association was common demand; the surviving half is real. **Fact 5 is
   now a confirmed model failure needing a MECHANISM** — the spillover knob is already
   ruled out (§30.5). This is the model's foremost open problem.
2. **The `nu` decision is open and is Sebastián's** (§30.6). `nu = 0.55` is the
   default; `nu = 0` is the version where Theorem 5 certifies rather than merely
   verifies. If `nu = 0` is ever adopted, **drop the solver damping to 0.15** — the
   admissible bound is 0.273 there and the default 0.25 has only 9% of room (§31.2).
3. **Intra-firm / related-party share of MNE exports** — not computable from the
   customs file; needs the Orbis affiliate roster (§27.7).
4. ~~**Interval arithmetic** to close the wage-uniqueness gap for real (§26.7).~~
   **BUILT 2026-08-26, FINAL 2026-08-27** (§32): `interval_certificate.jl` certifies
   gross substitutes and the Birkhoff contraction on the CONTINUUM of wage boxes
   around the ν = 0 solution — no between-points gap. **Certified through spread
   0.30: the continuum within ±15% (log) of the solved relative wages contains
   exactly the one equilibrium the solver found** (κ = 0.15, entry frozen, 2,275
   subboxes total). `run_interval_certificate.txt`.
5. ~~**Layer 4 (optimal tariff) predates endogenous entry.**~~ **REDONE WITH ENTRY
   2026-08-27 (§36):** the ownership monotonicity survives; at ν = 0 the optimal
   tariff rises 0 → ≥0.30 as ownership goes 0 → 1; at ν = 0.55 ownership NEUTRALISES
   (−1.03% → −0.02% at t = 0.10) without flipping the sign. The exit channel is live
   (86 → 73 affiliates at ν = 0, λ = 0). Still open on the policy side: the
   ownership-adjusted sufficient-statistic dW/dt formula in the entry model.
6. ~~**Report the empirical Cov(θ, S).**~~ **MEASURED 2026-08-27 (§34):** USA
   understatement 10.1%, sign-varying by owner (COL +24% … PER −30%). The referee's
   "single most damaging gap" is closed.
7. ~~**The Fact 5 mechanism is a well-posed modeling task (§35).**~~ **BUILT, TESTED,
   AND THE SIGN FLIPPED — ADOPTED as the six-fact λ configuration (§37):**
   row_L = 192, spill = 0.30, fspill = 0.50, λ = 0.071 → **+0.071 vs data +0.087**,
   extensive margin positive. `julia mne_model.jl lambda` → `run_model_lambda.txt`.
   Core keeps row_L = 0 (failure stays visible; 6 facts / 6 fitted vs 5 / 3 stated).
   Remaining discipline: identify λ from measured absorption shares.
8. **Math-referee pass (2026-08-27), five statement fixes applied to the guide:**
   thm:contract's proof gap repaired (integral MVT + convexity of the Dobrushin
   coefficient + δ ≤ tanh(Δ/4); D log-convex added as hypothesis; convergence now
   conditional on invariance, uniqueness not); thm:entry's one-half hypothesis
   restated on the comparison region + genericity of ties; prop:irrelevance cut back
   to income only; thm:simple's strictness fixed (strict GS needs Ē > 0, margin
   +0.024); two attribution nits. Everything else — every closed form, every
   first- and second-order comparative-static formula in the certificate —
   independently re-derived and CONFIRMED.

## 0.6 Uncommitted work outside this folder

`C:\Sebas BID\Orbis_DNB_Customs_Final\` is a git repo with **uncommitted changes**:
`README.md`, `src/09_`–`12_` (the destination and conduit pipelines) and five CSVs plus
two logs in `output/tables/`. Nothing was committed. The empirical results those
produce are §27 and §28.

## 0.7 Traps that have each cost a run

- **Julia closure scoping.** A nested helper that reuses an enclosing local silently
  destroys it (§17.6). Never reuse enclosing names in a nested function.
- **Always verify a solver satisfies its own defining equation** before believing
  anything derived from it. A solver that converges is not a solver that is correct.
- **Writing `.tex` or `.jl` from Python: build every backslash as `chr(92)`.** Heredocs
  eat `\t \n \f \v`, two of them invisibly (§25.1). And check that tabular rows still
  end with a DOUBLED terminator — a `\\` silently collapsing to `\` left
  `model_guide.tex` uncompilable for a week and a control-character scan cannot see it.
- **The driver runs outside the module**, so `NL` is not in scope there; use a literal
  newline in `@printf` inside `run_*` functions.
- **Check run files for `ERROR`**, not just exit codes — Julia scripts can print a
  stack trace and still exit 0 under a pipe.
- **Run every mode, not just the one you are working on.** `julia mne_model.jl quick`
  died with a `SingularException` on Fact 6 while `normal` was fine. Fixed 2026-08-21 by
  solving the tall system with QR (`X \ y`, `pinv` fallback) instead of through the
  normal equations `(X'X) \ (X'y)`, which square up the problem and lose conditioning
  for nothing. **The exact source of the singularity was NOT pinned down** — a
  reconstruction of that design matrix came out full rank and well conditioned
  (cond(X'X) ~ 200-300 at both N=4 and N=5), so it likely depends on the realised sample
  in that run. The QR solve is the right way to do it regardless, and it no longer
  crashes; if it ever misbehaves again, instrument the real matrix inside `run_facts`
  rather than rebuilding it outside.

---

New reader? Start with **`docs/model_guide.pdf`** — the full model in plain
language, auto-descriptive, with every modelling choice justified where it is
introduced. `docs/derivations.md` has the terse algebra.

> ⚠️ **READ §12 BEFORE WRITING ANY POLICY TEXT.** The GE model produces a result
> that runs **against** the project's central hypothesis: home ownership of the
> taxed multinationals makes the optimal tariff **higher**, not lower. The income
> decomposition (P2a) is untouched; the policy claim (P2b) is not established and
> may have the wrong sign.
**Last updated:** 2026-08-10

> **Two findings from the Layer-0 verification pass change the plan. Read §4.7 and
> §4.8 before writing any theory.** (a) The ownership-weighted-Herfindahl result is
> an identity about the *level* of profit income; it is not yet the optimal-tariff
> proposition P2 claims. (b) At Layer 0 optimal policy is an import *subsidy* for
> ordinary market structures, so Layer 0 cannot host the tariff result at all.

This file is the single source of truth for where the project stands. Read it fully
before writing code. If you change the model, update this file in the same commit.

---

## 1. The research question

Developed countries set tariffs on imports from Latin America. A large share of those
imports are produced by affiliates of multinational groups **headquartered in the
tariff-setting country**. The claim under investigation: this materially lowers the
optimal tariff, and standard models miss it because they have no firm-level ownership.

### 1.1 What is NOT the contribution

Be honest about this or the paper gets desk-rejected. "Foreign ownership lowers the
optimal tariff" is old and already quantified:

- Brecher & Bhagwati (1981, JPE) — "Foreign Ownership and the Theory of Trade and Welfare". The ancestor.
- Blanchard (2007, 2010) — foreign equity holdings lower the optimal tariff.
- Blanchard & Matschke (2015, REStat) — offshore investment makes a large importer internalise the terms-of-trade externality and cut tariffs on host countries.
- Blanchard, Bown & Johnson (2024, ReStud) — GVCs and trade policy.
- **Itskhoki & Mukhin (2025), "The Optimal Macro Tariff", NBER WP 33839** — the US's
  gross external dollar liabilities (including equity/FDI claims) cut the optimal
  tariff roughly three-fold, from 34% to 9%, via valuation effects. This is the
  closest rival and it already has the number. (Figure verified 2026-08-26.)
- Lu, Li & Li (2025) — optimal unilateral policy with multinational production and trade.

### 1.2 What IS the contribution

Every paper above uses a **country-level ownership share** with CES or representative
firms. The ownership correction in those models is a scalar.

We have global-ultimate-parent identity matched to transaction-level customs data for
nine LAC origins. That lets us compute the ownership share **market by market** and,
crucially, measure how ownership **covaries with market power**.

**Headline claim to defend:** the correct correction is not the average ownership share.
It is the ownership-weighted Herfindahl. Using the country-level share understates the
correction by exactly the ownership–concentration covariance. That covariance is large
and positive in our data (home-country parents are disproportionately the biggest players
in the most concentrated, most complex markets), and nobody with country-level data can
compute it.

This is now an **analytic result**, not a conjecture. See §4.3.

---

## 2. Empirical foundation

Source document: `docs/Multinational_Firms_and_Trade.pdf` (stylized facts, v. 2026-07-29).
Data: matched ORBIS / D&B firm records linked to customs transactions, 9 LAC origins,
2006–2022. Pipeline: `github.com/jsvp97/IADB_Orbis_DNB_Customs_Final`.

### 2.1 The six facts

| # | Fact | Model job | Status |
|---|------|-----------|--------|
| 1 | MNEs are a large share of export value everywhere (0.47–0.74), overwhelmingly foreign-owned | pins the ownership index | usable |
| 2 | Foreign MNEs concentrate in complex goods, domestic MNEs in primary | pins HQ-input intensity α_k | **restate, see below** |
| 3 | Parent countries are concentrated (GBR 24.7%, USA 22.5%, CAN 10.2%) | pins ownership weights θ | **fix conduits first** |
| 4 | Few large groups dominate; grouping affiliates by parent raises HHI (0.192→0.215; top share 36%→40%) | **justifies oligopoly. Core fact.** | usable, strongest |
| 5 | More MNE presence correlates with higher trade volumes | **the model currently fails it** | see below |
| 6 | Distance is a weaker barrier for MNEs, weakest for those already present at destination | pins export platforms / correlated draws | **needs firm FE** |

### 2.2 Known problems with the facts (do not paper over these)

- **The missing fact that the thesis needs.** The document never shows *where* MNE exports
  go. The whole "exports boomerang back to the parent country" story has zero evidence
  behind it. **Highest-priority empirical task:** tabulate (a) share of foreign-MNE export
  value destined for the parent's own country, (b) share that is intra-firm / related-party.
  If US-owned LAC affiliates ship mostly to Brazil, Europe and Asia, the boomerang framing
  dies and the paper becomes a pure profit-repatriation story. This is cheap and decisive.
- **Fact 3 contradicts the thesis as written.** GBR (24.7%) beats USA (22.5%), and
  PAN + BMU + IRL + NLD + CHE together are ~11%. Those are conduit jurisdictions, not
  economic owners. On top of that, ~half of foreign-MNE export value has no recorded
  parent at all (stated in the document's own note to Figure 4). A paper about US tariffs
  cannot rest on a figure where the top owner is Britain and half the value is missing.
  Reallocate conduits to ultimate controlling parent and show the robustness.
- **Fact 2 is mis-stated and the correct version is better.** Table A.4 Panel A (total MNE
  share on PCI): 0.0029 / 0.0055 / 0.0101 — essentially nothing. Panel B (foreign): +0.045.
  Panel C (domestic): −0.043. Complexity does not sort *multinational status*. It sorts
  *parent nationality*. Say that. It is an ownership fact, which is what the model needs.
- **Fact 6 has a specification smell.** A baseline log-distance coefficient of −0.16 at the
  firm level is an order of magnitude below any gravity estimate. Without firm fixed effects
  the interaction is contaminated by selection (MNEs are bigger; bigger firms have flatter
  distance gradients regardless of ownership). Add firm FE.
- **Fact 5 is not "descriptive only" — it is the one fact the model currently gets
  wrong, and that makes it the most informative fact in the document.** Table 1 Panel B
  shows *non-MNE* exports rising strongly with MNE presence (0.24 to 1.35 on the
  intensive margin, surviving origin×dest×product FE). A fixed-`E` oligopoly predicts
  the opposite sign: more MNE presence steals business from the fringe. Something
  outside the current model carries it — endogenous `E`, input–output linkages, or
  selection on market attractiveness. Note also that the document's identification
  claim is overstated: origin×dest×product FE do **not** absorb time-varying market
  shocks, and no column carries destination×product×year. If a positive-demand-shock
  story survives that, Fact 5 says the fixed-`E` closure is wrong and Layer 3 is
  load-bearing rather than cosmetic. **Decide whether Fact 5 is a target or a
  known-failed moment, and say so in the paper.**
- **Fact 4's real content is the level, not the change.** HHI ≈ 0.215 is an effective
  number of ~4.7 exporters per HS6 — that is what justifies oligopoly. The grouping
  *increment* (0.192→0.215, +12%) justifies something different and more specific:
  that the strategic agent is the **parent**, not the affiliate. Two claims, two pieces
  of evidence. The brief currently runs them together.
- **Coverage limit.** Nine LAC origins, half of foreign-MNE value without a parent. We
  **cannot** compute the US global optimal tariff. Frame the exercise as the ownership
  correction to US tariffs *on these nine countries*, explicitly a lower bound.

---

## 3. Model architecture and layer plan

Design rule: **each layer has exactly one empirical job, and a layer is not finished until
its tests pass.** Do not start a layer before the previous one is green.

| Layer | Content | Empirical job | Status |
|-------|---------|---------------|--------|
| **0-theory** | CES monopolistic competition GE benchmark; Ownership Irrelevance proposition | establishes the null | done (LaTeX doc, separate) |
| **0-code** | Single-market group Cournot solver, general η | the object with no closed form | **DONE, verified** |
| **1** | Many markets; ownership accounting; ownership–concentration covariance | Facts 1, 3, 4 | **DONE, verified** |
| **2** | HQ input with intensity α_k rising in complexity (Head–Mayer) | Fact 2 | **folded into Layer 3 cost function** |
| **3** | GE: labour market clearing, income=expenditure, endogenous E and wages | makes welfare meaningful | **DONE, verified** |
| **4** | Optimal tariff, ownership-adjusted; sufficient-statistic formula | the propositions | not started |
| **5** | Export platforms / correlated Fréchet draws across own locations | Fact 6 | optional |
| **6** | Endogenous location portfolios (Arkolakis–Eckert combinatorial choice) | extensive margin | **cut for v1** |

### Why not Ramondo–Rodríguez-Clare (2013) as the benchmark

RRC 2013 is a continuum of goods under perfect competition. No firms, no market shares,
no markups, no groups, zero profits. Ownership Irrelevance holds there *by construction*
and nothing can break it. Facts 4 and 6 are literally undefined (you cannot compute an
HHI over a continuum). And Cournot groups cannot be added incrementally to a Fréchet
continuum — it is a rewrite, not a layer. RRC is a benchmark to cite, not to code.

The one piece worth stealing from RRC is the **correlation parameter in the multivariate
Fréchet**, which is Layer 5. Note the trap: Tintelnot (2017) shows the RRC calibration only
matches US export-platform sales when within-firm draws are *uncorrelated*. That parameter
is contested territory.

Tintelnot (2017, QJE) is the closer benchmark — monopolistic competition, discrete
portfolios, export platforms — and it nests the Layer 0 theory. But it has constant
markups, so it also cannot host the punchline.

### Reconciling "start from Ramondo and Tintelnot" with the layer plan

These are not in conflict, and the apparent conflict has confused this file. The
Ramondo–Tintelnot line supplies the **multinational production skeleton**: who owns a
plant, where it produces, which markets it serves, and the multivariate-Fréchet
machinery for location portfolios (Ramondo & Rodríguez-Clare 2013; Ramondo,
Rodríguez-Clare & Tintelnot 2015 for the facts; Tintelnot 2017 for platforms and
discrete portfolios). What that line does **not** supply is market conduct: it is
perfectly or monopolistically competitive, so markups are constant or absent and
ownership is irrelevant by construction.

The correct architecture is therefore:

| Block | Source | What it delivers |
|---|---|---|
| location / sourcing | Tintelnot (2017), RRC (2013) | which affiliates exist in which market |
| **market conduct** | Atkeson–Burstein group Cournot | share-dependent markups, rents |
| **ownership** | this project | θ over global ultimate parents |

Take portfolios as **exogenous** in v1 (§ Cut from scope), so the Ramondo–Tintelnot
block enters as the *given list of affiliates per market* and a calibrated re-sourcing
elasticity. Layer 5 is where the Fréchet correlation parameter would actually be
estimated. So "start from Ramondo and Tintelnot" and "Layer 0 is Cournot" are the same
plan: RT gives the market lists, Cournot gives the conduct on those lists.

What must **not** happen is coding RRC's continuum first and trying to add groups to it.
Facts 4 and 6 are undefined on a continuum (there is no HHI over a continuum), and
Cournot groups are a rewrite of that model rather than a layer on it.

### Cut from scope for v1

Endogenous location portfolios under oligopoly (Yang 2023; Head & Mayer's
super/submodularity warning). With tariffs in the payoff, the modularity sign can flip and
there may be no existence proof. Replace with **exogenous portfolios plus a calibrated
re-sourcing elasticity** taken from Tintelnot or Flaaen–Hortaçsu–Tintelnot. Yang's
algorithm becomes a robustness appendix or the follow-up paper.

---

## 4. Layer 0 (PE): full specification

A **market** is one (destination `d`, product class `k`) cell.

### 4.1 Environment

- Upper tier CES(**η**) across classes, `E = D·P^(1-η)`, requiring `σ > η ≥ 1`.
  **η = 1 is Cobb–Douglas**: `E` is fixed, the layer is genuinely partial equilibrium, and
  the closed forms are cleanest. Report the η = 1 algebra, but check every headline number
  at η > 1 — η = 1 is a knife-edge exactly where the punchline lives (a monopolist's markup
  is unbounded and its pass-through goes to zero). The solver takes arbitrary η.
- Lower tier CES with elasticity `σ > 1` over varieties `i = 1..N`.
- Each variety belongs to a **group** `g` (a global ultimate parent). Groups compete in
  **quantities (Cournot)** and internalise cannibalisation across their own affiliates.
  This is the modelling choice that Fact 4 / Figure 6 justifies, and it is the single most
  important decision in the whole project.
- `c_i` is the **delivered** marginal cost of variety `i`: `c_i = mc_i · τ_i · (1 + t_i)`.

### 4.2 Derivation

Inverse demand (derived, not assumed):

```
A   = Σ_j q_j^((σ-1)/σ)
p_i = E · q_i^(-1/σ) / A
s_i = q_i^((σ-1)/σ) / A        (revenue share, Σ_i s_i = 1)
```

Group `g` maximises `Π_g = E·S_g − Σ_{i∈g} c_i q_i` where `S_g = Σ_{i∈g} s_i`. Using
`∂S_g/∂q_i = ((σ-1)/σ)(s_i/q_i)(1 − S_g)` and `E·s_i/q_i = p_i`, the FOC is

```
p_i = c_i · μ_g    with    1/μ_g = 1 - (1-S_g)/σ - S_g/η
                           μ_g   = σ/[(σ-1)(1-S_g)]   when η = 1
```

**The markup depends on the group's TOTAL share, not the variety's share, and is common
across all of a group's affiliates.** This is the model-level reason grouping affiliates by
parent raises both measured concentration and true markups. It is the theoretical
counterpart of Figure 6.

Profit, since `c_i q_i = r_i/μ_g`:

```
Π_g = E·S_g·(1 − 1/μ_g) = (E/σ) · S_g · [ 1 + (σ/η - 1)·S_g ]
```

Limits: `S_g → 0` gives `Π_g → E·S_g/σ` (the CES benchmark, revenue over σ);
`S_g → 1` gives `Π_g → E` (a monopolist captures all class expenditure under Cobb–Douglas).

### 4.3 The ownership result (this is Proposition 2)

With ownership weights `θ_g` for country H over groups:

```
Π_H = (E/σ) [ Σ_g θ_g S_g   +   (σ-1) Σ_g θ_g S_g² ]
              └── CES term ──┘   └─ granular correction ─┘
```

The first term is exactly what a CES / monopolistic-competition model predicts: ownership
share × revenue / σ. The second is an **ownership-weighted Herfindahl**. It vanishes in the
atomistic limit (recovering Ownership Irrelevance) and dominates when markets are granular.

Verified numerically (σ = 5, H owns 20% of groups, total profit income as a multiple of the
CES prediction):

| structure | H owns biggest 20% | H owns smallest 20% |
|---|---|---|
| 200 groups | 1.05× | 1.004× |
| 20 groups | 1.45× | 1.05× |
| 5 groups | **2.44×** | 1.26× |

Ownership Irrelevance is recovered in the limit and fails hard when granular — and it fails
**asymmetrically in which firms you own**. That asymmetry is the covariance term.

Decomposed against what country-level data can produce:

```
Σ_g θ_g S_g²  =  θ̄ · HHI  +  Cov_S(θ, S)          θ̄ = Σ_g θ_g S_g
```

`θ̄·HHI` is the most a researcher with a country ownership share and a published HHI
could build. `Cov_S(θ,S)` needs firm-level GUP identity. **That term is the contribution.**
Verified (`test_D`): owning the largest 20% of groups gives `total/naive` = 1.41–1.47×;
owning the smallest 20% gives 0.12–0.26×.

⚠️ **This is an identity about the LEVEL of profit income. It is not a policy result,
and as written P2 conflates the two.** Optimal tariffs depend on `dΠ_H/dt`, not `Π_H`.
See §4.7.

### 4.6 Comparative statics — the objects Layer 4 will actually need

Derived and verified this pass; see `docs/derivations.md` §0.4. With
`Λ_g = (1−S_g)/(1+(σ−2)S_g)`:

```
d ln S_g = Λ_g (d ln K_g − d ln A)
d ln A   = Σ_g ω_g d ln K_g          ω_g = S_g Λ_g / Σ_h S_h Λ_h     (Σ_g ω_g = 1)
d ln P / d ln(1+t_g) = ω_g
```

`Λ` is decreasing in `S`, so a dominant group has **`ω_g < S_g`**: it absorbs the tariff
in its markup and the price index rises by less than CES predicts. Atomistically
`Λ→1`, `ω_g→S_g`, pass-through → 1, and CES is recovered.

Rivals are **not** passive: `d ln S_h = Λ_h ω_g (σ−1) d ln(1+t_g) > 0`, so their markups
and prices rise too. A tariff on one group raises its rivals' prices. That is a real
welfare cost of tariffs that CES cannot represent, and it is a candidate result in its
own right.

### 4.7 Welfare, and the sign of optimal Layer-0 policy — **this changes the plan**

```
(1/E)·dW/dt|₀ = S_g/μ_g − ω_g + Σ_h θ_h (dΠ_h/dt)/E
                └ revenue ┘  └ prices ┘  └ ownership ┘
```

Verified against the full solver to 2.6e-10 with random θ.

Two structural facts the brief did not record:

1. **The tariff base is markup-deflated.** Ad valorem duties are levied on customs
   value `r_i/μ_g`, not market value `r_i`. Taxing a high-markup supplier collects
   little per unit of distortion. Invisible in any model without markups.
2. **Layer 0 has constant marginal cost, so it has NO terms-of-trade motive.** The
   only reason to tax is rent extraction; the only reason to subsidise is the markup
   distortion.

**Result.** Over 14,024 random (market, group) pairs with θ = 0, a tariff beats free
trade in **0.26%** of cases. Frontier (smallest dominant share at which a tariff pays,
against a fringe of `n` equal rivals) — identical in Julia and Python:

```
 sigma      n=1      n=2      n=4      n=9     n=19     n=49
   3.0        .        .        .        .        .        .
   5.0        .        .        .    0.361    0.201    0.135
   8.0        .        .    0.393    0.178    0.096    0.052
  20.0        .    0.593    0.283    0.127    0.062    0.026
```

Consequences, both blocking:

- **The competitive fringe is not a calibration nuisance — it is what creates the
  tariff motive.** Open decision 1 is therefore substantive, not cosmetic. A markup
  cap would suppress the mechanism. **Use a fringe.**
- **Layer 0 cannot host the optimal-tariff result.** For ordinary market structures
  optimal Layer-0 policy is an import *subsidy*, and ownership makes the subsidy
  larger (θ₁ = 0 → t* = −0.26; θ₁ = 1 → t* = −0.87). Ownership has the sign the
  project predicts, but it is correcting a subsidy. A positive optimal tariff needs
  upward-sloping foreign supply (GE, Layer 3) or an extensive margin. **Do not write
  the tariff propositions against Layer 0.**

### 4.8 Why the profit elasticity in Test 4 is the wrong statistic

§5 reads `dlnΠ/dln(1+t)` = −2.08 vs CES −2.95 as cutting against the thesis. Two
corrections. The comparison is not like-for-like (the CES benchmark is evaluated at
the *Cournot* share). And the elasticity is not what enters welfare — `dΠ_g/dt` in
**levels** does, and `Π_g` is far larger under Cournot.

The mechanism runs through **pass-through**, not the profit elasticity. The share of a
tariff borne by the foreign firm is `1 − ρ_g`, increasing in `S_g`. Under CES
pass-through is exactly 1 and **no rent is extracted at all** — rent extraction is a
pure oligopoly phenomenon, and it is exactly what home ownership neutralises. State
the mechanism that way.

### 4.4 Algorithm — existence and uniqueness

**Do not use a damped fixed-point iteration on shares.** It is the standard way people code
Atkeson–Burstein and it does not converge for all parameters. It hung during development.

Use nested monotone bisection. Let `A = Σ_i p_i^(1-σ)` and `K_g = Σ_{i∈g} c_i^(1-σ)`,
`B = (σ/(σ-1))^(1-σ)`.

- **Inner:** `S_g` solves `x = B·(1-x)^(σ-1)·K_g/A`. LHS rises 0→1, RHS falls. Unique root.
- **Outer:** `Σ_g S_g(A) = 1`. Each `S_g` is strictly decreasing in `A`. Unique `A`.

The construction **proves existence and uniqueness** of the Cournot equilibrium in the
market. Write it up as a lemma in the paper — it is free and referees like it.

### 4.5 Traps

- With η = 1 a **monopolist's markup is unbounded**. The solver asserts ≥ 2 groups. In the
  real data many HS6 × destination cells have one dominant group, so before this touches
  data you need a **competitive fringe** or a markup cap. Decide which and document it.
- Costs must be strictly positive or `c^(1-σ)` blows up.
- Model concentration overshoots the data badly (see §5), which is the same fringe problem
  showing up from the other side.

---

## 5. Code status

`src/cournot_pe.jl` — Layer 0 solver. Base Julia only, no packages. Run:

```bash
julia src/cournot_pe.jl
```

### Tests and what they prove

| Test | What it checks | Result |
|---|---|---|
| 1 | atomistic limit → CES markup σ/(σ-1) | N=10⁴ gives μ = 1.2501250 vs 1.25 |
| 2 | closed-form profit vs solver, 3000 random markets | max error 4.4e-16 |
| 3 | ownership irrelevance holds atomistically, fails when granular | table in §4.3 |
| 4 | tariff incidence on one group's profit | see below, and §4.8 |
| 5 | Fact 4 / Figure 6, grouping affiliates | see below |
| 6 | analytic incidence weights ω, Λ vs finite differences | max error 1.6e-9 |
| 7 | sign of optimal Layer-0 policy, frontier table | §4.7 |

`scripts/verify/cournot_pe.py` — independent Python mirror. Run:

```bash
python scripts/verify/cournot_pe.py
```

It contains a solver that uses **none** of the derived algebra: it finds the Nash point
by complex-step differentiation of the primitive profit function
`Σ_{i∈g}(p_i(q) − c_i)q_i`. Agreement with the Julia solver therefore validates the FOC,
the markup rule and the profit closed form simultaneously.

| Test | What it checks | Result |
|---|---|---|
| A | primitive profit gradient at the derived solution, 300 markets | 9.0e-16 |
| A | independent global solve from a CES start, 40 markets | agrees to 2.2e-16 |
| A | within-group markup spread (the common-markup prediction) | 2.2e-14 |
| B | **best-response check (discipline rule 3 — had never been run)** | worst gain −1.1e-12 |
| C | ω and pass-through vs finite differences | 6.3e-10 |
| D | ownership decomposition θ̄·HHI + Cov | ratios 1.41–1.47× / 0.12–0.26× |
| E | analytic dW/dt vs solver; sign of optimal policy | 2.6e-10; 0.26% positive |
| F | corrected Figure 6 analogue | 3.54× vs data 1.12× |

Test B closes the one discipline rule the project had written down and never executed.
The equilibrium is a genuine maximum, not just a stationary point.

### Test 4 — superseded. See §4.8.

Tariff on one group's delivered cost, σ = 5, 6 groups:

- pass-through to price: **0.55**, not 1
- profit elasticity `dlnΠ/dln(1+t)`: **−2.08**
- CES benchmark `−(σ-1)(1-S)`: **−2.95**

The group absorbs about half the tariff in its markup, and its **proportional** profit loss
is **smaller** than a CES model predicts, not bigger. Variable markups **cushion** the hit
on the intensive margin.

The intuition "oligopoly makes tariffs more damaging to home-owned MNEs" is **wrong on this
margin**. What survives is that the *level* of profit at stake is much higher under
oligopoly, so the absolute loss can still be larger. Do not write the amplification sentence
until Layer 3/4 shows which channel actually carries it. Candidates: levels rather than
elasticities, the HQ-input channel (Layer 2), GE terms-of-trade (Layer 3), or the extensive
margin.

### Test 5 — Fact 4

12 affiliates, σ = 5:

| counting | HHI | top share | mean markup |
|---|---|---|---|
| 12 independent firms | 0.129 | 0.206 | 1.370 |
| 3 parents, counted naively | 0.105 | — | — |
| 3 parents, grouped correctly | **0.373** | 0.475 | **1.934** |

**The presentation mixes two different experiments and the write-up compared the wrong
one to the data.** Figure 6 holds *behaviour* fixed and changes only the *accounting*.
Rows 1 vs 3 above change behaviour as well (12 independent firms is a different
equilibrium from 3 groups). The Figure-6 experiment is rows 2 vs 3:

```
0.105 → 0.373   ratio 3.54×        model
0.192 → 0.215   ratio 1.12×        data
```

so the overshoot is worse than recorded, not better. Two further points:

- The naive HHI **falls** (0.129 → 0.105) when firms coordinate, because output
  restriction equalises affiliate shares. **Grouping and coordination push measured
  concentration in opposite directions**, and Fact 4 nets the two. Any calibration to
  Figure 6 must respect that, or it will attribute the whole gap to the fringe.
- The data HHI and the model `S_g` were **not the same object** — Figure 6 is a share
  of *LAC exports within an HS6*, model `S_g` a share of *destination-market class
  expenditure*. **Resolved: see §5a.**

### 5a. The denominator problem — RESOLVED

Fixed by building the **measurement operator** into the model (`measured_hhi` in
`src/layer1_markets.jl`, mirrored in `scripts/verify/measurement.py`) rather than
trying to rebuild the model's denominator in data we do not have. The model now
simulates the customs records and runs *Figure 6's own estimator* on them: keep
in-sample origins only, pool across destinations within a product, renormalise, HHI,
value-weight across products. `level = :affiliate / :parent_country / :parent`
reproduces Figure 6's three bars exactly.

**What that immediately showed** (Julia; Python mirror agrees):

| n_fringe | λ | structural HHI | measured HHI (parent) |
|---|---|---|---|
| 0 | 1.000 | 0.244 | 0.079 |
| 16 | 0.548 | 0.092 | 0.080 |
| 256 | 0.133 | 0.010 | 0.107 |

Structural HHI falls **24×** (Python: 30.8×). Measured HHI moves **1.36×**
(Python: 1.22×). **Renormalising within the observed sample cancels the fringe out,
so Figure 6 is near-invariant to it and can never identify it.** That was the bug.

**The correct identification is two moments for two primitives:**

- **λ**, the in-sample share of destination absorption — LAC exports ÷ destination
  absorption (Comtrade imports + domestic production) — identifies the **fringe mass**.
- **Measured HHI** (Figure 6) identifies the **number and dispersion of LAC exporters**.

Match them **jointly**, not one at a time: measured HHI drifts up slightly with the
fringe because oligopolistic markup discipline compresses in-sample shares.

**The overshoot verdict changes.** Like-for-like, the model's grouping increment is
~1.7–1.8× versus the data's 1.12× — an overshoot of ~1.6×, not the ~3.2× that
CLAUDE.md recorded. Most of the apparent gap was the denominator bug, not the model.
The *level* is still too low (too many LAC exporters per product in the synthetic
panel), which is now a calibration target rather than a defect.

### Development discipline (learned the hard way)

A bug shipped in the first version: `μ` was computed from the old share vector and then the
share vector was overwritten, so `μ` and `S` were inconsistent whenever the iteration had
not converged. The closed-form check failed with error 1.77 instead of 1e-16. In Julia alone
this would have been blamed on the algebra.

**Rules:**
1. Every closed form goes into the paper only after a numerical check against the solver on
   ≥1000 random parameter draws, to ~1e-14.
2. Prototype in Python, port to Julia, keep the Python mirror in `scripts/verify/` as a
   cross-check. Two independent implementations catch what one cannot.
3. Every FOC gets a numerical best-response check (perturb the agent's own choice variables,
   confirm profit falls both directions) before it is trusted.
4. No layer is added until the previous layer's tests are green.

---

## 6. Repo layout (actual, 2026-08-19)

```
Orbis_DNB_Customs_Model/
|-- CLAUDE.md                     # this file -- keep current
|-- mne_model.jl                  # THE consolidated model, base Julia, no packages
|-- entry_uniqueness.jl           # entry in one market: the theorem + every test  [S17]
|-- entry_ge.jl                   # entry inside the GE loop                       [S18]
|-- entry_approaches.jl           # superseded prototype: 4 mechanisms, Bertrand cost
|-- entry_facts.jl                # superseded prototype: Poisson-Pareto parents
|-- run_entry_uniqueness.txt      # committed run of the theorem's tests
|-- run_entry_ge.txt              # committed run of the GE audit + facts
|-- run_calib_exportsample.txt    # fixed cost vs the size of the LAC export sample
|-- run_calib_joint.txt           # edge x capability-slope grid
|-- Model_Multinational_Firms_and_Trade.pdf
|-- chenying_yang_2023_08.pdf  Gaubert_2020.pdf  Gaubert_2021.pdf   # the 3 papers
|-- src/                          # pre-consolidation layers, kept for reference
|   |-- cournot_pe.jl  layer1_markets.jl  layer3_ge.jl
|   `-- stylized_facts.jl  uniqueness.jl
|-- scripts/verify/               # Python mirrors: cournot_pe.py, layer1_markets.py,
|                                 #                 measurement.py
`-- docs/
    |-- entry_options.md          # THE entry memo: architecture, theorem, evidence
    |-- model_guide.pdf / .tex    # the model in plain language
    |-- derivations.md            # the terse algebra
    |-- model_explained.md
    `-- Multinational_Firms_and_Trade.pdf   # stylized-facts document, v.2026-07-29
```

Run:

```bash
julia mne_model.jl facts        # the model without entry: CALIB block + scorecard
julia entry_uniqueness.jl       # the entry theorem and every test of it   (~4 min)
julia entry_ge.jl               # entry in GE: audit, uniqueness, facts    (~37 min)
julia entry_ge.jl full          # same, larger world
```

**Entry lives in `entry_uniqueness.jl` + `entry_ge.jl`, not in `mne_model.jl`.**
Folding it into the single consolidated file is deliberate remaining work, not an
oversight: it changes the cost function's calibration (§18.6), so `mne_model.jl` still
reproduces the no-entry results the earlier document was built on.

---

## 7. Layer 1 — DONE. What it delivered, and what comes next

`src/layer1_markets.jl`, mirrored in `scripts/verify/layer1_markets.py`.
Full algebra in `docs/derivations.md` §1.

**The headline is a three-term decomposition, not two.** With `w_m = E_m/ΣE_m`:

```
Σ_m w_m Σ_g θ_g S_gm²  =  θ̄_agg · H̄              (1) naive
                        + Cov_w(θ̄_m , H_m)        (2) BETWEEN markets
                        + Σ_m w_m Cov_S,m(θ , S)   (3) WITHIN markets
```

These map exactly onto a data-availability ladder, which is the cleanest possible
statement of the contribution:

| what you have | what you can compute |
|---|---|
| country ownership share + published HHI | (1) only |
| + ownership share market by market | (1) + (2) |
| + firm-level global-ultimate-parent identity | (1) + (2) + (3) |

Deliverables 1–5 are all green:

| # | Deliverable | Status |
|---|---|---|
| 1 | `solve_all(markets)` reusing Layer 0 unchanged | done |
| 2 | ownership accounting, aggregate `Π_H` | exact to 3.4e-16 vs brute force |
| 3 | the decomposition and the `total/naive` ratio | done, see traps below |
| 4 | competitive fringe | done, mass → ∞ recovers CES |
| 5 | tests (a) uniform θ, (b) fringe limit, (c) closed form | all pass |

### Two traps in reporting the headline — both would be easy to get wrong

- **`total/naive` does not converge to 1 as markets become atomistic.** It converges
  to the home-firm **size premium**, because numerator and denominator vanish
  together. The proportional understatement survives even as the correction dies.
  **Never report the ratio without the level of the granular correction beside it.**
- **A positive within-term appears with no ownership sorting at all**, purely because
  MNEs are larger than the fringe (1.08× in the synthetic panel). That is the
  **MNE-size channel**, economically real but distinct from ownership sorting.
  Report the two separately or they get conflated. The clean null is uniform θ over
  *all* firms, which gives exactly zero.

### Next task: Layer 2 (HQ input, α_k) — now unblocked

The denominator problem is resolved (§5a), so the cell definition Layer 2's α_k is
identified off is now settled. Build α_k against the **measured** statistics, never
the structural ones.

The remaining empirical prerequisite is unchanged and still the single highest-value
hour on the project: the destination-of-MNE-exports tabulation in §2.2. The
"boomerang" framing has no evidence behind it yet.

### Uniqueness — settled, see `src/uniqueness.jl`

The equilibrium is unique, and the proof is short: a group's many quantity choices
collapse to one sufficient statistic `X_g`; revenue is concave in `X_g` and cost is
convex, so group profit is **concave in the group's own quantity vector** and the
best response is unique; the game is aggregative and each `S_g(A)` is strictly
decreasing in `A`, so `Σ_g S_g(A) = 1` has exactly one solution. All five steps are
verified numerically, plus a decisive multi-start test: **1,000 runs** from starting
points spanning a factor of ~e¹², across five market structures (G = 2…8, η = 1…2.5),
all converge to the same equilibrium to ~1e-9.

Three assumptions carry the proof and all three are asserted in `solve`: `σ > η`,
at least two groups, strictly positive costs.

## 8. Literature map

**Machinery for oligopoly and granularity**
- Atkeson & Burstein (2008, AER 98:5) — Cournot with share-dependent markups. The cell game.
- Edmond, Midrigan & Xu (2015, AER 105:10) — GE markup aggregation, productivity-ordered entry.
- Gaubert & Itskhoki (2021, JPE 129:3) — finite Fréchet lists per market, granular decomposition, Poisson–Pareto primitives. Use for Layer 1 calibration of market lists.

**MNE block**
- Helpman, Melitz & Yeaple (2004, AER 94:1) — export vs affiliate margin.
- Horstmann & Markusen (1992, JIE 32) — Cournot MNE entry game.
- Tintelnot (2017, QJE 132:1) — export platforms, location portfolios, fixed-cost identification. Closest published benchmark.
- Head & Mayer (2019, AER 109:9) — HQ-input transfer cost, single-sourcing evidence, sub/supermodularity warning. Source for Layer 2's δ.
- Alviarez, Head & Mayer (2025, AEJ Micro) — oligopoly + multinational ownership + brands. **Vanessa Alviarez is at the IDB.** Talking to her is the highest-return hour available on this project. Do it before writing more theory.
- Yang (2023) — multi-plant oligopolist location choice, Arkolakis–Eckert submodular combinatorial algorithm. Layer 6, cut for v1.
- Ramondo & Rodríguez-Clare (2013, JPE 121:2); Arkolakis, Ramondo, Rodríguez-Clare & Yeaple (2018, AER 108:8) — cite as the quantitative MP benchmark, do not build on.

**Trade policy machinery**
- Lashkaripour & Lugovskyy (2023, AER) — optimal policy with markups and scale economies. This is what to take from Lashkaripour.
- Lashkaripour (2021, JIE) — sufficient statistics for tariff wars.
- Fajgelbaum, Goldberg, Kennedy & Khandelwal (2020, QJE) — tariff pass-through discipline.
- Ossa (2014, JPE) — trade wars and trade talks; profit-shifting / delocation motive.

**Ownership and tariffs (the direct ancestors — engage explicitly)**
- Brecher & Bhagwati (1981, JPE 89:3); Blanchard (2007, 2010); Blanchard & Matschke (2015, REStat); Blanchard, Bown & Johnson (2024, ReStud); Itskhoki & Mukhin (2025); Lu, Li & Li (2025).

Note: **Fajgelbaum and Lashkaripour do not work on multinationals.** Take the optimal-tariff
machinery from them; do not cite them as the MNE foundation.

---

## 9. Target propositions

- **P0 (Ownership Irrelevance).** In CES monopolistic competition with free entry, welfare
  and optimal policy are invariant to the ownership distribution. The null. Already proved.
- **P1 (Rent-extraction cancellation).** When the foreign supplier is H-owned, the classic
  rent-extraction motive for a tariff vanishes: the tariff moves rents from H's shareholders
  to H's treasury, leaving deadweight loss and the terms-of-trade term. Brecher–Bhagwati
  generalised to oligopoly. Anchor, not contribution.
- **P2 (Ownership–concentration covariance).** The correction is proportional to the
  ownership-weighted Herfindahl, not the average ownership share. Understatement from using
  the country-level share equals the covariance. **Analytic, §4.3. This is the paper.**
  ⚠️ **As stated this is a proposition about the LEVEL of profit income, and §4.3 proves
  exactly that and no more.** The policy claim needs `dΠ_H/dt`, whose share-dependence is
  `Λ_g`-weighted and is *not* `S_g²` (§4.6). Split P2 into:
  - **P2a (income).** `Π_H` correction = `θ̄·HHI + Cov_S(θ,S)`. Proved, verified, safe.
  - **P2b (policy).** The optimal-tariff correction is ownership-weighted and rises with
    concentration. **Not yet proved, and it cannot be proved at Layer 0** (§4.7).
  Writing P2b on the strength of §4.3 is the single biggest referee risk in the project.
- **P1 needs a health warning.** P1 presumes a positive rent-extraction motive to cancel.
  At Layer 0 that motive is absent for ordinary market structures (§4.7): with constant
  marginal cost there is no terms-of-trade term, and the markup distortion dominates, so
  optimal policy is a subsidy. P1 is only non-vacuous where the frontier in §4.7 is
  crossed — dominant group, fragmented fringe, high σ — or once Layer 3 supplies an
  upward-sloping foreign supply curve.
- **P3 (Downward-sloping optimal tariff in complexity).** α_k rises with complexity and
  foreign-MNE share rises with complexity, so the correction is biggest in complex goods —
  reversing standard tariff escalation. Testable against actual US/EU schedules.
- **P4 (Deflection blunts the terms-of-trade motive).** Export platforms make the
  export-supply elasticity facing H higher for MNE-supplied products, so the inverse-elasticity
  term is smaller exactly where P2 also bites. Two independent facts, one conclusion.

Frame the paper as **measuring** the correction, not as asserting tariffs are self-defeating.
Relocation can pull production into H and raise its wage; the sign is quantitative.

---

## 10. Open decisions

1. ~~Competitive fringe vs markup cap~~ — **resolved: use a fringe.** §4.7 shows the
   fringe is not a numerical convenience, it is what creates the tariff motive at all.
   The frontier depends on how *fragmented* the fringe is (`n`), not merely on whether a
   cap binds. A markup cap would suppress the mechanism the paper is about. What remains
   open is how to *calibrate* the fringe, which is blocked on the denominator problem in
   §5 (Figure-6 HHI is a share of LAC exports; model `S_g` is a share of destination
   expenditure). **That denominator fix is now the blocking item for Layer 1.**
2. Bertrand vs Cournot. Cournot chosen for the clean η = 1 closed form and because it is the
   AB convention. Bertrand gives `ε(s) = σ(1-s) + s`; worth a robustness appendix.
3. θ from GUO country only, or portfolio ownership shares? Benchmark against BEA's
   ownership-based current account.
4. Conduit reallocation rule for PAN, BMU, IRL, NLD, CHE, LUX. **Blocking for any θ that
   goes in a table.**
5. Sample: nine LAC origins, half of foreign-MNE value unmatched. Fix the framing as a lower
   bound and say so in the abstract.

## 11. Reality check

Full endogenous portfolios plus granular oligopoly plus endogenous optimal tariffs,
quantified, is three papers and a team. Scope v1 to: group Cournot + HQ input + ownership
accounting + GE + the ownership-adjusted optimal tariff formula, with exogenous portfolios.
That is a complete, defensible paper.

Also note the strategic cost. This is a trade-theory paper and the declared PhD field is
behavioural/development. It buys a substantive Volpe letter and a real technical asset, but
it does not close the field-credibility gap identified in earlier application post-mortems.
Budget time accordingly.


---

## 12. Layer 3 (GE) — built, and it reverses the headline

`src/layer3_ge.jl`. Multi-country, multi-sector general equilibrium with
multinational production, built on Ramondo–Rodríguez-Clare and Tintelnot.

### 12.1 What the GE closure adds

Marginal cost is no longer a number typed in from outside:

```
a[affiliate, n] = ( w_h^α_k · w_ℓ^(1-α_k) · γ_{hℓ} · d_{ℓn} ) / φ
```

- `w_h^α_k w_ℓ^(1-α_k)` — HQ services paid at the **parent's** wage, production at
  the **host's**. This is Layer 2's α_k, now load-bearing.
- `γ_{hℓ}` — the MP friction of Ramondo–Rodríguez-Clare (2013).
- `d_{ℓn}` — trade cost from the **production** country, so one plant serves many
  destinations: Tintelnot's (2017) export platforms.

Closed by labour market clearing, `X_n = w_n L_n + Σ_g θ_gn Π_g + T_n`, and Walras.

### 12.2 Accounting audit — the math adds up

All verified at **arbitrary, non-equilibrium wages**, which is the strong test:

| Identity | Error |
|---|---|
| **Walras' law, `Σ_n w_n Z_n = 0`, off equilibrium** | 4.3e-16 |
| Goods clearing `Σ_g S_g = 1` | 2.2e-15 |
| Revenue = factor payments + tariffs + profits | 4.8e-16 |
| Country budget `X = wL + profits owned + tariffs` | 4.0e-16 |
| World budget | 4.0e-16 |
| Homogeneity of degree 0 in wages | 4.3e-15 |
| Doubling endowments doubles income, wages unchanged | ratio 2.0000000000 |
| Labour clearing at the solution | 8.9e-13 |
| GE uniqueness, 60 random wage starts | identical to 3e-12 |

Market-level uniqueness is a **theorem** (§ uniqueness). GE wage uniqueness is now
**proved under two checked conditions** — Theorem 4, §27 — and gross substitutes is
shown NOT to hold unconditionally: there is a counterexample. Say both halves.

### 12.3 The finding that reverses the hypothesis

Country 1 taxes its imports; vary how much of the foreign MNEs it owns:

| Country 1 owns | dW/dt at free trade | optimal tariff |
|---|---|---|
| 0% | −0.209 | < −0.90 |
| 25% | −0.146 | < −0.90 |
| 50% | −0.012 | 0.00 |
| 75% | +0.190 | +0.20 |
| 100% | +0.459 | +0.55 |

**The optimal tariff RISES with home ownership.** Monotone across all five levels,
confirmed by two independent methods (marginal effect at free trade, which cannot
hit a corner; and a direct grid search).

**Mechanism.** Brecher–Bhagwati is present but outweighed by a channel specific to
multinational production:

1. a tariff depresses foreign **wages** (verified: 0.684 → 0.630 as t goes 0 → 40%);
2. the firms you own **produce abroad** with that labour;
3. so their costs fall and their **profits rise**, and those profits are yours.

Owning a foreign *exporter* ≠ owning a foreign *affiliate*. A tariff is partly a
device for depressing foreign wages, and if you own the capital hiring that labour
you capture the rents — collecting the terms-of-trade gain twice, as consumer and
as shareholder. This does not appear to be in the ownership-and-tariffs literature,
which uses country-level equity positions rather than affiliates hiring foreign
workers.

### 12.4 What to do about it

- **Keep P2a.** The income decomposition is an identity and is the measurement
  contribution. Untouched.
- **Do not write P2b.** "Ownership lowers the optimal tariff" is not established
  and may be false. It is one of two opposing forces.
- **The sign is empirical, and cheap to resolve.** It turns on α_k and on where
  the affiliates we observe actually produce and sell. If US-owned LAC affiliates
  ship mostly back to the US, Brecher–Bhagwati is strong; if they serve third
  markets while hiring LAC labour, the wage channel wins.
- **This makes the destination-of-MNE-exports tabulation (§2.2) decisive, not just
  useful.** The sign of the headline result depends on it. Do it first.

This may be a better paper than the original plan: "when does home ownership raise
rather than lower the optimal tariff, and why" is a sharper question than
re-deriving Brecher–Bhagwati with better data.

### 12.5 A bug worth remembering

A Julia closure-scoping bug cost a run: inside a nested function, assigning to a
name that also exists in the enclosing scope **overwrites the outer variable**. A
loop variable `h` (HQ country) silently clobbered a step size `h = 1e-4`, producing
a tariff of −2 and negative costs. Nested helper functions must not reuse outer
names.


---

## 13. Stylized-facts scorecard — `src/stylized_facts.jl`

Every fact tested by RUNNING the GE model. **One** parameter is calibrated (the MNE
productivity edge, to Fact 1), so Facts 4 and 6 are out-of-sample predictions.
**Superseded 2026-08-20:** the productivity edge is no longer fitted (it is 0), so the
count is TWO fitted parameters, not three. See §0.3 and §29.4.

| # | Fact | Verdict | What it needs |
|---|---|---|---|
| 1 | MNEs large share of exports | calibration target (0.45–0.66 vs 0.47–0.74) | endogenous MP entry |
| 2 | Foreign MNEs in complex goods | partial | capability channel |
| 3 | Parents from few countries | input | endogenous MP entry |
| 4 | Grouping raises concentration | **GENERATED** | — |
| 5 | MNE presence raises non-MNE exports | **FAILED** | I-O linkages |
| 6 | Distance weaker for MNEs | **GENERATED** | — |

**Fact 2 — the informative failure.** With no capability channel the gradient of the
foreign share on α_k is **−0.11**, essentially zero; the data need **+0.43**. Reason:
in the cost function a foreign affiliate pays its *parent's* wage on the α_k share,
and parents sit in high-wage countries, so raising α_k makes foreign ownership more
*expensive* — a force pushing against the fact. Fix: make the HQ input a **capability
local firms cannot buy**, not merely costly labour. A capability slope ≈0.8 reproduces
Figure 2 (0.48/0.43/0.57/0.69 vs data 0.52/0.52/0.63/0.70).

**Fact 4 — the core success.** Model 0.064 → 0.073 → 0.115 (ratio 1.78×); data
0.192 → 0.209 → 0.215 (1.12×). Ordering is a *prediction*. Overshoots the jump,
undershoots the level (too many firms per market) — calibration targets, not defects.

**Fact 5 — the outright failure.** Adding MNEs to a sector cuts non-MNE export value
by 42% (PE) / 45% (GE) at η=1; 31%/37% at η=2.5. Data say strongly positive. GE softens
but does not reverse. **This is the only contradiction and it goes first.** Needs
input–output linkages (MNEs supplying cheaper inputs to local exporters).

**Fact 6 — generated, at the affiliate level.** With origin/destination/sector FE:
gradients −0.93 (non-MNE) / −0.76 (MNE) / −0.72 (MNE present). Both interactions
positive and in the data's order (data: −0.16/−0.12/−0.05). Falls out of `d[ℓ,n]`
running from the *production* country. Magnitudes far too large, but note the data's
−0.164 is itself an order of magnitude below any gravity estimate.
**At the PARENT level the "present" term flips sign** (cannibalisation: a parent with
a plant at the destination has a high markup there and holds back distant plants).
→ **Sharp test for Table 2: re-run with firm FE and at the parent level.** Model
predicts attenuation survives at affiliate level, weakens at parent level.

### 13.1 Ranked next steps

1. **Fact 5**: input–output linkages. Only outright contradiction.
2. **Fact 2**: capability channel (Layer 2), slope identified off Figure 2.
3. **Facts 1 and 3**: endogenous MP entry (fixed cost + γ_hℓ + productivity). One
   extension buys both.
4. **Fact 6**: endogenous plant location. Hardest, least urgent.

### 13.2 GE uniqueness — strengthened

220 random wage starts spread over ~e⁶, across 5 economies (N=4–7, K=3–5, η=1–2.5,
tariffs 0–40%). All converge to the same wages (max gap 5.7e-12) and every one
genuinely clears the labour market (worst residual 1e-12), so a "match" cannot be
two failures agreeing. Since 2026-08-20 this sits alongside **Theorem 4** (§27):
proved under (H1) no group above half and (H2) demand beats exposure, both checked.


---

## 14. Input-output linkages, and the consolidated file

### 14.1 `mne_model.jl` — one file, no dependencies

Everything now lives in a single self-contained Julia file (stdlib only). Share it
as-is. `julia mne_model.jl quick` (~2 min) runs every section on a small economy;
`julia mne_model.jl` (~20 min) is the full thing; also `core`, `ge`, `facts`, `full`.
The split `src/*.jl` files are kept for reference but `mne_model.jl` is the source
of truth.

### 14.2 The I-O block

Cost now includes an intermediate bundle (Caliendo–Parro):

```
a[a,n] = (1/φ) · (w_h^α_k · w_ℓ^(1-α_k))^(1-ν_k) · (PIO[ℓ,k])^ν_k · γ_hℓ · d_ℓn
PIO[ℓ,k] = Π_k' P[ℓ,k']^ω[k',k]
```

Intermediates are bought **in the production country**, so multinational entry into
ℓ lowers local firms' input costs. That is the Fact 5 channel.

**New theorem, and it strengthens the uniqueness story.** The cost↔price loop is a
**contraction with modulus max ν < 1**, because (i) the market price index is
homogeneous of degree 1 in costs (scale all costs by λ → shares unchanged → P scales
by λ), and (ii) ln cost loads on ln PIO with coefficient ν, with ω columns summing
to 1. Banach ⇒ unique fixed point, global convergence, geometric rate ν.
Verified: iterations needed vs predicted log(ε)/log(ν) = **27/27, 45/47, 87/91**.

So the solution ladder is now: market equilibrium **proved** unique → I-O price loop
**proved** unique → sector shares closed form → E and X **two exact linear systems**
→ wages the only step resting on numerical evidence.

### 14.3 Fact 5: the I-O channel is NOT enough

| ν | same sector | other sectors | all |
|---|---|---|---|
| 0.00 | −43.7% | −6.6% | −16.4% |
| 0.60 | −38.5% | −5.9% | −14.8% |

Moves the number ~5 points; does not flip the sign. Two reasons: business stealing
is direct (σ=5) while the input-price benefit reaches only ν·ω ≈ 0.2 of cost; and
**in GE the new plants bid up local wages**, which is why even *other* sectors lose.
A PE model would have shown other sectors gaining and declared victory — the GE
closure changed the answer, not just the decimals.

**What it would take.** Adding a same-sector productivity spillover
`φ_local × (1 + #MNE plants)^spill`:

| spill | same sector |
|---|---|
| 0.00 | −40.9% |
| 0.10 | −12.6% |
| **0.20** | **+16.4%** |
| 0.30 | +38.9% |

**Sign flips at ≈0.15–0.20.** But that is a *horizontal* spillover, and Javorcik
(2004) finds horizontal spillovers to be zero or negative, with positive effects only
through *backward* linkages. **The channel the model needs is the one the literature
says is absent.**

Two readings: (i) the spillover is real and larger than estimated; or (ii) **Fact 5
is substantially selection** — good markets attract multinationals and local
exporters alike. Reading (ii) fits the evidence better and the empirics cannot rule
it out (no destination×product×year FE anywhere in Table 1).

**→ Top-ranked next step: add destination×product×year FE to Table 1.** Cheap, and it
decides whether Fact 5 is a target the model must hit or an artefact it is right to
miss.

### 14.4 Performance note

The naive implementation was unusably slow once I-O was added (~10 min per suite).
Two fixes, neither of which changes any answer: **safeguarded Newton** (bracket-
preserving, so the uniqueness guarantee is intact) replacing pure bisection in both
market loops, and **Newton on log wages** replacing 200-step tatonnement. Residuals
got *better* (5.7e-16 vs 9.1e-13), runtime fell ~15×.


---

## 15. Calibration

All parameters are in a single `CALIB` block at the top of `mne_model.jl` and are
printed by `julia mne_model.jl facts`. Three kinds:

| Parameter | Value | Kind | Basis |
|---|---|---|---|
| σ within-sector | 5.0 | external | trade elasticity 4–6 |
| η across-sector | 1.0 | external | Cobb–Douglas outer nest (Atkeson–Burstein) |
| distance exponent | 1/(σ−1) | external | **makes gravity trade elasticity = −1** |
| ν intermediate share | 0.55 | external | manufacturing I-O share (Caliendo–Parro) |
| ω own-sector | 0.45 | external | I-O tables are diagonal-heavy |
| α_k HQ share | 0.10–0.55 | external | rises with complexity (Head–Mayer) |
| γ MP friction | 1.18 | external | Ramondo–Rodríguez-Clare |
| MNE productivity edge | ~~solved~~ **0.0** | ~~internal~~ **not needed** | superseded 2026-08-20: with head-office services a capability the edge is unnecessary and Fact 1 still comes out at 0.689. See §29.4. |
| HQ capability gradient | solved | **internal** | set to match **Fact 2** |
| country productivity, w₁ | 2.2/1.0, 1.0 | normalised | relative wages, numeraire |

**Only two parameters are fitted, both to stylized facts.** That is what makes
Facts 4, 5 and 6 out-of-sample.

⚠️ **The EXTERNAL values are standard magnitudes for this literature, not
transcriptions from specific tables.** Verify each against its source paper and
replace before publication. They are collected in `CALIB` so this is a five-minute
job.

### 15.1 The gravity check — a free diagnostic

Setting the distance exponent to 1/(σ−1) makes the elasticity of *trade* with
respect to distance exactly −1, the central gravity estimate. The model's own
gravity regression then returns **−0.93** for ordinary firms: target hit without
being fitted to that regression.

**This reframes Fact 6.** Model −0.93, gravity literature ≈ −1, Table 2 **−0.164**.
The *data* coefficient is the outlier, an order of magnitude below any gravity
estimate — the specification concern already flagged. Judge Fact 6 on its
**interactions**, which is what the fact is about, not on the level.

### 15.2 Self-reporting verdicts (defect found and fixed)

The facts runner originally printed **hardcoded** verdict text. A reduced-economy
run printed "both interactions positive" while the numbers directly above showed
−0.231 and −0.250. Every verdict that asserts a sign is now **computed from the
output**, and reports "NOT REPRODUCED IN THIS RUN" with a power caveat when a small
sample lacks identification. Anything shared must not contain prose that can drift
from its own numbers.


### 15.3 Final verification run (default mode, 66 min)

| Check | Result |
|---|---|
| Walras off-equilibrium | 2.6e-16 |
| goods clearing / revenue identity / budgets / I-O identity | ≤ 1.1e-15 |
| homogeneity in wages | 2.0e-15 |
| market uniqueness, 200 starts × 5 structures | identical to 8.9e-10 |
| GE uniqueness, 110 starts × 5 economies | identical, resid ≤ 1.7e-12 |
| **Fact 4** measured HHI | 0.067 → 0.076 → **0.119** (1.77×) — GENERATED |
| **Fact 6** gradients | −0.93 / −0.77 / −0.71, both interactions **positive** — GENERATED |
| gravity calibration check | −0.93 vs target −1 — on target |
| **Fact 5** sign flip | between spillover 0.10 and 0.20 |
| calibrated MNE edge | 0.382 log points (1.47×) |

Runtimes (measured, now in the file header): `quick` 6 min, default 66 min,
`core` 10 s, `ge` 9 min, `facts` 55 min.


---

## 16. Final state (joint calibration)

`mne_model.jl` now calibrates **two** parameters to **two** facts jointly: the
gradient is scanned, the productivity edge re-solved at each point so Fact 1 always
holds exactly, then interpolated to Figure 2's slope. **Facts 3-6 are evaluated at
that single calibrated model** — previously they were evaluated with the gradient
set to zero, so the calibration table and the code disagreed.

**Calibrated:** capability gradient **0.588**, productivity edge **0.188** (1.21x).

| Fact | Model | Data | Verdict |
|---|---|---|---|
| 1 MNE export share | 0.44–0.67 | 0.47–0.74 | target, hit |
| 2 complexity gradient | 0.51/0.43/0.57/0.67, slope **+0.43** | 0.52/0.52/0.63/0.70, slope **+0.43** | target, hit exactly |
| 3 parent concentration | 0.43 / 0.57 | GBR/USA/CAN | input |
| 4 grouping HHI | 0.069 → 0.078 → **0.121** (1.75x) | 0.192 → 0.209 → 0.215 (1.12x) | **GENERATED** |
| 5 non-MNE exports | needs spillover ≈0.15 | positive | see §14.3 |
| 6 distance gradients | −0.93 / −0.78 / −0.73 | −0.164 / −0.118 / −0.051 | **GENERATED** |

Gravity check: model −0.93 against the calibration target of −1. On target.

### 16.1 Document restructure

Part IV ("Where this model comes from") is **dissolved**. Its content is now seven
**"Where this comes from"** notes placed at each piece's first appearance — CES and
the share-dependent markup (Atkeson–Burstein), HQ input (Head–Mayer), intermediates
(Caliendo–Parro), MP friction (Ramondo–Rodríguez-Clare), export platforms
(Tintelnot), ownership (Brecher–Bhagwati, Blanchard). Each says what it is in the
original, why the model needs it, what it does here, and how it is disciplined.
Our departure sits inside the conduct section; the location caveat sits where
locations are assumed. A one-page "Provenance at a glance" closes Part II.

### 16.2 Process defects found in this pass — read before editing the document

1. **Silent sync failures.** Earlier scripts reported success on a global flag
   while individual replacements failed. Fact 4 and Fact 5 tables were **two runs
   stale** while being reported as synced. **Every replacement must assert its own
   match.**
2. **CRLF.** The .tex has CRLF endings. Multi-line Python string matches fail
   silently. Use line-based replacement (`split(nl)`), not multi-line `str.replace`.
3. **Escape corruption.** Non-raw Python strings turned `	` into a TAB (rendering
   "8.9imes10^-10" in the PDF) and `` into a bell. **Always use r"" for LaTeX.**
4. **Hardcoded verdicts in code.** Fixed: every verdict asserting a sign is now
   computed from the run, and reports "NOT REPRODUCED IN THIS RUN" with a power
   caveat on small samples.


---

## 17. Firm entry — solved, and with a theorem rather than a selection rule

Papers in the folder: Yang (2023) multi-plant oligopoly location; Gaubert & Itskhoki
(2020 JPE) granular comparative advantage; Gaubert, Itskhoki & Vogler (2021 JME)
granular policies. Full memo: **`docs/entry_options.md`**.
Code: **`entry_uniqueness.jl`** (one market: the theorem and every test of it),
**`entry_ge.jl`** (entry inside the GE loop). Superseded prototypes kept for their
Bertrand and Poisson-Pareto experiments: `entry_approaches.jl`, `entry_facts.jl`.

### 17.1 What was wrong with the earlier recommendation — and it was wrong

The 2026-08-19 morning pass recommended a two-tier structure in which granular
parents played **Yang's order-selected entry game**. ~~Two-tier with order
selection~~ — **dead.** Christian's objection is correct: with G parents there are G!
orders, no principle picks one, and the outcome selected is not a property of the
model. The two-tier *economics* survives; the selection device does not.

The earlier pass also asked the wrong diagnostic question. It tested the
Gaubert–Itskhoki condition — does the marginal entrant's incremental profit fall as
the **market-wide** entrant count rises — which mixes a firm's own effect with its
rivals'. That is why it read as hopeless (226/300 violations with internalisation).

### 17.2 The theorem

Separate the two effects. For one market and one parent *g*,

```
K_g    = sum over ACTIVE own affiliates of c_i^(1-sigma)     capability stock
psi(S) = S mu(S)^(sigma-1)                                   S_g = psi^{-1}(K_g/A)
Omega(K; A) = Pi(psi^{-1}(K/A))  -  fixed costs
```

A parent's payoff depends on its own affiliates **only through the scalar `K_g`**.
Internalisation is already inside `K_g`, so the parent's entry problem is
one-dimensional however many affiliates it owns.

- **(A) own concavity** — `Omega` strictly concave in `K`, so the best response is a
  **cutoff on the parent's own list**. GI's cutoff, applied *within* the parent.
- **(B) strategic substitutes** — `d Omega_K/dA < 0`, i.e. `|eps_G| < eps_psi` with
  `G = Pi'/psi'`.

(A) and (B) ⇒ every `K_g` non-increasing in `A`, and `A` strictly increasing in every
`K_g` ⇒ **the entry equilibrium is unique.** No order, no selection rule, and
internalisation intact.

**The hypothesis has content.** (A) is unconditional; (B) needs no single **parent**
above a share frontier:

| sigma | eta | frontier | sharper Nash notion |
|---|---|---|---|
| 5 | 1 | **0.548** | 0.532 |
| 5 | 1.5 | 0.936 | 0.652 |
| 4 | 1 | 0.556 | 0.536 |
| 8 | 1 | 0.532 | 0.524 |

Past that, Cournot profit is convex enough in the share that a parent's marginal value
of capability rises with the aggregate. **Checked at the solution, not assumed** — the
largest parent share anywhere in the solved GE is **0.252**, under half the frontier.

**Equilibrium concept, stated:** parents internalise among their own affiliates when
setting *quantities* (the group markup — Fact 4 rests on it) and take the market
aggregate as given when deciding *entry*. That is Gaubert–Itskhoki's own free-entry
condition. The sharper alternative is implemented as `nash_refine`, and the gap is
reported rather than assumed away.

### 17.3 Evidence — `entry_uniqueness.jl`, committed run in `run_entry_uniqueness.txt`

```
CONDITION A  Omega concave in own K       violated   0 / 400
CONDITION B  Omega_K falling in A         violated 101 / 400  over ALL shares
                                                     1 / 300  inside the frontier
brute force over EVERY configuration, deviations scored on the RE-SOLVED market:
  markets with a pure-strategy equilibrium   120 / 120
    of those UNIQUE                          120        multiple  0
  uniqueness CERTIFICATE issued              120        false positives  0
cheap solve already an exact Nash eq          82 / 120
  after one nash_refine pass                 120 / 120
K*(A) not non-increasing, whole grid         193 / 1200
  restricted to shares inside the frontier     0 / 1200
sum_g S_g(A) not decreasing                    0 / 200
```

The certificate is what travels into the GE loop, where brute force is unaffordable:
the solver re-checks that the clearing function is decreasing with exactly one sign
change, **market by market**.

### 17.4 Two tiers, one theorem

- **Tier 1 — granular parents**: several potential plants per market, internalised,
  markup on the parent's *total* share. Fact 4 lives here.
- **Tier 2 — local fringe**: one plant, one variety, its own group.

Both are the same object (a group choosing a subset of its own list; tier 2 is the
one-item case), so the two tiers need **one** theorem. Unlike Yang, the fringe plays
rather than sitting passive. Computationally the theorem pays for itself: concavity
means only the Pareto frontier of `(K, F)` can be optimal, so a parent with *n* plants
is scanned in a handful of candidates rather than `2^n`.

### 17.5 Cost of switching to Bertrand — no longer relevant, but recorded

Profit as `(E/sigma) s [1 + kappa s]`; the ownership correction is proportional to
`kappa`. `kappa_Cournot = sigma/eta - 1 = 4.0`; `kappa_Bertrand = (sigma-eta)/sigma =
0.8` at sigma=5, eta=1 — **five times smaller**. **Not taken**: the theorem delivers
uniqueness under Cournot, so the trade is unnecessary. **Do not** use monopolistic
competition with constant markups: `kappa = 0`, the `S²` term vanishes, and the
ownership result disappears entirely.

### 17.6 Bugs worth remembering

The Julia closure-scoping trap bit again in `solve_bertrand`: `share_given` assigned
`lo, hi`, also locals of the enclosing function, destroying the outer bisection
bracket on every call. It returned a solution violating its own defining equation by
0.375 and produced a **fake finding**. **Nested helpers must never reuse enclosing
local names**, and **always verify a solver satisfies its defining equation** before
believing anything derived from it.

Two more from this pass:

- **Test the right condition.** The 226/300 "failure" was a true statement about the
  wrong object. Before concluding a mechanism is impossible, check that the condition
  under test is necessary and not merely sufficient.
- **A solver that converges is not a solver that is correct.** The first entry solver
  converged happily while scoring each candidate subset against a *different* rivals'
  aggregate. It matched brute force 88/120 and looked plausible. Only the brute-force
  referee caught it.
- **CRLF and multi-line string edits** (trap 3 in §5) recurred: multi-line
  `str.replace` against these files fails silently. Use line-based replacement, and
  never join lines by counting quotes — docstrings are triple-quoted and get eaten.

---

## 18. Entry inside the general equilibrium — `entry_ge.jl`

§17 solves one market. This puts that margin inside the GE of `mne_model.jl`, so
**who produces where is an outcome, not a list typed in from outside.**

### 18.1 What the fixed cost is, and why it is that

Serving market *n* costs

```
F[a,n] = f_k * w[l] * fdist[l,n]      # BASELINE since 2026-08-20; see §0.1.
                                      # was f_k*(w[h]^alpha*w[l]^(1-alpha))*fdist
                                      # and still is under hq_cost = true
```

paid in the **same factor bundle as production** — in the baseline that is the host's
wage only; under `hq_cost = true` it is HQ services at the parent's wage,
the rest at the host's. Three consequences, all load-bearing:

1. the system stays homogeneous of degree one in wages, so `w[1] = 1` is still a
   legitimate numeraire;
2. fixed costs are **real resources**: they enter labour demand, and profits
   distributed through `theta` are **net** of them;
3. entry responds to wages, to the I–O price loop and to market size, so the
   extensive margin of multinational production **responds to policy** — which is what
   Layer 4 will need.

### 18.2 Where entry sits, and what happened to the contraction proof

The worry in the old §17.6 was right: entry is discrete, so Banach lapses, and entry
depends on market size, so expenditure enters a loop it used to sit outside. The fix
is structural — **freeze the discrete part while the smooth part is solved exactly**,
at both levels:

```
OUTER   entry configuration
  IN    equilibrium WAGES given that configuration   <- the audited smooth model
    IN  the I-O price loop, configuration frozen     <- the old map, modulus max nu
  THEN  ask who wants to move, and let some of them move
```

Freezing is not a numerical convenience, it is what keeps the inner block provably
convergent: prices do not depend on expenditure, so with the configuration fixed the
continuous block is *exactly* the block already verified. Measured inner modulus
**0.550** against `nu = 0.550`.

**Damping matters and is not selection.** Moving every unhappy parent at once does not
converge: the first pass starts from all plants active, wages adjust to a far denser
economy than will survive, and the configuration oscillates between two very different
market structures (46–49 slots of 432 moving each pass, wages differing by 38% in
logs). Moving only the parents with the **largest payoff gains** each pass converges.
That ranks parents by their own gain — it changes the path, not the fixed point, and
it is not an exogenous entry order.

**One structural fact makes the outer loop behave, and it is verified.** Entry is
**invariant to a uniform change in costs**: scale every delivered cost by a common
factor and shares, markups and profits are all unchanged, so no margin moves. Confirmed
at ×0.5, ×0.8, ×1.25, ×2.0. Only *relative* cost changes move entry, and the I–O loop
moves costs largely in common.

### 18.3 The accounting still adds up, fixed costs and all

Checked at **arbitrary, non-equilibrium wages**, which is the strong test. If fixed
costs were counted as factor payments but not netted out of distributed profits (or
the reverse), Walras' law would break — which is exactly why it is the test.

| Identity | Error |
|---|---|
| revenue = factor + intermediates + tariffs + gross profit | 3.0e-16 |
| wage bill = variable labour + fixed-cost labour | 6.6e-16 |
| **Walras' law off equilibrium** | 2.2e-16 |
| world income = world final expenditure | 0.0 |
| country budget `X = wL + owned NET profit + tariffs` | 1.0e-16 |

Fixed costs are **0.9%** of revenue; gross profit 27.6%, profit net of entry costs
**26.7%**.

### 18.4 Is the GE with entry unique? — the check that could not be run before

Market-level uniqueness is a theorem and every market carries its certificate. GE
wage uniqueness is **proved under (H1)–(H2)** (Theorem 4, §27); gross substitutes does
not hold unconditionally and the counterexample is in §27.4. But entry adds a check
that did not exist before: wages could agree while a
**different set of firms** was operating, which is precisely the multiplicity the entry
order was invented to paper over.

```
random wage starts                              8
worst disagreement in equilibrium wages         0.0     (bit-identical)
worst labour-market residual at those solutions 1.55e-15
runs ending with an unresolved integer margin   0
runs where EVERY market carried the certificate 8 / 8
largest single-parent share anywhere            0.199   (frontier 0.548)
DISTINCT ENTRY CONFIGURATIONS ACROSS THE STARTS 1
```

**The same firms enter from every start.** Note also that the theorem's hypothesis is
satisfied with room to spare at the solution: 0.199 against a frontier of 0.548.

### 18.5 The integer problem — sized, not asserted away

Entry is a choice over whole plants, so an exact fixed point in integers need not
exist. When none is reached the solver keeps the configuration with the fewest firms
wanting to move and returns `regret`; it never breaks a tie by an entry order. In the
audit economy that is **1 slot out of 432**, and at the calibrated point the outer
loop reaches an EXACT fixed point (regret 0) in three passes. Scaling the economy:

| firms | potential slots | active | unresolved | share |
|---|---|---|---|---|
| 80 | 320 | 242 | 1 | 0.0031 |
| 160 | 640 | 369 | 0 | 0.0000 |
| 320 | 1280 | 582 | 3 | 0.0023 |
| 480 | 1920 | 768 | 2 | 0.0010 |

Note the residue was much larger (5–7 slots) at the *first* fixed-cost guess, which
is the same thing §18.6 diagnoses: too high a fixed cost puts many firms on the
margin at once.

This is a **near-miss on existence, not a multiplicity**: nothing here requires
choosing between equilibria. Say it that way — a referee will ask.

### 18.6 Calibration with entry — `run_calib_exportsample.txt`, `run_calib_joint.txt`

**Entry pins a parameter that used to be free.** Facts 2, 3 and 4 are all measured on
the LAC *export* sample. Without entry every affiliate exported by construction, so
that sample was rich whatever the parameters. With entry it is an outcome, and the
first guess `f = 0.006` collapsed it to **six plants** — at which point those three
facts are not measurable at all (parent HHI 1.00, affiliate→parent ratio exactly 1.00,
and every "degenerate" reading in the first facts run traces to this).

| fixed cost | LAC exporters | parents | parent countries | hhi affiliate → parent |
|---|---|---|---|---|
| 0.0060 | 6 | 6 | 2 | 0.555 → ×1.00 |
| 0.0020 | 31 | 24 | 3 | 0.148 → ×1.40 |
| **0.0006** | **59** | **43** | **4** | **0.080 → ×1.51** |
| 0.0002 | 71 | 53 | 4 | 0.071 → ×1.51 |

An order of magnitude below the first guess, and flat below it. **This is a gain:** a
free parameter is now disciplined by how many firms actually export. It also explains
the integer residue in §18.5 — too high a fixed cost puts many firms on the margin.

**Edge and capability slope, jointly, at `f = 0.0006`** (grid, not nested bisection,
so the trade-off is visible):

| slope | edge | Fact 1 | parent HHI | top parent | hhi aff → par | Fact 2 gradient |
|---|---|---|---|---|---|---|
| 0.0 | 0.10 | 0.304 | 0.340 | 0.447 | 0.088 → 0.096 (×1.09) | **−0.05** |
| 0.0 | 0.50 | 0.687 | 0.310 | 0.393 | 0.102 → 0.144 (×1.42) | −0.22 |
| 0.6 | 0.10 | 0.464 | 0.324 | 0.426 | 0.091 → 0.112 (×1.23) | +0.14 |
| **1.2** | **0.10** | **0.631** | **0.314** | **0.414** | **0.112 → 0.154 (×1.37)** | **+0.26** |
| 1.2 | 0.50 | 0.916 | 0.302 | 0.388 | 0.144 → 0.222 (×1.55) | −0.17 |
| *data* | | *0.47–0.74* | *0.130* | *0.250* | *0.192 → 0.215 (×1.12)* | *+0.43* |

**Calibrated with entry: fixed cost 0.0006, edge 0.10 (1.11×), capability slope 1.20.**
The edge falls from ~~0.188 (1.21×)~~ without entry to **0.10 (1.11×)** with it — the
extensive margin is far more sensitive, so Fact 1 is now much more informative about
the edge than it was with a fixed roster (0.30 → 0.63 → 0.92 as the edge goes 0.10 →
0.50). The slope rises from ~~0.588~~ to **1.20** and still undershoots.

### 18.7 Scorecard at the calibrated model with entry

672 potential affiliate-destination pairs, **498 active (74.1%)**, foreign entry rate
0.889, 3 unresolved slots.

| # | Fact | Model with entry | Data | Verdict |
|---|---|---|---|---|
| 1 | MNE share of LAC export value | 0.631 | 0.47–0.74 | calibration target, hit |
| 2 | Foreign MNEs in complex goods | 0.16 / 0.26 / 0.33 / 0.27, slope **+0.26** | +0.43 | target; needs slope 1.20 and still undershoots |
| 3 | Parents from few countries | HHI **0.314**, top **0.414** | 0.130 / 0.250 | **GENERATED**, overshoots in a 4-country world |
| 4 | Grouping affiliates raises HHI | 0.112 → 0.112 → **0.154** (×1.37) | 0.192 → 0.215 (×1.12) | **GENERATED — and it survived the rewrite** |
| 5 | MNE presence raises non-MNE exports | **−91.6%** | strongly positive | **WORSE with entry** |

**Fact 4 is the one that mattered.** Its ordering is a prediction, it needs
internalisation, and internalisation is exactly what the Gaubert–Itskhoki cutoff
cannot host. The whole theorem in §17 exists so that this fact could survive the
addition of entry. It did.

**Fact 5 — the answer to the question the brief flagged, and it is a NO.** The
conjecture was that cheaper local inputs would let more local firms clear their own
entry cutoff, pushing the sign the right way for the first time. Tested directly:
adding 12 multinationals to a sector cuts non-MNE export value by **91.6%**, against
**42%** with a fixed roster, and the count of active local plant-market pairs falls
86 → 48. **The extensive margin amplifies the contradiction rather than resolving it.**
The business-stealing effect operates on the entry margin too, and it dominates the
input-price channel. So the ≈0.15 spillover is still needed and now has *more* to
overturn, not less. Do not write that entry fixes Fact 5.

**Fact 2 note.** The foreign share is non-monotone across sectors (0.16 / 0.26 / 0.33 /
0.27) — it peaks in the second-most HQ-intensive sector rather than the most. The
linear gradient hides that; report the profile, not just the slope.

---

## 19. FINAL CONSOLIDATION (2026-08-19 evening) — read this before §17–18

§17–18 describe how entry was solved. This section records what changed when the
model was consolidated into the shareable version, and **three defects the review
found**. Where §17–18 conflict with this section, this section wins.

**Deliverables:** `mne_model.jl` (2,024 lines, one file, no packages) and
`docs/model_guide.tex` → `model_guide.pdf` (22 pp). Superseded files moved to
`superseded/`: `mne_model_no_entry.jl`, `entry_uniqueness.jl`, `entry_ge.jl`,
`entry_approaches.jl`, `entry_facts.jl`. `docs/model_guide_OLD.tex` is the
previous document.

### 19.1 Conduct is now a PARAMETER, not a commitment

The question "cannibalisation instead of Cournot" turned out to be a **false
trade-off**, and this is the most useful thing the review produced. Under nested
CES a parent that internalises its own plants charges a markup depending only on
its TOTAL share — **under both conducts** — and

```
S_g = psi^{-1}(K_g/A),   psi(S) = S mu(S)^(sigma-1)
```

is *identical*. Conduct enters ONLY through `mu(.)`. So the market solver, both
uniqueness theorems, the entry argument, the GE loop and the ownership
decomposition are all written for a general `mu` satisfying one assumption
(strictly increasing = cannibalisation). Cournot and Bertrand are two calibrations.

| | frontier for the entry theorem | kappa |
|---|---|---|
| Cournot | 0.548 | 4.00 |
| Bertrand | **0.680** (weaker hypothesis) | 0.80 |

**Cournot kept as baseline**, for two checkable reasons: the quadratic profit form
is EXACT under Cournot so P2a is an identity (verified 3e-16), and plant-level HHI
is closer to the data (0.206 vs 0.272; data 0.192). `julia mne_model.jl bertrand`
runs everything under Bertrand.

### 19.2 Defect found: the pool of potential parents was drawn UNIFORMLY

`entry_economy` drew each parent's HQ country uniformly. That is a departure from
Gaubert–Itskhoki and it broke Fact 1: **domestic MNEs came out at 0.245, LARGER
than foreign (0.248)**, against data where domestic is 0.02–0.28 and foreign
dominates everywhere.

Fix (GI's own mechanism): the pool scales with country size, `Mbar_h ∝ L_h z_h^ζ`.
The capability LAW stays identical everywhere — concentration comes from a bigger
pool having a better maximum. Domestic share falls 0.245 → 0.091 (ζ=1) → 0.015
(ζ=2). **Calibrated ζ = 1.5.** This also drives Fact 3.

### 19.3 Defect found: Fact 2 came out BACKWARDS, and the fix is structural

Foreign-share gradient on alpha_k was **−0.58** against data +0.43. The reason is a
real property of any Head–Mayer HQ input: a foreign affiliate pays its PARENT's
wage on the alpha_k share, parents sit in high-wage countries, so foreign ownership
is most expensive exactly where alpha_k is highest — and the extensive margin
amplifies it. Scaling the capability draw by alpha_k does not beat it
(E[ln xi] = 1/theta = 0.25 is too small).

Fix = the brief's own prescription, finally implemented **on the cost side**: a
stand-alone local firm carries a penalty rising in HQ intensity,
`phi_local *= exp(-hq_gap * alpha_k)`. Headquarter services are a capability not
available at arm's length.

| hq_gap | foreign share | gradient (foreign) |
|---|---|---|
| 0.0 | 0.310 | **−0.58** |
| 1.0 | 0.499 | +0.21 |
| **1.3** | **0.550** | **+0.49** (data +0.43) |
| 2.0 | 0.641 | +0.83 |

**Calibrated hq_gap = 1.3.**

### 19.4 Defect NOT fixed: the domestic half of Fact 2 is not robust

The domestic-MNE gradient is **−0.15 in a 4-country world and +0.73 in a
5-country one**. Reason: a domestic MNE is a LAC parent, so it escapes the hq_gap
penalty while still paying a LOW home wage on the alpha share — the wrong-sign
force again, now in its favour. **The model pins down that domestic MNEs are FEW,
not where they specialise.** Reported as an open item in the code and the
document; do not claim it.

### 19.5 Final calibration and scorecard

Fitted: `f = 0.0006` (to the SIZE of the export sample), edge `0.10` (Fact 1),
`hq_gap = 1.3` (Fact 2), `zeta = 1.5` (Fact 1 split). Facts 3, 4, 6 out-of-sample.

| # | Fact | Model (N=5) | Data | Verdict |
|---|---|---|---|---|
| 1 | MNE share; foreign ≫ domestic | 0.689 = 0.550 + 0.139 | 0.46–0.74 | target + **split GENERATED** |
| 2 | foreign in complex goods | +0.33 | +0.43 | target; **domestic half not robust** |
| 3 | parents from few countries | HHI 0.282, top 0.403 | 0.130 / 0.247 | **GENERATED** |
| 4 | grouping raises HHI | 0.091→0.177 (×1.95) | ×1.12 | **GENERATED — a prediction** |
| 5 | MNE presence raises local exports | **−100%** | positive | **FAILS, worse with entry** |
| 6 | distance weaker for MNEs | −0.375 / +0.724 / +0.389 | −0.164 / +0.046 / +0.067 | **GENERATED**, too large |

GE with entry: 10 random wage starts → wages agree to 1.5e-11, **1 distinct entry
configuration**, 0 unresolved slots, every market certified, largest parent share
0.376 < frontier 0.548. Accounting exact to 1e-16 off equilibrium. Runtime 7.5 min
full, 4.5 min quick.

### 19.6 What the review confirmed

Multinationals ✓ · entry of firms ✓ (677 of 1050 potential slots active) ·
cannibalisation ✓ (and it does not require Cournot) · general equilibrium ✓ ·
a solution that exists and is unique ✓ (two theorems + certificates + brute force)
· ideas from Ramondo–Rodríguez-Clare (γ), Tintelnot (d from the production
country; fixed cost of market access), Head–Mayer (α_k), Caliendo–Parro (I–O),
Gaubert–Itskhoki (granular pool, cutoff logic), Yang (multi-plant parents choosing
sets) ✓ · easy to solve ✓ (a nested sequence of one-dimensional monotone problems).

---

## 20. THE UNIQUENESS CONDITIONS ARE NOW PROVED, NOT ASSUMED (2026-08-20)

§17–19 treated the two entry conditions as assumptions checked numerically. They now
have **closed forms and proofs**. This is the single most important upgrade for
publication: a referee cannot accept "we verified it on a grid" as the foundation of
a uniqueness claim.

### 20.1 The closed forms (Cournot, eta = 1)

With `a = 2(sigma-1)`, `b = sigma-2`, and `G = Pi'/psi'`:

```
eps_psi(S)          = (1 + b S) / (1 - S)
eps_G(S)            = S [ a/(1+aS) - b/(1+bS) - sigma/(1-S) ]
eps_G + eps_psi     = aS/(1+aS) - bS/(1+bS) + (1-2S)/(1-S)
```

`(A)` concavity ⟺ `eps_G < 0`;  `(B)` substitutes ⟺ `eps_G + eps_psi > 0`.

### 20.2 The proofs

**(A) holds at EVERY share (sigma ≥ 2).** Write `eps_G = S f(S)`. Then `f(0) = a-b-sigma
= 0` identically, and `f'(S) < 0` because `a > b ≥ 0` implies
`a²/(1+aS)² > b²/(1+bS)²`. So `f < 0` on `(0,1)`, hence `eps_G < 0`. ∎

**(B) holds at EVERY share ≤ 1/2, for every sigma > 1.** In the expression above the
first two terms are strictly positive (`x/(1+x)` is increasing and `a > b`) and the
third, `(1-2S)/(1-S)`, is non-negative exactly when `S ≤ 1/2`. ∎

**And the slack at the boundary is exactly `1/sigma`**: at `S = 1/2` the third term
vanishes and the first two give `(sigma-1)/sigma - (sigma-2)/sigma = 1/sigma`.
Verified to machine precision (error 0.00e+00).

### 20.3 What this buys

The theorem's hypothesis is now **"no single parent holds more than half of any
market"** — economically meaningful, checkable in the data, and 20% from binding at
sigma = 5. Largest parent share at the solved model: **0.376**.

It is SUFFICIENT, not necessary: the exact threshold where (B) first fails is 0.545 at
sigma=5. And `S ≤ 1/2` was checked to be sufficient for other eta and under Bertrand
too (0 failures), so the eta=1 Cournot case proved above is the binding one.

`verify_conditions_analytic()` in `mne_model.jl` checks the closed forms against brute
numerical differentiation (agreement 4.35e-05, the finite-difference error) so the
proof is machine-checked, not just typed. It runs inside `julia mne_model.jl entry`.

### 20.4 The layer-by-layer honesty table (now in the document)

| Layer | Exists? | Unique? |
|---|---|---|
| one market | **proved** | **proved** (Thm 1) |
| entry | generic (integer problem, sized) | **proved** (Thm 2) |
| I–O prices | **proved** | **proved** (contraction, Banach) |
| spending and incomes | **proved** | **proved** (linear, rho(B) < 1) |
| wages | **proved** (Thm 3: continuity + homogeneity + Walras + boundary) | **proved** (Thm 4, §27) under two CHECKED conditions |

~~The only unproved uniqueness is the wage vector~~ — **superseded 2026-08-20 by
Theorem 4, §27.** Wage uniqueness now has a proof, conditional on (H1) no group above
half of any market and (H2) demand beats exposure, both of which are CHECKED at the
solution. What remains unproved is the *unconditional* statement: gross substitutes is
false in general for this model class and §27.4 exhibits the counterexample. Evidence
that still stands alongside the theorem: 10 starts → same wages to 1.5e-11 AND the
same set of operating firms.

### 20.5 Document

`docs/model_guide.tex` rewritten (26 pp): **no author names**, plain-language
explanation before every equation, a **provenance table** naming the source of each
component, and all proofs written out. `docs/model_guide_OLD.tex` is the previous
version. Note: the provenance table's exact years/titles and the CALIB values are
flagged in the document as needing verification against sources before circulation.

---

## 21. CITATIONS VERIFIED (2026-08-20) — two were wrong

Checked against published sources. **Two corrections that were live in §14, §17 and
the document:**

- ~~Gaubert & Itskhoki (2020)~~ → **Gaubert & Itskhoki (2021)**, JPE 129(3), 871–939.
- ~~Blanchard, Bown & Johnson (2024)~~ → **(2026)**, ReStud 93(1), 181–214.

**One attribution was loose and is fixed:** the headquarter input share `alpha_k` was
credited to "Head–Mayer". Head & Mayer (2019, AER 109(9), 3073–3124, *Brands in
Motion*) is about MP frictions in the car industry, not the HQ-intensity concept. The
HQ input belongs to **Antràs (2003)**, QJE 118(4), 1375–1418 and **Antràs & Helpman
(2004)**, JPE 112(3), 552–580. Head & Mayer now sits next to Ramondo–Rodríguez-Clare
on the MP friction, which is what it actually measures.

**Verified as already correct:** Atkeson & Burstein (2008) AER 98(5) 1998–2031 ·
Ramondo & Rodríguez-Clare (2013) JPE 121(2) 273–322 · Tintelnot (2017) QJE 132(1)
157–209 · Caliendo & Parro (2015) ReStud 82(1) 1–44 · Gaubert, Itskhoki & Vogler
(2021) JME 121, 95–112 · Javorcik (2004) AER 94(3) 605–627 · Brecher & Bhagwati
(1981) JPE 89(3) 497–511 · Blanchard (2007) BEJEAP 7(1) · Blanchard (2010) JIE 82(1)
63–72 · Blanchard & Matschke (2015) REStat 97(4) 839–854 · Itskhoki & Mukhin (2025)
NBER WP 33839, *The Optimal Macro Tariff*.

**Yang (2023)** is *"Location Choices of Multi-plant Oligopolists: Theory and Evidence
from the Cement Industry"*, working paper. Worth knowing: **the paper itself uses the
word "cannibalization"** for the mechanism, which is the term this project adopted
independently. Also added to the reference list: Milgrom & Shannon (1994) Econometrica
62(1) 157–180 and Novshek (1985) ReStud 52(1) 85–98, the standard results behind the
two steps of Theorem 2.

Citation years in `mne_model.jl` were corrected to match.

## 22. BALANCE AUDIT ON THE CALIBRATED WORLD (2026-08-20)

§18.3 audited the small demo economy. Redone on the economy actually used, under
**both** conducts, at three wage vectors including one spread over e², and at the
solution:

| worst error over three wage vectors | Cournot | Bertrand |
|---|---|---|
| sales = wages + inputs + tariffs + profit | 3.0e-16 | 9.0e-16 |
| wage bill = production + entry-cost labour | 1.2e-15 | 2.4e-15 |
| **Walras, off equilibrium** | 4.8e-16 | 5.9e-16 |
| world income = world spending | 9.5e-16 | 3.9e-16 |
| country budget | 2.6e-16 | 4.0e-16 |
| all five at the solution | ≤1.0e-15 | ≤1.1e-15 |

Also at the solution: every market certified unique under both conducts; largest parent
share **0.344** (Cournot, frontier 0.548) and **0.420** (Bertrand, frontier 0.680) — so
the theorem's hypothesis holds with room to spare either way; unresolved integer slots
2/1050 and 1/1050.

**The model is balanced under both conducts.** That is a stronger statement than §18.3
made, because it is the calibrated economy rather than a synthetic one.

## 23. DOCUMENT (2026-08-20)

`docs/model_guide.pdf` — **27 pp, ONE version only** (`model_guide_OLD.tex` deleted at
Sebastián's request). No author names anywhere; the only LaTeX warning is
"No uthor given", which is the intended state. Plain-language explanation before
every equation, provenance table with verified years, full reference list, and all
proofs written out. Balance table now reports exact maxima from §22 rather than
rounded figures.

---

## 24. DOCUMENT: FINAL EXPOSITION PASS (2026-08-20)

`docs/model_guide.pdf` — **31 pp, one version, no author names.** This pass was about
exposition, not content: no result changed. What changed:

**Every central equation now carries the same three-part gloss, in this order:**
`\says{}` *What it says* (plain words) · `\why{}` *Why we model it this way* ·
`rom{}` *Where it comes from* (the source paper). Applied to all nine numbered
equations: markup, profit, adding-up, strength/share map, delivered cost, entry cost,
income, the decomposition, and the error term. 19 / 12 / 7 instances respectively.

**Everything is defined.** A full notation table (§4) lists every symbol used anywhere
in the document, grouped by indices / demand / competition / the two tractability
objects / costs / countries. Proof-internal symbols (rho, X_g, B_g, e, C, a, b, T(A),
D(A), z(w), epsilon, B) are each defined at first use — several were previously
introduced without definition.

**The six facts are now stated up front (§3),** with a one-line description each, plus
which are fitted and which are predictions. Previously they were referenced by number
from p.5 but not defined until Part V, which was a real readability failure.

**Added:** a formal Definition 1 of equilibrium (six numbered conditions); a one-page
summary table of all equations in the appendix; a note on which facts are out-of-sample.

**Structure:** I question/notation/sources · II model equation by equation · III
existence and uniqueness · IV the result · V model vs data · VI limits · appendix
(equation summary, references, how to run).

Only LaTeX warning is `No uthor given` — the intended state.

### 24.1 State of the deliverables

| File | State |
|---|---|
| `mne_model.jl` | 2,040 lines, one file, no packages. `core`/`entry`/`ge`/`facts`/`bertrand`/`quick` |
| `docs/model_guide.pdf` | 31 pp, the shareable document |
| `docs/model_guide.tex` | source, 1,745 lines |
| `superseded/` | five earlier prototypes, all folded in |

Re-verified after the citation edits: ownership identity 3.4e-16, quadratic form
3.2e-16, miss identity 1.3e-12; entry uniqueness 40/40 with 0 false certificates;
closed forms 4.35e-05 against numerics; (A) and (B) 0 violations; slack at 1/2 exact.

---

## 25. THREE-PAGE SUMMARY, AND A TOOLING TRAP THAT BIT (2026-08-20)

**New file: `docs/model_summary.pdf` (3 pp) + `.tex`.** A condensed companion for
reviewers: the question, the mechanism, the seven equations, both theorems with
proof sketches, the ownership proposition with its proof, the proved/not-proved
status, and the facts scorecard. It is a *different document*, not a second copy of
the 31-page guide, so it does not violate the "one version" rule --- the guide still
has exactly one version.

### 25.1 TOOLING TRAP — this cost several rebuilds, add it to §5

Editing `.tex` via `python - <<'PYEOF'` heredocs **silently converts backslash
escapes inside single-backslash Python string literals**. Observed damage:

| written | became | rendered as |
|---|---|---|
| `
oindent` | newline + `oindent` | literal "oindent" |
| `	extbf{x}` | TAB + `extbf{x}` | literal "extbf{x}" |
| `	imes` | TAB + `imes` | literal "imes" |
| `rac12` | FF + `rac12` | LaTeX error U+000C |
| `arepsilon` | VT + `arepsilon` | LaTeX error U+000B |
| `
ef` | CR + `ef` | broken reference |

Two of these (`	`, `
`) are invisible to a control-character scan because tabs
and newlines are legal in the file. **Rule: when writing LaTeX (or Julia) command
sequences from Python, build every backslash as `BS=chr(92)` and concatenate.**
Never type `'\command'` in these heredocs.

~~`model_guide.tex` was audited for the same damage and is clean.~~ **FALSE — it was
not.** On 2026-08-20 the file would not compile: seven table-row terminators in the
"six facts" table, and one in the summary, had been written as a two-character escape
in a Python string and arrived as a SINGLE backslash, which LaTeX reads as a space
command rather than a row break. A control-character scan cannot see this, and neither
can a reader. **Add to the audit: check that every tabular row still ends with a
doubled terminator, not just that there are no stray control characters.** Repaired and
both documents rebuilt (guide 36 pp, summary 4 pp).

### 25.2 State of the deliverables

| File | Pages / lines | Purpose |
|---|---|---|
| `mne_model.jl` | 2,040 lines | the model; `core`/`entry`/`ge`/`facts`/`bertrand`/`quick` |
| `docs/model_guide.pdf` | 31 pp | the full document, one version, no names |
| `docs/model_summary.pdf` | 3 pp | condensed companion for review |

---

# 26. WAGE UNIQUENESS — SOLVED, CONDITIONALLY (2026-08-20)

**Status change.** Wages were the one layer whose uniqueness was only verified. They
now have **Theorem 4**: a proof, with two hypotheses that are *checked* rather than
assumed, plus an explicit **counterexample** showing the second hypothesis cannot be
dropped. What is NOT proved is the unconditional statement — and that is not a gap in
the argument, it is a fact about the model: gross substitutes is false in general here.

Code: `wage_uniqueness.jl` (committed output `run_wage_uniqueness.txt`) and
`verify_wage_theorem()` / `wage_hypotheses()` / `passthrough()` in `mne_model.jl`
PART 7b, which run inside `julia mne_model.jl ge`. Document: `docs/model_guide.tex`
Part III, Layer 5. The earlier probe `uniqueness_probe.jl` is superseded but kept — its
Part 1 threshold table is still the quickest way to see the one-half result.

## 26.1 The object

Write `W_n = w_n LD_n` for country n's wage bill and `M[n,j] = d ln W_n / d ln w_j`.
Homogeneity of degree one in wages makes **every row of M sum to one**, and since
`z_n = W_n/w_n - L_n`,

```
gross substitutes  <=>  M[n,j] > 0 for all n != j
                   <=>  a country's wage bill rises when ANY OTHER country's wage rises
```

That reformulation is what made the problem tractable: it puts the whole question into
one matrix with unit row sums, and it is the object every route below acts on.

## 26.2 LEMMA 3 — pass-through is a row-stochastic matrix. UNCONDITIONAL.

`d ln P / d ln w` is **non-negative with unit row sums**, at any wage vector, for both
conducts, any α, any ν < 1, any σ > η ≥ 1. Proof in three steps, each an averaging
argument:

1. **Cost.** `ĉ_a = (1-ν)[α ŵ_h + (1-α) ŵ_l] + ν Σ ω ê P̂` — coefficients non-negative,
   summing to one. The head-office term lives here: it changes *which* wage a plant is
   exposed to, not the fact that the weights are a probability distribution.
2. **Market.** From `S_g = μ(S_g)^(1-σ) K_g / A`, `Ŝ_g = -χ_g (ĉ_g - P̂)` with
   `χ_g = (σ-1)/(1+(σ-1) ε_μ) > 0`. Shares sum to one ⇒ `Σ_g S_g Ŝ_g = 0` ⇒
   **`P̂ = Σ_g ξ_g ĉ_g` with `ξ_g = S_g χ_g / Σ S χ` — a convex combination.**
   *This is where the markup channel goes: it changes the weights and nothing else.*
3. **Stack.** `P̂ = Γ ŵ + N P̂`, Γ,N ≥ 0, `(Γ+N)1 = 1`, row sums of N alone = ν_k < 1
   ⇒ ρ(N)<1 ⇒ `Λ = (I-N)^{-1} Γ ≥ 0` and `Λ1 = (I-N)^{-1}(1-ν) = 1`. ∎

**So §26-old's channels 1 and 2 cannot make a price index fall when a wage rises.**
Machine-checked: min entry +3.1e-02, row sums exact to 1.3e-15, analytic vs numerical
1.2e-07.

## 26.3 LEMMA 4 — the three-term decomposition, and two corollaries

```
V̂_a = Ê_nk  -  (σ-1)(λ_a - λ_g)  -  [1 - ε_μ(S_g)] χ_g (λ_g - Λ_nk)
        market      within group          between groups
```

- **Corollary A — the one half, again.** The between-group coefficient
  `[1-ε_μ]χ_g ≥ 0` **iff ε_μ ≤ 1**, which under Cournot η=1 is **exactly S_g ≤ 1/2**.
  *One condition — no group above half of any market — carries Theorem 2 AND Theorem 4.*
  The §26-old lead was right.
- **Corollary B — HEAD-OFFICE NEUTRALITY, and this is the surprise.** All of a group's
  plants report to the same head office (it is a property of the parent — state this,
  it is the corollary's one hypothesis), so that country is paid α_k of the group's
  **entire** bill, and `Σ_{a∈g}(s_a/S_g)(λ_a-λ_g)=0`.
  **The within-group reallocation term — the multinational linkage that was supposed to
  be the obstacle — drops out of the head-office channel completely.**

Machine-checked: share response 2.2e-07, plant bill 2.1e-07, country wage bill 1.4e-07.

## 26.4 THEOREM 4

Raise `w_m` alone. Let `ω_{a,n'}` be the share of country n's wage bill paid by plant a
out of its sales in market (n',k) (production and entry cost), and

```
Ēn^(m) = Σ ω · Ê_{n'k}   (entry-cost part: λ^F_a)         the demand term
Ξ_nm   = Σ ω · [ (σ-1)(λ_a-λ_g)^+ + (1-ε_μ)χ_g(λ_g-Λ_{n'k})^+ ]   the exposure penalty
```

**(H1)** `ε_μ(S_g) ≤ 1` in every market (⟺ no parent above 1/2 under Cournot η=1).
**(H2)** `Ēn^(m) > Ξ_nm` for every ordered pair n ≠ m.

> **THEOREM 4.** With the operating set fixed and η = 1: if (H1) and (H2) hold at every
> wage vector of spread ≤ 2R, then gross substitutes holds there, and there is **at most
> one equilibrium wage vector of spread ≤ R**.

Proof: (H1) makes both reallocation coefficients non-negative, so each negative
contribution is at least minus its own coefficient times the positive part of its gap;
`M[n,m] ≥ Ē - Ξ > 0`; then the Arrow–Block–Hurwicz argument (written out in the guide).
The spread bookkeeping: every point on the ABH path has spread ≤ the sum of the spreads
of the two endpoints, hence the factor of two.

**At the calibrated economy, both conducts:**

| | Cournot | Bertrand |
|---|---|---|
| largest parent share (H1 needs ≤ 0.50) | 0.344 | 0.420 |
| largest ε_μ (H1 needs ≤ 1) | 0.525 | 0.218 |
| (H2) ordered pairs satisfied | 20 / 20 | 20 / 20 |
| (H2) tightest margin | +0.0237 | +0.0254 |
| gross substitutes, min off-diagonal | +0.0999 | +0.0915 |
| index +1 | yes | yes |

## 26.5 THE COUNTEREXAMPLE — gross substitutes is NOT a theorem here

Three countries, one sector. Country 1 has local firms; **country 2 owns parents whose
only plants are in country 3**, so every euro country 2 earns is a head-office payment
for production abroad. Raise `w_3`: those parents get dearer, lose share to country 1,
and country 2's whole income falls with them. Enough local rivals are put in each
market that **no parent holds more than a third of any market — (H1) is satisfied
throughout and cannot be what fails.**

```
design                                           maxS    M[3,2] min offdg  H2 slack index
14 rivals in 1, 10 in 3, alpha 0.15, 3 parents  0.177   -0.2192   -1.4288   -1.4288    +1
10 rivals in 1,  6 in 3, alpha 0.30, 2 parents  0.248   -0.3096   -0.8270   -0.8270    +1
10 rivals in 1,  6 in 3, alpha 0.50, 2 parents  0.293   -0.3546   -0.3920   -0.4482    +1
10 rivals in 1,  6 in 3, alpha 0.80, 2 parents  0.308   +0.0975   +0.0565   -0.3362    +1
 6 rivals in 1,  3 in 3, alpha 0.30, 2 parents  0.291   -0.2694   -0.6344   -0.6344    +1
10 rivals in 1,  6 in 3, alpha 0.30, Bertrand   0.325   -0.5030   -1.2263   -1.2263    +1
10 rivals in 1,  6 in 3, alpha 0.30, nu = 0.55  0.227   -0.1145   -0.5116   -0.5116    +1
```

**THE OBVIOUS GUESS IS WRONG, AND THIS IS THE PART TO REMEMBER.** It is not that alpha
is too large. Inside this family the failure is **worst at alpha = 0.15 and has
disappeared by alpha = 0.80** — with a big head-office share the group's cost is
dominated by its OWN parent's wage, so the host's wage barely moves it. And in ordinary
ownership geographies GS never failed at any alpha tested (175 points each at
alpha = 0, 0.25, 0.55, 0.9: zero failures). What breaks GS is a country whose income is
**concentrated on one foreign partner**. The discriminating statistic is bilateral
exposure `psi[n,m]` = share of n's labour income paid by m-linked plants (host or
parent):

```
counterexample : psi = 1.000
calibration    : max psi = 0.380 (Cournot)   0.492 (Bertrand)
```

**Trap recorded:** the first version of this counterexample reported the largest share
in market 1 only (0.371) and claimed (H1) held; the max over ALL markets was 0.52-0.82,
so (H1) did NOT hold and the claim was false. `wage_hypotheses().Smax` is the max over
every market and is the number to quote.

## 26.6 WHAT UNIQUENESS ACTUALLY NEEDS — and it survives the counterexample

GS is sufficient, not necessary. The real requirement is **index +1 at every
equilibrium** (indices sum to +1). Two things worth recording:

- **Walras' law makes the index numeraire-free.** At an equilibrium
  `Σ_n w_n L_n (M[n,m] - δ) = 0`, so the wage-bill vector is the LEFT NULL VECTOR of
  `I - M`; since `adj(I-M) = 1·(wL)'·c`, **every diagonal cofactor of `I-M` is
  proportional to a wage bill**, so all N choices of numeraire give the same index sign.
- **In 50 random economies from the counterexample family: GS fails at the equilibrium
  in 34, index is +1 in 50/50, numeraire-independent in 50/50, and 25 dispersed starts
  find a single equilibrium in every one. Zero cases of multiplicity, anywhere.**

So the counterexample kills the standard *argument*, not the *conclusion*.
**The remaining open problem is much smaller than the one this section used to state:
prove index +1 directly.** It needs `det[(I-M)_{-1,-1}] > 0`, equivalently that no
non-unit real eigenvalue of M exceeds one — economically, that no wage perturbation is
self-reinforcing.

## 26.7 ROUTES TRIED, AND WHERE EACH GOT TO

| route | outcome |
|---|---|
| **1. Bound the markup channel, push GS through the chain** | **PARTLY WORKED — this is Theorem 4.** The chain is Lemma 3 → Lemma 4 → (H1)+(H2). The markup channel is fully bounded and the ε_μ<1 ⟺ S<1/2 lead was correct. The head-office channel turned out to be *neutral* for the parent's country (Corollary B), which was not expected. What could NOT be bounded is a country's *concentration* of income on one partner, and that is the counterexample. **Global unconditional GS: DEAD, with a counterexample, not merely unproven.** |
| **2. Index theory** | **Right requirement, still open.** Established: index +1 ⟺ det of the reduced Jacobian has sign (-1)^(N-1); Walras ⇒ numeraire-independent; GS ⇒ index +1. Verified +1 in 50/50 economies *including the 34 where GS fails*. Not proved analytically: doing so needs a bound on M that is not GS, and every norm bound tried (∞-norm; the wage-bill-weighted 1-norm, the natural one because wL is the left null vector) collapses back to "M ≥ 0", i.e. to GS itself. That is a real obstruction, recorded here so it is not re-tried blind. |
| **3. Restricted cases + continuity** | **Not needed, and it would not have helped.** α=0 does not restore GS as a theorem — GS holds at α=0 numerically but so does it at α=0.9, so α is not the margin that matters (§26.5). The restricted case that *is* informative is the counterexample's, and it goes the other way. |
| **4. Computer-assisted / interval arithmetic** | **Not attempted in full; partly substituted.** Instead of interval Newton, (H1) and (H2) are evaluated on a grid of the wage box (§26.4 table, PART 3 of the probe). That is a certificate on the grid, not on the continuum, and the document says so. A genuine interval-arithmetic enclosure would have to propagate through the price contraction and the market Newton solve; it remains the honest way to close the last gap and is now a *smaller* job, because Theorem 4 reduces what must be enclosed to two scalar inequalities per country pair. |
| **5. (tried, not on the list) Nonlinear Perron–Frobenius / Hilbert metric on the wage-bill map** | **Dominated, dropped.** An order-preserving, degree-one-homogeneous map has a unique positive eigenvector; equilibrium is exactly `W(w) = w⊙L`. But order preservation needs `M ≥ 0` INCLUDING the diagonal, whereas ABH needs only the off-diagonals. Strictly stronger hypothesis, same proof burden. Recorded so it is not re-derived. |

## 26.8 HOW FAR THE VERIFIED REGION REACHES

```
conduct   spread    pts    H1 slack    H2 slack min M offdg  GS fails
cournot   0.00        1     +0.4749     +0.0237     +0.0999         0
cournot   0.15       20     +0.4504     +0.0143     +0.0970         0
cournot   0.30       20     +0.4352     +0.0084     +0.0952         0
cournot   0.50       20     +0.4026     -0.0079     +0.0914         0
cournot   0.80       20     +0.3771     -0.0392     +0.0818         0
bertrand  0.00        1     +0.7815     +0.0254     +0.0915         0
bertrand  0.15       20     +0.7573     +0.0157     +0.0882         0
bertrand  0.30       20     +0.7404     +0.0098     +0.0847         0
bertrand  0.50       20     +0.7057     -0.0060     +0.0723         0
bertrand  0.80       20     +0.6663     -0.0442     +0.0531         0
```

Reading. (H1)+(H2) are verified out to spread 0.30 under both conducts, so by the
corollary there is at most one equilibrium of spread <= 0.15. GS ITSELF is verified
directly out to spread 0.80 (zero failures at any of the 20 points per row), which by
the same corollary gives at most one equilibrium of spread <= 0.40. Both are grid
certificates, not continuum ones. The value of (H1)+(H2) is that they are ANALYTIC
sufficient conditions, not that they reach further — they reach less far, because
they discard the favourable terms.

(H1) and (H2) are *sufficient* for GS, so they are conservative; the last column shows
how much the sufficient condition gives up relative to GS itself.

## 26.9 WHAT CHANGED IN THE REPOSITORY

`mne_model.jl` PART 7b (new: `eps_markup`, `chi_share`, `passthrough`,
`cost_response`, `wage_hypotheses`, `wage_bill_elasticity`, `verify_wage_theorem`),
called from `run_ge`. `wage_uniqueness.jl` + `run_wage_uniqueness.txt` (new).
`docs/model_guide.tex`: Theorem 4, Lemmas 3–4, Corollaries, Proposition
(counterexample), layer-table row 5, limits list, and the whole "what is not proved"
subsection rewritten. `docs/model_summary.tex`: proof-status section rewritten around
Theorem 4. `CLAUDE.md`: §12.2, §13.2, §18.4, §20.4 honesty table, and this section.
Brain brief P3: §3, scorecard, decision log.

---

# 27. WHERE MNE EXPORTS ACTUALLY GO — THE MISSING FACT, NOW MEASURED (2026-08-20)

This was the highest-priority empirical item in the brief and the one that decides the
sign of the headline policy result. §2's warning: *"if US-owned LAC affiliates ship
mostly back to the US, Brecher–Bhagwati is strong; if they serve third markets while
hiring LAC labour, the wage channel wins."* **The answer is: they serve third markets.**

Code: `Orbis_DNB_Customs_Final\src\09_mne_export_destinations_build.do` (reads the
20.5 GB transaction file once, saves a 202k-row cube) and
`10_mne_export_destinations_tables.do` (the tables). Outputs and the run log are in
`Orbis_DNB_Customs_Final\output\tables\`. The MNE definitions are copied verbatim from
`05_descriptive_stats.do`, so these numbers sit on the same classification as every
published table in the project.

## 27.1 Coverage

Total export value in the ten-origin sample, 2006–2022: **2,586.6 bn USD FOB**.

| | value (bn) | share |
|---|---|---|
| matched to the corporate database (`MNE_total`) | 1,678.3 | 0.649 |
| foreign multinational affiliate (`MNE_ext`) | 1,267.1 | 0.490 |
| domestic multinational (`MNE_dom`) | 396.0 | 0.153 |
| matched but no parent country recorded | 15.2 | 0.006 |

**The "~half of foreign-MNE export value has no recorded parent" problem in §6 of the
brief is GONE** in this vintage of the file: only 15.2 bn of 1,678.3 bn matched value
lacks a parent country. The Apr-2026 rebuild added `parent_country` /
`parent_country_conf` / `parent_country_flagged` and closed it. Use `iso3_parent`;
`parent_country` is populated mainly where `iso3_parent` was missing, so the two agree
on only ~10% of rows and are complements, not duplicates.

## 27.2 THE HEADLINE

```
foreign-MNE export value                :  1,267.1 bn
of which shipped TO THE PARENT'S COUNTRY:    116.1 bn
SHARE                                   :    0.0916
```

**Only 9.2% of foreign-MNE export value from LAC goes to the parent's own country.**
The boomerang story has essentially no support in the aggregate.

**But it is very heterogeneous, and the split is economically legible:**

| origin | share to parent country |
|---|---|
| SLV | **0.402** |
| DOM | **0.300** |
| CRI | **0.281** |
| ECU | 0.204 |
| COL | 0.075 |
| PER | 0.059 |
| ARG | 0.058 |
| URY | 0.057 |
| CHL | 0.053 |
| PRY | 0.023 |

The Central American / Caribbean maquila economies do ship home; the South American
commodity exporters do not, by a factor of five. **That is a better paper than a single
average**, and it maps onto the model: the boomerang case and the wage case are both
present in the data, in different countries.

By year the share is stable at 0.08–0.10 through 2011–2019. Ignore 2006–08 (0.34–0.38)
and 2020–22 (0.04): those years contain only SLV and only CHL/COL/PRY respectively, so
the movement is sample composition, not behaviour.

## 27.3 WHERE THEY GO INSTEAD — and it is the opposite of the story

Destination shares of export value, foreign-MNE affiliates versus everyone else:

| destination | MNE share | non-MNE share |
|---|---|---|
| CHN | **0.150** | 0.113 |
| USA | **0.139** | **0.250** |
| BRA | **0.104** | 0.049 |
| JPN | 0.041 | 0.023 |
| CHL | 0.032 | 0.028 |

**Foreign multinational affiliates in LAC are LESS US-oriented than local exporters
(0.139 against 0.250) and much more China- and Brazil-oriented.** Whatever the
affiliates are there for, it is not serving the parent's home market.

## 27.4 THE NUMBER THE OPTIMAL-TARIFF CORRECTION ACTUALLY NEEDS

For destination X, `theta_X` = share of X's imports from these ten LAC origins that is
produced by affiliates of X-headquartered parents. This is the ownership share that
enters the Brecher–Bhagwati / Blanchard correction, market by market — the object the
whole measurement contribution is about.

| destination | LAC imports (bn) | of which own-affiliate (bn) | theta |
|---|---|---|---|
| USA | 506.0 | 67.8 | **0.134** |
| CHN | 339.5 | 1.2 | 0.004 |
| BRA | 196.8 | 5.5 | 0.028 |
| JPN | 82.1 | 3.5 | 0.043 |
| CAN | 61.7 | 15.7 | **0.255** |
| ESP | 69.3 | 4.6 | 0.067 |
| DEU | 50.3 | 4.3 | 0.086 |
| GBR | 31.3 | 2.1 | 0.068 |

**theta_USA = 0.134.** Canada is the outlier at 0.255 (mining). China, the largest
single buyer of LAC MNE exports, is 0.004.

## 27.5 WHAT THIS MEANS FOR THE MODEL — read with §2

- The ownership correction to the US optimal tariff runs on `theta = 0.13`, not on
  anything close to a half. **Brecher–Bhagwati is weak here.**
- Which means the GE model's uncomfortable result — home ownership raises the optimal
  tariff, because the wage channel dominates — **is the empirically relevant case for
  the US**, not an artefact. §2's "CRITICAL WARNING" now has a sign attached to it.
- P2b is still not proved and still must not be written as a general claim. What has
  changed is that the *empirical* precondition for the wage channel is satisfied.
- The origin split (SLV/DOM/CRI at 0.28–0.40 against ARG/CHL/PER at 0.05–0.06) is a
  real result and probably the sharpest framing available: **"when does home ownership
  raise rather than lower the optimal tariff"** is answerable in the cross-section.

## 27.6 CONDUITS ARE STILL A PROBLEM — §6 item 2 stands

Top parent countries by foreign-MNE export value, and how much each ships home:

| parent | share of MNE export value | share shipped home |
|---|---|---|
| USA | 0.275 | 0.194 |
| GBR | 0.179 | **0.009** |
| CAN | 0.065 | 0.192 |
| NLD | 0.051 | 0.015 |
| **LIE** | 0.044 | 0.000005 |
| DEU | 0.037 | 0.091 |
| CHE | 0.032 | 0.003 |
| PAN | 0.031 | 0.037 |

GBR at 17.9% of value with 0.9% shipped home is not an economic owner, it is a holding
structure — and LIE (Liechtenstein) at 4.4%, CHE 3.2%, PAN 3.1%, LUX 1.0%, CUW 0.8%,
CYM 0.8%, IRL 0.7% are conduits by construction. **Conduit jurisdictions are roughly
11% of foreign-MNE export value.** Reallocating them to the ultimate controlling parent
would move `theta_USA` UP and is now the blocking task for any theta that goes in a
table.

## 27.7 WHAT IS STILL NOT MEASURED

**The intra-firm / related-party share (item (b) of the brief's priority 1) is NOT
computable from this dataset.** The customs records carry no related-party flag, and
"destination equals parent country" is a different object — it is a lower bound on
intra-firm trade at best. Getting (b) needs either a customs related-party field (US
imports have one; the LAC export declarations here do not) or the Orbis affiliate
roster, so that a shipment can be scored against whether the group owns an entity in
the destination. The second is feasible with the D&B/Orbis files already on disk and is
the natural follow-up.

---

# 28. CONDUIT REALLOCATION TO THE ULTIMATE PARENT — DONE, AND IT BARELY MOVES (2026-08-20)

§6 item 2 of the brief called this **blocking for any θ that goes in a table**: GBR is
17.9% of foreign-MNE export value and ships 0.9% of it home; LIE 4.4%; CHE 3.2%;
PAN 3.1%. The worry was that the top "owners" are holding structures, not economic
owners, and that a paper about US tariffs cannot rest on that.

**It is now done, and the answer is reassuring in a way that has to be stated carefully:
the correction is real, it is worth making, and it does not change the headline.**

Code: `Orbis_DNB_Customs_Final\src\11_ultimate_parent_build.do` (reads the 20.5 GB file
once, saves `mne_ucp_cube.dta`, 278k rows, keeping every ownership source separately so
any rule can be applied afterwards without re-reading) and `12_ultimate_parent_tables.do`.
Log and CSVs in `output\tables\` (`V1`–`V3`, `ultimate_parent.log`).

## 28.1 Three owner concepts, carried side by side

| | definition |
|---|---|
| **A** `iso3_parent` | the status quo: the parent country as recorded |
| **B** `iso3_ucp` | naive chain: first available of GUO50 > GUO25 > D&B global ultimate > parent |
| **C** `iso3_ucp2` | **improved**: the first **non-conduit** country among those four |

Sources and their reach, as a share of value that has a recorded parent:
Orbis **GUO50C 0.672**, **GUO25C 0.714**, D&B `globalultimatecountry` **0.462**.
Orbis codes are ISO2 and D&B gives a country *name*, so both are crosswalked to ISO3
first (`iso2_to_iso3.csv`, `cname_to_iso3.csv`, built from `pycountry`).

**Concept B is not good enough, and the flows show why.** The naive chain happily hands
you a holding company: it moves DEU → PAN (1.8 bn), CAN → PAN (1.1 bn), NLD → LUX
(1.2 bn), LIE → LUX (1.2 bn), SGP → LUX (1.6 bn), JOR → CYM (4.1 bn). **A rule that
always takes the GUO makes the conduit problem worse in places.** Concept C is the one
to use.

## 28.2 What actually moves

```
total export value                                :  2,586.6 bn
value with a recorded parent                      :  1,663.2 bn
  concept B moves                                 :    139.8 bn   (8.4% of parented)
  concept C moves                                 :    127.9 bn   (7.7%)
value whose PARENT is a conduit jurisdiction      :    157.9 bn
  STILL a conduit after the full chain (residual) :    136.0 bn   (86% of it)
```

**The single biggest and most defensible correction is GBR → AUS, 52.5 bn** — the
dual-listed mining groups. After it, GBR's owner share falls **0.135 → 0.109** and AUS
rises **0.007 → 0.038**. PAN falls 0.023 → 0.016, LUX 0.007 → 0.005.

But note what this says about GBR: **76% of GBR-parented value has GBR as its genuine
ultimate owner.** GBR is not a shell problem. It is LSE-listed mining and commodity
groups whose *residence* is British. That is a nationality-of-listing issue, and no
ownership chain can fix it — if it matters, it needs a different argument (residence of
the controlling shareholders), not a better GUO field.

## 28.3 The number that matters barely moves

`theta_X` = share of X's imports from the ten LAC origins produced by X-owned affiliates.

| destination | A (parent) | B (naive) | C (non-conduit) | unresolved |
|---|---|---|---|---|
| USA | 0.1341 | 0.1353 | **0.1356** | 0.034 |
| CHN | 0.0035 | 0.0029 | 0.0029 | 0.091 |
| BRA | 0.0281 | 0.0325 | 0.0325 | 0.021 |
| CAN | 0.2549 | 0.2506 | 0.2506 | 0.014 |
| DEU | 0.0861 | 0.0771 | 0.0771 | 0.066 |
| ESP | 0.0667 | 0.0759 | 0.0759 | 0.054 |
| JPN | 0.0428 | 0.0429 | 0.0429 | 0.139 |

And the headline share of foreign-MNE exports going to the owner's own country:
**0.0916 (A) → 0.0913 (B) → 0.0914 (C).** It does not move at all.

**State θ_USA as a RANGE, not a point.** The 3.4% of US-destined LAC imports whose owner
cannot be resolved past a conduit is the disclosure. If none of it is US-owned,
θ_USA = 0.136; if all of it is, θ_USA = 0.170.

```
theta_USA  in  [0.136, 0.170]
```

Either end is small. **The conduit worry does not rescue Brecher–Bhagwati**, and §27's
conclusion stands unchanged: the ownership correction to the US optimal tariff runs on a
θ near 0.14, so the wage channel is the empirically relevant one.

## 28.4 What is now closed and what is not

- **Closed.** "Reallocate conduits to the ultimate controlling parent and show
  robustness" — done, with three concepts and a disclosed residual. It is no longer
  blocking for θ.
- **Not closed, and now the sharper worry.** Residence versus control: GBR (10.9% of
  owner value even after the chain) and CHE (2.6%) are real jurisdictions hosting groups
  whose economic owners may be elsewhere. That is not fixable with GUO fields. If a
  referee presses, the answer is the range in §28.3, not a better chain.
- **Still open from §27.7.** The intra-firm / related-party share, which needs the Orbis
  affiliate roster rather than the customs file.

---

# 29. A SIMPLER MODEL THAT IS EASIER TO PROVE UNIQUE — AND FITS BETTER (2026-08-20)

**The question:** can the model be simplified so that its wage equilibrium is unique for
a better reason, without losing the stylized facts? **Yes, and the simplification that
works is not the obvious one.**

Code: `simple_model.jl` (+ `run_simple_model.txt`), `reallocation_floor()` in
`mne_model.jl` PART 7b, and the switch `world_economy(...; hq_cost = false)`.
Theorem 5 is in `docs/model_guide.tex` Part III, Layer 5.

## 29.1 Dropping Caliendo–Parro is the SECOND step, not the first

The counterexample of §26.5 runs entirely on one modelling choice: a factory pays its
**parent's** wage on the head-office share `alpha_k` of its costs. Nothing else in the
model does that, and nothing else in the counterexample matters.

`alpha_k` does **two** jobs and only one of them is the culprit:

| job | what it does | is it the problem? |
|---|---|---|
| **cost share** | the fraction of a plant's cost paid at its PARENT's wage | **yes — this is the whole counterexample** |
| **complexity index** | the local-firm penalty `exp(-hq_gap alpha_k)` and the parent capability gradient `xi(1 + adv_slope alpha_k)` | no — it never touches wages |

`world_economy(...; hq_cost = false)` keeps the second and drops the first. **The
productivity draws are bit-for-bit identical**, so nothing behind Facts 1–2 moves for a
mechanical reason (checked: `phi` identical).

## 29.2 THEOREM 5, and what each simplification buys

> **THEOREM 5.** With head-office services a capability (`alpha_k = 0` in the cost
> function), `nu = 0`, `eta = 1`, and **(H1)** `eps_mu <= 1`: every factory paying
> country `n != m` is LOCATED in `n`, so `lam_a = 0` exactly, and its whole reallocation
> term is bounded below by `(sigma-1) min{lam_g, Lam} >= 0`. Hence
> `d ln W_n / d ln w_m >= Ebar_n^(m)` and gross substitutes follows from a pure DEMAND
> condition — no market's nominal spending falls.

Proof: `(H1)` gives `0 <= (1-eps_mu) chi_g <= sigma-1`; then the two cases
`lam_g >= Lam` and `lam_g < Lam` both give the bound. Written out in the guide.

**With `nu > 0` the same conclusion needs one extra condition,
`lam_a <= min(lam_g, Lam)`** — country n's input prices do not rise by more than its
group's or its market's. `nu = 0` makes that vacuous. **That is exactly what dropping
Caliendo–Parro buys: it turns a conditional statement into an unconditional one. It is
not what breaks anything.**

At the calibrated economy (N=5, K=4, with entry), all three variants:

| model | maxS | max eps_mu | **floor** | (*) violation | min GS | index |
|---|---|---|---|---|---|---|
| baseline: HQ cost, nu = .55 | 0.344 | 0.525 | **−1.786** | 0.45 | +0.0999 | +1 |
| HQ a capability, nu = .55 | 0.297 | 0.423 | **−0.106** | 0.030 | +0.1516 | +1 |
| **HQ a capability, nu = 0** | 0.305 | 0.440 | **+0.0136** | **0.0** | **+0.3599** | +1 |

The floor is the smallest single-factory reallocation term. Theorem 5 says it is
non-negative in the third row, and it is — **and the gross-substitutes margin widens by
a factor of 3.6.** Over a box of wage vectors the floor stays non-negative at every
spread out to 2.0 (+0.0136, +0.0043, +0.0008, +0.0001, +0.0000), and no variant violates
gross substitutes anywhere in that box.

## 29.3 The stress test, split by whether (H1) holds

Deliberately nasty random economies: scrambled cross-ownership, countries with no
domestic production, extreme size asymmetry, tariffs. **Split by whether (H1) — the same
"no group above half" condition Theorem 2 already needs — holds at the point.**

```
family                              pts | H1 ok  GSfail  worst GS | H1 bad GSfail | floor|H1
BASELINE  HQ cost  nu .55  N=3      180 |   150       8   -0.3702 |     30      2 |  -1.1740
BASELINE  HQ cost  nu .55  N=4 K=2  160 |   100      11   -0.2467 |     60      9 |  -1.1539
SIMPLE    capability nu 0   N=3     160 |   129       0   +0.0232 |     31      0 |  +0.0000
SIMPLE    capability nu 0   N=4 K=2 150 |    70       0   +0.0041 |     80      9 |  -0.0000
SIMPLE    capability nu 0   sigma12 150 |    36       3   -0.1204 |    114     15 |  -0.0000
SIMPLE    capability nu 0   Bertrand150 |    86       0   +0.0033 |     64      8 |  -0.0000
SIMPLE    capability nu 0   tariff.3150 |    55       0   +0.0036 |     95     10 |  -0.0000
SIMPLE    capability nu .55 N=4 K=2 150 |    83       0   +0.0084 |     67      5 |  -0.2935
```

**Read the (H1)-ok columns.** The baseline breaks gross substitutes at **19 of 250**
points where (H1) holds — that is Theorem 4's second hypothesis failing, exactly as the
counterexample predicts. The simplified model with `nu = 0` breaks it at **0 of 340**
points at the calibrated elasticity, across N=3 and N=4, Cournot and Bertrand, with and
without tariffs. And `floor|H1` is machine zero in every `nu = 0` row and −0.29 in the
`nu = .55` row — Theorem 5, confirmed and delimited.

**At the equilibrium itself** (each economy solved from many dispersed starts, PART 4
of `run_simple_model.txt`): in the baseline family 28 economies gave **0 multiple
equilibria**, 1 gross-substitutes failure at the solution and index +1 everywhere; in the
simplified family 26 economies gave **0 multiple equilibria, 0 gross-substitutes failures
at the solution** (worst off-diagonal +0.0754) and index +1 everywhere. Gross substitutes
holding AT every equilibrium is enough for index +1 at every equilibrium, which with
Poincare-Hopf gives uniqueness — so for the simplified model the chain is complete on
every economy tested.

**The honest exception: `sigma = 12`, 3 failures of 36 even under (H1).** The income
effect through repatriated profits is still there and it bites at very high trade
elasticity. `sigma` is calibrated at 5 (literature 4–6), so this is outside the
paper's range — but **(H1) alone is not a theorem, and this is why.**

## 29.4 THE FACTS GET BETTER, AND ONE FITTED PARAMETER DISAPPEARS

Same seed, same parameters, N=5, K=4, with entry:

| | data | baseline (HQ cost) | capability, mne_adv = .10 | **capability, mne_adv = 0** |
|---|---|---|---|---|
| Fact 1 total | 0.46–0.74 | 0.689 | 0.755 | **0.689** |
| Fact 1 foreign / domestic | foreign dominant | 0.550 / 0.139 | 0.598 / 0.157 | 0.543 / 0.146 |
| **Fact 2 foreign gradient** | **+0.43** | **+0.33** | +0.37 | **+0.42** |
| Fact 3 parent HHI / top | 0.130 / 0.250 | 0.282 / 0.403 | 0.275 / 0.353 | 0.273 / 0.344 |
| Fact 4 concentration | ×1.12 | ×1.95 | ×2.01 | — |
| Fact 5 local exports | positive | −100% | −100% | — |
| Fact 6 distance / ×MNE | −0.16 / +0.05 | −0.375 / +0.724 | −0.419 / +0.862 | **−0.515 / +0.701** |
| integer residue (of 1050) | — | 2 | 0 | 1 |

**The headline: at `mne_adv = 0` the simplified model reproduces Fact 1 at exactly the
baseline's 0.689 AND gets Fact 2's gradient to +0.42 against a target of +0.43 — with
the MNE productivity edge switched OFF.** The edge was one of the three internally
calibrated parameters. Removing the head-office wage penalty already makes
multinationals competitive enough, so it is no longer needed.

**Three fitted parameters become two.** Facts 2 and 3 improve; Fact 1 is unchanged;
Facts 4 and 6 move marginally in the wrong direction from an already-overshooting level;
Fact 5 fails either way. **Nothing that currently works is lost.**

Why Fact 2 improves is exactly the reason recorded in the code: under the head-office
cost share a foreign affiliate pays its parent's HIGH wage on the `alpha_k` share, and
parents sit in high-wage countries, so the wage channel pushes foreign shares DOWN in
complex sectors — the wrong sign. Removing it removes a force that was fighting the
fact.

## 29.5 THE RECOMMENDATION — **ADOPTED 2026-08-20, see §30**

**Adopt `hq_cost = false` as the baseline and keep the head-office cost share as a
robustness variant.** Both are one keyword apart and both are supported.

- It is the only change that removes the mechanism behind the counterexample.
- It buys Theorem 5, a *signed* reallocation block under (H1) alone.
- It widens the gross-substitutes margin at the calibration by 3.6x.
- It improves Fact 2 from +0.33 to +0.42 and drops a fitted parameter.
- The economics is standard: multinationals share a non-rival capability
  (Ramondo–Rodriguez-Clare, Tintelnot) instead of buying head-office services at the
  parent's wage (Antras). Both are in the literature; the paper should say it checked
  both.

**On `nu = 0`: optional, and a real trade-off.** It converts Theorem 5 from conditional
to unconditional, and the block it removes was added for Fact 5, which fails with or
without it. But it also removes the input–output realism a referee may expect. **Keep
`nu = 0.55` for the quantitative work and report the `nu = 0` case as the one where the
theorem is unconditional** — the floor is −0.29 at `nu = .55` and exactly 0 at `nu = 0`,
so the reader can see precisely what the assumption buys.

**What is still NOT proved, and it is now the textbook obstacle.** A country whose firms
lose profits when a foreign wage rises spends less. That is the classical income effect,
common to every model with non-labour income, and it is not peculiar to multinationals.
It is what the `sigma = 12` failures are.

---

# 30. THE BASELINE AS ADOPTED, AND AN AUDIT AGAINST FOUR QUESTIONS (2026-08-20)

**`hq_cost = false` is now the DEFAULT in `world_economy`, and `mne_adv = 0.0`.**
Head-office services are a non-rival capability, not a factor input bought at the
parent's wage. The old cost-share model is one keyword away (`hq_cost = true`) and is
still reported everywhere. Delivered cost is now

```
a[a,n] = (1/phi_a) * w[l]^(1-nu_k) * PIO[l,k]^nu_k * gamma[h,l] * d[l,n]
F[a,n] = f_k * w[l] * fdist[l,n]
```

Re-run at the new baseline: `run_model_full.txt`, `run_wage_uniqueness.txt`,
`run_simple_model.txt`; `docs/model_guide.pdf` (38 pp) with Equation 5 and its new
subsection "Where the head office is, and why it is not in the cost equation".

## 30.1 Does it have a unique solution — as a fact, not as a hypothesis?

| layer | unique? | conditional on anything? |
|---|---|---|
| one market | **proved** (Thm 1) | no |
| entry | **proved** (Thm 2) | no group above half a market — CHECKED, max 0.376 |
| I–O prices | **proved** (contraction) | no |
| spending and incomes | **proved** (linear) | no |
| **wages** | **proved** (Thm 4; Thm 5 in the baseline) | (H1) the same half-share condition, plus a demand condition |

**As a fact, yes.** Across everything run today — hundreds of economies, each solved
from 12–25 wage vectors spread over `e^4` — **the number of distinct equilibria found
was one, every time. Zero cases of multiplicity, in any family, under either conduct.**
And the index was `+1` at **100%** of equilibria, which is what uniqueness actually
requires (indices sum to `+1`); it is also numeraire-independent by Walras' law.
At the calibrated economy gross substitutes holds outright — margin `+0.152` (N=5,
Cournot), `+0.145` (N=5, Bertrand), `+0.195` (N=4 driver), `+0.360` (N=5 with nu=0) —
and it survives a box of relative wages of spread 2.0. Every one of these is WIDER than
under the old baseline (`+0.100` Cournot, `+0.092` Bertrand).

**THE IMPORTANT CAVEAT, AND IT POINTS AT `nu`.** At the adopted baseline the FACT got
stronger and the SUFFICIENT CONDITIONS got weaker, and those are different things.

The fact, from `run_wage_uniqueness.txt` PART 3 at the new baseline:

```
conduct   spread  pts   H1 slack   (H2) slack   min GS   GS fails
cournot     0.00    1    +0.5767     +0.0094    +0.1516      0
cournot     0.15   20    +0.5748     -0.0077    +0.1466      0
cournot     0.30   20    +0.5615     -0.0222    +0.1413      0
cournot     0.50   20    +0.5502     -0.0601    +0.1369      0
cournot     0.80   20    +0.5393     -0.0724    +0.1119      0
bertrand    0.00    1    +0.8377     -0.0195    +0.1449      0
bertrand    0.80   20    +0.8074     -0.1250    +0.1024      0
```

**Gross substitutes never fails — 0 of 20 at every spread, both conducts, margins
+0.10 to +0.15, all wider than the old baseline's +0.09 to +0.10.** (H1) has enormous
slack (0.54 to 0.84 against a threshold of 1).

But **Theorem 4's condition (H2) now goes negative away from the Cournot equilibrium,
and at the Bertrand equilibrium itself.** The bound got looser precisely because the
head-office cost share went away: with `alpha = 0` a plant's own cost response is
`lam_a = nu * (I-O in its host)`, no longer identically zero, and it can exceed its
group's average. That is exactly the extra condition `lam_a <= min(lam_g, Lam)` in
Theorem 5's remark. At the calibrated economy its violation is 0.030 at `nu = 0.55`
and **exactly zero at `nu = 0`**, where the reallocation floor is `+0.0136` and stays
non-negative across the whole box.

**So Theorem 4 is the wrong tool for the new baseline and Theorem 5 is the right one --
and Theorem 5 bites only at `nu = 0`.** At `nu = 0.55` the position is "verified with a
wide margin, sufficient conditions too conservative to certify it"; at `nu = 0` it is
"proved under (H1) alone". That is the real trade-off on the input-output block, and it
is sharper than it looked before this run. See section 30.6.

**As an unconditional theorem, no — and further simplification will not buy one.**
After `hq_cost = false` the only channel left that can break gross substitutes is the
ordinary income effect: a country whose firms lose profits when a foreign wage rises
spends less. **That channel IS the paper's subject.** Removing it means spending profits
where they are earned instead of where they are owned, which deletes the contribution.
So the sequence stops here, deliberately:

- the channel peculiar to this model (head-office wages) — **removed**, and Theorem 5
  now signs the whole reallocation block under (H1) alone;
- the markup channel — **signed** by the same half-share condition Theorem 2 needs;
- the input–output channel — signed too if `nu = 0`, which is offered but not adopted;
- the income channel — **kept, because it is the model's point**, and measured.

That is the honest answer, and it is a much smaller residual than the one this file
described this morning.

## 30.2 Does it show the stylized facts?

At the new baseline (N=5, K=4, with entry, Cournot), `run_model_full.txt`:

| # | fact | data | model | verdict |
|---|---|---|---|---|
| 1 | MNE share of exports; foreign dominant | 0.46–0.74 | **0.689** = 0.543 + 0.146 | matched, level is the one target |
| 2 | foreign MNEs in complex goods | gradient **+0.43** | **+0.42** | **matched — was +0.33** |
| 3 | parents from few countries | HHI 0.130, top 0.250 | 0.273, 0.344 | generated, overshoots in a 5-country world |
| 4 | grouping by owner raises concentration | ×1.12 | ×1.94 | generated, a prediction |
| 5 | MNE presence raises local exports | positive | **negative** | **FAILS** |
| 6 | distance weaker for MNEs | −0.16 / +0.05 | too large | generated, magnitudes too large |

**Five of six, and Fact 2 is now essentially exact.** Fact 5 is the one outright
contradiction; it was failing before this change and it fails after, so the change did
not cause it. See §30.4.

Active plant-market pairs rose from 677 to **778 of 1050** and the unresolved integer
residue fell from 2 to **1**.

## 30.3 Does it use Ramondo, Tintelnot, Yang and Gaubert?

**Yes — and after this change it uses them MORE purely, because the one channel removed
was the only one that was not theirs.**

| source | the idea | where it is in the model |
|---|---|---|
| **Ramondo & Rodríguez-Clare (2013)** | producing away from head office is less efficient | `gamma[h,l]` multiplies delivered cost, = 1 at home, 1.18 abroad (`delivered_cost`) |
| **Tintelnot (2017)** | export platforms; a fixed cost per market served | `d[l,n]` runs from the PRODUCTION country, so one plant serves third markets — this is what generates Fact 6; `fixed_cost(m,w,a,n)` per plant-market |
| **Yang (2023)** | multi-plant oligopoly choosing SETS of locations | every granular parent has a potential plant in every country; entry picks the subset; conduct is at the PARENT level, so a group internalises competition among its own plants (`EntryMarket`, `subset_table`, `solve_market_entry`) |
| **Gaubert & Itskhoki (2021)** | granular firms; Pareto capability; the pool of potential entrants scales with country size; a cost-ranked cutoff | `xi ~ Pareto(theta_p = 4)`; pool `∝ L_h z_h^zeta`, `zeta = 1.5`; the cutoff is Theorem 2's condition (A) applied WITHIN the parent |
| Caliendo & Parro (2015) | input–output loop | `nu_k`, `omega[k',k]`, the `PIO` term |
| Antràs (2003), Antràs–Helpman (2004) | head-office intensity rises with complexity | now the COMPLEXITY INDEX `alpha_k` inside productivity (`hq_gap`, `adv_slope`), **not** a cost share. The cost-share version is the `hq_cost = true` variant |

What is new and not borrowed: parent-level conduct with **variable** markups (so
`kappa > 0` and ownership matters), firm-level ownership `theta[g,n]`, and the two
uniqueness theorems.

## 30.4 What is still wrong: Fact 5

Fact 5 remains the one failure and it is not caused by this change. In the data more
multinationals in a country-sector goes with MORE exports by local firms; in the model
business stealing on the ENTRY margin beats the cheaper-input channel. The Javorcik
spillover knob (`spill`) is the candidate fix and the size it needs is measured in
§30.5. The alternative reading — that the empirical coefficient is contaminated because
no specification carries destination × product × year fixed effects — is cheap to test
and belongs in the empirics, not the model.

**This is the item to put in the abstract, not a footnote.** It was true before today
and it is true now.

## 30.5 THE SPILLOVER CANNOT RESCUE FACT 5 — MEASURED, NOT ASSUMED

Grid over the Javorcik knob at the new baseline (N=4, K=4, entry on; each row is two
full GE solves; `scratchpad s7.jl`). `spill` raises a local firm's productivity with the
number of multinational plants in its own country and sector.

```
spill     non-MNE exports, no extra MNEs -> with extra MNEs     change
0.00           0.04840                       0.00162            -96.7%
0.05           0.05786                       0.00525            -90.9%
0.10           0.06557                       0.00783            -88.1%
0.15           0.07339                       0.01137            -84.5%
0.25           0.09082                       0.01992            -78.1%
0.40           0.11000                       0.04415            -59.9%

head-office-cost variant, spill = 0:
0.00           0.04930                       0.00495            -90.0%
```

**The response improves monotonically but far too slowly to be rescued.** At
`spill = 0.40` -- more than twice the 0.15 the fixed-roster model needed, and far above
any Javorcik-type estimate -- the sign is still firmly negative at -60%. Extrapolating,
flipping it would take a spillover somewhere near 0.7-1.0, which is not a number anyone
can defend.

The mechanism is visible in the two columns. Raising the spillover raises the LEVEL of
local exports a lot (0.048 -> 0.110) and the RESPONSE to multinational entry much less,
because business stealing works on the ENTRY margin: the extra multinationals drive
local plants OUT, and a plant that has exited exports nothing however productive it
would have been.

**Note also that the new baseline is slightly WORSE on Fact 5** than the head-office-cost
variant (-96.7% against -90.0% at `spill = 0`), because removing the parent-wage penalty
makes multinationals cheaper and so sharpens business stealing. That is a real cost of
the change and it is the only one found.

**Conclusion: do not fit a spillover to Fact 5.** The two honest options are (i) report
Fact 5 as the model's outright failure, which is the current position, or (ii) test the
alternative reading on the empirical side — the published coefficient has no
destination x product x year fixed effect, so a market-level demand shock that raises
both multinational entry and local exports would produce it. **(ii) is cheap and is the
decisive test; it belongs in the empirics and it is now the top model-facing empirical
task.**

## 30.6 THE `nu` DECISION, WITH THE NUMBERS — AND WHY THE DEFAULT DID NOT MOVE

Same seed, same everything, only `nu` changes (N=5, K=4, entry on, Cournot,
`hq_cost = false`, `mne_adv = 0`):

| nu | Fact 1 | Fact 2 grad | Fact 3 HHI / top | Fact 4 | floor | (*) viol | min GS | residue |
|---|---|---|---|---|---|---|---|---|
| **0.55** | **0.689** | **+0.42** | 0.273 / 0.344 | 1.94 | −0.108 | 3.2e-2 | +0.147 | 1 |
| 0.30 | 0.707 | +0.44 | 0.271 / 0.344 | 1.94 | −0.065 | 2.5e-2 | +0.306 | 0 |
| 0.00 | 0.727 | +0.47 | 0.269 / 0.339 | 1.95 | **+0.014** | **0.0** | **+0.365** | 2 |
| data | 0.46–0.74 | +0.43 | 0.130 / 0.250 | 1.12 | | | | |

**At `nu = 0` the theorem CERTIFIES uniqueness on a box.** Both of Theorem 5's inputs
hold there, not just the first:

```
nu = 0   spread   pts   min Ebar   min floor    min GS   H1 slack   GS fails
          0.00      1    +0.0899     +0.0137   +0.3650    +0.5773       0
          0.40     20    +0.0687     +0.0052   +0.2414    +0.5623       0
          0.80     20    +0.0242     +0.0013   +0.1600    +0.5397       0
          1.40     20    -0.0157     +0.0001   +0.0329    +0.3212       0
```

The same scan at `nu = 0.55`, for contrast:

```
nu = .55 spread   pts   min Ebar   min floor    min GS   H1 slack   GS fails
          0.00      1    +0.0397     -0.1084   +0.1470    +0.5968       0
          0.40     20    -0.0047     -0.1455   +0.1314    +0.5859       0
          0.80     20    -0.0100     -0.2263   +0.1233    +0.5588       0
          1.40     20    -0.0615     -0.3418   +0.0759    +0.5199       0
```

**At `nu = 0.55` BOTH of Theorem 5's inputs fail away from the solution** — the floor is
negative everywhere (the `(*)` condition) and the demand term turns negative by spread
0.40 — while gross substitutes itself never fails (0 of 20 at every spread, margin
+0.076 to +0.147). That is the cleanest statement of the trade-off: at `nu = 0` the
theorem certifies what is true; at `nu = 0.55` the same thing is true but the theorem
cannot see it.

(H1) holds with slack 0.54 and the demand condition `Ebar >= 0` holds with slack
+0.024 out to spread 0.80. **Theorem 5 then gives gross substitutes on that box, and
the corollary gives AT MOST ONE equilibrium of spread <= 0.40.** That is a certified
uniqueness statement for the calibrated economy, with both hypotheses verified rather
than assumed. At spread 1.40 the demand term turns slightly negative while the floor
stays positive and gross substitutes still holds (0 failures, +0.033).

**The default stays at `nu = 0.55` anyway, and here is the reason.** Intermediates are
55% of gross output and the paper's policy question is how a TARIFF propagates. Dropping
the input-output block to buy a proof would change the object the policy analysis is
about, and that is a bad trade for a gain the evidence says is not needed: at
`nu = 0.55` gross substitutes holds with margin +0.147 at the solution and 0 failures
across every box tested, the index is +1 everywhere, and no multiplicity has ever been
found. What `nu = 0.55` loses is only the ability of the SUFFICIENT CONDITION to certify
what is anyway true.

Note also that at `nu = 0` the capability gradient stops identifying Fact 2: the
gradient is +0.46 to +0.47 across `adv_slope` from 0.8 to 1.2, so it is pinned by
`hq_gap` instead and the fit is slightly worse than `nu = 0.55`'s +0.42.

**So the position to write down is:** uniqueness is PROVED, with hypotheses verified on
a box, in a version of the model that differs from the baseline only in the
input-output block — and in the baseline itself it is verified with a wide margin and
has never failed. **Flipping the default is one keyword (`nu = 0.0`) if a referee
prefers the proof to the input-output realism.**

---

# 31. UNIQUENESS, CONSTRUCTIVELY: THE SOLVER'S OWN MAP IS A CONTRACTION (2026-08-21)

**This is now the statement to quote, and it replaces "gross substitutes, therefore
Arrow–Block–Hurwicz" as the headline.** Theorems 4 and 5 rule out a second equilibrium
by importing a sufficient condition from outside the model. Theorem 6 does something
more direct: **the map the solver iterates is a contraction**, so the model has one
solution because of how it is solved, and the algorithm converges to it from anywhere.
The only thing imported is Birkhoff's theorem, which is exact.

Code: `hilbert_metric`, `birkhoff_coefficient`, `tatonnement_jacobian`, `max_damping`,
`contraction_certificate`, `verify_contraction` — `mne_model.jl` PART 7c, reported by
`verify_wage_theorem` and by `run_ge`. Document: guide Part III, Layer 5,
§"Uniqueness the constructive way".

## 31.1 The argument

Given `w`, everything else in the model is ALREADY a unique function of it — markets
(Thm 1), entry (Thm 2), prices (a contraction), incomes (a linear solve). So the whole
model is one map on wages, and it is the one the solver iterates:

```
Psi_k(w)_n = w_n * (LD_n(w)/L_n)^k = w_n^(1-k) * (W_n(w)/L_n)^k
```

- **Homogeneous of degree one**, so it acts on the PROJECTIVE space of wage vectors,
  whose natural distance is Hilbert's `d(w,w') = max_n ln(w_n/w'_n) − min_n ln(w_n/w'_n)`
  — exactly the "spread" used everywhere else, and unchanged by rescaling either vector.
- **Log-Jacobian `A_k = (1−k) I + k M`**, and `M` has **unit row sums** — the model's own
  homogeneity, not an assumption — so `A_k` does too.
- `A_k` is entrywise positive iff **(i)** `M[n,j] > 0` for `n != j` (gross substitutes)
  and **(ii)** `k < 1/(1 − min_n M[n,n])`.
- **Birkhoff (1957):** a strictly positive matrix contracts the Hilbert metric with
  coefficient `tanh(Delta/4) < 1`. So `Psi_k` is a strict contraction, has **at most one**
  fixed point up to scale, and the iteration converges geometrically.

## 31.2 THE DAMPING IS NOT A NUMERICAL CONVENIENCE — this is the new economics

**Labour demand is elastic.** At the calibrated economy `M[n,n] = −1.489`: a country's
own wage bill FALLS by about 1.5% when its wage rises 1%. So the UNDAMPED map (`k = 1`)
is **not order-preserving at all**, and the Hilbert/Perron–Frobenius route applied to it
fails outright — measured, not guessed: min entry of `M` is −1.49 at every point tested.

The model itself then says how much damping is needed, `k < 1/(1 − min M[n,n])`:

| model | min `M[n,n]` | largest admissible `k` | solver uses |
|---|---|---|---|
| baseline, nu = 0.55, Cournot | −1.489 | **0.402** | 0.25 ✓ |
| baseline, nu = 0.55, Bertrand | −1.568 | 0.390 | 0.25 ✓ |
| nu = 0, Cournot | **−2.669** | **0.273** | 0.25 — only 9% of room |

**Actionable:** at `nu = 0` the default damping is close to the edge; `0.15` would be the
safer default there. At `nu = 0.55` there is plenty of room.

## 31.3 The certificate, over boxes

```
model              spread  min M_nn  max damping  min A_k   Delta   Birkhoff
nu=.55 cournot       0.00    -1.4890      0.4018  +0.0367   4.6512    0.8219
nu=.55 cournot       0.40    -1.6291      0.3804  +0.0344   4.8862    0.8401
nu=.55 cournot       0.80    -1.6685      0.3747  +0.0296   5.5664    0.8835
nu=.55 cournot       1.40    -1.7041      0.3698  +0.0179   6.4429    0.9233
nu=.55 cournot       2.00    -1.6666      0.3750  +0.0075   7.1405    0.9452
nu=.55 bertrand      0.00    -1.5677      0.3895  +0.0350   4.6459    0.8215
nu=.55 bertrand      2.00    -1.7602      0.3623  +0.0077   7.0234    0.9420
nu=0   cournot       0.00    -2.6691      0.2725  +0.0827   1.0187    0.2493
nu=0   cournot       0.80    -2.7305      0.2681  +0.0381   2.7947    0.6035
nu=0   cournot       2.00    -2.9477      0.2533  +0.0020  11.4719    0.9936
```

**The map contracts on every box tested, under both conducts, out to a spread of 2.0 —
a factor of e^2 between the highest and lowest relative wage.** So that whole box
contains at most one equilibrium, and the solver reaches it from any start in it.

Note `nu = 0` has a far better modulus near the solution (0.249 against 0.822) but
degrades faster far from it, and its admissible damping is much tighter. Two different
things, both worth knowing.

## 31.4 CHECKED, NOT TRUSTED

Birkhoff's bound is a theorem, so the useful test is whether the model actually obeys
it. Taking random PAIRS of wage vectors and applying the solver's own map to both:

```
model            spread  pairs  worst observed ratio  certified bound  respected
nu=.55 cournot     0.40     25                0.4051           0.8481        yes
nu=.55 cournot     0.80     25                0.4285           0.8722        yes
nu=.55 cournot     1.40     25                0.4989           0.9283        yes
nu=0   cournot     0.40     25                0.1526           0.4179        yes
nu=0   cournot     0.80     25                0.1598           0.6803        yes
nu=0   cournot     1.40     25                0.3048           0.9541        yes
```

**The bound was respected in all 150 pairs, with room to spare** — worst observed 0.50
against a guarantee of 0.93 at nu = 0.55, and 0.30 against 0.95 at nu = 0. The map
contracts, and by more than the certificate promises.

## 31.5 WHAT THIS DOES AND DOES NOT SETTLE

**Does:** uniqueness now follows from the model's own solution map, with an explicit
rate and a guarantee of convergence, rather than from an external sufficient condition
plus a fixed-point index argument. It also explains a piece of the code that used to
look arbitrary — the damping — and gives it a model-derived bound.

**Does not:** condition (i) is still gross substitutes, and gross substitutes is still
not unconditional (§26.5's counterexample stands; it is about the `hq_cost = true`
variant). So the honest chain is: *the tâtonnement contracts wherever the wage-bill
elasticity matrix is positive; that matrix is positive everywhere tested, on boxes of
spread 2.0, under both conducts; and the one channel that could make it fail is the
ordinary income effect, which is the paper's own subject and cannot be removed without
removing the contribution.*

---

# 32. THE CONTINUUM CERTIFICATE: INTERVAL ARITHMETIC CLOSES THE
#     BETWEEN-POINTS GAP (2026-08-26)

**What this adds to §31.** Every §31 certificate evaluates the contraction hypotheses
at FINITELY MANY points (grids, random draws) — a certificate on the sample, not on
the region. `interval_certificate.jl` encloses M[n,j] = d ln W_n / d ln w_j over the
WHOLE of a wage box at once, with outward-rounded interval arithmetic, so a certified
row is a theorem about the continuum: **that box contains exactly the one equilibrium
the solver found** (at most one from the contraction; at least one because the solved
wages sit in it). This is §26.7 route 4, executed.

## 32.1 Scope and honesty

- Model: the **ν = 0 capability baseline** (hq_cost = false, Cournot, η = 1, zero
  tariffs) — the version §30.6 designates as "the case where the theorem certifies".
  There the model is closed-form in wages given market shares, which is what makes a
  rigorous enclosure feasible without enclosing the price contraction.
- Entry configuration **frozen at the solution**, exactly as Theorem 6 is stated.
- Rounding: IEEE ops outward-rounded by 1 ulp; exp/log by 4 ulps (they are faithfully
  rounded — the cushion is 4x their documented error).
- Damping κ = 0.15 in the committed run — §31.2's own recommendation at ν = 0; the
  fixed points of Ψ_κ are the same equilibria for every κ, so nothing in the
  uniqueness conclusion depends on the choice.

## 32.2 The method — what it took to make intervals survive a real box

Naive interval evaluation of M blows up by spread 0.01 (dependency). Four structural
fixes, in the order they mattered:

1. **Centered form with subdivision (decisive).** Per subbox: M is evaluated EXACTLY
   at the subbox center (error = rounding) plus an interval GRADIENT times the radius,
   with the box certificate the entrywise hull over subboxes. The gradient needed the
   full second-order comparative statics in closed form (∂κ, ∂Λ via
   εΛ = −(σ−1)S/[(1−S)(1+(σ−2)S)], ∂ω with Σ∂ω = 0 exploited, a second income solve
   with the same Neumann bound, and ∂M[n,m] = ∂Ŵ/W − M[n,m]M[n,q]). Validated against
   finite differences of the point evaluation to 1e-07 before any certificate prints.
2. **Centered first-order inputs inside the gradient** (dS, dA, dμ, X̂ each enclosed
   as center + own-second-order × radius), plus a two-pass bootstrap for the M·M term
   — this pushes the dependency error to second order in the subbox width, so
   subdivision converges quadratically.
3. **Group × location collapse**: every plant sum rewritten so each wage occurs once
   per expression (K_g = Σ_l w_l^{1−σ}C_gl with exact constants; kappa via the tight
   a/(a+c) form) — near-exact level enclosures.
4. **Corner-solved shares**: S_g is increasing in own K and decreasing in rivals'
   (Λ > 0, ω ∈ (0,1) at every interior share — theorems of the model), so its exact
   range over the box is bracketed by re-solving the market at two capability corners,
   each root bracketed by bisection on certified interval signs.

## 32.3 Results (committed run: `run_interval_certificate.txt`)

Validation gate: interval X and W contain the solver's values; interval M matches
`wage_bill_elasticity` finite differences to 1.4e-10; analytic gradient matches finite
differences to 1.1e-07; M row sums = 1 (homogeneity) to 1e-06. **All passed.**

**FINAL (committed run, κ = 0.15 — §31.2's own ν = 0 damping; same equilibria):**

| spread | s | boxes | minGS | minA | contraction τ | certified |
|---|---|---|---|---|---|---|
| 0.02 | 1 | 1 | +0.315 | +0.047 | 0.57 | YES |
| 0.05 | 2 | 16 | +0.273 | +0.041 | 0.61 | YES |
| 0.10 | 3 | 81 | +0.164 | +0.024 | 0.71 | YES |
| 0.15 | 4 | 256 | +0.106 | +0.016 | 0.78 | YES |
| 0.20 | 5 | 625 | +0.091 | +0.014 | 0.82 | YES |
| **0.30** | 6 | 1296 | **+0.040** | **+0.006** | **0.90** | **YES** |

**The continuum within ±15% (log) of the solved relative wages contains exactly the
one equilibrium the solver found.** The 0.40 row (4,096 subboxes) was stopped for
machine time and can be extended by re-running the script. At κ = 0.25 the binding
entry is the diagonal (budget −3.0; 0.02/0.05 certify, 0.10 misses by 0.02); at
κ = 0.15 the binding entry is gross substitutes itself, as it should be.

**How to quote it:** "uniqueness within a ±ρ/2 band of relative wages around the
solved equilibrium is computer-certified on the continuum, by outward-rounded interval
arithmetic; beyond that band the §31 grid and random-pair certificates apply." The
un-certified remainder is the same honest gap as before — but the sampled gap now
starts where the proved region ends, not at the solution.

## 32.4 Traps met while building it (so they are not re-met)

- **The raw interval evaluation is not usable** beyond spread ~0.01 even with tight
  collapses; the centered form is not an optimisation, it is the difference between
  failing at 0.01 and certifying real boxes.
- The certificate's binding entry at κ = 0.25 is the DIAGONAL of (1−κ)I + κM (labour
  demand is elastic, M_nn ≈ −2.7 at ν = 0); at κ = 0.15 it is the off-diagonal (GS).
  Choose κ accordingly — the equilibria do not depend on it.
- `include`-ing `mne_model.jl` is safe (the driver is guarded), but the certificate
  loads only the module part via `include_string` to keep the runner out of scope.

---

# 33. FACT 5 SURVIVES THE SATURATED SPECIFICATION — THE DECISIVE TEST,
#     RUN (2026-08-26)

**The question §30.5 left open:** is Table 1 Panel B's positive coefficient a real
fact, or contamination from the missing destination × product × year fixed effect?

**The answer: real, at half size.** `Orbis_DNB_Customs_Final/src/13_fact5_dpy_fe.do`
re-runs Panel B on the v4 cube (`collapsed_odpy.dta`, 1,546,580 cells, from the
predecessor project's Intermediate_v4) with `dpt_id = group(country_dest hs6 year)`
added to the published tightest column (odp_id + odt_id, cluster od_id):

| margin | published controls | + dest×product×year |
|---|---|---|
| intensive, ln(# MNE firms) | 0.166 (0.014) | **0.087 (0.017)** |
| extensive, any MNE present | 0.117 (0.011) | **0.061 (0.011)** |
| PPML on the level, full FE | — | 0.229 (0.051) |

All significant at 1%. Notes: (a) the benchmark on this cube is 0.166, not the
published 0.2391 — the published table used a later (July-2026) build; the cube
vintage difference is documented in the README. (b) The cube holds positive-trade
cells only, so margins are conditional on the cell exporting at all. (c) reghdfe
drops singletons — the saturated column runs on 153k cells.

**Consequences.** Half the published association was common demand; the surviving
half is precisely estimated with the same sign on every margin, PPML agreeing. So
Fact 5 is a REAL target at ~half its published size, the model's −97% to −100% is a
genuine failure that can no longer be attributed to the specification, and §30.5
already shows the spillover knob cannot fix it. **Finding the mechanism that carries
the positive sign is now the model's foremost open problem.** Candidate directions
NOT yet tested: complementarities through the extensive margin of varieties (entry of
MNEs expanding the market's product scope), local input-market thickening that lowers
the FIXED cost (the current spillover works through marginal productivity only), and
demand spillovers through quality signalling. Committed: script 13 (repo, commit
6600ac8), outputs `output/tables/fact5_dpy_fe.{log,csv,tex}` (gitignored, local).

---

# 34. Cov(θ,S) MEASURED IN THE DATA — THE HEADLINE OBJECT EXISTS AND IS
#     SIGN-VARYING (2026-08-27)

**The referee pass called this "the single most damaging gap": the paper's central
object was never computed in the data the team owns. It now is.**
`Orbis_DNB_Customs_Final/src/14_ownership_covariance.do` (commit 079e30e), outputs
`output/tables/ownership_cov_bymarket.dta` + `ownership_cov_owners.csv` + log.

**Construction.** Market = destination × HS6 × year. Group = ultimate parent FIRM
(Orbis GUO BvD id `guo25`; unmatched exporters are their own singleton groups — the
same convention as the six facts; the GUO id covers ALL matched value, USD 1.23tn of
2.59tn total). Shares are within-LAC-sample shares — Figure 6's own measurement
operator; §5a maps them into destination-absorption shares. Owner of a group =
`iso3_parent` (origin country for unmatched). Value-weighted aggregation gives the
Layer-1 three-term decomposition: naive θ̄·H̄ + between-market Cov + within-market Cov.

**Headline numbers** (H̄ = 0.373 value-weighted mean market HHI; 35 groups/market):

| owner | θ̄ | total | naive | between | within | total/naive |
|---|---|---|---|---|---|---|
| USA | 0.135 | 0.0555 | 0.0503 | +0.0032 | +0.0020 | **1.10** |
| COL | 0.133 | 0.0614 | 0.0496 | +0.0072 | +0.0046 | 1.24 |
| CHL | 0.106 | 0.0434 | 0.0394 | +0.0033 | +0.0006 | 1.10 |
| ARG | 0.096 | 0.0273 | 0.0357 | −0.0072 | −0.0012 | 0.76 |
| GBR | 0.088 | 0.0296 | 0.0328 | −0.0035 | +0.0003 | 0.90 |
| PER | 0.067 | 0.0176 | 0.0251 | −0.0058 | −0.0017 | 0.70 |

**Readings.** (1) For the US the country-level calculation UNDERSTATES the
ownership-weighted concentration by 10%, both covariance terms positive — US parents
are disproportionately the big players in the concentrated markets, the paper's
hypothesized sign. (2) The correction is owner-specific and SIGN-VARYING (COL +24%
vs PER −30%) — no country-level share can express this; that is the contribution in
one line. (3) θ̄_USA = 0.135 independently reproduces §27's 0.134 from a different
cut of the file — a free cross-check that both pipelines are right.
**Traps disclosed:** shares are within-sample (λ-renormalised); singleton treatment
of unmatched firms follows the stylized-facts convention; a group occasionally
carries two owner strings and is labelled by its largest-value owner.

---

# 35. FACT 5's MECHANISM, ROUND ONE: THE FIXED-COST SPILLOVER REPAIRS THE
#     ENTRY MARGIN AND STILL CANNOT FLIP THE SIGN (2026-08-27)

**The candidate.** §30.5's diagnosis was that business stealing dominates ON THE
ENTRY MARGIN — an exited plant exports nothing however productive. So the natural
next mechanism attacks that margin directly: multinational presence in a (country,
sector) thickens export infrastructure and lowers LOCAL plants' market-access cost,
`F *= (1 + n_mne(loc, sec))^(-fspill)`. Implemented as `fspill` in `world_economy`
via a per-plant fixed-cost multiplier `fmult` in `GEEntry` (new fourth field;
2- and 3-argument constructors preserved). The count is over the POTENTIAL roster,
the same convention as `spill`, so the multiplier is a constant of the entry game:
**Theorem 2 and all uniqueness results apply verbatim**, homogeneity in wages is
untouched, and the accounting stays exact (Walras 1.4e-16 with the knob on).

**The result (`run_fspill.txt`, N = 4, +8 MNE parents into sector 2):**

| fspill | spill | Δ non-MNE exports | local pairs, before → after |
|---|---|---|---|
| 0.00 | 0 | −96.7% | 19 → 1 |
| 0.50 | 0 | −84.3% | 29 → 15 |
| 1.00 | 0 | −80.6% | 36 → 32 |
| 1.50 | 0 | −80.2% | **36 → 36** |
| 1.00 | 0.15 | −71.1% | 36 → 36 |

**Reading: at fspill ≥ 1.5 the extensive margin is FULLY repaired — every local
plant-market pair survives the MNE influx — and the value response is still −80%.**
Even stacked with the productivity spillover it is −71%. So the sharpened verdict:

> **No cost-side spillover — marginal or fixed, of any defensible or indefensible
> size — can produce Fact 5 in this market structure.** With market expenditure
> E_nk fixed, entry is zero-sum in value: once exits are prevented, pure
> intensive-margin share-stealing still delivers −80%.

**What this narrows the problem to.** The data say a destination×product×year
cell is NOT zero-sum for the origin: Panel A total exports rise strongly with MNE
presence (+0.69 with the saturated FE) — the origin's suppliers jointly capture
more of the destination market. The model's GE markets contain only the N countries'
plants; there is no rest-of-world fringe inside a market for origin suppliers to
displace. **The mechanism therefore has to be the §5a λ margin: an outside (rest of
world) supply in every market, with multinational presence raising the origin's
penetration against it** — visibility/certification of the origin, or platform
access. That is a well-posed modeling task: one fringe supplier per market,
calibrated by λ = LAC exports / destination absorption (Comtrade + production), the
identification §5a already prescribes. Business stealing then hits the fringe, not
only fellow locals. NOT yet implemented.

**Side findings.** Facts 1–4 are essentially invariant to fspill (Fact 1 0.673–0.676,
gradient +0.41, Fact 4 ×1.94), gross substitutes holds (minGS +0.157 at fspill = 1),
and the integer residue is 0 at fspill = 1 — the knob is safe to keep as a reported
option. `fspill` stays OFF in the baseline.

---

# 36. LAYER 4 REDONE WITH ENTRY: THE OWNERSHIP GRADIENT SURVIVES, AND THE
#     I-O BLOCK DECIDES WHETHER IT FLIPS THE SIGN (2026-08-27)

**The exercise** (`experiments.jl tariff` → `run_tariff_entry.txt`): country 1 (rich)
taxes ALL its imports at rate t; country 1 owns a share λ of every
foreign-headquartered parent (θ[g,1] = λ, θ[g,hq] = 1−λ); full GE re-solved WITH
entry at every (t, λ); welfare = X₁/P₁, P₁ = Π_k P₁ₖ^βₖ. The §12.3 table was
fixed-roster; this is the version with the exit channel live.

## 36.1 ν = 0 (the certified variant): the reversal SURVIVES entry, cleanly

| λ | dW% at t=.05 | t=.10 | t=.20 | t=.30 | grid-optimal t |
|---|---|---|---|---|---|
| 0.00 | −0.06 | −0.23 | −0.97 | −1.81 | 0 (or a subsidy) |
| 0.25 | +0.02 | −0.01 | −0.45 | −1.25 | ≈ 0.05 |
| 0.50 | +0.15 | +0.22 | −0.10 | −0.58 | ≈ 0.10 |
| 0.75 | +0.26 | +0.47 | +0.59 | +0.36 | ≈ 0.20 |
| 1.00 | +0.69 | +1.05 | +1.38 | +1.48 | ≥ 0.30 |

**The grid-optimal tariff rises monotonically in ownership — §12.3's uncomfortable
result is confirmed with entry.**

## 36.2 ν = 0.55 (quantitative baseline): ownership NEUTRALISES, does not flip

| λ | dW% at t=.10 | t=.20 |
|---|---|---|
| 0.00 | −1.03 | −2.61 |
| 0.50 | −0.67 | −1.81 |
| 1.00 | **−0.02** | −0.38 |

With intermediates, a tariff also taxes the input bundle, so it is costlier
everywhere; the ownership gradient is intact (the welfare cost of a 10% tariff
shrinks 50-fold from λ = 0 to λ = 1) but at the calibrated ν the optimal tariff at
FULL ownership is ≈ 0, not positive. **The honest headline: home ownership raises
the optimal tariff monotonically; whether it turns positive depends on the
input–output structure.** P2b's health warning stands — do not write a general sign.

## 36.3 The new channel, sized

Affiliates are driven OUT of market 1 by tariffs: at λ = 0, ν = 0, the count serving
country 1 falls 86 → 73 as t goes 0 → 0.30 (ν = 0.55: 116 → 107 by t = 0.20). And a
GE income effect the fixed-roster table could not show: exit varies with λ not through
firm behaviour (θ never enters entry decisions) but through incomes — at λ = 1 country
1 is rich enough that its market holds every affiliate at every tariff tested
(115 → 115; 138 → 139).

**Caveats for any write-up.** Grid optima, not first-order conditions; the ν = 0
corner subsidies of §4.7 still lurk below t = 0; and the λ = 1 economy is a stark
thought experiment (country 1 owns every foreign parent outright). The cross-section
reframing of §27 (SLV/DOM/CRI vs ARG/CHL/PER) remains the paper-ready version of the
question.

---

# 37. THE λ-MARGIN OUTSIDE SUPPLIER — BUILT, AND IT MOVES FACT 5 AT FIRST
#     ORDER (2026-08-27)

**What was built.** `world_economy(...; row_L > 0)` appends a REST-OF-WORLD country:
a large labour force (`row_L`) and a dense world-class fringe in every sector
(`row_ndom = 18` per sector, productivity `row_z = 2.2`, no hq_gap penalty — it
proxies ALL non-sample supply, multinationals included), but NO parents and NO
sample-MNE plants. Every market then contains a big outside supply, and sample
exporters hold only the in-sample absorption share — the λ of the §5a measurement
operator, now a real object inside the GE. Distance to everyone `row_dist = 2.5`.
The ROW roster is drawn AFTER the sample roster, so the sample's random draws are
bit-for-bit identical with and without it (checked). Every uniqueness theorem is
dimension-agnostic and ROW firms are ordinary single-plant groups: **nothing in
Part III changes**, and the largest parent share anywhere FALLS (0.376 → 0.164 at
row_L = 12) — deeper inside every frontier. Accounting exact (Walras 3.5e-16).

**The Fact-5 experiment with the margin on (`run_rowfact5.txt`):**

| row_L | spill | fspill | Δ non-MNE exports | pairs before → after |
|---|---|---|---|---|
| 0 | 0 | 0 | −96.7% | 19 → 1 |
| 12 | 0 | 0 | −71.2% | 16 → 7 |
| 12 | 0.15 | 0 | −50.0% | 23 → 16 |
| 12 | 0.15 | 0.5 | −42.6% | **33 → 36** |
| 12 | 0.30 | 0.5 | **−30.3%** | **40 → 44** |
| 24 | 0.15 | 0.5 | −36.6% | 31 → 32 |

**Three findings.** (1) The λ margin is the first channel that moves the response at
first order ON ITS OWN (−97 → −71 with no spillover at all) — it changes the
INCIDENCE of business stealing, which no spillover can. (2) **The extensive margin
flips to the data's sign**: with the margin plus moderate spillovers, adding MNEs
INCREASES the number of local plant-market export pairs (33 → 36, 40 → 44) — the
data's +0.061 extensive fact, reproduced in sign for the first time. (3) The
intensive (value) response is still negative at these doses, but for a measurable
reason: at row_L = 12–24 the sample still holds λ ≈ 0.2–0.4 of each market, whereas
the DATA's λ is ~0.02–0.1. In the λ → 0 limit sample suppliers are atomistic, entry
barely moves the market aggregate, and any spillover makes the response POSITIVE.

**The dose–response, measured (rows appended to the run file):**

| row_L | λ (in-sample share) | Δ non-MNE exports | pairs |
|---|---|---|---|
| 12 | ≈ 0.28 | −42.6% | 33 → 36 |
| 24 | ≈ 0.20 | −36.6% | 31 → 32 |
| 48 | 0.138 | −28.0% | 24 → 32 |
| 96 | **0.091** | **−23.4%** | **20 → 29** |

At λ = 0.091 — squarely in the data's range — the response is −23%, with the
extensive margin strongly positive (+45% more local export pairs).

**AND THE SIGN FLIPS (2026-08-27, second pass — `run_rowcalib.txt`):** raising the
productivity spillover to 0.30 crosses zero:

| row_L | λ | spill | fspill | Δ non-MNE exports | pairs |
|---|---|---|---|---|---|
| 96 | 0.108 | 0.30 | 0.50 | **+0.7%** | 29 → 38 |
| 96 | 0.109 | 0.30 | 1.00 | −0.2% | 38 → 47 |
| 192 | 0.059 | 0.15 | 0.50 | −20.9% | 16 → 24 |
| **192** | **0.071** | **0.30** | **0.50** | **+8.9%** | **21 → 38** |

**ADOPTED as the SIX-FACT λ CONFIGURATION** (user instruction 2026-08-27), after two
N = 5 refinements the N = 4 grid could not see:

1. **The spillovers must accrue to SINGLE-PLANT LOCALS ONLY** (`spill_fringe = true`,
   new switch): at Fact-5-sized spillovers the unrestricted version supercharges
   domestic MNEs' home plants and INVERTS Fact 1's split (foreign 0.171 / domestic
   0.263 at N = 5). Fringe-only restores it — and is the population AHH 1997 actually
   measured. (First fringe-only run crashed: `nplants` was built before the ROW
   roster is appended → KeyError; fixed with a defaulting lookup.)
2. **spill retuned 0.30 → 0.17 on the N = 5 world** (`run_lamtune.txt`): at 0.30 the
   concentrated boost overshoots Fact 5 (+152%) and sinks Fact 1's level (0.29);
   grid: spill .10 → Fact5 −15%, .15 → +5%, .20 → +27%.

**FINAL: row_L = 192 (λ ≈ 0.07), spill = 0.17 fringe-only, fspill = 0.50.**
Committed run `run_model_lambda.txt` (`julia mne_model.jl lambda`, 16 min, 0 ERROR):

| fact | core | λ configuration | data |
|---|---|---|---|
| 1 | 0.69 (0.54/0.15) | **0.49 (0.39/0.11)** | 0.46–0.74, foreign ≫ |
| 2 | +0.42 | +0.58 | +0.43 |
| 3 | 0.27/0.34 | 0.29/0.42 | 0.13/0.25 |
| 4 | ×1.94 | **×1.67** | ×1.12 |
| 5 | −97% | **+45.5% (POSITIVE, both margins)** | positive (+0.087) |
| 6 | signs 3/3, too big | signs 2/3 (present-term flips to −0.39) | — |

**The core calibration keeps row_L = 0** so Fact 5's failure there, and what it took
to fix it, stay visible. Honest accounting (stated in the guide): the configuration
costs three more fitted parameters, a Fact 2 overshoot, and one flipped sign inside
Fact 6; Fact 5's magnitude is within a factor of ~3 of the saturated estimate at the
experiment's dose. Remaining discipline: identify λ from measured absorption shares
(Comtrade + production) instead of fitting it alongside Fact 5.

**Costs, disclosed.** At (row 12, spill .15, fspill .5) without recalibration:
Fact 1 = 0.572 (foreign 0.348) — inside the data range but lower; Fact 2 gradient
+0.30 (was +0.42); **Fact 4 ×1.60 — closer to the data's ×1.12 than the baseline's
×1.94**; minGS +0.046 (uniqueness fine); integer residue 7 of 1050. Adopting the
margin as baseline means a joint recalibration (fcost, hq_gap, zeta, row_L against
the six facts plus measured λ) — that is the next substantive session, not this one.
`row_L` stays OFF in the baseline until then.
