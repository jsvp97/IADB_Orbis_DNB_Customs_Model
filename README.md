# Multinational Ownership, Market Power, and Trade Policy — the model

A static general-equilibrium model of multinational production with **large firms**
(markups rise with a group's market share), **firm-level ownership**, and
**endogenous entry**, built to confront six stylized facts measured in matched
Orbis/D&B–customs data for ten Latin American origin countries (2006–2022).

**The contribution in one line:** a country's share of foreign profit depends not on
what fraction of *firms* it owns but on what fraction of the *big* ones — the correct
adjustment is an ownership-weighted concentration index, and the error of the standard
country-level calculation is exactly the covariance between ownership and firm size
(an identity, measured in our data at **+10% for the United States** and sign-varying
across owners).

*Coauthors: Sebastián Velásquez Palacios and Christian Volpe Martincus (IDB).*

## Read this first

| Document | What it is |
|---|---|
| [`docs/model_guide.pdf`](docs/model_guide.pdf) | **The full model, 45 pp.** Every central equation carries the same three-part gloss — *what it says / why we model it this way / where it comes from* — every proof is written out, and everything only checked numerically is collected in one section rather than left to be discovered. |
| [`docs/model_summary.pdf`](docs/model_summary.pdf) | The 5-page condensed companion. |
| [`CLAUDE.md`](CLAUDE.md) | The complete working record: every decision, every dead end, every trap, section by section. §0 is a one-screen current-state briefing. |
| [`docs/derivations.md`](docs/derivations.md) | The terse algebra. |

## What is proved

| Layer | Existence / uniqueness |
|---|---|
| One market (group Cournot/Bertrand) | **Proved, unconditional** (a decreasing curve crosses a level once) |
| Which factories operate (entry) | **Proved** — at most one outcome, *with* cannibalisation and *without* an assumed order of moves, under "no group above half of any market" (checked at the solution: largest share ≈ 0.30) |
| Input prices | **Proved** — contraction, modulus ν < 1 |
| Spending and incomes | **Proved** — linear system, ρ(B) < 1 |
| Wages | Contraction of the solver's own map (Birkhoff) wherever its Jacobian is positive; that positivity is **computer-certified on the continuum** of every wage vector within ±15 % of the solution (outward-rounded interval arithmetic, no sampling gap) and grid-checked far beyond. The honest limit — why no unconditional theorem exists — is stated in the guide. |

## The six facts

Five of six reproduced (three of them out-of-sample), one instructive failure:

| # | Fact | Model | Data | Status |
|---|---|---|---|---|
| 1 | MNE export share; foreign ≫ domestic | 0.69 (0.54/0.15) | 0.46–0.74 | level fitted; **split produced** |
| 2 | Foreign MNEs in complex goods | +0.42 | +0.43 | fitted |
| 3 | Parents from few countries | 0.27 | 0.13 | **produced**, overshoots |
| 4 | Grouping by owner raises concentration | ×1.94 | ×1.12 | **produced** — a prediction |
| 5 | MNE presence raises local exports | negative | +0.09 | **fails**; see below |
| 6 | Distance weaker for MNEs | signs+ordering | — | **produced**, too large |

**Fact 5 is handled the honest way.** The empirical fact survives a saturated
destination×product×year specification at half its published size, so it is real.
Cost-side spillovers (productivity *and* fixed-cost) are implemented and *measured
insufficient* — with fixed market spending, entry is zero-sum in value. The missing
margin is an **outside (rest-of-world) supplier in every market** (`row_L` switch):
with it, the extensive margin flips to the data's sign and the intensive elasticity
moves from ≈ −2.9 to ≈ −0.2 against the target +0.087. The joint recalibration with
that margin on is the model's next task; the guide documents the full dose–response.

## Running it

Base Julia (≥ 1.9), **no packages**. Run the programs one at a time.

```bash
julia mne_model.jl              # everything: theorems, GE audit, facts (~10 min)
julia mne_model.jl quick        # the same on a smaller world (~4 min)
julia wage_uniqueness.jl        # Theorem 4 + the counterexample
julia simple_model.jl           # Theorem 5 + the baseline comparison
julia interval_certificate.jl   # the continuum uniqueness certificate
julia experiments.jl <name>     # spillover | fspill | rowfact5 | tariff | nu | ...
```

Every committed `run_*.txt` is the exact output of the corresponding program; if you
change code, regenerate them and `grep -c ERROR` each (a Julia script can print a
stack trace and still exit 0 under a pipe).

The documents rebuild with `pdflatex` (three passes) in `docs/`; check
`grep -c "^!" model_guide.log` is 0 afterwards.

## Repository map

```
mne_model.jl              THE model: 3,000 lines, one file, no dependencies.
                          Calibration in one CALIB block at the top.
wage_uniqueness.jl        Theorem 4 (gross substitutes under two checked
                          conditions) and the counterexample that shows the
                          second condition cannot be dropped.
simple_model.jl           Theorem 5: the capability baseline, and what the
                          simplification buys.
interval_certificate.jl   Outward-rounded interval arithmetic + centred forms:
                          uniqueness certified over whole boxes of wage vectors.
experiments.jl            The parameter scans quoted in the documents (Fact-5
                          spillover grids, the lambda margin, tariff-with-entry,
                          the nu comparison).
run_*.txt                 Committed outputs of all of the above.
docs/                     The guide, the summary, the algebra, the entry memo,
                          and the stylized-facts document.
superseded/               The historical record: entry prototypes, the
                          pre-consolidation layer files, and stale runs.
                          Kept deliberately — struck results beat deleted ones.
```

The empirical pipeline (Stata/Python, firm-level data) lives in a separate
repository: [`IADB_Orbis_DNB_Customs_Final`](https://github.com/jsvp97/IADB_Orbis_DNB_Customs_Final).
The three reference papers cited throughout are not redistributed here.
