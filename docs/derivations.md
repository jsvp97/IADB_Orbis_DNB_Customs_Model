# Derivations

One section per layer. Every closed form here has been checked against the solver
on random parameter draws in **two independent implementations** (`src/cournot_pe.jl`,
`scripts/verify/cournot_pe.py`). The check that matters is stated after each result.

---

## Layer 0 — single market, group Cournot, nested CES

A market is one (destination `d`, class `k`) cell. Upper tier Cobb–Douglas (η = 1),
so class expenditure `E` is fixed. Lower tier CES(σ) over varieties `i = 1..N`.
Variety `i` belongs to group `g(i)`. Groups choose quantities, internalising
cannibalisation across their own affiliates. `c_i` is the delivered marginal cost.

### 0.1 Demand

With `Q = (Σ_j q_j^((σ-1)/σ))^(σ/(σ-1))` and `P·Q = E`, homogeneity gives
`p_i = E·Q_i/Q` where `Q_i = ∂Q/∂q_i = Q^(1/σ) q_i^(-1/σ)`. Writing
`A ≡ Σ_j q_j^((σ-1)/σ) = Q^((σ-1)/σ)`:

```
p_i = E · q_i^(-1/σ) / A
s_i = p_i q_i / E = q_i^((σ-1)/σ) / A          Σ_i s_i = 1
```

### 0.2 The group first-order condition

`Π_g = E·S_g − Σ_{i∈g} c_i q_i` with `S_g = Σ_{i∈g} s_i`. Differentiating `S_g`:

```
∂S_g/∂q_i = ((σ-1)/σ)·(q_i^(-1/σ)/A)·(1 − S_g)
```

and since `E·q_i^(-1/σ)/A = p_i`, the FOC `E·∂S_g/∂q_i = c_i` gives

```
p_i = c_i · μ_g          μ_g = σ / [ (σ-1)(1 − S_g) ]
```

**The markup depends on the group's TOTAL share, not the variety's, and is common
across all of a group's affiliates.** This is Atkeson–Burstein with η = 1, applied
to groups rather than firms. It is the model-level counterpart of Figure 6.

Since `c_i q_i = r_i/μ_g`:

```
Π_g = E·S_g·(1 − 1/μ_g) = (E/σ)·S_g·[ 1 + (σ-1)·S_g ]
```

Limits: `S_g → 0` gives `E·S_g/σ` (CES: revenue over σ); `S_g → 1` gives `E`.

> **Checked.** The primitive profit function `Σ_{i∈g}(p_i(q) − c_i)q_i`, which uses
> none of the above, has own-quantity gradient `9.0e-16·E` at the point these
> formulas produce, over 300 random markets (`test_A`). Random own-quantity
> deviations never raise profit (worst gain `−1.1e-12`, `test_B`), so this is a
> maximum and not merely a stationary point. An independent global solve from a
> CES starting point returns the same equilibrium to `2.2e-16` (`test_A`).

### 0.3 Equilibrium: existence and uniqueness

Let `A = Σ_i p_i^(1-σ)`, `K_g = Σ_{i∈g} c_i^(1-σ)`, `B = (σ/(σ-1))^(1-σ)`. Then
`S_g = μ_g^(1-σ) K_g / A = B(1−S_g)^(σ-1) K_g / A`.

- **Inner:** `x = B(1−x)^(σ-1)K_g/A`. LHS rises 0→1, RHS falls. Unique root.
- **Outer:** `Σ_g S_g(A) = 1`. Each `S_g` strictly decreasing in `A`. Unique `A`.

Consistency: `Σ_i p_i^(1-σ) = Σ_g μ_g^(1-σ)K_g = A·Σ_g S_g = A`. The construction
**proves existence and uniqueness** — write it up as a lemma.

Do **not** use a damped fixed point on shares; it does not converge for all
parameters.

### 0.4 Comparative statics (new — the useful part)

From `ln S_g = ln B + (σ-1)ln(1−S_g) + ln K_g − ln A`:

```
d ln S_g = Λ_g ( d ln K_g − d ln A )        Λ_g ≡ (1−S_g)/(1+(σ-2)S_g)
```

`Σ_g S_g = 1` implies `Σ_g S_g d ln S_g = 0`, which pins the aggregator:

```
d ln A = Σ_g ω_g d ln K_g                   ω_g ≡ S_g Λ_g / Σ_h S_h Λ_h
```

The `ω_g` are **incidence weights** and they sum to one. Since `P = A^(1/(1-σ))`
and a tariff on group `g` has `d ln K_g = (1−σ) d ln(1+t_g)`:

```
d ln P / d ln(1+t_g) = ω_g                              (price-index incidence)
d ln p_g / d ln(1+t_g) = 1 + S_g Λ_g (1-σ)(1-ω_g)/(1−S_g)   (own pass-through)
```

`Λ_g` is decreasing in `S_g`, so **a dominant group has `ω_g < S_g`**: it absorbs the
tariff in its markup and the consumer price index rises by less than CES predicts.
In the atomistic limit `Λ_g → 1`, `ω_g → S_g`, and pass-through → 1, recovering CES.

Rivals are not passive: `d ln S_h = Λ_h ω_g (σ-1) d ln(1+t_g) > 0` for `h ≠ g`.
Their markups rise, so a tariff on one group **raises its rivals' prices**. That
strategic price increase is a real welfare cost of tariffs that CES cannot see.

> **Checked.** Max `|d ln P − ω_g|` = `1.0e-9` and max pass-through error `1.6e-9`
> against central differences of the solver, 200 markets in Julia and 60 in Python.
> Floor is the finite-difference step.

### 0.5 Ownership accounting

With ownership weights `θ_g` for country H:

```
Π_H = (E/σ) [ Σ_g θ_g S_g  +  (σ-1) Σ_g θ_g S_g² ]
              └ CES term ┘     └ granular correction ┘
```

Decompose the granular term against what country-level data can produce:

```
Σ_g θ_g S_g²  =  θ̄ · HHI  +  Cov_S(θ, S)          θ̄ ≡ Σ_g θ_g S_g
```

where `Cov_S` is the **share-weighted** covariance of ownership with market share.
`θ̄ · HHI` is the most a researcher with a country-level ownership share and a
published HHI could construct. `Cov_S(θ, S)` requires firm-level global-ultimate-parent
identity. **That term is the contribution.**

> **Checked.** `test_D`. With H owning the largest 20% of groups the ratio
> `total/naive` is 1.41–1.47× across atomistic/moderate/granular structures; owning
> the smallest 20% gives 0.12–0.26×. The correction is **asymmetric in which firms
> you own**, which is the whole point.

⚠️ **This is an identity about the LEVEL of profit income, not a policy result.**
Optimal tariffs depend on `dΠ_H/dt`, not `Π_H`. See §0.6. The current CLAUDE.md
Proposition P2 conflates the two.

### 0.6 Welfare and the sign of optimal policy (new — read this before Layer 4)

Money-metric welfare of the tariff-setting destination, this class only:

```
W = −E ln P + T + Σ_g θ_g Π_g          T = Σ_i t_i a_i q_i,  c_i = a_i(1+t_i)
```

At free trade, for an ad valorem tariff on group `g`:

```
(1/E)·dW/dt|₀  =   S_g/μ_g              tariff revenue
                 − ω_g                  consumer price index
                 + Σ_h θ_h (dΠ_h/dt)/E  home-owned profit
```

Two things to notice.

1. **The tariff base is markup-deflated.** An ad valorem tariff is levied on
   customs value `a_i q_i = r_i/μ_g`, not on the market value `r_i`. Taxing a
   high-markup supplier collects little revenue per unit of distortion. This term
   alone makes ad valorem tariffs a poor rent-extraction instrument, and it is
   invisible in models without markups.

2. **Layer 0 has constant marginal cost, so there is no terms-of-trade motive at
   all.** The only reason to tax is rent extraction; the only reason to subsidise
   is the markup distortion. So Layer 0 answers a clean question: which wins?

> **Checked.** Analytic expression matches finite differences of the full solver to
> `2.6e-10` over 60 markets with random `θ` (`test_E`).

**Result.** Over 14,024 random (market, group) pairs with `θ = 0`, a tariff beats
free trade in **0.26%** of cases. The frontier — smallest dominant share `S` at
which a tariff pays, against a fringe of `n` equal rivals — is identical in the
Julia and Python implementations:

```
 sigma      n=1      n=2      n=4      n=9     n=19     n=49
   2.5        .        .        .        .        .        .
   3.0        .        .        .        .        .        .
   5.0        .        .        .    0.361    0.201    0.135
   8.0        .        .    0.393    0.178    0.096    0.052
  12.0        .        .    0.318    0.144    0.073    0.034
  20.0        .    0.593    0.283    0.127    0.062    0.026
```

Reading: **a tariff pays only against a dominant group facing a fragmented fringe,
and only at high σ.** Against one equal-sized rival it never pays, at any σ or `S`.

Two consequences the project must absorb:

- **The competitive fringe is not a calibration nuisance. It is what creates the
  tariff motive.** Open decision 1 in CLAUDE.md is therefore not a modelling
  convenience question; the answer determines whether there is a tariff to correct.
  A markup cap would suppress the mechanism. Use a fringe.
- **Layer 0 cannot host the optimal-tariff result.** For ordinary market structures
  optimal Layer-0 policy is an import *subsidy*, and ownership makes the subsidy
  larger. Ownership has the sign the project predicts, but it is correcting a
  subsidy. A positive optimal tariff needs either upward-sloping foreign supply
  (GE, Layer 3) or an extensive margin. Do not write the tariff propositions
  against Layer 0.

Worked example (σ = 5, four groups, `c = exp(linspace(0,0.6))`, shares
`[.377 .289 .204 .130]`), optimal policy on group 1:

| `θ₁` | `t*` | `dW/dt` at 0 |
|---|---|---|
| 0.00 | −0.261 | −0.086 |
| 0.25 | −0.462 | −0.150 |
| 0.50 | −0.648 | −0.215 |
| 1.00 | −0.868 | −0.343 |

### 0.7 Why proportional profit loss is the wrong statistic

CLAUDE.md §5 records that `d lnΠ_g/d ln(1+t)` is −2.08 under Cournot versus −2.95
under CES, and reads this as cutting against the thesis. Two corrections.

- The comparison is not like-for-like: the CES benchmark `−(σ-1)(1−S)` is evaluated
  at the *Cournot* equilibrium share. A CES firm with the same cost would have a
  different share.
- More importantly, the elasticity is not the object in the welfare formula. What
  enters is `dΠ_g/dt` in **levels**, and `Π_g` is much larger under Cournot. The
  mechanism that makes ownership matter is incomplete pass-through: the share of a
  tariff borne by the foreign firm is `1 − ρ_g`, increasing in `S_g`. Under CES
  pass-through is exactly 1 and **no rent is extracted at all**. Rent extraction is
  a pure oligopoly phenomenon, and it is precisely what home ownership neutralises.

That is the correct statement of the mechanism, and it runs through pass-through,
not through the profit elasticity.

---

## Layer 0b — general upper-tier elasticity η

`η = 1` (Cobb–Douglas) is a knife-edge sitting exactly where the paper's punchline
lives: at `S → 1` the markup is unbounded and tariff pass-through goes to zero.
Both are artefacts. The solver now takes arbitrary `η` with `σ > η ≥ 1`.

Upper tier CES(η), so class expenditure `E = D·P^(1-η)` with `D` a fixed demand
shifter (the PE approximation: the class is small enough that the destination's
aggregate price index does not move). Inverse demand becomes

```
A_q = Σ_j q_j^((σ-1)/σ)        P = D^(1/η) A_q^(σ/((1-σ)η))
p_i = D^(1/σ) P^((σ-η)/σ) q_i^(-1/σ)
```

Revenue shares are unchanged, `s_i = q_i^((σ-1)/σ)/A_q`. Differentiating,
`∂lnP/∂ln q_i = −s_i/η`, and the group FOC becomes

```
1/μ_g = 1 − (1−S_g)/σ − S_g/η                (Atkeson–Burstein, applied to groups)
Π_g   = (E/σ)·S_g·[ 1 + (σ/η − 1)·S_g ]
```

`η = 1` recovers `μ_g = σ/((σ-1)(1−S_g))` and the `(σ−1)` coefficient. Requiring
`σ > η` is what makes markups increase in share; it also preserves monotonicity of
the inner bisection, so **existence and uniqueness carry over unchanged**. For
`η > 1` the monopoly markup is `η/(η−1)`, finite.

`Λ_g` generalises to `Λ_g = 1/[1 + (σ−1)S_g μ_g (1/η − 1/σ)]`, and `ω_g`, the
price-index incidence and the welfare formula are otherwise unchanged.

> **Checked.** The primitive-profit gradient vanishes to 3.7e-16–5.8e-16 at
> `η ∈ {1, 1.5, 2.5}`; closed-form profit matches the solver to 3.6e-15; `ω` and
> pass-through match finite differences to ~1e-9 at every `η`.

**The ownership result does not depend on the knife-edge.** Only the prefactor
`(σ/η − 1)` moves; the `Σ_g θ_g S_g²` object and its decomposition are unchanged.

One bug this exposed. Money-metric consumer surplus is `E/(η−1)`, whose derivative
is `−E dlnP` for any `η`. Using `−E ln P` as the *level* is correct only at `η = 1`,
where `E` is constant — `ln P` is not scale-free, so the spurious `−lnP·dE/dt` term
appears otherwise. That error showed up as a 3.1e-01 discrepancy in the welfare
cross-check at `η = 1.5` and is now fixed.

---

## Layer 1 — many markets, ownership, covariance

`src/layer1_markets.jl`, mirrored in `scripts/verify/layer1_markets.py`.

A **panel** is a list of markets. Groups have **global identity** — parent `g` may
operate in many markets — and ownership `θ_g` is a global object, constant across
markets. What varies is which parents are present and how big they are.

### 1.1 The aggregate decomposition

With `w_m = E_m/ΣE_m` and `T_m = Σ_g θ_g S_gm²`:

```
Σ_m w_m T_m  =  θ̄_agg · H̄               (1) naive
              + Cov_w(θ̄_m , H_m)         (2) BETWEEN markets
              + Σ_m w_m Cov_S,m(θ , S)    (3) WITHIN markets
```

`θ̄_m = Σ_g θ_g S_gm` is market `m`'s ownership share of revenue, `H_m` its
Herfindahl, `θ̄_agg = Σ_m w_m θ̄_m` the aggregate value-weighted ownership share,
`H̄ = Σ_m w_m H_m`. **This is the paper's headline in one equation**, because the
three terms map exactly onto a data-availability ladder:

| what you have | what you can compute |
|---|---|
| country ownership share + published HHI | (1) only |
| + ownership share market by market | (1) + (2) |
| + firm-level global-ultimate-parent identity | (1) + (2) + (3) |

(2) asks whether home parents sit in the *more concentrated markets*.
(3) asks whether, *inside* a market, home parents are the *bigger players*.
Only the ORBIS/D&B match delivers (3).

> **Checked.** `Π_H` from the aggregate formula matches a brute-force
> `Σ_m Σ_g θ_g Π_gm` over the solver to 3.4e-16 (Julia) / 4.3e-16 (Python) across
> 24 panels at `η ∈ {1, 1.5, 2.5}`. Uniform `θ` drives **both** covariance terms to
> ~1e-17 and the ratio to exactly 1.000000 — the correct null. A fringe of growing
> mass drives `H̄ → 0`, mean markup → `σ/(σ-1)`, and the granular correction → 0,
> recovering Ownership Irrelevance.

### 1.2 Two traps in reporting the headline ratio

Both established numerically; both would be easy to get wrong in the paper.

- **`total/naive` does not converge to 1 as markets become atomistic.** It
  converges to the home-firm **size premium**: with `S_home = a/N` against a fringe
  at `1/N`, the ratio tends to `a` while numerator and denominator both vanish. The
  proportional understatement survives even as the correction dies. **Never report
  the ratio without the level of the granular correction beside it.**
- **A positive within-term arises with no ownership sorting at all**, purely because
  MNEs are larger than the fringe. In the synthetic panel that alone gives 1.08×.
  This is the **MNE-size channel** and it is economically real, but it is not the
  ownership-sorting channel and must be reported separately or the two get
  conflated. The clean null is uniform `θ` over *all* firms, which gives exactly 0.

### 1.3 Uniqueness of the Cournot equilibrium

Full statement and numerical verification in `src/uniqueness.jl`. Let
`ρ = (σ-1)/σ`, `X_g = Σ_{i∈g} q_i^ρ`, `B_g = Σ_{j∉g} q_j^ρ`.

**Step 1 (sufficient statistic).** Group revenue depends on its own quantities only
through `X_g`:

```
R_g = D^(1/η) · X_g · (X_g + B_g)^e         e = (η-σ)/((σ-1)η) ∈ (-1, 0)
```

**Step 2.** `dR/dX = (X+B)^(e-1)[(1+e)X + B] > 0` and
`d²R/dX² = e(X+B)^(e-2)[(1+e)X + 2B] < 0`. Increasing and strictly concave.

**Step 3.** Cost minimisation within the group (`min Σc_i q_i` s.t. `Σq_i^ρ = X`)
gives `q_i ∝ c_i^(-σ)` and

```
C_g(X) = X^(σ/(σ-1)) · K_g^(-1/(σ-1))       K_g = Σ_{i∈g} c_i^(1-σ)
```

Strictly convex, since `σ/(σ-1) > 1`. The same `K_g` the solver uses — it is the
group's cost-efficiency index, and it is *all* that matters about a group's internal
cost structure.

**Step 4.** `Π_g = R_g − C_g` = concave − strictly convex = strictly concave, so the
FOC is necessary *and* sufficient and the best response is unique. Directly in the
choice vector: `X(q) = Σq_i^ρ` is concave (`ρ<1`), `R` is concave and increasing, and
a concave increasing function of a concave function is concave, so **`Π_g(q)` is
concave in the group's own quantity vector**. No corner: as `q_i → 0`, revenue falls
like `q_i^ρ` and cost like `q_i` with `ρ<1`, so `dΠ/dln q_i > 0` — producing zero is
never optimal and every equilibrium is interior.

**Step 5 (aggregate).** With `ψ(x) = x·μ(x)^(σ-1)` strictly increasing on `[0,1)`
(this needs `σ > η`), `S_g(A) = ψ^(-1)(K_g/A)` is unique and strictly decreasing in
`A`. So `Σ_g S_g(A)` falls strictly from `> 1` to `0` and crosses 1 **exactly once**.
Unique `A` ⟹ unique `S_g` ⟹ unique `μ_g`, `p_i`, `r_i`, `q_i`. Existence is free
because the construction is constructive.

> **Checked.** Step 1 to 8.5e-16 by reshuffling output within a group at constant `X`.
> Steps 2–3: zero sign violations in 2,000 draws. Step 4: zero concavity violations
> over 400 markets × groups × 5 chords, worst chord−midpoint gap −1.8e-7. Step 5: `ψ`
> monotone in 500/500 draws. **Decisive test:** 200 random starts × 5 market
> structures (G = 2…8, η = 1…2.5), starts spanning ~e¹², every one converging to the
> same equilibrium to ~1e-9.

This is why the solver bisects rather than iterating a fixed point: monotonicity
means bisection cannot land on a different root, because there is no different root.

### 1.4 The measurement operator (the denominator fix)

The model's `S_g` is a share of **destination-market absorption**. Figure 6's HHI is
a share of **LAC export value within an HS6**, pooled across destinations. Different
populations; both ≈5 effective firms, which is why the comparison looked fine.

Rather than rebuild the model's denominator in data we do not have, build the data's
estimator in the model:

```
1. keep in-sample varieties only (orig > 0)
2. pool across destinations within a product
3. aggregate to the firm concept (affiliate / parent×country / parent)
4. renormalise so shares sum to 1 within the observed sample
5. HHI per product, value-weighted mean across products
```

Step 4 is where the fringe cancels. Hence:

| n_fringe | λ | structural HHI | measured HHI (parent) |
|---|---|---|---|
| 0 | 1.000 | 0.244 | 0.079 |
| 16 | 0.548 | 0.092 | 0.080 |
| 256 | 0.133 | 0.010 | 0.107 |

Structural falls 24× (Python mirror: 30.8×); measured moves 1.36× (mirror: 1.22×).
**Figure 6 is near-invariant to the fringe and cannot identify it.**

Identification, corrected: **λ → fringe mass** (observable as LAC exports ÷
destination absorption), **measured HHI → number and dispersion of LAC exporters**.
Match jointly, since measured HHI drifts up slightly with the fringe through markup
compression.

### 1.5 Still open

- Calibrating the fringe is blocked on the **denominator problem**: Figure 6's HHI
  is a share of LAC exports within an HS6, while model `S_g` is a share of
  destination-market class expenditure. Different objects. Fix before targeting the
  0.192/0.215 moments.
- The Julia and Python panels use different RNGs, so Tests 3–4 agree
  qualitatively, not digit for digit. The cross-check that binds is Test 1
  (closed form vs brute force), which is exact in both.
