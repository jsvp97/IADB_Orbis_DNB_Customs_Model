# Firm entry with granular competition — the architecture, and why it is unique

**Status: decided and built.** Two-tier entry, embedded in the general equilibrium.
Uniqueness comes from a theorem about the *market*, not from an entry order.

Code: `entry_uniqueness.jl` (one market: the theorem and every test of it),
`entry_ge.jl` (entry inside the GE loop). Both run standalone.
Superseded prototypes: `entry_approaches.jl`, `entry_facts.jl` — kept because their
Bertrand comparison and Poisson–Pareto experiments are still the source for §6 and §7.

---

## 1. The problem, and why the obvious answers failed

Facts 1, 2 and 3 were calibration targets or inputs because the model had no entry.
Adding entry to an Atkeson–Burstein oligopoly is a discrete game of strategic
substitutes, which normally has many equilibria.

| Paper | Entry | Multiplicity |
|---|---|---|
| **Gaubert–Itskhoki (2020 JPE)** | fixed market-access cost; potential entrants Poisson–Pareto | **solved** — rank entrants by marginal cost, lowest moves first ⇒ unique cutoff |
| **Gaubert–Itskhoki–Vogler (2021 JME)** | same | same cutoff, plus firm-specific tariffs |
| **Yang (2023)** | multi-plant parents choose *sets* of locations | **not solved** — proves existence (quasi-aggregative + submodular, Jensen 2010), then *selects* by imposing an entry order |

Both answers were unusable here.

**Gaubert–Itskhoki needs every entrant to be a separate competitor.** Their condition
(c) — every incumbent's share falls when one more firm enters — fails the instant a
parent **internalises** several affiliates, because adding its own affiliate *raises*
its share. Internalisation is exactly what makes Fact 4 work, so it cannot be given up.
Measured: the weaker condition that would still deliver a cutoff (the marginal
entrant's incremental profit monotone in K) fails in **226/300** cases with
internalisation and **0/300** with independent varieties.

**Yang's order selection is not acceptable.** With G parents there are G! orders, no
principle picks one, and the outcome selected is not a property of the model. This was
Christian's objection and it is correct.

---

## 2. The fix: split the firm's own effect from its rivals'

The earlier pass asked the wrong question. The Gaubert–Itskhoki condition mixes
together what a firm does to itself and what its rivals do to it. Separate them and
both behave.

Write, for one market and one parent *g*,

```
K_g    = sum over its ACTIVE affiliates of c_i^(1-sigma)      capability stock
psi(S) = S mu(S)^(sigma-1)                                    so S_g = psi^{-1}(K_g/A)
Pi(S)  = E S [ (1-S)/sigma + S/eta ]                          gross operating profit
```

A parent's payoff depends on its own affiliates **only through the scalar `K_g`**.
That one line is the whole trick: internalisation is already inside `K_g`, so the
parent's entry problem is one-dimensional however many affiliates it owns. Its payoff
is `Omega(K; A) = Pi(psi^{-1}(K/A))` minus fixed costs, and:

- **(A) Own concavity.** `Omega` is strictly concave in `K`. So a parent's best
  response is a **cutoff on its own list** — Gaubert–Itskhoki's cutoff applied *within
  the parent* rather than across the market. Nothing is assumed about order across
  parents.
- **(B) Strategic substitutes.** `d Omega_K / dA < 0`, i.e. `|eps_G| < eps_psi` with
  `G = Pi'/psi'`.

Given (A) and (B), each parent's optimal `K_g` is non-increasing in `A`, while `A` is
strictly increasing in every `K_g`. Two equilibria with `A < A'` would need every
parent weakly smaller at `A'` and the total strictly larger. Contradiction.

**The entry equilibrium is unique.** No entry order, no selection rule, and parents
still internalise their affiliates.

### The hypothesis has content — state it, do not hide it

Condition (A) is unconditional. Condition (B) is not:

| sigma | eta | share frontier | sharper Nash notion |
|---|---|---|---|
| 5 | 1 | **0.548** | 0.532 |
| 5 | 1.5 | 0.936 | 0.652 |
| 5 | 2.5 | 1.000 | 0.624 |
| 4 | 1 | 0.556 | 0.536 |
| 8 | 1 | 0.532 | 0.524 |
| 3 | 1 | 0.572 | 0.540 |

At the calibration the theorem needs **no single parent above about 55% of a market**.
Past that, Cournot profit is convex enough in the share that a parent's marginal value
of capability *rises* with the aggregate and substitutability lapses. The solver checks
the realised shares at the solution rather than assuming; the frontier is only
sufficient, and §3 finds uniqueness beyond it too.

### Equilibrium concept, stated plainly

Parents internalise competition among their own affiliates when setting **quantities**
— that is the group markup `mu(S_g)`, and Fact 4 rests on it. When deciding **entry**
they take the market aggregate as given: a firm compares its equilibrium profit with
the fixed cost. That is Gaubert–Itskhoki's own free-entry condition and standard in
quantitative granular-entry models. The sharper alternative — score every deviation on
a fully re-solved market — is implemented as `nash_refine` and the gap is reported (§3).

---

## 3. Evidence

`entry_uniqueness.jl`. All figures from the committed run.

**The two conditions**, by brute numerical differentiation of `Omega`, using none of
the algebra above:

```
CONDITION A  Omega strictly concave in own K   :  violated   0 / 400
CONDITION B  Omega_K strictly falling in A     :  violated 101 / 400  over ALL shares
                                                            1 / 300  inside the frontier
```

**Uniqueness, by brute force over every configuration** — each parent free to choose
any subset of its own affiliates, every deviation scored on the **re-solved** market
equilibrium. This is precisely the case the GI cutoff cannot handle:

```
markets with a pure-strategy entry equilibrium  : 120 / 120
  of those, UNIQUE                              : 120
  of those, multiple                            :   0
solver issued its uniqueness CERTIFICATE        : 120
certificate issued where brute force found many :   0
```

**The cheap solve versus the sharper notion:**

```
cheap answer already an exact Nash equilibrium : 82 / 120
after one nash_refine pass it is one           : 120 / 120
```

**The mechanism, tested directly on the discrete problem:**

```
K*(A) not non-increasing, whole grid           : 193 / 1200
  ... restricted to shares inside the frontier :   0 / 1200
markets where sum_g S_g(A) is not decreasing   :   0 / 200
```

The certificate is what the model carries into the GE loop, where brute force is
unaffordable: the solver re-evaluates the clearing function on a wide grid and confirms
it is decreasing with exactly one sign change, **market by market**.

---

## 4. Two tiers, one theorem

- **Tier 1 — granular multinational parents.** Several potential plants in the same
  market, different production locations. They internalise: the markup is a function of
  the parent's *total* share. Fact 4 lives here.
- **Tier 2 — the local fringe.** One plant, one variety, its own group, no
  internalisation. Gaubert–Itskhoki's entrants.

Both are the same object — a group choosing a subset of its own list, tier 2 being the
one-item case — so the two tiers need **one** theorem, not two. Unlike Yang (2023),
where the fringe is passive with fixed locations, here the fringe plays.

Computationally the theorem pays for itself: concavity means only the Pareto frontier of
`(K, F)` can ever be optimal, so a parent with *n* potential plants is scanned in a
handful of candidates rather than `2^n`.

---

## 5. Entry inside the general equilibrium

`entry_ge.jl`. Serving market *n* costs

```
F[a,n] = f_k * ( w[h]^alpha_k * w[l]^(1-alpha_k) ) * fdist[l,n]
```

paid in the same factor bundle as production — HQ services at the **parent's** wage,
the rest at the **host's**. Three consequences, all load-bearing:

1. the system stays homogeneous of degree one in wages, so the numeraire is still
   legitimate;
2. fixed costs are real resources: they enter labour demand, and profits distributed
   through `theta` are **net** of them;
3. entry responds to wages, to the I–O price loop, and to market size — so the
   extensive margin of multinational production responds to policy.

**Where entry sits, and what it does to the old contraction proof.** The loop is
arranged so the proof survives where it can:

```
OUTER    the entry configuration (discrete)
  INNER  the I-O price loop, configuration FROZEN  -> the old map, modulus max nu
  THEN   expenditures and incomes, one linear solve
  THEN   one pass of entry decisions at the converged (P, E)
```

Freezing the configuration is not a convenience: it is what keeps the inner block
provably convergent. Prices do not depend on expenditure, so with the configuration
fixed the continuous block is *exactly* the block verified before. Measured: inner
modulus **0.550** against `nu = 0.550` — on the nose.

The outer discrete map has no proof and is measured instead. One structural fact makes
it behave, and it is verified: **entry is invariant to a uniform change in costs.**
Scale every delivered cost by a common factor and shares, markups and profits are all
unchanged, so no margin moves. Only *relative* cost changes move entry, and the I–O
loop moves costs largely in common.

**The integer problem, sized rather than asserted away.** Entry is a choice over whole
plants, so an exact fixed point in integers need not exist, and the outer loop can cycle
over one or two marginal plants. The loop does not pick a winner by fiat: it keeps the
configuration with the fewest firms wanting to move and reports the residue. In the
audit economy that is **2 slots out of 432**. `run_integer` scales the economy up and
shows the share falling. The point worth keeping: this is a **near-miss on existence,
not a multiplicity** — nothing here requires choosing between equilibria.

---

## 6. What entry cost, and what it bought

**Cost of switching conduct (unchanged from the earlier pass).** Writing profit as
`(E/sigma) s [1 + kappa s]`, the ownership correction is proportional to `kappa`:

| sigma | eta | kappa Cournot | kappa Bertrand | ratio |
|---|---|---|---|---|
| 5 | 1 | 4.00 | 0.80 | **5.0×** |
| 5 | 1.5 | 2.33 | 0.70 | 3.3× |
| 8 | 1 | 7.00 | 0.88 | 8.0× |

Bertrand buys cleaner entry and costs headline magnitude by a factor of five. **Not
taken** — the theorem above delivers uniqueness under Cournot, so the trade is no longer
necessary.

**Ruled out on the maths, not on taste:** monopolistic competition with constant
markups. `kappa = 0`, the `S²` term vanishes, and the ownership result disappears
entirely.

---

## 6b. Calibration with entry — one parameter gained, two re-fitted

**Entry pins the market-access fixed cost, which used to be free.** Facts 2, 3 and 4
are all measured on the LAC *export* sample. Without entry every affiliate exported by
construction, so that sample was rich whatever the parameters. With entry it is an
outcome — and the first guess of `f = 0.006` collapsed it to **six plants**, at which
point those three facts are not measurable at all (parent HHI 1.00, affiliate→parent
ratio exactly 1.00). Scanning it:

| fixed cost | LAC exporters | parents | parent countries | hhi affiliate → parent |
|---|---|---|---|---|
| 0.0060 | 6 | 6 | 2 | 0.555 → ×1.00 |
| 0.0020 | 31 | 24 | 3 | 0.148 → ×1.40 |
| **0.0006** | **59** | **43** | **4** | **0.080 → ×1.51** |
| 0.0002 | 71 | 53 | 4 | 0.071 → ×1.51 |

An order of magnitude below the first guess, and it stops moving below 0.0006. This is
a **gain**: a parameter that used to be unconstrained is now disciplined by how many
firms actually export.

**Then the edge and the capability slope, jointly, at `f = 0.0006`:**

| slope | edge | Fact 1 | parent HHI | top parent | hhi aff → par | Fact 2 gradient |
|---|---|---|---|---|---|---|
| 0.0 | 0.10 | 0.304 | 0.340 | 0.447 | 0.088 → 0.096 (×1.09) | **−0.05** |
| 0.0 | 0.30 | 0.483 | 0.316 | 0.400 | 0.085 → 0.106 (×1.24) | −0.14 |
| 0.6 | 0.10 | 0.464 | 0.324 | 0.426 | 0.091 → 0.112 (×1.23) | +0.14 |
| **1.2** | **0.10** | **0.631** | **0.314** | **0.414** | **0.112 → 0.154 (×1.37)** | **+0.26** |
| 1.2 | 0.30 | 0.788 | 0.310 | 0.409 | 0.127 → 0.187 (×1.48) | +0.08 |
| 1.2 | 0.50 | 0.916 | 0.302 | 0.388 | 0.144 → 0.222 (×1.55) | −0.17 |
| — | — | *0.47–0.74* | *0.130* | *0.250* | *0.192 → 0.215 (×1.12)* | *+0.43* |

Reading, fact by fact:

- **Fact 1 — hit**, and note how *steep* it is: the MNE share runs 0.30 → 0.63 → 0.92
  as the edge goes 0.10 → 0.50. With the extensive margin a small productivity edge
  tips whole markets, so Fact 1 is far more informative about the edge than it was
  when the roster was fixed. The calibrated edge falls from **0.188 (1.21×) without
  entry to 0.10 (1.11×) with it.**
- **Fact 3 — GENERATED.** Nothing tells the model which countries host parents: the
  capability law is identical everywhere and each potential parent's origin is drawn
  uniformly. It still overshoots (0.314 vs 0.130) in a four-country world, exactly as
  the earlier prototype found it converging as the world grows.
- **Fact 4 — GENERATED, and it survived the rewrite.** ×1.37 against ×1.12 in the
  data. This is the fact that forced the whole theorem: it needs internalisation, and
  internalisation is what the Gaubert–Itskhoki cutoff cannot host.
- **Fact 2 — still a target, and entry makes it harder.** Without the capability
  channel the gradient is **negative** (−0.05, and −0.22 at a larger edge), because a
  foreign affiliate pays its *parent's* wage on the `alpha_k` share and parents sit in
  high-wage countries; the extensive margin amplifies that wrong-sign force. It takes a
  slope of 1.2 to reach +0.26 against the data's +0.43 — twice the 0.588 needed without
  entry, and it still undershoots.


## 7. Facts at the calibrated model, and the one that got worse

672 potential affiliate-destination pairs, **498 active (74.1%)**, foreign entry rate
0.889. Full output in `run_entry_ge.txt`.

| # | Fact | Model with entry | Data | Verdict |
|---|---|---|---|---|
| 1 | MNE share of LAC export value | 0.631 | 0.47–0.74 | target, hit |
| 2 | Foreign MNEs in complex goods | 0.16 / 0.26 / 0.33 / 0.27, slope **+0.26** | +0.43 | target; slope 1.20 and still short |
| 3 | Parents from few countries | HHI **0.314**, top **0.414** | 0.130 / 0.250 | **GENERATED**, overshoots in a 4-country world |
| 4 | Grouping affiliates raises HHI | 0.112 → 0.112 → **0.154** (×1.37) | ×1.12 | **GENERATED — and it survived the rewrite** |
| 5 | MNE presence raises non-MNE exports | **−91.6%** | strongly positive | **WORSE with entry** |

**Fact 4 is why the theorem was worth the trouble.** Its ordering is a prediction, it
requires internalisation, and internalisation is precisely what the Gaubert–Itskhoki
cutoff cannot host. The theorem exists so this fact could survive entry. It did.

**Fact 5 — the open question from the last pass, answered, and the answer is no.** The
conjecture was that multinationals buying inputs locally would lower the price index
enough to let *more* local firms clear their own entry cutoff, flipping the sign for
the first time. Tested: adding 12 multinationals to a sector cuts non-MNE export value
by **91.6%** against **42%** with a fixed roster, and active local plant-market pairs
fall 86 → 48. **The extensive margin amplifies the contradiction.** Business stealing
operates on the entry margin too and dominates the input-price channel. The ≈0.15
spillover is still needed, and it now has *more* to overturn. Do not write that entry
fixes Fact 5.

---

## 8. Open items
1. **Fact 5 is now the only outright contradiction, and entry made it worse.** Find
   what carries the positive sign: the direct Javorcik spillover is the placeholder,
   but its required size has gone up, not down.
2. **Fact 2 still undershoots** (+0.26 against +0.43) at a capability slope of 1.20,
   twice what the no-entry model needed. The profile is also non-monotone across
   sectors. Either the capability channel needs a different shape, or `alpha_k` does.
3. **Fact 3 overshoots in a small world** (0.314 against 0.130). The earlier prototype
   showed this converging as the world grows; re-run at N = 9–12 to confirm the
   overshoot is the world and not the mechanism.
4. **Fact 1's level is still one fitted parameter.** Making it fully endogenous needs a
   fixed cost of *becoming* multinational, not just of market access.
5. **Correlation of a parent's capability draws across locations** (`corr` in
   `entry_economy`) is the discrete analogue of the multivariate Fréchet correlation in
   Ramondo–Rodríguez-Clare. Tintelnot (2017) argues it must be near zero to match US
   export-platform sales. It is exposed as a knob and reported, not estimated.
6. **Layer 4 (optimal tariff) is now unblocked in a stronger form**: the extensive
   margin of multinational production responds to tariffs, so a tariff can drive
   affiliates out rather than only shrinking them. That channel does not exist in the
   fixed-roster model and it should be signed before any policy text is written.
