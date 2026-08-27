# The model, explained from scratch

This document explains every piece of the model in plain language: what it is, why
we chose it, and what would break if we chose otherwise. It is written so you can
read it once and then explain the model to someone else at a whiteboard.

`docs/derivations.md` has the algebra. This file has the reasoning. The code is
`src/cournot_pe.jl` (one market), `src/layer1_markets.jl` (many markets and the
measurement), `src/uniqueness.jl` (the proof that there is only one equilibrium).

---

## 1. The question, in one paragraph

The United States puts tariffs on imports from Latin America. But a large share of
those imports is produced by affiliates of companies **owned by Americans**. When
the US taxes those imports, part of what it "takes" is taken from its own
shareholders. So the correct tariff should be lower. Everyone knows this. What
nobody has been able to measure is that ownership is **not spread evenly**: US
parents tend to be the *biggest firms in the most concentrated markets*, and that is
exactly where profits are largest. We have firm-level ownership matched to customs
transactions, so we can measure it. The model exists to say precisely what
"it" is and how much country-level data misses.

---

## 2. The building blocks

### 2.1 A "market" is one destination × one product

**What.** A market is a cell: Brazil × HS6 870323, say. Firms compete inside a cell
and not across cells.

**Why.** Two Colombian firms shipping auto parts to Brazil compete. A Colombian
flower exporter to the Netherlands is irrelevant to them. Concentration and market
power only mean something inside a cell.

**Cost of this choice.** It rules out competition across products. A firm cannot
respond to a tariff on auto parts by moving into flowers. Fine for the tariff
questions we ask; it would matter for a study of entry.

### 2.2 Demand has two tiers

**What.** Buyers first split spending across products (elasticity **η**), then
across varieties within a product (elasticity **σ**).

**Why two tiers.** We need substitution *within* a product to be much easier than
*across* products. Two brands of the same auto part are near-substitutes; an auto
part and a banana are not. One tier cannot express that.

**Why CES.** It is the standard workhorse, it gives closed forms, and — more
importantly — it makes our benchmark honest. Under plain CES with many small firms,
ownership does not matter at all. So CES gives us the **null hypothesis for free**.
If we found the ownership effect under a functional form that *builds it in*, the
result would be worthless. Here we recover "ownership is irrelevant" exactly in the
limit, and everything we report is a departure from it.

**σ > η is a real assumption, not a technicality.** It says goods within a product
are closer substitutes than goods across products. If it were violated, a firm's
markup would *fall* as it got bigger, which is economically backwards, and the
uniqueness proof would collapse. The code asserts it.

### 2.3 η = 1 is a special case we use on purpose, and check

When η = 1 (Cobb–Douglas across products), spending on each product is **fixed**.
That is what makes this "partial equilibrium": we can study one cell without solving
the whole world economy. It also gives the cleanest formulas.

But η = 1 is a knife-edge exactly where our story lives. With η = 1:

- a firm with 100% of a market has an **infinite** markup;
- and it absorbs a tariff **completely** (zero pass-through).

Both are artifacts. So the solver takes any η, and every headline number gets
checked at η > 1, where a monopolist's markup is the sensible η/(η−1). The main
ownership result survives untouched — only a prefactor moves.

### 2.4 The competing agent is the PARENT, not the affiliate

**What.** Varieties are grouped by global ultimate parent. A parent choosing output
for its Colombian and its Peruvian affiliate picks both together.

**Why.** This is the single most important modelling decision, and Fact 4 is what
justifies it. A parent that owns three exporters of the same product does not let
them undercut each other. Counting them as three independent firms understates
concentration and understates markups.

**What it implies, and this is testable.** The markup depends on the **group's total
share**, not the variety's share, and is therefore **the same across all of a
group's affiliates**. That is a sharp prediction. It is also why grouping affiliates
raises both measured concentration *and* true market power at once.

### 2.5 Cournot (quantities), not Bertrand (prices)

**Why.** Three reasons. It is the convention in this literature (Atkeson–Burstein),
so results are comparable. With η = 1 it gives a clean closed form. And quantity
competition is the natural fit for goods produced in fixed plants with capacity.

**Honest note.** Bertrand would give a similar-looking but different markup rule.
This belongs in a robustness appendix, not swept under the rug.

### 2.6 Constant marginal cost

**What.** Producing the tenth unit costs the same as the first.

**Why.** Simplicity, and it isolates the mechanism we care about.

**This one has a big consequence, and it changed our research plan.** With constant
costs, taxing a foreign supplier cannot push down its production cost. So there is
**no terms-of-trade motive** in this layer at all. The only reason to tax is to grab
oligopoly rents; the only reason to subsidise is that markups already make imports
too expensive. We tested which wins: for ordinary market structures the **subsidy
motive wins**, so optimal policy here is an import subsidy. Home ownership makes
that subsidy larger.

That is not a failure. It tells us precisely where the optimal-tariff result can and
cannot come from: **not** from this layer. It needs general equilibrium (where
foreign costs respond) or firm entry. Knowing this before writing the theory saved a
wrong turn.

### 2.7 The competitive fringe

**What.** Each market also contains many small suppliers: domestic producers in the
destination and exporters from countries we do not observe. Each is tiny, none is
home-owned.

**Why it is essential, not cosmetic.** Two reasons.

1. Without it, real HS6 cells often have one dominant group and the η = 1 markup
   explodes.
2. More deeply, **the fringe is what creates the tariff motive at all.** A tariff
   only pays against a dominant group facing a *fragmented* fringe. Against one
   equal-sized rival, a tariff never pays at any elasticity. So how fragmented the
   fringe is directly determines whether there is a tariff to correct.

This is why "competitive fringe or markup cap?" is not a numerical convenience
question. A markup cap would silently delete the mechanism the paper is about.

---

## 3. The whole model is three equations

Let `S_g` be group g's share of the market's spending.

**1. The markup rule.** How much a group marks up over cost:

```
1/μ_g = 1 − (1 − S_g)/σ − S_g/η
```

Read it: a tiny firm (`S_g → 0`) charges the textbook CES markup `σ/(σ−1)`. A big
firm charges more, because it knows that expanding drives down its own price.

**2. Profit.**

```
Π_g = (E/σ) · S_g · [ 1 + (σ/η − 1) · S_g ]
      └ what CES predicts ┘ └ granular correction ┘
```

The first piece is what a standard model would say: revenue over σ. The second is
extra profit that exists **only because the firm is big**. It is proportional to
`S_g²` — a Herfindahl.

**3. Market clearing.** Shares add to one: `Σ_g S_g = 1`.

That is the entire model. Everything else is bookkeeping.

---

## 4. Why there is exactly ONE equilibrium

This matters: if the model had many equilibria, no counterfactual would mean
anything, because we could never say which one the tariff moved us to. There is one.
The proof is in `src/uniqueness.jl`, and every step is checked numerically. Here it
is in words.

**Step 1 — a group's many choices collapse to one number.**
A group might run five affiliates, so it seems to choose five quantities. But its
*revenue* depends on those five only through a single combination, call it `X`
(its total "output bundle"). Verified by reshuffling output across a group's
affiliates while holding `X` fixed: revenue does not move.

**Step 2 — revenue is concave in X.** Selling more always helps, but by less and
less. (Diminishing returns to expanding.)

**Step 3 — cost is convex in X.** Producing a bigger bundle costs more, and
increasingly so, because the group loads more onto its less efficient plants.

**Step 4 — so profit is concave, and the best response is unique.**
Concave revenue minus convex cost is a strictly concave hill. A hill has exactly one
top. There is no second local peak to get stuck on. Checked directly: profit always
lies above the chord between any two points, over 400 markets, zero violations.

*(Also: a firm never wants to produce zero. As output goes to zero its price goes to
infinity fast enough that a little production always pays. So every equilibrium is
interior and the calculus always applies — no corner cases.)*

**Step 5 — the aggregate has one fixed point.**
Every group only cares about one summary of what everyone else does, call it `A`
(the price aggregate). Each group's share `S_g(A)` is **strictly decreasing** in `A`:
tougher competition, smaller share. So the total `Σ_g S_g(A)` is strictly decreasing
too — from more than 1 when `A` is tiny, down to 0 when `A` is huge. A strictly
decreasing line crosses the level 1 **exactly once**.

One `A` ⟹ one set of shares ⟹ one set of markups ⟹ one set of prices, revenues and
quantities. Unique in every object. And because we *construct* it, existence is free.

**This is why the solver uses bisection.** Bisection on a monotone function cannot
land on a different root, because there is no different root. A damped fixed-point
iteration — the usual way people code these models — is *not* guaranteed to converge
here, and in fact hung during development. Bisection is not a stylistic preference;
it is the algorithm the mathematics hands you.

**The decisive numerical check.** 200 random starting points, spread over a factor
of ~e¹², for each of five market structures (2 to 8 groups, η from 1 to 2.5), each
solved by best-response iteration that uses none of the closed forms. All 1,000 runs
converge to the same equilibrium, to about 1e-9.

**Where it could break.** Three assumptions carry the proof, and all three are
asserted in the code: `σ > η` (else markups fall with size and monotonicity dies),
at least two groups (else a monopolist with η = 1 shrinks output without bound), and
strictly positive costs.

---

## 5. How the model meets the data: the denominator problem

This is subtle and it caused a real error, so it is worth being slow.

**The model's share** `S_g` has as its denominator **everything sold in that
destination market**: local producers, other exporters, everyone.

**Figure 6's HHI** has as its denominator **only LAC export value in that product**,
pooled across all destinations — because customs data from nine LAC countries is all
we observe.

These count different populations. Both happen to work out to "about 5 effective
firms," which is exactly why comparing them looked fine. Calibrating the model's
fringe to Figure 6's 0.192 / 0.215 was a category error.

**The fix: build the estimator into the model.** Rather than trying to reconstruct
the model's denominator in the data (impossible — we cannot see Brazil's domestic
producers), we simulate the data the customs authority would record and run *the same
estimator on it*: keep only observed origins, pool across destinations within a
product, renormalise, take the HHI, value-weight across products. Now both sides are
the same object.

**What this immediately revealed.** Renormalising *within* the observed sample
cancels the fringe out. So:

| fringe size | structural HHI | measured HHI (parent) |
|---|---|---|
| 0 | 0.244 | 0.079 |
| 16 | 0.092 | 0.080 |
| 256 | 0.010 | 0.107 |

The structural HHI collapses by a factor of 24. The measured one barely moves.
**Figure 6 is essentially invariant to the fringe, so it can never identify it.**

**The correct identification is two moments for two objects:**

- **λ**, the observed share of destination absorption (LAC exports ÷ what the
  destination buys, from Comtrade plus production data) → identifies **fringe mass**.
- **Measured HHI** (Figure 6) → identifies **how many LAC exporters there are and how
  unequal they are**.

Figure 6 alone was never enough. Now the comparison is like-for-like, and the model's
grouping effect overshoots the data by far less than we thought — most of the
apparent gap was the denominator bug, not the model.

---

## 6. What we are actually measuring: the data ladder

Home country profit income is

```
Π_H = Σ_m (E_m/σ) [ Σ_g θ_g S_gm  +  (σ/η − 1) Σ_g θ_g S²_gm ]
```

The first term is standard. The second — the ownership-weighted Herfindahl — is the
whole paper. Split it three ways:

```
Σ_m w_m Σ_g θ_g S²_gm = θ̄·H̄ + Cov(θ̄_m, H_m) + Σ_m w_m Cov(θ, S)
                        (1)      (2) BETWEEN      (3) WITHIN
```

In words:

1. **naive** — the average ownership share times the average concentration.
2. **between markets** — are home parents in the *more concentrated markets*?
3. **within markets** — inside a market, are home parents the *bigger firms*?

And these line up exactly with what different researchers can compute:

| what you have | what you can compute |
|---|---|
| country ownership share + published HHI | (1) only |
| + ownership share market by market | (1) + (2) |
| + firm-level parent identity (**us**) | (1) + (2) + (3) |

That table is the contribution in one picture. Every rival paper sits on row 1.

**Two traps when reporting this**, both found by testing rather than by thinking:

- The ratio `total/naive` **does not go to 1** in a competitive market. It goes to
  the home-firm *size premium*, because numerator and denominator vanish together.
  So the ratio alone is meaningless — always report it next to the **level**.
- A positive "within" term appears **even with no ownership sorting at all**, purely
  because multinationals are bigger than the fringe. That is the *MNE-size* channel.
  It is real, but it is not the ownership-sorting channel, and reporting them
  together would overstate the contribution.

---

## 7. What this model does not do

Being explicit about this is what keeps referees friendly.

- **No terms-of-trade channel** (constant costs). So no positive optimal tariff here.
- **No entry or exit.** Who is in a market is taken as given.
- **No choice of where to produce.** Location portfolios are exogenous in version 1.
- **It does not reproduce Fact 5.** In the data, non-multinational exports *rise*
  where multinationals are present. With fixed spending per product, this model
  predicts the opposite. Something outside it — market growth, supply-chain linkages,
  or simply that good markets attract everyone — is doing that work. This is the one
  fact the model currently fails, and it should be stated, not hidden.

---

## 8. Symbol glossary

| symbol | meaning | where it comes from |
|---|---|---|
| `σ` | substitution between varieties of the same product | calibrated / literature |
| `η` | substitution across products (`σ > η ≥ 1`) | calibrated; 1 = PE case |
| `E` | spending on the product in the destination | data (absorption) |
| `c_i` | delivered cost of variety i, including tariff | `mc × freight × (1+t)` |
| `g` | global ultimate parent — the strategic agent | ORBIS/D&B match |
| `S_g` | group g's share of the destination market | model outcome |
| `μ_g` | group g's markup; same for all its affiliates | model outcome |
| `θ_g` | share of parent g owned by the home country | ORBIS/D&B match |
| `K_g` | group g's cost-efficiency index, `Σ c_i^(1−σ)` | model object |
| `A` | price aggregate, the thing bisection solves for | model object |
| `λ` | observed share of destination absorption | customs ÷ Comtrade |
| `Λ_g`, `ω_g` | how a cost shock to g spreads to shares and prices | derived |
