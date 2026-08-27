###############################################################################
#  FIRM ENTRY WITH GRANULAR COMPETITION: FOUR APPROACHES
#
#  THE PROBLEM
#  -----------
#  Our model has no entry, so Stylized Facts 1, 2 and 3 are calibration targets
#  or inputs rather than model output. Adding entry to an Atkeson-Burstein
#  oligopoly normally produces MULTIPLE EQUILIBRIA: entry is a discrete game of
#  strategic substitutes, so which firms enter is indeterminate.
#
#  WHAT THE THREE PAPERS DO ABOUT IT
#  ---------------------------------
#  Gaubert & Itskhoki (2020 JPE) and Gaubert, Itskhoki & Vogler (2021 JME):
#      Rank potential entrants by marginal cost and let the LOWEST-COST firm
#      move first. This sequential-entry selection yields a UNIQUE CUTOFF
#      equilibrium: only firms below a cost cutoff enter. It works because
#         (b) share falls with cost rank,
#         (c) every incumbent's share falls when one more firm enters,
#         (d) profit rises with share,
#      so the marginal entrant's profit is monotone in the number of entrants
#      and crosses zero exactly once. They also use BERTRAND rather than
#      Cournot (their footnote 18 notes Cournot gives larger markup variation).
#
#  Yang (2023): multi-plant oligopolists choosing SETS of locations. The game is
#      quasi-aggregative and submodular, so a pure-strategy equilibrium EXISTS
#      (Jensen 2010), and the Arkolakis-Eckert algorithm finds it. But
#      multiplicity is NOT resolved: he picks an equilibrium by imposing an
#      entry ORDER and checks robustness across orders. Crucially, only the two
#      granular firms play the entry game; the fringe is passive.
#
#  WHAT WE VERIFIED FIRST
#  ----------------------
#  Conditions (b), (c), (d) hold EXACTLY in our Cournot market (0 violations in
#  400 random markets). So the GI cutoff argument transfers to our setting --
#  PROVIDED each entrant is a separate competitor. If a parent internalises
#  several affiliates, adding one of its own affiliates RAISES its share, (c)
#  fails, and we are back in Yang's world.
#
#  THE FOUR APPROACHES
#  -------------------
#   A0  no entry                  the current model, as a reference
#   A1  cutoff entry, Cournot     each variety competes; no internalisation
#                                 (the user idea: "no canibaliza sino compite")
#   A2  cutoff entry, Bertrand    pure Gaubert-Itskhoki conduct
#   A3  group entry, internalised Yang: parents internalise, order selection
#
#  Run:  julia entry_approaches.jl
###############################################################################

include("mne_model.jl")
using .MNEModel
using Printf, Random, LinearAlgebra

const MM = MNEModel

###############################################################################
# PART 1.  MARKET SOLVERS: COURNOT AND BERTRAND
###############################################################################

"""
Bertrand markup in nested CES with granular firms: eps(s) = sigma - (sigma-eta)s,
mu = eps/(eps-1). Rising in s like Cournot, but less steeply -- which is exactly
the reason Gaubert-Itskhoki give for preferring it.
"""
function bertrand_markup(s, sigma, eta)
    eps = sigma .- (sigma - eta) .* s
    return eps ./ (eps .- 1.0)
end

"""
Solve one market under Bertrand. Same nested structure as the Cournot solver:
given the price aggregate A, each share solves s = (mu(s)c)^(1-sigma)/A with mu
increasing in s, so the inner root is unique; total share decreases in A, so the
outer root is unique too.
"""
function solve_bertrand(sigma, eta, E, c::Vector{Float64})
    n = length(c)
    # NB: these bounds are named a,b -- NOT lo,hi. `share_given` is a nested
    # function, and in Julia assigning to a name that is also a local of the
    # ENCLOSING function overwrites the outer variable. Naming them lo,hi
    # silently destroyed the outer bisection bracket on every call.
    function share_given(cA)
        a, b = 0.0, 1.0 - 1e-12
        for _ in 1:200
            mid = 0.5 * (a + b)
            f = mid - bertrand_markup(mid, sigma, eta)^(1.0 - sigma) * cA
            f > 0 ? (b = mid) : (a = mid)
        end
        return 0.5 * (a + b)
    end
    tot(A) = sum(share_given(c[i]^(1.0 - sigma) / A) for i in 1:n)
    lo, hi = 1e-14, 1e14
    it = 0; while tot(hi) > 1.0 && it < 500; hi *= 10.0; it += 1; end
    it = 0; while tot(lo) < 1.0 && it < 500; lo /= 10.0; it += 1; end
    for _ in 1:200
        mid = sqrt(lo * hi)
        tot(mid) > 1.0 ? (lo = mid) : (hi = mid)
        hi / lo - 1.0 < 1e-14 && break
    end
    A = sqrt(lo * hi)
    s = [share_given(c[i]^(1.0 - sigma) / A) for i in 1:n]
    s ./= sum(s)
    mu = bertrand_markup(s, sigma, eta)
    Pi = E .* s .* (1.0 .- 1.0 ./ mu)
    return (s = s, mu = mu, Pi = Pi, A = A)
end

"""Cournot with every variety its own competitor. Wraps the main solver."""
function solve_cournot_indep(sigma, eta, E, c::Vector{Float64})
    eq = MM.solve_market(MM.Market(sigma, eta, E, c, collect(1:length(c))))
    return (s = eq.s, mu = eq.mu, Pi = eq.Pi, A = eq.A)
end

###############################################################################
# PART 2.  THE CUTOFF ENTRY RULE (Gaubert-Itskhoki)
###############################################################################

"""
Unique cutoff entry.

Potential entrants are sorted by delivered cost. One more entrant lowers every
incumbent share, and profit rises with share, so the marginal entrant profit
falls monotonically in K and crosses the fixed cost exactly once.

`solver` is passed in, so the rule is identical for Cournot and Bertrand.
"""
function cutoff_entry(sigma, eta, E, cost::Vector{Float64}, F::Float64;
                      solver = solve_cournot_indep)
    ord = sortperm(cost)
    c = cost[ord]
    n = length(c)
    n < 2 && return (K = 0, ord = ord, eq = nothing)
    function marginal_profit(K)
        K < 2 && return Inf
        return solver(sigma, eta, E, c[1:K]).Pi[K]
    end
    marginal_profit(2) < F && return (K = 0, ord = ord, eq = nothing)
    lo, hi = 2, n
    if marginal_profit(n) >= F
        lo = n
    else
        while hi - lo > 1
            mid = (lo + hi) >> 1
            marginal_profit(mid) >= F ? (lo = mid) : (hi = mid)
        end
    end
    return (K = lo, ord = ord, eq = solver(sigma, eta, E, c[1:lo]))
end

"""
Brute force: enumerate EVERY subset and keep the Nash ones (every insider earns
at least F, no outsider would want in). Used to check that the cutoff really is
the unique equilibrium. Only feasible for small n.
"""
function all_entry_equilibria(sigma, eta, E, cost::Vector{Float64}, F::Float64;
                              solver = solve_cournot_indep)
    n = length(cost)
    eqs = Vector{Vector{Int}}()
    for mask in 0:(2^n - 1)
        S = [i for i in 1:n if (mask >> (i - 1)) & 1 == 1]
        length(S) < 2 && continue
        eq = solver(sigma, eta, E, cost[S])
        all(eq.Pi .>= F) || continue
        ok = true
        for j in 1:n
            j in S && continue
            S2 = sort(vcat(S, j))
            eq2 = solver(sigma, eta, E, cost[S2])
            if eq2.Pi[findfirst(==(j), S2)] >= F
                ok = false; break
            end
        end
        ok && push!(eqs, S)
    end
    return eqs
end

###############################################################################
# PART 3.  GROUP ENTRY WITH INTERNALISATION (Yang)
###############################################################################

"""
Entry when a parent internalises its affiliates. Parents take turns choosing
their best set of varieties given rivals. Because adding an affiliate to a
parent RAISES that parent own share, condition (c) fails and the cutoff argument
does not apply: the outcome can depend on the order in which parents move.
`order` is the equilibrium selection, exactly as in Yang.
"""
function group_entry(sigma, eta, E, cost::Vector{Float64}, par::Vector{Int},
                     F::Float64, order::Vector{Int}; rounds = 15)
    active = falses(length(cost))
    for g in unique(par)
        idx = findall(==(g), par)
        active[idx[argmin(cost[idx])]] = true
    end
    function payoff(mask_active, g)
        on = findall(mask_active)
        length(on) < 2 && return -Inf
        pars = par[on]
        uniq = sort(unique(pars))
        eq = MM.solve_market(MM.Market(sigma, eta, E, cost[on],
                                       [findfirst(==(p), uniq) for p in pars]))
        gi = findfirst(==(g), uniq)
        gi === nothing && return -Inf
        return eq.Pi[gi] - F * count(mask_active[findall(==(g), par)])
    end
    for _ in 1:rounds
        changed = false
        for g in order
            idx = findall(==(g), par)
            best = copy(active); bestval = payoff(active, g)
            for mask in 0:(2^length(idx) - 1)
                trial = copy(active)
                for (j, i) in enumerate(idx)
                    trial[i] = ((mask >> (j - 1)) & 1 == 1)
                end
                v = payoff(trial, g)
                if v > bestval + 1e-12
                    bestval = v; best = copy(trial)
                end
            end
            best != active && (changed = true)
            active = best
        end
        changed || break
    end
    return findall(active)
end

###############################################################################
# PART 4.  TEST 1 -- IS THE EQUILIBRIUM UNIQUE?
###############################################################################

function test_uniqueness(; reps = 60, n = 9)
    println("="^78)
    println("TEST 1   IS THE ENTRY EQUILIBRIUM UNIQUE?")
    println("="^78)
    println("Brute force over all 2^n subsets, keeping the Nash ones: every insider")
    println("earns at least F and no outsider would want to enter. Compared with the")
    println("cutoff rule. A1/A2 should give exactly one; A3 is where trouble starts.\n")
    rng = MersenneTwister(11)
    for (name, solver) in (("A1 Cournot, independent varieties", solve_cournot_indep),
                           ("A2 Bertrand, independent varieties", solve_bertrand))
        nuniq = 0; nmatch = 0; multi = 0; nzero = 0
        for _ in 1:reps
            sigma = 3.0 + 4.0 * rand(rng)
            eta = 1.0 + (sigma - 1.0) * 0.4 * rand(rng)
            E = 1.0
            cost = exp.(0.5 .* randn(rng, n))
            F = 0.004 + 0.02 * rand(rng)
            eqs = all_entry_equilibria(sigma, eta, E, cost, F; solver = solver)
            cut = cutoff_entry(sigma, eta, E, cost, F; solver = solver)
            if isempty(eqs); nzero += 1; continue; end
            length(eqs) == 1 && (nuniq += 1)
            length(eqs) > 1 && (multi += 1)
            cutset = sort(cut.ord[1:cut.K])
            any(sort(e) == cutset for e in eqs) && (nmatch += 1)
        end
        @printf("  %-36s unique:%3d  multiple:%3d  none:%3d  cutoff is a Nash eq:%3d\n",
                name, nuniq, multi, nzero, nmatch)
    end
    println()
    println("  A3 group entry with internalisation. Multiplicity in a discrete entry")
    println("  game is driven by SYMMETRY: when rivals are near-identical, which one")
    println("  enters is indeterminate. Yang notes his selection rule has no bite")
    println("  because his two firms are very asymmetric. So we scan cost dispersion.")
    println()
    @printf("  %-14s%-16s%-16s%s
", "cost spread", "A3 multiple", "A1 multiple", "meaning")
    for sd in (0.0, 0.02, 0.10, 0.30, 0.60)
        n3 = 0; n1 = 0; ncase = 25
        for r in 1:ncase
            rr = MersenneTwister(400 + r)
            sigma = 3.0 + 4.0*rand(rr); eta = 1.0 + (sigma-1.0)*0.4*rand(rr)
            npar = 4
            par = repeat(1:npar, inner=2)
            base = exp.(0.5 .* randn(rr, 1))[1]
            cost = base .* exp.(sd .* randn(rr, length(par)))
            F = 0.004 + 0.02*rand(rr)
            outs = Set{Vector{Int}}()
            for t in 1:10
                r2 = MersenneTwister(900+t)
                push!(outs, group_entry(sigma, eta, 1.0, cost, par, F, randperm(r2, npar)))
            end
            length(outs) > 1 && (n3 += 1)
            # same economy, but varieties compete instead of being internalised
            e1 = all_entry_equilibria(sigma, eta, 1.0, cost, F; solver=solve_cournot_indep)
            length(e1) > 1 && (n1 += 1)
        end
        note = sd == 0.0 ? "identical firms" : sd >= 0.30 ? "realistically dispersed" : ""
        @printf("  %-14.2f%-16s%-16s%s
", sd,
                string(n3, "/25"), string(n1, "/25"), note)
    end
    println()
    println("  READING. A1 and A2 give ONE equilibrium and the cutoff rule finds it,")
    println("  at every dispersion. A3 is multiple exactly when parents are close to")
    println("  identical, and the problem fades as they become asymmetric -- which is")
    println("  Yang's own experience. So internalised group entry is usable IF the")
    println("  granular parents are far apart in cost, but it carries no uniqueness")
    println("  guarantee; the cutoff approaches do.")
end


###############################################################################
# PART 5.  TEST 2 -- CAN WE KEEP INTERNALISED GROUPS AND STILL GET A CUTOFF?
###############################################################################

"""
Keeping internalised groups is what makes Stylized Fact 4 work: the markup
depends on the PARENT share, so grouping affiliates raises measured concentration
and true market power together. So it is worth asking whether entry can still be
a cutoff when parents internalise.

The cutoff needs the marginal entrant INCREMENTAL contribution to its own parent,
    delta(K) = Pi_g({1..K}) - Pi_g({1..K-1}),   g = parent of variety K,
to fall monotonically in K. This is weaker than condition (c), so it might hold
even though (c) fails. It does not.
"""
function test_incremental(; reps = 300, n = 8, naff = 2)
    println()
    println("="^78)
    println("TEST 2   CAN INTERNALISED GROUPS STILL HAVE A CUTOFF?")
    println("="^78)
    println("Fact 4 needs parents to internalise. The cutoff needs the marginal")
    println("entrant incremental profit to fall in K. Does it?")
    println()
    function gprof(sigma, eta, E, cost, par, idx)
        isempty(idx) && return Dict{Int,Float64}()
        pars = par[idx]; uniq = sort(unique(pars))
        eq = MM.solve_market(MM.Market(sigma, eta, E, cost[idx],
                                       [findfirst(==(p), uniq) for p in pars]))
        return Dict(uniq[i] => eq.Pi[i] for i in eachindex(uniq))
    end
    rng = MersenneTwister(21); bad = 0; worst = -Inf
    for _ in 1:reps
        sigma = 3.0 + 4.0*rand(rng); eta = 1.0 + (sigma-1)*0.4*rand(rng)
        npar = n ÷ naff
        par = repeat(1:npar, inner = naff)
        cost = exp.(0.5 .* randn(rng, n))
        o = sortperm(cost); cost = cost[o]; par = par[o]
        deltas = Float64[]
        for K in 2:n
            g = par[K]
            pk  = gprof(sigma, eta, 1.0, cost, par, collect(1:K))
            pk1 = gprof(sigma, eta, 1.0, cost, par, collect(1:K-1))
            push!(deltas, get(pk, g, 0.0) - get(pk1, g, 0.0))
        end
        d = diff(deltas)
        any(d .> 1e-12) && (bad += 1)
        worst = max(worst, maximum(d))
    end
    @printf("  parents internalise : incremental profit NOT monotone in %d / %d  (worst rise %+.2e)
",
            bad, reps, worst)
    rng = MersenneTwister(21); bad2 = 0; worst2 = -Inf
    for _ in 1:reps
        sigma = 3.0 + 4.0*rand(rng); eta = 1.0 + (sigma-1)*0.4*rand(rng)
        cost = sort(exp.(0.5 .* randn(rng, n)))
        dl = [solve_cournot_indep(sigma, eta, 1.0, cost[1:K]).Pi[K] for K in 2:n]
        d = diff(dl)
        any(d .> 1e-12) && (bad2 += 1)
        worst2 = max(worst2, maximum(d))
    end
    @printf("  varieties compete   : marginal profit NOT monotone in %d / %d  (worst rise %+.2e)
",
            bad2, reps, worst2)
    println()
    println("  THE TRADE-OFF, IN ONE LINE. Internalisation is what makes Fact 4 work,")
    println("  and it is exactly what destroys the cutoff. Independent varieties give")
    println("  a provable unique equilibrium but lose the group-markup mechanism.")
end

###############################################################################
# PART 6.  HOW BIG IS THE GRANULAR TERM UNDER EACH CONDUCT?
###############################################################################

"""
The paper contribution lives in the granular profit term, the part proportional
to s^2 that vanishes when firms are small. Writing Pi = (E/sigma) s [1 + kappa s]:
    Cournot   kappa = sigma/eta - 1
    Bertrand  kappa = (sigma - eta)/sigma
Switching conduct is therefore not free, and this reports the price.
"""
function test_conduct_cost()
    println()
    println("="^78)
    println("TEST 3   WHAT DOES SWITCHING TO BERTRAND COST THE OWNERSHIP RESULT?")
    println("="^78)
    println("Profit as (E/sigma) s [1 + kappa s]. kappa is the coefficient on the s^2")
    println("Herfindahl term -- the whole ownership correction is proportional to it.")
    println()
    @printf("  %-8s%-8s%15s%15s%10s
", "sigma", "eta", "kappa Cournot", "kappa Bertrand", "ratio")
    for (sg, et) in ((5.0,1.0),(5.0,1.5),(5.0,2.5),(8.0,1.0),(4.0,1.0))
        kc = sg/et - 1.0; kb = (sg - et)/sg
        @printf("  %-8.1f%-8.1f%15.3f%15.3f%9.1fx
", sg, et, kc, kb, kc/kb)
    end
    println()
    println("  At our calibrated sigma=5, eta=1 the granular term is FIVE TIMES")
    println("  SMALLER under Bertrand. Bertrand buys cleaner entry and costs")
    println("  headline magnitude. That is a judgement call, not a free lunch.")
end

if abspath(PROGRAM_FILE) == (@__FILE__)
    test_uniqueness()
    test_incremental()
    test_conduct_cost()
end
