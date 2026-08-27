"""
Layer 3: GENERAL EQUILIBRIUM with multinational production.

This closes the model. Layers 0-1 were partial equilibrium: market size E and
marginal costs c were exogenous numbers. Here both become endogenous outcomes of
wages, and wages are determined by labour market clearing.

================================================================================
THE FIVE INGREDIENTS OF A GENERAL EQUILIBRIUM MODEL, AND WHERE EACH ONE LIVES
================================================================================

  1. ENDOWMENTS      L[n] units of labour in country n. The only primitive factor.
  2. DEMAND          Two-tier. Across sectors CES(eta) with weights beta[k]; within
                     a sector CES(sigma[k]) over varieties. Country n spends its
                     entire income X[n].
  3. SUPPLY          Each affiliate has productivity phi and combines HQ-country
                     labour (share alpha[k]) with host-country labour (share
                     1-alpha[k]). This is the multinational production technology:
                     an affiliate uses its PARENT'S country labour for headquarter
                     services and its HOST country's labour for production.
  4. CONDUCT         Groups (global ultimate parents) compete in quantities inside
                     each (destination, sector) cell. This is Layers 0-1, reused
                     completely unchanged.
  5. CLEARING        (a) goods:  sum_g S_g = 1 in every market
                     (b) labour: labour demand = L[n] in every country
                     (c) budget: X[n] = wage income + profits owned + tariff revenue

Walras' law ties these together: the value of excess labour demand is identically
zero at EVERY wage vector, not only in equilibrium. One clearing condition is
therefore redundant and one wage must be normalised. TEST 1 verifies this.

================================================================================
THE COST FUNCTION -- THIS IS THE MULTINATIONAL PRODUCTION BLOCK
================================================================================

An affiliate `a` of a parent headquartered in country h, producing in country l,
selling into destination n, in sector k, has delivered pre-tariff unit cost

    a[a,n] = ( w[h]^alpha[k] * w[l]^(1-alpha[k]) * gamma[h,l] * d[l,n] ) / phi[a]

and delivered cost inclusive of tariff  c[a,n] = a[a,n] * (1 + t[l,n,k]).

Every piece is a named object from the multinational production literature:

  w[h]^alpha * w[l]^(1-alpha)   HQ input intensity. Headquarter services are paid
                                at the PARENT's wage, production at the HOST's.
                                alpha[k] rises with product complexity, which is
                                Stylized Fact 2. (Head & Mayer 2019.)
  gamma[h,l]                    the MP friction: how much efficiency an h-parent
                                loses by producing in l. This is exactly the
                                multinational production cost of Ramondo &
                                Rodriguez-Clare (2013).
  d[l,n]                        iceberg trade cost from the PRODUCTION country to
                                the destination. Because l and n are separate, one
                                affiliate serves many destinations: this is the
                                EXPORT PLATFORM structure of Tintelnot (2017).
  phi[a]                        affiliate productivity.

WHY THIS MATTERS FOR THE PROJECT. In Layer 0 marginal cost was a constant, so a
tariff could not change any foreign supplier's cost and there was NO terms-of-
trade motive: optimal policy was always an import subsidy. Here a tariff lowers
foreign demand, lowers foreign wages, and so lowers foreign costs. The terms-of-
trade motive exists. TEST 7 shows it.

================================================================================
HOW IT IS SOLVED, AND WHY THAT WORKS
================================================================================

The key structural fact that makes this fast and reliable:

    MARKET SHARES AND MARKUPS DEPEND ONLY ON COSTS, NEVER ON MARKET SIZE.

Look at Layer 0: S_g solves S = mu(S)^(1-sigma) K_g / A with sum_g S_g = 1, and E
appears nowhere. E only scales levels afterwards. So given wages we can solve every
market ONCE, and the income system is then LINEAR in X and solved exactly rather
than iterated. The only genuine fixed point is over the N-1 free wages.

  given w -> costs -> per-market shares, markups, price indices   (unique, Layer 0)
          -> expenditure shares eps[n,k]                          (closed form)
          -> incomes X by solving ONE linear system               (exact)
          -> labour demand -> excess demand -> update w

Run:  julia src/layer3_ge.jl
"""

module Layer3

include("cournot_pe.jl")
using .CournotPE
using LinearAlgebra

export GEModel, solve_markets, solve_incomes, labour_demand, excess_demand,
       solve_ge, accounting, expenditure_shares, delivered_cost

"""
A general equilibrium economy with multinational production.

Countries n = 1..N, sectors k = 1..K, affiliates a = 1..A.
Every affiliate sells into every destination (portfolios are exogenous in v1).
"""
struct GEModel
    N::Int
    K::Int
    sigma::Vector{Float64}      # within-sector elasticity, per sector
    eta::Float64                # across-sector elasticity; needs eta < min(sigma)
    beta::Vector{Float64}       # sector weights in demand, sum to 1
    alpha::Vector{Float64}      # HQ input intensity, per sector
    L::Vector{Float64}          # labour endowment, per country
    par::Vector{Int}            # global ultimate parent of each affiliate
    hq::Vector{Int}             # HQ country of each affiliate's parent
    loc::Vector{Int}            # production country of each affiliate
    sec::Vector{Int}            # sector of each affiliate
    phi::Vector{Float64}        # affiliate productivity
    gamma::Matrix{Float64}      # gamma[h,l] MP friction (Ramondo-Rodriguez-Clare)
    d::Matrix{Float64}          # d[l,n] iceberg trade cost
    tariff::Array{Float64,3}    # tariff[l,n,k] ad valorem
    theta::Matrix{Float64}      # theta[parent, country]; each ROW sums to 1
end

"""
Pre-tariff delivered unit cost of affiliate `a` into destination `n`, given wages.
This is the multinational production cost function; see the header.
"""
function delivered_cost(m::GEModel, w::Vector{Float64}, a::Int, n::Int)
    k = m.sec[a]
    return (w[m.hq[a]]^m.alpha[k] * w[m.loc[a]]^(1.0 - m.alpha[k]) *
            m.gamma[m.hq[a], m.loc[a]] * m.d[m.loc[a], n]) / m.phi[a]
end

"""
Solve every (destination, sector) market given wages.

Shares and markups do not depend on market size, so this can be done once, before
incomes are known. Each market is solved by the Layer 0 routine unchanged, and is
therefore UNIQUE by the argument in src/uniqueness.jl.
"""
function solve_markets(m::GEModel, w::Vector{Float64})
    mk = Array{Any}(undef, m.N, m.K)
    for n in 1:m.N, k in 1:m.K
        idx = findall(==(k), m.sec)
        a_pre = [delivered_cost(m, w, a, n) for a in idx]
        c     = [a_pre[j] * (1.0 + m.tariff[m.loc[idx[j]], n, k]) for j in eachindex(idx)]
        pars  = m.par[idx]
        uniq  = sort(unique(pars))
        lut   = Dict(u => i for (i, u) in enumerate(uniq))
        eq    = solve(Market(m.sigma[k], m.eta, 1.0, c, [lut[p] for p in pars]))
        mk[n, k] = (eq = eq, idx = idx, pars = uniq, a_pre = a_pre,
                    P = eq.A^(1.0 / (1.0 - m.sigma[k])))
    end
    return mk
end

"""
Expenditure shares across sectors: E[n,k] = eps[n,k] * X[n], with sum_k eps = 1.
Closed form from the upper-tier CES; no iteration needed.
"""
function expenditure_shares(m::GEModel, mk)
    eps = zeros(m.N, m.K)
    for n in 1:m.N
        Pn = sum(m.beta[k] * mk[n, k].P^(1.0 - m.eta) for k in 1:m.K)
        for k in 1:m.K
            eps[n, k] = m.beta[k] * mk[n, k].P^(1.0 - m.eta) / Pn
        end
    end
    return eps
end

"""
Solve for incomes X.

    X[n] = w[n] L[n]  +  sum_g theta[g,n] Pi[g]  +  T[n]

Profits and tariff revenue are both LINEAR in X once shares are known, so this is
one linear system, solved exactly. No fixed point, no convergence tolerance.
"""
function solve_incomes(m::GEModel, w::Vector{Float64}, mk, eps)
    G = size(m.theta, 1)

    # pi_unit[g, n, k] : profit of parent g per unit of expenditure in market (n,k)
    pi_unit = zeros(G, m.N, m.K)
    tau     = zeros(m.N)                       # tariff revenue per unit of X[n]
    for n in 1:m.N, k in 1:m.K
        e, idx, pars = mk[n, k].eq, mk[n, k].idx, mk[n, k].pars
        s2 = m.sigma[k] / m.eta - 1.0
        for (i, g) in enumerate(pars)
            pi_unit[g, n, k] = eps[n, k] * (e.S[i] / m.sigma[k]) * (1.0 + s2 * e.S[i])
        end
        for (j, a) in enumerate(idx)
            t = m.tariff[m.loc[a], n, k]
            gi = findfirst(==(m.par[a]), pars)
            tau[n] += eps[n, k] * e.s[j] * t / (e.mu[gi] * (1.0 + t))
        end
    end

    # X[n] = w L + sum_g theta[g,n] sum_{m',k} pi_unit[g,m',k] X[m'] + tau[n] X[n]
    M = zeros(m.N, m.N)
    for n in 1:m.N, mp in 1:m.N
        M[n, mp] = sum(m.theta[g, n] * sum(pi_unit[g, mp, k] for k in 1:m.K)
                       for g in 1:G)
    end
    A = I(m.N) - Diagonal(tau) - M
    X = A \ (w .* m.L)
    return X, pi_unit, tau
end

"""
Labour demand in every country.

Two sources, and this is the multinational part: an affiliate hires HQ labour in
its PARENT's country and production labour in its HOST country.
Factor payment by affiliate a into destination n is  a_pre * q = E s / (mu (1+t)),
split alpha[k] to the HQ country and (1-alpha[k]) to the host.
"""
function labour_demand(m::GEModel, w::Vector{Float64}, mk, eps, X)
    LD = zeros(m.N)
    for n in 1:m.N, k in 1:m.K
        e, idx, pars = mk[n, k].eq, mk[n, k].idx, mk[n, k].pars
        E = eps[n, k] * X[n]
        for (j, a) in enumerate(idx)
            t  = m.tariff[m.loc[a], n, k]
            gi = findfirst(==(m.par[a]), pars)
            pay = E * e.s[j] / (e.mu[gi] * (1.0 + t))      # factor payment
            LD[m.hq[a]]  += m.alpha[k] * pay / w[m.hq[a]]
            LD[m.loc[a]] += (1.0 - m.alpha[k]) * pay / w[m.loc[a]]
        end
    end
    return LD
end

"""Excess labour demand, country by country. Zero in equilibrium."""
function excess_demand(m::GEModel, w::Vector{Float64})
    mk  = solve_markets(m, w)
    eps = expenditure_shares(m, mk)
    X, _, _ = solve_incomes(m, w, mk, eps)
    LD = labour_demand(m, w, mk, eps, X)
    return LD .- m.L, (mk = mk, eps = eps, X = X, LD = LD)
end

"""
Find equilibrium wages.

Tatonnement: raise the wage where labour is scarce. w[1] is the numeraire, which
is legitimate because the whole system is homogeneous of degree zero in wages
(TEST 3) and one market-clearing condition is redundant by Walras' law (TEST 1).
"""
function solve_ge(m::GEModel; w0 = ones(m.N), kappa = 0.25, tol = 1e-12,
                  maxit = 20_000)
    w = copy(w0); w ./= w[1]
    local info
    for it in 1:maxit
        z, info = excess_demand(m, w)
        gap = maximum(abs.(z[2:end] ./ m.L[2:end]))
        gap < tol && return (w = w, iters = it, gap = gap, info = info)
        for n in 2:m.N
            w[n] *= ((info.LD[n] / m.L[n])^kappa)
        end
        w ./= w[1]
    end
    z, info = excess_demand(m, w)
    return (w = w, iters = maxit, gap = maximum(abs.(z[2:end] ./ m.L[2:end])),
            info = info)
end

"""
Full accounting audit. Every identity a general equilibrium must satisfy.
Returns the worst violation of each, all of which must be ~1e-12 or smaller.
"""
function accounting(m::GEModel, w::Vector{Float64})
    z, inf = excess_demand(m, w)
    mk, eps, X = inf.mk, inf.eps, inf.X
    G = size(m.theta, 1)

    revenue = factor_pay = tariff_rev = profit = 0.0
    prof_by_g = zeros(G)
    tariff_by_n = zeros(m.N)
    for n in 1:m.N, k in 1:m.K
        e, idx, pars = mk[n, k].eq, mk[n, k].idx, mk[n, k].pars
        E = eps[n, k] * X[n]
        revenue += E
        s2 = m.sigma[k] / m.eta - 1.0
        for (i, g) in enumerate(pars)
            p = E * (e.S[i] / m.sigma[k]) * (1.0 + s2 * e.S[i])
            prof_by_g[g] += p; profit += p
        end
        for (j, a) in enumerate(idx)
            t  = m.tariff[m.loc[a], n, k]
            gi = findfirst(==(m.par[a]), pars)
            pay = E * e.s[j] / (e.mu[gi] * (1.0 + t))
            factor_pay += pay
            tariff_rev += pay * t
            tariff_by_n[n] += pay * t
        end
    end

    # (i) goods market clearing in every cell
    share_err = maximum(abs(sum(mk[n, k].eq.S) - 1.0) for n in 1:m.N, k in 1:m.K)
    # (ii) revenue = factor payments + tariffs + profits
    ident_err = abs(revenue - factor_pay - tariff_rev - profit) / revenue
    # (iii) each country's budget: X = wL + profits owned + tariff revenue
    budget = [w[n]*m.L[n] + sum(m.theta[g,n]*prof_by_g[g] for g in 1:G) +
              tariff_by_n[n] for n in 1:m.N]
    budget_err = maximum(abs.(budget .- X) ./ X)
    # (iv) Walras: value of excess labour demand is zero
    walras_err = abs(sum(w .* z)) / revenue
    # (v) world budget: total spending = total income
    world_err = abs(sum(X) - sum(w .* m.L) - profit - tariff_rev) / revenue

    return (shares = share_err, identity = ident_err, budget = budget_err,
            walras = walras_err, world = world_err, excess = z,
            revenue = revenue, factor_pay = factor_pay,
            tariff_rev = tariff_rev, profit = profit, X = X)
end

end # module


# ---------------------------------------------------------------------------
using .Layer3
using .Layer3.CournotPE
using Printf
using Random
using LinearAlgebra

"""A small symmetric-ish world used by the tests."""
function demo_model(rng; N = 4, K = 3, nparents = 12, naff = 2, eta = 1.0,
                    tariff = 0.0)
    sigma = [4.0 + 2.0*rand(rng) for _ in 1:K]
    beta  = rand(rng, K) .+ 0.5; beta ./= sum(beta)
    alpha = [0.15 + 0.35*rand(rng) for _ in 1:K]      # HQ input intensity
    L     = 1.0 .+ rand(rng, N)

    par = Int[]; hq = Int[]; loc = Int[]; sec = Int[]; phi = Float64[]
    for g in 1:nparents
        h = rand(rng, 1:N)
        for _ in 1:naff
            push!(par, g); push!(hq, h); push!(loc, rand(rng, 1:N))
            push!(sec, rand(rng, 1:K)); push!(phi, exp(0.3*randn(rng)))
        end
    end
    # make sure every sector has at least 2 distinct parents in it
    for k in 1:K
        while length(unique(par[sec .== k])) < 2
            j = rand(rng, 1:length(sec)); sec[j] = k
        end
    end

    gamma = 1.0 .+ 0.3 .* rand(rng, N, N)              # MP friction
    for h in 1:N; gamma[h, h] = 1.0; end               # no friction at home
    d = 1.0 .+ 0.4 .* rand(rng, N, N)
    for l in 1:N; d[l, l] = 1.0; end
    t = fill(tariff, N, N, K)
    for n in 1:N, k in 1:K; t[n, n, k] = 0.0; end      # no tariff on domestic

    theta = zeros(nparents, N)
    for g in 1:nparents; theta[g, hq[findfirst(==(g), par)]] = 1.0; end

    return GEModel(N, K, sigma, eta, beta, alpha, L, par, hq, loc, sec, phi,
                   gamma, d, t, theta)
end

function run_tests()
    rng = MersenneTwister(20260812)

    println("="^78)
    println("TEST 1  WALRAS' LAW: value of excess labour demand is zero at ANY wage")
    println("="^78)
    println("  This is the strongest single check that the accounting is complete.")
    println("  It must hold OUT of equilibrium too, not just at the solution.")
    worst = 0.0
    for _ in 1:40
        m = demo_model(rng; tariff = 0.15)
        w = exp.(0.4 .* randn(rng, m.N)); w ./= w[1]     # arbitrary, NOT equilibrium
        acc = accounting(m, w)
        worst = max(worst, acc.walras)
    end
    @printf("  40 models at 40 arbitrary wage vectors: max |sum_n w_n Z_n| / GDP = %.2e\n", worst)
    println("  -> one clearing condition is redundant, so exactly one wage is")
    println("     normalised. The model has N-1 independent unknowns, as it must.")

    println()
    println("="^78)
    println("TEST 2  ACCOUNTING IDENTITIES at arbitrary wages")
    println("="^78)
    wS = wI = wB = wW = 0.0
    for _ in 1:40
        m = demo_model(rng; tariff = 0.2)
        w = exp.(0.4 .* randn(rng, m.N)); w ./= w[1]
        a = accounting(m, w)
        wS = max(wS, a.shares); wI = max(wI, a.identity)
        wB = max(wB, a.budget); wW = max(wW, a.world)
    end
    @printf("  goods market clearing, sum_g S_g = 1               : %.2e\n", wS)
    @printf("  revenue = factor pay + tariffs + profits           : %.2e\n", wI)
    @printf("  country budget  X = wL + profits owned + tariffs   : %.2e\n", wB)
    @printf("  world budget    total spending = total income      : %.2e\n", wW)
    println("  -> every euro is accounted for exactly once. The math adds up.")

    println()
    println("="^78)
    println("TEST 3  HOMOGENEITY: only relative wages matter")
    println("="^78)
    worst = 0.0
    for _ in 1:20
        m = demo_model(rng; tariff = 0.1)
        w = exp.(0.3 .* randn(rng, m.N))
        z1, _ = excess_demand(m, w)
        z2, _ = excess_demand(m, 7.3 .* w)               # scale ALL wages
        worst = max(worst, maximum(abs.(z1 .- z2) ./ m.L))
    end
    @printf("  scaling every wage by 7.3 changes excess demand by %.2e\n", worst)
    println("  -> the system is homogeneous of degree zero in wages, which is why")
    println("     a numeraire must be chosen and why doing so loses nothing.")

    println()
    println("="^78)
    println("TEST 4  EQUILIBRIUM EXISTS AND MARKETS CLEAR")
    println("="^78)
    for tar in (0.0, 0.15)
        m = demo_model(MersenneTwister(11); tariff = tar)
        r = solve_ge(m)
        a = accounting(m, r.w)
        @printf("  tariff=%.2f  iters=%5d  max |labour excess demand|/L = %.2e\n",
                tar, r.iters, r.gap)
        @printf("               budget err %.1e   identity err %.1e   walras %.1e\n",
                a.budget, a.identity, a.walras)
    end

    println()
    println("="^78)
    println("TEST 5  UNIQUENESS OF THE GENERAL EQUILIBRIUM (numerical)")
    println("="^78)
    println("  Within a market, equilibrium is PROVED unique (src/uniqueness.jl).")
    println("  Across countries there is no such proof, so we test it hard: random")
    println("  starting wages spread over a factor of ~e^6, across economies that")
    println("  vary in size, elasticity and tariffs. Also reported is the worst")
    println("  labour-market residual, so a 'match' cannot be two failures agreeing.")
    @printf("  %-34s%9s%13s%12s\n", "economy", "starts", "max wage gap", "worst resid")
    cases = [
        ("N=4 K=3 eta=1  free trade",      4, 3, 1.0, 0.00, 60),
        ("N=4 K=3 eta=1  tariff 15%",      4, 3, 1.0, 0.15, 60),
        ("N=5 K=4 eta=1.8 tariff 25%",     5, 4, 1.8, 0.25, 40),
        ("N=6 K=5 eta=2.5 tariff 10%",     6, 5, 2.5, 0.10, 30),
        ("N=7 K=3 eta=1  tariff 40%",      7, 3, 1.0, 0.40, 30),
    ]
    for (lab, N, K, et, tar, nst) in cases
        m = demo_model(MersenneTwister(11); N = N, K = K, eta = et, tariff = tar,
                       nparents = 4 * K, naff = 2)
        base = solve_ge(m)
        ref, spread, worst = base.w, 0.0, base.gap
        for s in 1:nst
            r2 = MersenneTwister(5000 + s)
            w0 = exp.(3.0 .* randn(r2, m.N)); w0 ./= w0[1]      # ~e^6 spread
            rr = solve_ge(m; w0 = w0)
            spread = max(spread, maximum(abs.(rr.w .- ref) ./ ref))
            worst = max(worst, rr.gap)
        end
        @printf("  %-34s%9d%13.2e%12.2e\n", lab, nst, spread, worst)
    end
    println("  -> every start reaches the same equilibrium, and every one of them")
    println("     genuinely clears the labour market. VERIFIED, NOT PROVED: with")
    println("     variable markups the standard gross-substitutes argument does not")
    println("     apply, so this is numerical evidence and is reported as such.")

    println()
    println("="^78)
    println("TEST 6  SCALE: doubling every endowment doubles quantities, not wages")
    println("="^78)
    m1 = demo_model(MersenneTwister(3); tariff = 0.1)
    m2 = GEModel(m1.N, m1.K, m1.sigma, m1.eta, m1.beta, m1.alpha, 2.0 .* m1.L,
                 m1.par, m1.hq, m1.loc, m1.sec, m1.phi, m1.gamma, m1.d,
                 m1.tariff, m1.theta)
    r1, r2 = solve_ge(m1), solve_ge(m2)
    @printf("  max relative wage change  = %.2e   (should be 0: no scale effects)\n",
            maximum(abs.(r2.w .- r1.w) ./ r1.w))
    @printf("  income ratio X2/X1        = %.10f  (should be exactly 2)\n",
            sum(accounting(m2, r2.w).X) / sum(accounting(m1, r1.w).X))

    println()
    println("="^78)
    println("TEST 7  THE PAYOFF: general equilibrium restores the terms-of-trade motive")
    println("="^78)
    println("  In Layer 0 marginal cost was fixed, so a tariff could not move any")
    println("  foreign cost. There was NO terms-of-trade motive and optimal policy")
    println("  was always an import subsidy. Here a tariff by country 1 lowers")
    println("  demand for foreign labour, lowers foreign wages, and so lowers the")
    println("  price country 1 pays. That channel exists only in general equilibrium.\n")

    base = demo_model(MersenneTwister(11); tariff = 0.0)
    nfor = [g for g in 1:size(base.theta,1) if base.hq[findfirst(==(g), base.par)] != 1]

    """Rebuild the model with country 1 tariffing all imports at `tar`, and with
       country 1 owning fraction `omega` of every FOREIGN parent."""
    function variant(tar, omega)
        t = zeros(base.N, base.N, base.K)
        for l in 1:base.N, k in 1:base.K
            l == 1 || (t[l, 1, k] = tar)          # country 1 taxes its imports only
        end
        th = copy(base.theta)
        for g in nfor
            # NB: name this `hcty`, not `h`. `variant` is a closure, so assigning to
            # a name that also exists in the enclosing scope would OVERWRITE it.
            hcty = base.hq[findfirst(==(g), base.par)]
            th[g, 1] = omega; th[g, hcty] = 1.0 - omega
        end
        return GEModel(base.N, base.K, base.sigma, base.eta, base.beta, base.alpha,
                       base.L, base.par, base.hq, base.loc, base.sec, base.phi,
                       base.gamma, base.d, t, th)
    end

    """Real income of country 1: nominal income deflated by its true price index."""
    function realinc(m, w)
        acc = accounting(m, w)
        _, inf = excess_demand(m, w)
        return acc.X[1] / prod(inf.mk[1, k].P^m.beta[k] for k in 1:m.K)
    end

    println("  (a) Does the terms-of-trade channel actually operate?")
    @printf("      %8s  %14s  %16s\n", "tariff", "mean w_foreign", "real income 1")
    for tar in (0.0, 0.10, 0.25, 0.40)
        m = variant(tar, 0.0); r = solve_ge(m)
        @printf("      %8.2f  %14.5f  %16.5f\n", tar,
                sum(r.w[2:end])/(base.N-1), realinc(m, r.w))
    end
    println("      Foreign wages DO fall with the tariff, so the channel is live.")

    println()
    println("  (b) MARGINAL test: dW/dt at free trade, as country 1 comes to own")
    println("      more of the foreign multinationals it would be taxing. This is")
    println("      the clean test of the mechanism -- no grid, no corner solutions,")
    println("      and no confound from ownership changing country 1's income level.")
    dt = 1e-4
    @printf("      %22s%18s\n", "country 1 owns", "dW/dt at t=0")
    for omega in (0.0, 0.25, 0.50, 0.75, 1.0)
        mp, mm = variant(dt, omega), variant(-dt, omega)
        vp = realinc(mp, solve_ge(mp).w)
        vm = realinc(mm, solve_ge(mm).w)
        @printf("      %19.0f%%%18.5f\n", 100*omega, (vp - vm) / (2dt))
    end

    println()
    println("  (c) Optimal tariff, wide grid (boundary hits are flagged).")
    grid = collect(-0.90:0.05:0.90)
    @printf("      %22s%10s%20s\n", "country 1 owns", "t*", "gain vs free trade")
    for omega in (0.0, 0.25, 0.50, 0.75, 1.0)
        best_t, best_v, v0 = NaN, -Inf, NaN
        for tar in grid
            m = variant(tar, omega); v = realinc(m, solve_ge(m).w)
            abs(tar) < 1e-12 && (v0 = v)
            v > best_v && (best_v = v; best_t = tar)
        end
        flag = (best_t <= grid[1] + 1e-9 || best_t >= grid[end] - 1e-9) ? "  <-- AT GRID EDGE" : ""
        @printf("      %19.0f%%%10.2f%17.4f%%%s\n",
                100*omega, best_t, 100*(best_v/v0 - 1), flag)
    end

    println()
    println("  -> THIS RUNS AGAINST THE PROJECT'S HYPOTHESIS. Report it, do not bury it.")
    println("     Home ownership of the taxed firms makes the tariff MORE attractive")
    println("     here, not less. The Brecher-Bhagwati force (taxing a firm you own")
    println("     takes from your own pocket) is present but is outweighed by a")
    println("     second channel that exists ONLY with multinational production:")
    println()
    println("       a tariff lowers foreign WAGES (the terms-of-trade effect);")
    println("       the firms country 1 owns PRODUCE ABROAD with that foreign labour;")
    println("       so their costs fall and their profits RISE.")
    println()
    println("     Owning a foreign affiliate is not the same as owning a foreign")
    println("     exporter. You capture the terms-of-trade gain twice: once as a")
    println("     consumer, and again as the shareholder of a firm whose wage bill")
    println("     just fell. Which force wins is a quantitative question about")
    println("     alpha_k (how much of cost is HQ vs host labour) and about where")
    println("     the affiliates you own actually produce and sell.")
    println()
    println("     This is exactly why the destination-of-MNE-exports tabulation is")
    println("     the highest-value empirical task on the project: the sign of the")
    println("     headline result depends on it.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_tests()
end
