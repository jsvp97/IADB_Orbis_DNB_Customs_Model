"""
Layer 0 (Partial Equilibrium): single-market group Cournot with nested CES demand.

Market = one (destination d, product class k) cell.

  upper tier   CES(eta) across classes, so class expenditure  E = D * P^(1-eta).
               eta = 1 is Cobb-Douglas: E = D is FIXED, which is what makes the
               layer partial equilibrium and gives the cleanest closed forms.
  lower tier   CES(sigma) over varieties. Variety i belongs to GROUP g (a global
               ultimate parent). Groups play Cournot in quantities and
               internalise cannibalisation across their own affiliates.
  requires     sigma > eta >= 1, so that markups rise with market share.

DERIVED RESULTS (all verified numerically to machine precision, both here and in
scripts/verify/cournot_pe.py against a solver that uses none of this algebra):

  revenue share    sh_i = q_i^((s-1)/s) / A_q,   sum_i sh_i = 1
  group FOC        p_i = c_i * mu_g,   1/mu_g = 1 - (1 - S_g)/s - S_g/eta
  group profit     Pi_g = (E/s) * S_g * (1 + (s/eta - 1) * S_g)

where S_g = sum_{i in g} sh_i is the GROUP's total market share.

  * the markup depends on the group's TOTAL share, not the variety's share, and
    is common across all affiliates of the group. This is why grouping affiliates
    by parent raises both measured concentration and true markups.
  * profit is the CES term (E/s)*S_g plus a granular correction
    (E/s)(s/eta-1)*S_g^2. Summed with ownership weights, that second term is an
    ownership-weighted HHI. The eta only moves the prefactor, so the ownership
    result does NOT rest on the eta = 1 knife-edge.
  * eta = 1 recovers mu_g = s/((s-1)(1-S_g)). Note that at eta = 1 a monopolist's
    markup is unbounded and its tariff pass-through goes to zero. Both are
    artefacts of the knife-edge; with eta > 1 the monopoly markup is eta/(eta-1).

ALGORITHM: nested monotone bisection.
  Inner: given A, each group's S_g solves  x = mu(x)^(1-s) * K_g/A  with
         K_g = sum_{i in g} c_i^(1-s). mu is increasing in x when s > eta, so the
         RHS is decreasing while the LHS rises 0->1 => unique root.
  Outer: sum_g S_g(A) = 1. Each S_g is strictly decreasing in A => unique A.
  So the equilibrium exists and is unique, and the construction proves it.
  (A damped fixed-point iteration on S does NOT converge for all parameters.
   Do not use one.)

See docs/derivations.md for the full algebra and the welfare results.

Run:  julia src/cournot_pe.jl
"""

module CournotPE

export Market, MarketEq, solve, hhi_naive, hhi_grouped, owner_income,
       tariffed_cost, markup, lambda_g, omega_g, passthrough, dW_dt,
       ownership_decomposition

struct Market
    sigma::Float64          # within-class CES elasticity over varieties
    eta::Float64            # across-class elasticity; 1 <= eta < sigma
    D::Float64              # demand shifter: class expenditure E = D * P^(1-eta)
    c::Vector{Float64}      # delivered marginal cost of each variety
    gid::Vector{Int}        # group index of each variety, values in 1:G
end

"""Backward-compatible constructor: eta = 1 (Cobb-Douglas), so D is exactly E."""
Market(sigma::Float64, E::Float64, c::Vector{Float64}, gid::Vector{Int}) =
    Market(sigma, 1.0, E, c, gid)

struct MarketEq
    S::Vector{Float64}      # group shares (length G)
    s::Vector{Float64}      # variety shares (length N)
    p::Vector{Float64}      # prices
    mu::Vector{Float64}     # group markups (length G)
    q::Vector{Float64}      # quantities
    r::Vector{Float64}      # variety revenues
    Pi::Vector{Float64}     # group profits (length G)
    A::Float64              # sum_i p_i^(1-sigma)
    P::Float64              # lower-tier CES price index, A^(1/(1-sigma))
    E::Float64              # class expenditure (equals D when eta == 1)
end

"""
Group markup as a function of the group's TOTAL share:  1/mu = 1 - (1-S)/s - S/eta.

Read it as a weighted average of two elasticities. A group with share S faces the
within-class elasticity sigma on the fraction (1-S) of the market it does NOT own,
and the across-class elasticity eta on the fraction S it does. Expanding output
cannibalises its own sales, so the bigger it is the less elastic its residual
demand and the higher its markup.

  S -> 0   =>  mu -> sigma/(sigma-1)      the textbook CES markup
  S -> 1   =>  mu -> eta/(eta-1)          a monopolist (infinite if eta = 1)

Requires sigma > eta, otherwise the markup FALLS with share, which is backwards
and breaks uniqueness.
"""
markup(S, sigma::Float64, eta::Float64) =
    1.0 ./ (1.0 .- (1.0 .- S) ./ sigma .- S ./ eta)

"""
INNER LOOP. Given the price aggregate A, find one group's share.

Solves  x = mu(x)^(1-sigma) * KA  for x in (0,1), where KA = K_g / A.

Why the root is unique: the left side rises from 0 to 1. On the right, mu is
increasing in x and the exponent (1-sigma) is negative, so the right side is
DECREASING. A rising line and a falling line cross at most once, and they do
cross because the right side starts positive. Bisection is therefore exact and
cannot be fooled.
"""
function inner_share(KA::Float64, sigma::Float64, eta::Float64;
                     tol::Float64 = 1e-15, maxit::Int = 200)
    lo, hi = 0.0, 1.0
    for _ in 1:maxit
        mid = 0.5 * (lo + hi)
        if mid - markup(mid, sigma, eta)^(1.0 - sigma) * KA > 0.0
            hi = mid
        else
            lo = mid
        end
        hi - lo < tol && break
    end
    return 0.5 * (lo + hi)
end

function group_K(m::Market)
    G = maximum(m.gid)
    K = zeros(Float64, G)
    for i in eachindex(m.c)
        K[m.gid[i]] += m.c[i]^(1.0 - m.sigma)
    end
    return K
end

total_share(A::Float64, K::Vector{Float64}, sigma::Float64, eta::Float64) =
    sum(inner_share(K[g] / A, sigma, eta) for g in eachindex(K))

"""
Solve one market. Returns the unique equilibrium.

The three assertions below are not defensive padding: each one is a hypothesis of
the uniqueness theorem in src/uniqueness.jl. Remove any of them and the model can
have several equilibria, or none.
"""
function solve(m::Market)
    sigma, eta = m.sigma, m.eta

    # sigma > eta makes the markup INCREASE with market share. If it were violated
    # the markup would fall with size, psi below would not be monotone, and
    # uniqueness would fail. See uniqueness.jl STEP 5.
    @assert sigma > eta >= 1.0 "need sigma > eta >= 1"
    # c_i > 0 or the cost index K_g blows up.
    @assert all(m.c .> 0.0) "marginal costs must be strictly positive"
    # With one group and eta = 1, revenue is constant in output, so the monopolist
    # shrinks output without bound: the markup is unbounded. Needs >= 2 groups
    # (or eta > 1, where the monopoly markup is the finite eta/(eta-1)).
    @assert maximum(m.gid) >= 2 || eta > 1.0 "a monopolist has an unbounded markup when eta = 1"

    # K_g is the group's COST-EFFICIENCY INDEX. It is the only thing about a group's
    # internal cost structure that matters: two groups with the same K_g behave
    # identically no matter how many affiliates they have or how costs are spread
    # across them. (uniqueness.jl STEP 3 shows why: K_g is exactly what appears in
    # the group's cost function after cost minimisation.)
    K = group_K(m)

    # ------------------------------------------------------------------
    # OUTER LOOP. Find the price aggregate A = sum_i p_i^(1-sigma).
    #
    # Given A, every group's share S_g(A) is pinned down independently (the inner
    # loop). Each S_g(A) is STRICTLY DECREASING in A: a tougher market means a
    # smaller share. So total_share(A) falls monotonically from above 1 (as A -> 0
    # every share -> 1, so the total -> G > 1) to 0 (as A -> infinity).
    #
    # A strictly decreasing function crosses the level 1 EXACTLY ONCE. That single
    # crossing is the equilibrium, and bisection cannot miss it or land on another
    # root because there is no other root. This is why we bisect instead of
    # iterating a fixed point: a damped fixed point on shares is NOT guaranteed to
    # converge here and hung during development.
    # ------------------------------------------------------------------
    lo, hi = 1e-14, 1e14
    while total_share(hi, K, sigma, eta) > 1.0     # widen until total < 1 at hi
        hi *= 10.0
    end
    while total_share(lo, K, sigma, eta) < 1.0     # widen until total > 1 at lo
        lo /= 10.0
    end
    for _ in 1:300
        mid = sqrt(lo * hi)                        # geometric: A spans many decades
        if total_share(mid, K, sigma, eta) > 1.0
            lo = mid                               # shares too big => raise A
        else
            hi = mid
        end
        hi / lo - 1.0 < 1e-15 && break
    end
    A = sqrt(lo * hi)

    # ------------------------------------------------------------------
    # Unwind the equilibrium. Every step below is a one-to-one map, which is why
    # a unique A gives a unique equilibrium in EVERY object, not just in shares.
    # ------------------------------------------------------------------
    S  = [inner_share(K[g] / A, sigma, eta) for g in eachindex(K)]   # group shares
    mu = markup(S, sigma, eta)          # ONE markup per group, shared by its affiliates
    p  = mu[m.gid] .* m.c               # price = markup x delivered cost
    w  = p .^ (1.0 - sigma)
    s  = w ./ sum(w)                    # variety revenue shares, CES demand
    P  = A^(1.0 / (1.0 - sigma))        # lower-tier CES price index
    E  = m.D * P^(1.0 - eta)            # class expenditure; = D exactly when eta = 1
    r  = E .* s                         # revenue
    q  = r ./ p                         # quantity
    Pi = E .* S .* (1.0 .- 1.0 ./ mu)   # = (E/sigma) S (1 + (sigma/eta - 1) S)
    return MarketEq(S, s, p, mu, q, r, Pi, A, P, E)
end

hhi_naive(eq::MarketEq)   = sum(eq.s .^ 2)     # each affiliate counted as a firm
hhi_grouped(eq::MarketEq) = sum(eq.S .^ 2)     # affiliates grouped by global parent

"""Delivered cost: mc * iceberg * (1 + tariff), elementwise."""
tariffed_cost(mc::Vector{Float64}, iceberg::Vector{Float64}, tariff::Vector{Float64}) =
    mc .* iceberg .* (1.0 .+ tariff)

"""
Profit income accruing to a country with ownership weights theta over groups.
Returns (ces, granular, total). `ces` is what a CES / monopolistic-competition
model would predict; `granular` is the ownership-weighted HHI correction, with
prefactor (sigma/eta - 1).
"""
function owner_income(eq::MarketEq, theta::Vector{Float64},
                      sigma::Float64, E::Float64, eta::Float64 = 1.0)
    ces      = (E / sigma) * sum(theta .* eq.S)
    granular = (E / sigma) * (sigma / eta - 1.0) * sum(theta .* eq.S .^ 2)
    return (ces = ces, granular = granular, total = ces + granular)
end

"""
Decompose the granular ownership term.

    sum_g theta_g S_g^2  =  thetabar * HHI  +  Cov_S(theta, S)

`thetabar = sum_g theta_g S_g` is the value-weighted country ownership share, the
only object constructible from country-level data. `Cov_S` is the share-weighted
covariance of ownership with market share and requires firm-level global-ultimate-
parent identity. `ratio = total / naive` is the headline statistic of CLAUDE.md §7.
"""
function ownership_decomposition(S::Vector{Float64}, theta::Vector{Float64})
    thetabar = sum(theta .* S)
    hhi      = sum(S .^ 2)
    total    = sum(theta .* S .^ 2)
    naive    = thetabar * hhi
    return (total = total, thetabar = thetabar, hhi = hhi,
            naive = naive, cov = total - naive,
            ratio = naive > 0 ? total / naive : NaN)
end

# --------------------------------------------------------------------------
# Analytic comparative statics.
#
#   d ln S_g = Lambda_g (d ln K_g - d ln A)
#   sum_g S_g d ln S_g = 0  =>  d ln A = sum_g omega_g d ln K_g
#   omega_g = S_g Lambda_g / sum_h S_h Lambda_h,   sum_g omega_g = 1
#
# Verified against finite differences of `solve` in TEST 6 below and in
# scripts/verify/cournot_pe.py TEST C, for eta in {1, 1.5, 2.5}.
# --------------------------------------------------------------------------

"""Elasticity damping factor. Lambda -> 1 in the atomistic limit."""
function lambda_g(S::Vector{Float64}, sigma::Float64, eta::Float64 = 1.0)
    mu = markup(S, sigma, eta)
    return 1.0 ./ (1.0 .+ (sigma - 1.0) .* S .* mu .* (1.0 / eta - 1.0 / sigma))
end

"""
Incidence weights: `d ln P / d ln(1+t_g) = omega_g` for a tariff on group g.
Reduces to `omega_g = S_g` in the atomistic limit. Because Lambda is decreasing
in S, a dominant group has omega_g < S_g: it absorbs the tariff in its markup, so
the consumer price index rises by LESS than the CES benchmark.
"""
function omega_g(S::Vector{Float64}, sigma::Float64, eta::Float64 = 1.0)
    L = lambda_g(S, sigma, eta)
    return (S .* L) ./ sum(S .* L)
end

"""Tariff pass-through into group g's own price, `d ln p_g / d ln(1+t_g)`."""
function passthrough(S::Vector{Float64}, sigma::Float64, g::Int, eta::Float64 = 1.0)
    L, w, mu = lambda_g(S, sigma, eta), omega_g(S, sigma, eta), markup(S, sigma, eta)
    dlnS = L[g] * (1.0 - sigma) * (1.0 - w[g])
    return 1.0 + S[g] * mu[g] * (1.0 / eta - 1.0 / sigma) * dlnS
end

"""
Marginal welfare of an ad valorem tariff on group g at FREE TRADE, per unit of
class expenditure E:

    (1/E) dW/dt|_0  =  S_g/mu_g            tariff revenue (markup-deflated base!)
                     - omega_g             consumer price index
                     + sum_h theta_h dPi_h/dt / E     home-owned profit

Layer 0 has constant marginal cost, hence NO terms-of-trade motive: the only
reason to tax is rent extraction, the only reason to subsidise is the markup
distortion. Pass `theta = nothing` for the no-ownership benchmark.

Money-metric consumer surplus is E/(eta-1), whose derivative is -E dlnP for any
eta. Using -E ln(P) as the level is correct ONLY at eta = 1, where E is constant.
"""
function dW_dt(S::Vector{Float64}, sigma::Float64, g::Int,
               theta::Union{Nothing,Vector{Float64}} = nothing,
               eta::Float64 = 1.0)
    mu = markup(S, sigma, eta)
    L, w = lambda_g(S, sigma, eta), omega_g(S, sigma, eta)
    out = S[g] / mu[g] - w[g]
    if theta !== nothing
        dlnS = L .* (w[g] * (sigma - 1.0))          # rivals h != g expand
        dlnS[g] = L[g] * (1.0 - sigma) * (1.0 - w[g])
        k    = sigma / eta - 1.0
        dlnE = (1.0 - eta) * w[g]                   # E moves unless eta == 1
        Pi   = (1.0 / sigma) .* S .* (1.0 .+ k .* S)
        dPi  = Pi .* dlnE .+ (1.0 / sigma) .* (1.0 .+ 2.0 .* k .* S) .* S .* dlnS
        out += sum(theta .* dPi)
    end
    return out
end

end # module


# ---------------------------------------------------------------------------
using .CournotPE
using Printf
using Random

function run_tests()
    println("="^70)
    println("TEST 1  atomistic limit recovers the CES markup sigma/(sigma-1)")
    println("="^70)
    sigma = 5.0
    for N in (10, 100, 1000, 10000)
        eq = solve(Market(sigma, 1.0, ones(N), collect(1:N)))
        @printf("  N=%6d   mu=%.10f   CES=%.10f   share=%.2e\n",
                N, eq.mu[1], sigma / (sigma - 1.0), eq.S[1])
    end

    println()
    println("="^70)
    println("TEST 2  closed form  Pi_g = (E/s) S_g (1 + (s/eta-1) S_g)  vs solver")
    println("="^70)
    rng = MersenneTwister(7)
    for eta in (1.0, 1.5, 2.5)
        err = 0.0
        for _ in 1:1000
            sg  = eta + 0.5 + 9.0 * rand(rng)
            G   = rand(rng, 2:8)
            gid = vcat([fill(g, rand(rng, 1:3)) for g in 1:G]...)
            c   = exp.(0.6 .* randn(rng, length(gid)))
            D   = 0.5 + 2.5 * rand(rng)
            eq  = solve(Market(sg, eta, D, c, gid))
            cf  = (eq.E / sg) .* eq.S .* (1.0 .+ (sg / eta - 1.0) .* eq.S)
            err = max(err, maximum(abs.(cf .- eq.Pi)))
        end
        @printf("  eta=%-4.1f 1000 random markets, max |closed form - solver| = %.3e\n",
                eta, err)
    end

    println()
    println("="^70)
    println("TEST 3  ownership irrelevance holds atomistically, fails when granular")
    println("="^70)
    E = 1.0
    for (G, lab) in ((200, "atomistic"), (20, "moderate "), (5, "granular "))
        c   = exp.(range(0.0, 0.8, length = G))
        eq  = solve(Market(sigma, E, collect(c), collect(1:G)))
        ord = sortperm(eq.S, rev = true)
        k   = max(1, G ÷ 5)
        big = zeros(G); big[ord[1:k]] .= 1.0
        sml = zeros(G); sml[ord[end-k+1:end]] .= 1.0
        for (th, nm) in ((big, "owns biggest 20%"), (sml, "owns smallest 20%"))
            oi = owner_income(eq, th, sigma, E)
            d  = ownership_decomposition(eq.S, th)
            @printf("  %s G=%3d | %-18s total/CES=%.3fx   naive=%.5f cov=%+.5f  total/naive=%.2fx\n",
                    lab, G, nm, oi.total / oi.ces, d.naive, d.cov, d.ratio)
        end
    end
    println("  total/naive is the headline statistic: how badly country-level")
    println("  ownership data understates the correction. Independent of eta.")

    println()
    println("="^70)
    println("TEST 4  tariff incidence: pass-through, not the profit elasticity")
    println("="^70)
    G  = 6
    c0 = exp.(range(0.0, 0.5, length = G))
    eq0 = solve(Market(sigma, E, collect(c0), collect(1:G)))
    p0, pi0, S0 = eq0.p[1], eq0.Pi[1], eq0.S[1]
    for t in (0.10, 0.25)
        c = collect(c0); c[1] *= (1.0 + t)
        eq = solve(Market(sigma, E, c, collect(1:G)))
        pt = log(eq.p[1] / p0) / log(1.0 + t)
        el = log(eq.Pi[1] / pi0) / log(1.0 + t)
        @printf("  t=%.2f  pass-through=%.4f   dlnPi/dln(1+t)=%.4f   CES bench=%.4f\n",
                t, pt, el, -(sigma - 1.0) * (1.0 - S0))
    end
    println("  The profit ELASTICITY is smaller than CES, but it is the wrong")
    println("  statistic (see CLAUDE.md 4.8). The mechanism is pass-through: the")
    println("  foreign firm bears 1-rho, rising in S. Under CES rho = 1 exactly and")
    println("  NO rent is extracted, so rent extraction is a pure oligopoly effect --")
    println("  which is precisely what home ownership neutralises.")

    println()
    println("="^70)
    println("TEST 5  Fact 4 / Figure 6: grouping affiliates by parent")
    println("="^70)
    c  = exp.(range(0.0, 0.9, length = 12))
    eA = solve(Market(sigma, E, collect(c), collect(1:12)))
    gB = vcat([fill(g, 4) for g in 1:3]...)
    eB = solve(Market(sigma, E, collect(c), gB))
    @printf("  12 independent firms       HHI=%.4f  top=%.4f  mean mu=%.4f\n",
            hhi_naive(eA), maximum(eA.s), sum(eA.mu) / length(eA.mu))
    @printf("  3 parents, counted naively HHI=%.4f\n", hhi_naive(eB))
    @printf("  3 parents, grouped         HHI=%.4f  top=%.4f  mean mu=%.4f\n",
            hhi_grouped(eB), maximum(eB.S), sum(eB.mu) / length(eB.mu))
    println("  NOTE: Figure 6 holds BEHAVIOUR fixed and changes only ACCOUNTING.")
    @printf("  That experiment is %.4f -> %.4f (ratio %.2fx) vs data 0.192 -> 0.215 (1.12x).\n",
            hhi_naive(eB), hhi_grouped(eB), hhi_grouped(eB) / hhi_naive(eB))
    @printf("  Grouping RAISES measured HHI; coordination LOWERS it (%.4f -> %.4f)\n",
            hhi_naive(eA), hhi_naive(eB))
    println("  because output restriction equalises affiliate shares. Fact 4 nets these.")

    println()
    println("="^70)
    println("TEST 6  analytic comparative statics vs finite differences")
    println("="^70)
    rng6 = MersenneTwister(11)
    for eta in (1.0, 1.5, 2.5)
        errP, errR = 0.0, 0.0
        for _ in 1:100
            sg  = eta + 0.5 + 7.5 * rand(rng6)
            G   = rand(rng6, 2:5)
            gid = vcat([fill(g, rand(rng6, 1:3)) for g in 1:G]...)
            c   = exp.(0.6 .* randn(rng6, length(gid)))
            eq  = solve(Market(sg, eta, 1.0, c, gid))
            w   = omega_g(eq.S, sg, eta)
            for g in 1:G
                h  = 1e-6
                cu = [gid[i] == g ? c[i]*(1+h) : c[i] for i in eachindex(c)]
                cd = [gid[i] == g ? c[i]*(1-h) : c[i] for i in eachindex(c)]
                up = solve(Market(sg, eta, 1.0, cu, gid))
                dn = solve(Market(sg, eta, 1.0, cd, gid))
                errP = max(errP, abs((log(up.P) - log(dn.P)) / (2h) - w[g]))
                i    = findfirst(==(g), gid)
                errR = max(errR, abs((log(up.p[i]) - log(dn.p[i])) / (2h)
                                     - passthrough(eq.S, sg, g, eta)))
            end
        end
        @printf("  eta=%-4.1f max|dlnP - omega| = %.2e   max|rho - analytic| = %.2e\n",
                eta, errP, errR)
    end

    println()
    println("="^70)
    println("TEST 7  sign of optimal Layer-0 policy: dominant group vs a fringe of n")
    println("="^70)
    println("  Smallest dominant share S at which a TARIFF beats free trade ('.' = never).")
    ns = (1, 2, 4, 9, 19, 49)
    for eta in (1.0, 1.5)
        @printf("   eta = %.1f\n", eta)
        print("     sigma"); for n in ns; @printf("%9s", "n=$n"); end; println()
        for sg in (3.0, 5.0, 8.0, 12.0, 20.0)
            @printf("%10.1f", sg)
            for n in ns
                star = NaN
                for Sx in range(0.02, 0.998, length = 1500)
                    S = vcat([Sx], fill((1 - Sx) / n, n))
                    if dW_dt(S, sg, 1, nothing, eta) > 0
                        star = Sx; break
                    end
                end
                isnan(star) ? @printf("%9s", ".") : @printf("%9.3f", star)
            end
            println()
        end
    end
    println("  Layer 0 has no terms-of-trade motive, so a tariff pays only when rent")
    println("  extraction beats the markup distortion: high sigma, high S, fragmented")
    println("  fringe. Otherwise optimal Layer-0 policy is a SUBSIDY, and home")
    println("  ownership makes the subsidy LARGER. Layer 0 cannot host the optimal-")
    println("  tariff result; it needs GE or an extensive margin. Not an eta artefact.")

    println()
    println("="^70)
    println("TEST 8  what eta > 1 fixes")
    println("="^70)
    @printf("  %6s%16s%16s%16s\n", "S_g", "mu(eta=1)", "mu(eta=1.5)", "mu(eta=3)")
    for S in (0.5, 0.9, 0.99)
        @printf("  %6.2f%16.4f%16.4f%16.4f\n", S,
                markup(S, 5.0, 1.0), markup(S, 5.0, 1.5), markup(S, 5.0, 3.0))
    end
    println("  monopoly markup is eta/(eta-1) for eta > 1, unbounded at eta = 1.")
    println("  eta = 1 is a knife-edge exactly where the punchline lives. Report the")
    println("  eta = 1 closed forms, but check every headline number at eta > 1.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_tests()
end
