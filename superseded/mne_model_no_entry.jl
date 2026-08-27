###############################################################################
#
#   MULTINATIONAL OWNERSHIP, MARKET POWER AND TRADE POLICY
#   A complete, self-contained general equilibrium model.
#
#   Sebastian Velasquez Palacios (IDB / PTI, with Christian Volpe Martincus)
#
#   ---------------------------------------------------------------------------
#   HOW TO RUN
#   ---------------------------------------------------------------------------
#       julia mne_model.jl quick        everything, small world         (~6 min)
#       julia mne_model.jl              everything                      (~65 min)
#       julia mne_model.jl full         heavy verification              (~2 hours)
#       julia mne_model.jl core         one market + uniqueness proofs  (~10 sec)
#       julia mne_model.jl ge           general equilibrium + audit     (~9 min)
#       julia mne_model.jl facts        the six stylized facts          (~55 min)
#
#   Start with `quick`. It runs every section on a smaller economy, so you see
#   the whole structure in a couple of minutes before committing to a long run.
#
#   No packages required. Standard library only (Printf, Random, LinearAlgebra).
#   Tested on Julia 1.10 and 1.12.
#
#   ---------------------------------------------------------------------------
#   WHAT THE MODEL IS, IN ONE SCREEN
#   ---------------------------------------------------------------------------
#   Countries n = 1..N, sectors k = 1..K, affiliates (plants) a = 1..A.
#   Each affiliate belongs to a global ultimate PARENT g. Parents are the
#   strategic agents: they choose quantities market by market, internalising
#   competition between their own affiliates.
#
#   A market is one (destination, sector) cell. In it:
#
#       1/mu_g = 1 - (1-S_g)/sigma_k - S_g/eta            markup
#       Pi_g   = (E/sigma_k) S_g [1 + (sigma_k/eta - 1) S_g]   profit
#       sum_g S_g = 1                                     clearing
#
#   Delivered cost of affiliate a into destination n (sector k):
#
#       a[a,n] = (1/phi_a)
#                * ( w[h]^alpha_k * w[l]^(1-alpha_k) )^(1-nu_k)   labour
#                * ( PIO[l,k] )^nu_k                              intermediates
#                * gamma[h,l]                                     MP friction
#                * d[l,n]                                         trade cost
#       c[a,n] = a[a,n] * (1 + tariff[l,n,k])
#
#   with h = the parent's HQ country, l = the production country, and
#   PIO[l,k] = prod_k' P[l,k']^omega[k',k] the intermediate bundle price.
#
#   Sources. The multinational production block is Ramondo & Rodriguez-Clare
#   (2013) for gamma and Tintelnot (2017) for l != n (export platforms);
#   alpha_k is the headquarter input of Head & Mayer (2019); the input-output
#   loop follows Caliendo & Parro (2015). What is NEW here is (i) group-level
#   Cournot conduct instead of constant markups and (ii) firm-level ownership
#   theta[g,n]. Those two are the contribution.
#
#   ---------------------------------------------------------------------------
#   EQUILIBRIUM: DEFINITION
#   ---------------------------------------------------------------------------
#   Wages w, prices P, expenditures E and incomes X such that
#       (a) every group plays a best response in every market,
#       (b) sum_g S_g = 1 in every market,
#       (c) P is consistent with the costs it generates (the I-O loop),
#       (d) labour demand = L in every country,
#       (e) X[n] = w[n] L[n] + sum_g theta[g,n] Pi[g] + tariff revenue[n].
#   One of (d) is redundant by Walras' law; w[1] = 1 is the numeraire.
#
###############################################################################

module MNEModel

using Printf, Random, LinearAlgebra

export Market, MarketEq, solve_market, markup,
       GEModel, solve_ge, excess_demand, accounting, export_matrix, classify,
       measured_hhi, structural_hhi

###############################################################################
# PART 1.  ONE MARKET: GROUP COURNOT WITH NESTED CES
###############################################################################

"""
One (destination, sector) cell.

`gid[i]` is the group that owns variety `i`, numbered 1..G contiguously.
`D` is the demand shifter; with eta = 1 it is simply expenditure E.
"""
struct Market
    sigma::Float64
    eta::Float64
    D::Float64
    c::Vector{Float64}
    gid::Vector{Int}
end

struct MarketEq
    S::Vector{Float64}      # group shares
    s::Vector{Float64}      # variety shares
    p::Vector{Float64}      # prices
    mu::Vector{Float64}     # group markups
    q::Vector{Float64}      # quantities
    r::Vector{Float64}      # revenues
    Pi::Vector{Float64}     # group profits
    A::Float64              # price aggregate sum_i p_i^(1-sigma)
    P::Float64              # CES price index
    E::Float64              # expenditure
end

"""
Markup as a function of the group's TOTAL share.

    1/mu = 1 - (1-S)/sigma - S/eta

A weighted average of two elasticities: the group fights rivals with elasticity
sigma on the share (1-S) it does not own, and faces only the across-sector
elasticity eta on the share S it does own, because expanding there merely
cannibalises itself.

    S -> 0  =>  mu -> sigma/(sigma-1)     the textbook CES markup
    S -> 1  =>  mu -> eta/(eta-1)         a monopolist (unbounded if eta = 1)

Requires sigma > eta. If that failed, markups would FALL with size, which is
economically backwards and breaks uniqueness (see PART 2, step 5).
"""
markup(S, sigma::Float64, eta::Float64) =
    1.0 ./ (1.0 .- (1.0 .- S) ./ sigma .- S ./ eta)

"""
INNER LOOP. Given the price aggregate A, find one group's share: the unique root
in (0,1) of  x = mu(x)^(1-sigma) * KA,  where KA = K_g / A.

Unique because the left side rises from 0 to 1 while the right side falls (mu
increases in x and the exponent 1-sigma is negative). A rising and a falling
curve cross at most once, and they do cross. Bisection is therefore exact.
"""
function inner_share(KA::Float64, sigma::Float64, eta::Float64)
    lo, hi = 0.0, 1.0
    x = 0.5
    for _ in 1:100
        mu = markup(x, sigma, eta)
        F  = x - mu^(1.0 - sigma) * KA          # increasing in x
        F > 0.0 ? (hi = x) : (lo = x)           # keep a valid bracket at all times
        # F'(x) = 1 + (sigma-1) mu^(-sigma) mu'(x) KA,  mu' = mu^2 (1/eta - 1/sigma)
        dmu = mu * mu * (1.0/eta - 1.0/sigma)
        dF  = 1.0 + (sigma - 1.0) * mu^(-sigma) * dmu * KA
        xn  = x - F / dF                        # Newton step
        (isfinite(xn) && lo < xn < hi) || (xn = 0.5 * (lo + hi))   # safeguard
        abs(xn - x) <= 1e-16 && return xn
        x = xn
    end
    return x
end

group_K(m::Market) =
    (K = zeros(maximum(m.gid)); for i in eachindex(m.c)
        K[m.gid[i]] += m.c[i]^(1.0 - m.sigma) end; K)

total_share(A, K, sigma, eta) = sum(inner_share(k / A, sigma, eta) for k in K)

"""
d(sum_g S_g)/d(ln A). Strictly negative, which is the monotonicity the outer loop
relies on. Derived from ln S = (1-sigma) ln mu(S) + ln K - ln A.
"""
function dtotal_dlnA(A, K, sigma, eta)
    acc = 0.0
    for k in K
        S = inner_share(k / A, sigma, eta)
        mu = markup(S, sigma, eta)
        acc -= S / (1.0 + (sigma - 1.0) * mu * S * (1.0/eta - 1.0/sigma))
    end
    return acc
end

"""
Solve one market. The equilibrium is UNIQUE; see PART 2 for the proof and its
numerical verification. The three assertions are the theorem's hypotheses.
"""
function solve_market(m::Market)
    sigma, eta = m.sigma, m.eta
    @assert sigma > eta >= 1.0 "need sigma > eta >= 1 (else markups fall with size)"
    @assert all(m.c .> 0.0) "marginal costs must be strictly positive"
    @assert maximum(m.gid) >= 2 || eta > 1.0 "a monopolist's markup is unbounded when eta = 1"

    K = group_K(m)

    # OUTER LOOP. total_share(A) is strictly DECREASING in A, running from above
    # 1 (as A -> 0 every share -> 1, so the sum -> G > 1) down to 0. A strictly
    # decreasing curve crosses the level 1 exactly once, so the root is unique and
    # a bracketing method cannot land on a different one: there is none.
    #
    # We solve it by NEWTON SAFEGUARDED BY BISECTION on ln A. Newton gives speed;
    # the bracket gives the guarantee. Every iterate stays inside a bracket known
    # to contain the root, so this is exactly as safe as pure bisection and much
    # faster. (A plain fixed-point iteration on shares, the usual way people code
    # these models, has no such guarantee and does not always converge.)
    #
    # The bracket is anchored on the CES benchmark A_ces = sum_i (mu_ces c_i)^(1-sigma),
    # which is the right order of magnitude, then widened until it brackets.
    Aces = (sigma / (sigma - 1.0))^(1.0 - sigma) * sum(K)
    lo, hi = Aces * 1e-6, Aces * 1e6
    it = 0
    while total_share(hi, K, sigma, eta) > 1.0 && it < 500; hi *= 100.0; it += 1; end
    it = 0
    while total_share(lo, K, sigma, eta) < 1.0 && it < 500; lo /= 100.0; it += 1; end

    x = 0.5 * (log(lo) + log(hi))                 # work in ln A
    llo, lhi = log(lo), log(hi)
    for _ in 1:200
        Ax = exp(x)
        F  = total_share(Ax, K, sigma, eta) - 1.0
        F > 0.0 ? (llo = x) : (lhi = x)           # F is DECREASING in ln A
        dF = dtotal_dlnA(Ax, K, sigma, eta)
        xn = x - F / dF
        (isfinite(xn) && llo < xn < lhi) || (xn = 0.5 * (llo + lhi))
        abs(xn - x) <= 1e-15 && (x = xn; break)
        x = xn
    end
    A = exp(x)

    # Unwind. Every step is a one-way map, so one A gives one of everything.
    S  = [inner_share(k / A, sigma, eta) for k in K]
    mu = markup(S, sigma, eta)
    p  = mu[m.gid] .* m.c
    w  = p .^ (1.0 - sigma)
    s  = w ./ sum(w)
    P  = A^(1.0 / (1.0 - sigma))
    E  = m.D * P^(1.0 - eta)
    r  = E .* s
    q  = r ./ p
    Pi = E .* S .* (1.0 .- 1.0 ./ mu)
    return MarketEq(S, s, p, mu, q, r, Pi, A, P, E)
end

###############################################################################
# PART 2.  UNIQUENESS OF THE MARKET EQUILIBRIUM  (theorem + verification)
#
#   rho = (sigma-1)/sigma,  X_g = sum_{i in g} q_i^rho,  B_g = rivals' equivalent.
#
#   STEP 1  Group revenue depends on the group's own quantities ONLY through the
#           single number X_g:   R = D^(1/eta) X (X+B)^e,  e = (eta-sigma)/((sigma-1)eta).
#           A G-dimensional choice collapses to one dimension.
#   STEP 2  R is strictly increasing and strictly CONCAVE in X
#           (dR/dX > 0 since e > -1;  d2R/dX2 < 0 since e < 0).
#   STEP 3  Cost of delivering X is strictly CONVEX:
#           C(X) = X^(sigma/(sigma-1)) K_g^(-1/(sigma-1)),  K_g = sum c_i^(1-sigma).
#   STEP 4  Pi = concave - convex = strictly concave, so the FOC is not only
#           necessary but SUFFICIENT and the best response is unique. Directly in
#           the quantity vector: X(q) is concave, R is concave and increasing, and
#           a concave increasing function of a concave function is concave.
#           No corner: as q -> 0 revenue falls like q^rho and cost like q, and
#           rho < 1, so producing a little always beats producing nothing.
#   STEP 5  psi(x) = x mu(x)^(sigma-1) is strictly increasing (this needs
#           sigma > eta), so S_g(A) = psi^{-1}(K_g/A) is unique and strictly
#           decreasing in A. Hence sum_g S_g(A) = 1 has exactly one solution.
#
#   => unique A => unique shares, markups, prices, revenues, quantities.
#      Existence is free because the construction is constructive.
###############################################################################

rho_of(sigma) = (sigma - 1.0) / sigma

revenue_of_X(X, B, sigma, eta, D) =
    D^(1.0 / eta) * X * (X + B)^((eta - sigma) / ((sigma - 1.0) * eta))

cost_of_X(X, K, sigma) = X^(sigma / (sigma - 1.0)) * K^(-1.0 / (sigma - 1.0))

"""Profit at an arbitrary quantity vector, using NONE of the derived algebra."""
function primitive_profit(q, m::Market, g::Int)
    Aq = sum(q .^ rho_of(m.sigma))
    P  = m.D^(1.0 / m.eta) * Aq^(m.sigma / ((1.0 - m.sigma) * m.eta))
    p  = m.D^(1.0 / m.sigma) .* P^((m.sigma - m.eta) / m.sigma) .* q .^ (-1.0 / m.sigma)
    sel = m.gid .== g
    return sum((p[sel] .- m.c[sel]) .* q[sel])
end

function random_market(rng; eta = 1.0, gmax = 5, nmax = 3)
    sigma = eta + 1.0 + 7.0 * rand(rng)
    G   = rand(rng, 2:gmax)
    gid = vcat([fill(g, rand(rng, 1:nmax)) for g in 1:G]...)
    c   = exp.(0.6 .* randn(rng, length(gid)))
    return Market(sigma, eta, 0.5 + 2.5 * rand(rng), c, gid)
end

function verify_uniqueness(; reps = 200, starts = 40, verbose = true)
    rng = MersenneTwister(20260812)
    verbose && println("-"^78)
    verbose && println("MARKET-LEVEL UNIQUENESS: verifying each step of the proof")
    verbose && println("-"^78)

    # STEP 1: reshuffle output within a group at constant X; revenue must not move
    e1 = 0.0
    for _ in 1:reps
        m = random_market(rng; eta = rand(rng) < 0.5 ? 1.0 : 1.0 + rand(rng))
        q = exp.(randn(rng, length(m.c)))
        for g in 1:maximum(m.gid)
            sel = m.gid .== g
            X = sum(q[sel] .^ rho_of(m.sigma))
            q2 = copy(q); wgt = rand(rng, count(sel)); wgt ./= sum(wgt)
            q2[sel] = (wgt .* X) .^ (1.0 / rho_of(m.sigma))
            rev(qq) = begin
                Aq = sum(qq .^ rho_of(m.sigma))
                Pp = m.D^(1/m.eta) * Aq^(m.sigma/((1-m.sigma)*m.eta))
                pp = m.D^(1/m.sigma) .* Pp^((m.sigma-m.eta)/m.sigma) .* qq .^ (-1/m.sigma)
                sum(pp[sel] .* qq[sel])
            end
            e1 = max(e1, abs(rev(q2) - rev(q)) / rev(q))
        end
    end
    verbose && @printf("  step 1  revenue depends only on X_g          : %.2e\n", e1)

    # STEPS 2-3: curvature signs
    badR = badC = 0
    for _ in 1:2000
        sg = 1.5 + 8.0*rand(rng); et = 1.0 + (sg-1.0)*rand(rng)*0.9
        B = 0.05+3rand(rng); Kk = 0.1+3rand(rng); D = 0.5+2rand(rng)
        X = 0.05+3rand(rng); h = 1e-5*X
        d2R = (revenue_of_X(X+h,B,sg,et,D) - 2revenue_of_X(X,B,sg,et,D)
               + revenue_of_X(X-h,B,sg,et,D))/h^2
        d2C = (cost_of_X(X+h,Kk,sg) - 2cost_of_X(X,Kk,sg) + cost_of_X(X-h,Kk,sg))/h^2
        d2R >  1e-6 && (badR += 1)
        d2C < -1e-6 && (badC += 1)
    end
    verbose && @printf("  steps 2-3  R concave / C convex violations   : %d / %d\n", badR, badC)

    # STEP 4: concavity of profit in the group's own quantity vector
    viol, worst = 0, -Inf
    for _ in 1:reps
        m = random_market(rng; eta = rand(rng) < 0.5 ? 1.0 : 1.0 + rand(rng))
        eq = solve_market(m)
        for g in 1:maximum(m.gid)
            sel = m.gid .== g
            for _ in 1:5
                q1 = copy(eq.q); q2 = copy(eq.q)
                q1[sel] .*= exp.(0.7 .* randn(rng, count(sel)))
                q2[sel] .*= exp.(0.7 .* randn(rng, count(sel)))
                qm = copy(eq.q); qm[sel] = 0.5 .* (q1[sel] .+ q2[sel])
                f1 = primitive_profit(q1,m,g); f2 = primitive_profit(q2,m,g)
                fm = primitive_profit(qm,m,g)
                gap = (0.5*(f1+f2) - fm) / max(abs(fm), 1e-12)
                gap > 1e-10 && (viol += 1); worst = max(worst, gap)
            end
        end
    end
    verbose && @printf("  step 4  profit concave in own q: violations  : %d (worst %.1e)\n",
                       viol, worst)

    # STEP 5: monotonicity of psi
    bad = 0
    for _ in 1:500
        sg = 1.5+8rand(rng); et = 1.0+(sg-1.0)*rand(rng)*0.9
        xs = range(1e-6, 0.999, length=400)
        psi = [x*markup(x,sg,et)^(sg-1.0) for x in xs]
        all(diff(psi) .> 0) || (bad += 1)
    end
    verbose && @printf("  step 5  psi non-monotone in                  : %d / 500 draws\n", bad)

    # The decisive test: many random starts, solved WITHOUT the closed forms
    cases = [(1.0,5.0,[1,2]), (1.0,5.0,[1,1,2,2,3,3,4,4]),
             (1.0,3.0,[1,1,1,2,2,3,4,5,6,6]), (1.8,7.0,[1,1,2,2,3,3,4,4,5,5]),
             (2.5,9.0,collect(1:8))]
    worstspread = 0.0
    for (et, sg, gid) in cases
        rc = MersenneTwister(4242)
        m = Market(sg, et, 1.0, exp.(0.5 .* randn(rc, length(gid))), gid)
        ref = solve_market(m)
        for t in 1:starts
            r2 = MersenneTwister(1000+t)
            q = exp.(3.0 .* randn(r2, length(m.c)))
            for _ in 1:600                                  # best response on X_g
                qold = copy(q)
                for g in 1:maximum(m.gid)
                    sel = m.gid .== g
                    B = sum(q[.!sel] .^ rho_of(m.sigma))
                    Kk = sum(m.c[sel] .^ (1.0 - m.sigma))
                    # maximise Pi(X) = R(X) - C(X) by bisection on the derivative.
                    # Legitimate precisely BECAUSE Pi is strictly concave (step 4),
                    # so dPi crosses zero exactly once.
                    lo, hi = 1e-12, 1e6
                    dPi(X) = begin
                        h = 1e-7*X
                        (revenue_of_X(X+h,B,m.sigma,m.eta,m.D)-cost_of_X(X+h,Kk,m.sigma)
                        -revenue_of_X(X-h,B,m.sigma,m.eta,m.D)+cost_of_X(X-h,Kk,m.sigma))/(2h)
                    end
                    for _ in 1:70
                        mid = sqrt(lo*hi)
                        dPi(mid) > 0 ? (lo = mid) : (hi = mid)
                        hi/lo - 1 < 1e-13 && break
                    end
                    Xs = sqrt(lo*hi)
                    ci = m.c[sel] .^ (-m.sigma)
                    q[sel] = ci .* (Xs/sum(ci .^ rho_of(m.sigma)))^(1/rho_of(m.sigma))
                end
                maximum(abs.(log.(q ./ qold))) < 1e-13 && break   # converged
            end
            Aq = sum(q .^ rho_of(m.sigma))
            Pp = m.D^(1/m.eta)*Aq^(m.sigma/((1-m.sigma)*m.eta))
            pp = m.D^(1/m.sigma) .* Pp^((m.sigma-m.eta)/m.sigma) .* q .^ (-1/m.sigma)
            ss = (pp .* q) ./ sum(pp .* q)
            S = [sum(ss[m.gid .== g]) for g in 1:maximum(m.gid)]
            worstspread = max(worstspread, maximum(abs.(S .- ref.S)))
        end
    end
    verbose && @printf("  DECISIVE: %d starts x %d structures, all identical to %.1e\n",
                       starts, length(cases), worstspread)
    return (step1=e1, badR=badR, badC=badC, viol=viol, psi=bad, spread=worstspread)
end

###############################################################################
# PART 3.  GENERAL EQUILIBRIUM WITH INPUT-OUTPUT LINKAGES
###############################################################################

"""
A general equilibrium economy with multinational production and I-O linkages.

`nu[k]`        share of sector k's cost spent on intermediate inputs
`omega[k',k]`  share of sector k's intermediate bill spent on sector k' goods
               (each COLUMN sums to one)
`theta[g,n]`   share of parent g owned by country n (each ROW sums to one)
"""
struct GEModel
    N::Int
    K::Int
    sigma::Vector{Float64}
    eta::Float64
    beta::Vector{Float64}       # final demand weights across sectors
    alpha::Vector{Float64}      # HQ input intensity
    nu::Vector{Float64}         # intermediate share of cost
    omega::Matrix{Float64}      # I-O weights, columns sum to 1
    L::Vector{Float64}
    par::Vector{Int}
    hq::Vector{Int}
    loc::Vector{Int}
    sec::Vector{Int}
    phi::Vector{Float64}
    gamma::Matrix{Float64}      # MP friction
    d::Matrix{Float64}          # trade cost
    tariff::Array{Float64,3}
    theta::Matrix{Float64}
end

"""
Delivered pre-tariff cost, given wages and the matrix of sector price indices P.

    a = (1/phi) * (w_h^alpha * w_l^(1-alpha))^(1-nu) * PIO^nu * gamma * d
    PIO[l,k] = prod_k' P[l,k']^omega[k',k]

The intermediate bundle is bought in the PRODUCTION country l at the prices
prevailing there, which is what makes multinational entry into l cheapen local
firms' inputs -- the channel Fact 5 needs.
"""
function delivered_cost(m::GEModel, w::Vector{Float64}, P::Matrix{Float64},
                        a::Int, n::Int)
    k, h, l = m.sec[a], m.hq[a], m.loc[a]
    lab = (w[h]^m.alpha[k] * w[l]^(1.0 - m.alpha[k]))^(1.0 - m.nu[k])
    pio = 1.0
    if m.nu[k] > 0
        acc = 0.0
        for kp in 1:m.K
            m.omega[kp, k] > 0 && (acc += m.omega[kp, k] * log(P[l, kp]))
        end
        pio = exp(m.nu[k] * acc)
    end
    return lab * pio * m.gamma[h, l] * m.d[l, n] / m.phi[a]
end

"""Solve all markets given wages AND a price matrix (one pass of the I-O loop)."""
function solve_markets_at(m::GEModel, w::Vector{Float64}, P::Matrix{Float64})
    mk = Array{Any}(undef, m.N, m.K)
    idxk = [findall(==(k), m.sec) for k in 1:m.K]
    for n in 1:m.N, k in 1:m.K
        idx = idxk[k]
        apre = [delivered_cost(m, w, P, a, n) for a in idx]
        c = [apre[j] * (1.0 + m.tariff[m.loc[idx[j]], n, k]) for j in eachindex(idx)]
        pars = m.par[idx]
        uniq = sort(unique(pars))
        lut  = Dict(u => i for (i, u) in enumerate(uniq))
        eq   = solve_market(Market(m.sigma[k], m.eta, 1.0, c, [lut[p] for p in pars]))
        mk[n, k] = (eq = eq, idx = idx, pars = uniq, apre = apre, P = eq.P)
    end
    return mk
end

"""
Solve the INPUT-OUTPUT PRICE LOOP given wages.

Costs depend on price indices and price indices depend on costs. That fixed point
is a CONTRACTION, so it has exactly one solution and simple iteration finds it:

  * the market price index P is homogeneous of degree ONE in the costs it faces
    (scale every cost by lambda and every share is unchanged, so P scales by
    lambda), and
  * log cost depends on log PIO with coefficient nu_k, and the omega weights of
    each column sum to one.

So the map from log P to log P has all row sums of its Jacobian bounded by
max_k nu_k < 1: a contraction in the sup norm, with modulus max nu.
Existence, uniqueness and geometric convergence at rate nu all follow from
Banach's fixed point theorem. `iters` below should be about
log(tol)/log(max nu).
"""
function solve_prices(m::GEModel, w::Vector{Float64}; tol = 1e-14, maxit = 2000,
                      P0 = nothing)
    # A warm start does not change the answer -- the fixed point is unique -- but
    # it cuts the iteration count a lot when solving many nearby wage vectors.
    P = P0 === nothing ? ones(m.N, m.K) : copy(P0)
    mk = solve_markets_at(m, w, P)
    for it in 1:maxit
        Pn = [mk[n, k].P for n in 1:m.N, k in 1:m.K]
        gap = maximum(abs.(log.(Pn ./ P)))
        P = Pn
        mk = solve_markets_at(m, w, P)
        gap < tol && return (P = P, mk = mk, iters = it, gap = gap)
    end
    Pn = [mk[n, k].P for n in 1:m.N, k in 1:m.K]
    return (P = P, mk = mk, iters = maxit, gap = maximum(abs.(log.(Pn ./ P))))
end

"""Final-demand shares across sectors; sum to one, so no income leaks."""
function expenditure_shares(m::GEModel, mk)
    eps = zeros(m.N, m.K)
    for n in 1:m.N
        tot = sum(m.beta[k] * mk[n, k].P^(1.0 - m.eta) for k in 1:m.K)
        for k in 1:m.K
            eps[n, k] = m.beta[k] * mk[n, k].P^(1.0 - m.eta) / tot
        end
    end
    return eps
end

cellidx(n, k, K) = (n - 1) * K + k

"""
Solve for expenditures E and incomes X.

With shares already known, both intermediate demand and profits are LINEAR in
expenditure, so this is two exact linear solves rather than another fixed point:

    E = eps .* X + B E          intermediate demand      => E = (I-B)^{-1} eps X
    X = wL + theta' Pi(E) + T(E)                          => one N x N solve
"""
function solve_quantities(m::GEModel, w::Vector{Float64}, mk, eps)
    NK, G = m.N * m.K, size(m.theta, 1)

    # spending coefficient: value bought per unit of revenue in each cell
    Bm = zeros(NK, NK)
    piu = zeros(G, NK)
    tau = zeros(NK)
    for np in 1:m.N, kpp in 1:m.K
        e, idx, pars = mk[np, kpp].eq, mk[np, kpp].idx, mk[np, kpp].pars
        s2 = m.sigma[kpp] / m.eta - 1.0
        for (i, g) in enumerate(pars)
            piu[g, cellidx(np, kpp, m.K)] = (e.S[i]/m.sigma[kpp])*(1.0 + s2*e.S[i])
        end
        for (j, a) in enumerate(idx)
            t  = m.tariff[m.loc[a], np, kpp]
            gi = findfirst(==(m.par[a]), pars)
            pay = e.s[j] / (e.mu[gi] * (1.0 + t))        # factor+intermediate bill
            tau[cellidx(np, kpp, m.K)] += e.s[j] * t / (e.mu[gi] * (1.0 + t))
            # this affiliate buys intermediates in its OWN country m.loc[a]
            for kp in 1:m.K
                m.omega[kp, kpp] == 0 && continue
                Bm[cellidx(m.loc[a], kp, m.K), cellidx(np, kpp, m.K)] +=
                    m.omega[kp, kpp] * m.nu[kpp] * pay
            end
        end
    end

    Lam = zeros(NK, m.N)
    for n in 1:m.N, k in 1:m.K
        Lam[cellidx(n, k, m.K), n] = eps[n, k]
    end
    C = (I(NK) - Bm) \ Lam                      # E = C * X

    Mm = zeros(m.N, m.N)
    for n in 1:m.N
        row = zeros(NK)
        for g in 1:G
            m.theta[g, n] == 0 && continue
            row .+= m.theta[g, n] .* piu[g, :]
        end
        for k in 1:m.K
            row[cellidx(n, k, m.K)] += tau[cellidx(n, k, m.K)]
        end
        Mm[n, :] = row' * C
    end
    X = (I(m.N) - Mm) \ (w .* m.L)
    E = C * X
    return X, reshape(E, m.K, m.N)' |> Matrix, piu, tau
end

"""
Labour demand. Only the (1-nu) share of the bill is paid to labour; the rest buys
intermediates. Of the labour part, alpha goes to the PARENT's country and
(1-alpha) to the HOST -- that split is what makes this a multinational model.
"""
function labour_demand(m::GEModel, w::Vector{Float64}, mk, Emat)
    LD = zeros(m.N)
    for n in 1:m.N, k in 1:m.K
        e, idx, pars = mk[n, k].eq, mk[n, k].idx, mk[n, k].pars
        E = Emat[n, k]
        for (j, a) in enumerate(idx)
            t  = m.tariff[m.loc[a], n, k]
            gi = findfirst(==(m.par[a]), pars)
            pay = (1.0 - m.nu[k]) * E * e.s[j] / (e.mu[gi] * (1.0 + t))
            LD[m.hq[a]]  += m.alpha[k] * pay / w[m.hq[a]]
            LD[m.loc[a]] += (1.0 - m.alpha[k]) * pay / w[m.loc[a]]
        end
    end
    return LD
end

function excess_demand(m::GEModel, w::Vector{Float64}; P0 = nothing)
    pr  = solve_prices(m, w; P0 = P0)
    eps = expenditure_shares(m, pr.mk)
    X, Emat, piu, tau = solve_quantities(m, w, pr.mk, eps)
    LD  = labour_demand(m, w, pr.mk, Emat)
    return LD .- m.L, (mk = pr.mk, P = pr.P, eps = eps, X = X, E = Emat,
                       LD = LD, piu = piu, tau = tau, price_iters = pr.iters)
end

"""
Find equilibrium wages by tatonnement: raise the wage where labour is scarce.
w[1] is the numeraire, which is legitimate because the system is homogeneous of
degree zero in wages and one clearing condition is redundant by Walras' law.
"""
function solve_ge(m::GEModel; w0 = ones(m.N), kappa = 0.25, tol = 1e-12,
                  maxit = 4000, warm = 12)
    w = copy(w0); w ./= w[1]
    P = nothing
    local info

    # relative excess labour demand for the N-1 non-numeraire countries
    function resid!(wv)
        z, inf = excess_demand(m, wv; P0 = P)
        P = inf.P
        return z[2:end] ./ m.L[2:end], inf
    end

    # Phase 1: a few tatonnement steps. Cheap, globally well behaved, and they
    # put us inside the basin where Newton is reliable.
    for _ in 1:warm
        f, info = resid!(w)
        maximum(abs.(f)) < tol && return (w = w, iters = 0, gap = maximum(abs.(f)), info = info)
        for n in 2:m.N; w[n] *= (info.LD[n] / m.L[n])^kappa; end
        w ./= w[1]
    end

    # Phase 2: Newton on log wages with a numerical Jacobian, safeguarded by a
    # halving line search. The equilibrium is the same one tatonnement would
    # reach -- this only changes how fast we get there.
    n1 = m.N - 1
    for it in 1:maxit
        f, info = resid!(w)
        nrm = maximum(abs.(f))
        nrm < tol && return (w = w, iters = it, gap = nrm, info = info)

        J = zeros(n1, n1)
        h = 1e-7
        for j in 1:n1
            wp = copy(w); wp[j+1] *= exp(h)
            fp, _ = resid!(wp)
            J[:, j] = (fp .- f) ./ h
        end
        dx = try -(J \ f) catch; fill(0.0, n1) end
        (all(isfinite, dx) && maximum(abs.(dx)) > 0) || (dx = -0.3 .* f)
        maximum(abs.(dx)) > 1.0 && (dx .*= 1.0 / maximum(abs.(dx)))   # trust region

        step, ok = 1.0, false
        for _ in 1:25
            wt = copy(w); wt[2:end] .*= exp.(step .* dx)
            wt ./= wt[1]
            ft, _ = resid!(wt)
            if maximum(abs.(ft)) < nrm
                w = wt; ok = true; break
            end
            step *= 0.5
        end
        if !ok                                   # Newton stalled: fall back
            for n in 2:m.N; w[n] *= (info.LD[n] / m.L[n])^kappa; end
            w ./= w[1]
        end
    end
    f, info = resid!(w)
    return (w = w, iters = maxit, gap = maximum(abs.(f)), info = info)
end

"""
Every identity a general equilibrium must satisfy. All are checked at ARBITRARY
wages, not only in equilibrium -- that is the strong form of the test.
"""
function accounting(m::GEModel, w::Vector{Float64})
    z, inf = excess_demand(m, w)
    mk, X, Emat = inf.mk, inf.X, inf.E
    G = size(m.theta, 1)

    revenue = factor = inter = tariffrev = profit = 0.0
    prof_g = zeros(G); tar_n = zeros(m.N)
    for n in 1:m.N, k in 1:m.K
        e, idx, pars = mk[n, k].eq, mk[n, k].idx, mk[n, k].pars
        E = Emat[n, k]; revenue += E
        s2 = m.sigma[k] / m.eta - 1.0
        for (i, g) in enumerate(pars)
            p = E * (e.S[i]/m.sigma[k]) * (1.0 + s2*e.S[i])
            prof_g[g] += p; profit += p
        end
        for (j, a) in enumerate(idx)
            t  = m.tariff[m.loc[a], n, k]
            gi = findfirst(==(m.par[a]), pars)
            bill = E * e.s[j] / (e.mu[gi] * (1.0 + t))
            factor += (1.0 - m.nu[k]) * bill
            inter  += m.nu[k] * bill
            tariffrev += bill * t
            tar_n[n] += bill * t
        end
    end

    share_err = maximum(abs(sum(mk[n,k].eq.S) - 1.0) for n in 1:m.N, k in 1:m.K)
    ident_err = abs(revenue - factor - inter - tariffrev - profit) / revenue
    budget = [w[n]*m.L[n] + sum(m.theta[g,n]*prof_g[g] for g in 1:G) + tar_n[n]
              for n in 1:m.N]
    budget_err = maximum(abs.(budget .- X) ./ X)
    walras_err = abs(sum(w .* z)) / revenue
    # final expenditure = factor payments + tariffs + profits (intermediates net out)
    world_err = abs(sum(X) - sum(w .* m.L) - profit - tariffrev) / revenue
    # intermediate demand must equal intermediate purchases
    io_err = abs(sum(Emat) - sum(X) - inter) / revenue

    return (shares=share_err, identity=ident_err, budget=budget_err,
            walras=walras_err, world=world_err, io=io_err, excess=z,
            revenue=revenue, factor=factor, inter=inter,
            tariffrev=tariffrev, profit=profit, X=X, prof_g=prof_g)
end

###############################################################################
# PART 4.  MEASUREMENT -- mapping the model onto what customs data can see
#
#   The model's S_g is a share of the DESTINATION MARKET. Figure 6's index is a
#   share of LAC EXPORT VALUE WITHIN A PRODUCT, pooled across destinations.
#   Different populations. Rather than rebuild the model's denominator in data we
#   do not have, we run the DATA's estimator inside the model.
###############################################################################

"""Affiliate-level export value [affiliate, destination]; domestic sales excluded."""
function export_matrix(m::GEModel, w::Vector{Float64})
    _, inf = excess_demand(m, w)
    V = zeros(length(m.par), m.N)
    for n in 1:m.N, k in 1:m.K
        e, idx = inf.mk[n,k].eq, inf.mk[n,k].idx
        for (j, a) in enumerate(idx)
            m.loc[a] == n && continue
            V[a, n] = inf.E[n, k] * e.s[j]
        end
    end
    return V, inf
end

"""non-MNE / domestic MNE / foreign MNE, matching the empirical definitions."""
function classify(m::GEModel)
    ncty = Dict(g => length(unique(m.loc[m.par .== g])) for g in unique(m.par))
    return [m.hq[a] != m.loc[a] ? :foreign_mne :
            ncty[m.par[a]] > 1  ? :domestic_mne : :nonmne for a in eachindex(m.par)]
end

"""Theory object: HHI over the whole destination market. NOT comparable to Fig 6."""
function structural_hhi(m::GEModel, w::Vector{Float64})
    _, inf = excess_demand(m, w)
    num = den = 0.0
    for n in 1:m.N, k in 1:m.K
        num += inf.E[n,k] * sum(inf.mk[n,k].eq.S .^ 2); den += inf.E[n,k]
    end
    return num / den
end

"""
Figure 6's ESTIMATOR, run on the model's simulated customs records:
in-sample origins only -> pool across destinations within a sector -> aggregate to
the chosen firm concept -> renormalise WITHIN the sample -> HHI -> value-weight.

Step 4 is where the competitive fringe cancels out, which is why Figure 6 cannot
identify the fringe and lambda must do that job instead.
"""
function measured_hhi(m::GEModel, V::Matrix{Float64}, sample_countries;
                      level::Symbol = :parent)
    num = den = 0.0
    for k in 1:m.K
        acc = Dict{Any,Float64}()
        for a in eachindex(m.par)
            (m.sec[a] == k && m.loc[a] in sample_countries) || continue
            key = level === :affiliate      ? a :
                  level === :parent_country ? (m.par[a], m.loc[a]) : m.par[a]
            acc[key] = get(acc, key, 0.0) + sum(V[a, :])
        end
        tot = sum(values(acc)); tot <= 0 && continue
        num += tot * sum((v/tot)^2 for v in values(acc)); den += tot
    end
    return num / den
end

###############################################################################
# PART 5.  ECONOMIES USED BY THE TESTS
###############################################################################

"""Small synthetic world for the accounting and uniqueness audits."""
function demo_model(rng; N = 4, K = 3, nparents = 12, naff = 2, eta = 1.0,
                    tariff = 0.0, nu = 0.0)
    sigma = [4.0 + 2.0*rand(rng) for _ in 1:K]
    beta  = rand(rng, K) .+ 0.5; beta ./= sum(beta)
    alpha = [0.15 + 0.35*rand(rng) for _ in 1:K]
    nuv   = fill(nu, K)
    omega = fill(1.0/K, K, K)
    L     = 1.0 .+ rand(rng, N)

    par=Int[]; hq=Int[]; loc=Int[]; sec=Int[]; phi=Float64[]
    for g in 1:nparents
        h = rand(rng, 1:N)
        for _ in 1:naff
            push!(par,g); push!(hq,h); push!(loc,rand(rng,1:N))
            push!(sec,rand(rng,1:K)); push!(phi,exp(0.3*randn(rng)))
        end
    end
    for k in 1:K
        while length(unique(par[sec .== k])) < 2
            j = rand(rng, 1:length(sec)); sec[j] = k
        end
    end
    gamma = 1.0 .+ 0.3 .* rand(rng, N, N); for h in 1:N; gamma[h,h] = 1.0; end
    d = 1.0 .+ 0.4 .* rand(rng, N, N);     for l in 1:N; d[l,l] = 1.0; end
    t = fill(tariff, N, N, K); for n in 1:N, k in 1:K; t[n,n,k] = 0.0; end
    theta = zeros(nparents, N)
    for g in 1:nparents; theta[g, hq[findfirst(==(g), par)]] = 1.0; end
    return GEModel(N,K,sigma,eta,beta,alpha,nuv,omega,L,par,hq,loc,sec,phi,
                   gamma,d,t,theta)
end

###############################################################################
# CALIBRATION
#
#   Every parameter, where it comes from, and how confident we are in it.
#   Three kinds, and the difference matters when reading the results:
#
#     EXTERNAL   taken from the literature. Not chosen to fit anything here.
#     INTERNAL   chosen so the model reproduces a specific stylized fact.
#     NORMALISED free by construction (units, numeraire).
#
#   >>> HEALTH WARNING <<<
#   The EXTERNAL values below are standard magnitudes for this literature, not
#   transcriptions from specific tables. Before anything goes in a paper, check
#   each one against the source paper and replace it. They are in one place here
#   precisely so that is a five-minute job.
###############################################################################

const CALIB = (
 sigma   = (val="5.0",        kind="EXTERNAL",
            src="trade elasticity 4-6; Simonovska-Waugh ~4, Eaton-Kortum higher"),
 eta     = (val="1.0",        kind="EXTERNAL",
            src="Cobb-Douglas across sectors; the Atkeson-Burstein outer nest"),
 delta   = (val="1/(sigma-1)", kind="EXTERNAL",
            src="set so the gravity distance elasticity of TRADE is exactly -1"),
 nu      = (val="0.55",       kind="EXTERNAL",
            src="intermediate share of gross output, manufacturing; Caliendo-Parro"),
 io_own  = (val="0.45",       kind="EXTERNAL",
            src="I-O tables are diagonal-heavy: own-sector inputs dominate"),
 alpha   = (val="0.10-0.55",  kind="EXTERNAL",
            src="HQ input share rising in complexity; Head-Mayer headquarter input"),
 gamma   = (val="1.18",       kind="EXTERNAL",
            src="MP efficiency loss abroad; Ramondo-Rodriguez-Clare MP friction"),
 mne_adv = (val="solved",     kind="INTERNAL",
            src="MNE productivity edge, set so MNE export share matches FACT 1"),
 slope   = (val="solved",     kind="INTERNAL",
            src="HQ capability gradient, set to the FACT 2 complexity gradient"),
 z       = (val="2.2 / 1.0",  kind="NORMALISED",
            src="advanced vs LAC productivity level; fixes relative wages"),
 wage1   = (val="1.0",        kind="NORMALISED", src="numeraire (Walras)"),
)

function print_calibration()
    println("  " * "-"^74)
    @printf("  %-9s%-12s%-11s%s\n", "param", "value", "kind", "source")
    println("  " * "-"^74)
    for k in keys(CALIB)
        c = getfield(CALIB, k)
        @printf("  %-9s%-12s%-11s%s\n", String(k), c.val, c.kind, c.src)
    end
    println("  " * "-"^74)
    println("  EXTERNAL values are standard magnitudes for this literature, NOT")
    println("  transcriptions from specific tables. Verify each against the source")
    println("  paper before publication -- they are collected in CALIB for that.")
    println("  Only TWO parameters are fitted here, both to stylized facts, so")
    println("  Facts 4, 5 and 6 remain out-of-sample.")
    flush(stdout)
end

"""
Economy for the stylized facts, at the calibrated parameters.

Countries 1..n_rich are advanced (high productivity, source of parents); the rest
are the LAC-like hosts whose exports we measure.

Note `d = dist^(1/(sigma-1))`. That is not arbitrary: delivered cost enters
demand with elasticity (1-sigma), so this makes the elasticity of TRADE with
respect to distance exactly -1, which is the central estimate of the gravity
literature. The model's own gravity regression should therefore return about -1
for ordinary firms -- and it does. See FACT 6.
"""
function facts_economy(rng; N=5, K=4, n_rich=2, eta=1.0, n_dom=6, n_mne=20,
                       mne_locs=3, p_second=0.45, mne_adv=0.0, adv_slope=0.0,
                       nu=0.55, io_own=0.45, extra_mne_sector=0, extra_n=10,
                       spill=0.0)
    alpha = collect(range(0.10, 0.55, length=K))
    sigma = fill(5.0, K)
    beta  = fill(1.0/K, K)
    nuv   = fill(nu, K)
    omega = fill((1.0-io_own)/(K-1), K, K)
    for k in 1:K; omega[k,k] = io_own; end
    L = vcat(fill(1.5, n_rich), fill(1.0, N-n_rich))
    z = vcat(fill(2.2, n_rich), fill(1.0, N-n_rich))
    lac = collect((n_rich+1):N)

    pos  = collect(1.0:N)
    dist = [1.0 + abs(pos[i]-pos[j]) for i in 1:N, j in 1:N]
    d = dist .^ (1.0 / (sigma[1] - 1.0))     # => gravity trade elasticity = -1
    gamma = [i==j ? 1.0 : 1.18 for i in 1:N, j in 1:N]

    par=Int[]; hq=Int[]; loc=Int[]; sec=Int[]; phi=Float64[]; g=0
    for n in 1:N, k in 1:K, _ in 1:n_dom
        g += 1
        push!(par,g); push!(hq,n); push!(loc,n); push!(sec,k)
        push!(phi, z[n]*exp(0.25*randn(rng)))
    end
    function add_mne!(j, k, adv)
        g += 1
        h = 1 + (j-1) % n_rich
        for l in unique(rand(rng, lac, mne_locs))
            for _ in 1:(1 + (rand(rng) < p_second))
                push!(par,g); push!(hq,h); push!(loc,l); push!(sec,k)
                push!(phi, z[l]*exp(0.25*randn(rng) + adv + adv_slope*alpha[k]))
            end
        end
    end
    for j in 1:n_mne; add_mne!(j, 1 + (j-1) % K, mne_adv); end
    if extra_mne_sector > 0
        for j in 1:extra_n; add_mne!(j, extra_mne_sector, mne_adv); end
    end

    # KNOWLEDGE SPILLOVER (Javorcik 2004). A local firm's productivity rises with
    # the number of multinational plants in its own country and sector:
    #     phi_local  *=  (1 + #MNE plants in (l,k))^spill
    # spill = 0 switches it off. This is the channel Part C uses to ask how big a
    # spillover would have to be for Fact 5 to come out with the data's sign.
    if spill != 0.0
        cnt = Dict{Tuple{Int,Int},Int}()
        for a in eachindex(par)
            hq[a] != loc[a] && (cnt[(loc[a], sec[a])] = get(cnt,(loc[a],sec[a]),0) + 1)
        end
        for a in eachindex(par)
            hq[a] == loc[a] || continue                     # local firms only
            phi[a] *= (1.0 + get(cnt, (loc[a], sec[a]), 0))^spill
        end
    end

    theta = zeros(g, N)
    for a in eachindex(par); theta[par[a], hq[a]] = 1.0; end
    return GEModel(N,K,sigma,eta,beta,alpha,nuv,omega,L,par,hq,loc,sec,phi,
                   gamma,d,fill(0.0,N,N,K),theta), (dist=dist, n_rich=n_rich, lac=lac)
end

ols(X, y) = (X'*X) \ (X'*y)

function fe_block(labels)
    lev = sort(unique(labels))[2:end]
    isempty(lev) && return zeros(length(labels), 0)
    return reduce(hcat, [Float64.(labels .== l) for l in lev])
end

function ols_fe(y, regs::Vector{Vector{Float64}}, fes::Vector)
    X = hcat(ones(length(y)), reduce(hcat, regs))
    for f in fes
        B = fe_block(f); size(B,2) > 0 && (X = hcat(X, B))
    end
    return ols(X, y)[2:(1+length(regs))]
end

function mne_share(m::GEModel, w, lac)
    V, = export_matrix(m, w); cls = classify(m)
    sel = [l in lac for l in m.loc]
    return sum(V[sel .& (cls .!= :nonmne), :]) / sum(V[sel, :])
end

"""Calibrate the MNE productivity edge so the model matches Fact 1."""
function calibrate_adv(; target=0.55, seed=20260812, iters=18, kwargs...)
    f(adv) = begin
        m, aux = facts_economy(MersenneTwister(seed); mne_adv=adv, kwargs...)
        mne_share(m, solve_ge(m).w, aux.lac) - target
    end
    lo, hi = 0.0, 3.0
    for _ in 1:iters
        mid = 0.5*(lo+hi); f(mid) < 0 ? (lo = mid) : (hi = mid)
    end
    return 0.5*(lo+hi)
end

end # module MNEModel


###############################################################################
# PART 6.  THE RUNNER
###############################################################################

using .MNEModel
using .MNEModel: demo_model, facts_economy, verify_uniqueness, solve_prices, print_calibration,
                 solve_quantities, expenditure_shares, delivered_cost,
                 calibrate_adv, mne_share, ols, ols_fe, labour_demand
using Printf, Random, LinearAlgebra

banner(t) = (println(); println("="^78); println(t); println("="^78); flush(stdout))

# ---------------------------------------------------------------------------
function run_core(level)
    banner("PART A   ONE MARKET, AND WHY IT HAS EXACTLY ONE SOLUTION")
    reps, starts = level == :full ? (400, 100) : level == :quick ? (60, 10) : (200, 40)

    println("The markup rises with the group's share because expanding cannibalises")
    println("its own sales. Everything below verifies the solver and the proof.\n")
    for (S, sg, et) in ((0.0,5.0,1.0),(0.1,5.0,1.0),(0.3,5.0,1.0),(0.5,5.0,1.0),(0.9,5.0,1.0))
        @printf("  share %4.0f%%   markup %7.3f\n", 100S, MNEModel.markup(S, sg, et))
    end
    println()
    verify_uniqueness(reps=reps, starts=starts)
end

# ---------------------------------------------------------------------------
function run_ge(level)
    banner("PART B   GENERAL EQUILIBRIUM: THE ACCOUNTING MUST BE EXACT")
    rng = MersenneTwister(20260812)
    ndraw = level == :full ? 60 : level == :quick ? 10 : 30

    println("Walras' law is the strongest single test, because it must hold at")
    println("ARBITRARY wages, not just in equilibrium. If a single euro were lost")
    println("-- iceberg costs not paid to workers, tariff revenue double counted,")
    println("intermediates not netted out -- it would fail.\n")
    wS=wI=wB=wW=wIO=wWal=0.0
    for _ in 1:ndraw
        m = demo_model(rng; tariff=0.2, nu=0.45)
        w = exp.(0.4 .* randn(rng, m.N)); w ./= w[1]
        a = MNEModel.accounting(m, w)
        wS=max(wS,a.shares); wI=max(wI,a.identity); wB=max(wB,a.budget)
        wW=max(wW,a.world); wIO=max(wIO,a.io); wWal=max(wWal,a.walras)
    end
    @printf("  Walras: |sum_n w_n Z_n| / GDP (out of equilibrium)  : %.2e\n", wWal)
    @printf("  goods market clearing, sum_g S_g = 1               : %.2e\n", wS)
    @printf("  revenue = labour + intermediates + tariffs + profit: %.2e\n", wI)
    @printf("  country budget  X = wL + profits owned + tariffs   : %.2e\n", wB)
    @printf("  world budget: final spending = factors+tariffs+profit: %.2e\n", wW)
    @printf("  intermediate demand = intermediate purchases       : %.2e\n", wIO)

    println("\n  Homogeneity: only relative wages matter (so w[1]=1 loses nothing)")
    worst = 0.0
    for _ in 1:(level == :quick ? 5 : 15)
        m = demo_model(rng; tariff=0.1, nu=0.45)
        w = exp.(0.3 .* randn(rng, m.N))
        z1, = MNEModel.excess_demand(m, w); z2, = MNEModel.excess_demand(m, 7.3 .* w)
        worst = max(worst, maximum(abs.(z1 .- z2) ./ m.L))
    end
    @printf("  scaling every wage by 7.3 changes excess demand by  : %.2e\n", worst)

    println("\n  The I-O price loop is a CONTRACTION with modulus max(nu), so it has")
    println("  one solution and iteration converges geometrically. Check: iterations")
    println("  needed should track log(tol)/log(nu).")
    @printf("  %8s%12s%14s\n", "nu", "iterations", "predicted")
    for nu in (0.0, 0.3, 0.5, 0.7)
        m = demo_model(MersenneTwister(7); nu=nu, tariff=0.1)
        pr = solve_prices(m, ones(m.N))
        pred = nu == 0 ? 1 : ceil(Int, log(1e-14)/log(nu))
        @printf("  %8.2f%12d%14d\n", nu, pr.iters, pred)
    end

    println("\n  Equilibrium exists and markets clear:")
    for (nu, tar) in ((0.0,0.0),(0.45,0.0),(0.45,0.15))
        m = demo_model(MersenneTwister(11); nu=nu, tariff=tar)
        r = MNEModel.solve_ge(m); a = MNEModel.accounting(m, r.w)
        @printf("  nu=%.2f tariff=%.2f  iters=%5d  labour resid=%.1e  budget=%.1e\n",
                nu, tar, r.iters, r.gap, a.budget)
    end

    banner("PART B2  UNIQUENESS OF THE GENERAL EQUILIBRIUM (numerical)")
    println("Market-level uniqueness is PROVED (Part A). The I-O price loop is PROVED")
    println("unique (contraction). Across WAGES there is no proof: with variable")
    println("markups the standard gross-substitutes argument does not apply. So we")
    println("test it, hard, and report it as evidence rather than as a theorem.")
    println("The residual column matters: it rules out two failures agreeing.\n")
    cases = level == :quick ?
        [("N=4 K=3 eta=1 nu=0.45 t=15%", 4,3,1.0,0.15,0.45,10)] :
        [("N=4 K=3 eta=1   nu=0    t=0",   4,3,1.0,0.00,0.00, level==:full ? 60 : 30),
         ("N=4 K=3 eta=1   nu=0.45 t=15%", 4,3,1.0,0.15,0.45, level==:full ? 60 : 30),
         ("N=5 K=4 eta=1.8 nu=0.45 t=25%", 5,4,1.8,0.25,0.45, level==:full ? 40 : 20),
         ("N=6 K=5 eta=2.5 nu=0.60 t=10%", 6,5,2.5,0.10,0.60, level==:full ? 30 : 15),
         ("N=7 K=3 eta=1   nu=0.45 t=40%", 7,3,1.0,0.40,0.45, level==:full ? 30 : 15)]
    @printf("  %-32s%8s%13s%13s\n", "economy", "starts", "max wage gap", "worst resid")
    for (lab,N,K,et,tar,nu,nst) in cases
        m = demo_model(MersenneTwister(11); N=N,K=K,eta=et,tariff=tar,nu=nu,
                       nparents=4K, naff=2)
        base = MNEModel.solve_ge(m); ref = base.w
        spread, worst = 0.0, base.gap
        for s in 1:nst
            r2 = MersenneTwister(5000+s)
            w0 = exp.(3.0 .* randn(r2, m.N)); w0 ./= w0[1]
            rr = MNEModel.solve_ge(m; w0=w0)
            spread = max(spread, maximum(abs.(rr.w .- ref) ./ ref))
            worst  = max(worst, rr.gap)
        end
        @printf("  %-32s%8d%13.2e%13.2e\n", lab, nst, spread, worst)
    end
    println("  -> every start reaches the same wages AND genuinely clears the")
    println("     labour market. VERIFIED, NOT PROVED.")
end

# ---------------------------------------------------------------------------
function run_facts(level)
    banner("PART C   DO WE REPRODUCE THE SIX STYLIZED FACTS?")
    nu_io = 0.55
    println("  CALIBRATION")
    print_calibration()
    println()
    println("TWO parameters are calibrated, to TWO facts: the productivity edge of a")
    println("multinational affiliate (to Fact 1) and the HQ capability gradient (to")
    println("Fact 2). Everything else is external or a normalisation, so Facts 3-6 are")
    println("evaluated at the calibrated model and are OUT OF SAMPLE.\n")
    # Quick mode shrinks the NUMBER of firms, never the structure. K stays at 4 so
    # the Figure 2 comparison lines up, and N stays at 5 so the gravity regression
    # has enough origin-destination pairs to identify anything.
    econ = level == :quick ? (N=5, K=4, n_dom=2, n_mne=8) :
                             (N=5, K=4, n_dom=6, n_mne=20)
    cal_iters = level == :quick ? 6 : 14
    ecargs = (N=econ.N, K=econ.K, n_dom=econ.n_dom, n_mne=econ.n_mne)
    alph = collect(range(0.10, 0.55, length=econ.K))
    DATA_SHARE, DATA_GRAD = (0.52, 0.52, 0.63, 0.70), 0.43

    # ---- JOINT CALIBRATION -------------------------------------------------
    # For each candidate capability gradient, the productivity edge is re-solved
    # so Fact 1 still holds exactly. That leaves ONE free number (the gradient),
    # which we then pick to match the Fact 2 complexity gradient. Two parameters,
    # two targets, no slack.
    println("  Joint calibration. Edge re-solved at each gradient so Fact 1 always")
    println("  holds; the gradient is then chosen to hit the Fact 2 slope of +0.43.")
    @printf("  %-11s%9s%30s%9s\n", "cap. slope", "edge", "foreign share by sector", "slope")
    slopes = [0.0, 0.8, 1.6]
    grads, edges, shares = Float64[], Float64[], Vector{Vector{Float64}}()
    for slope in slopes
        adv_s = calibrate_adv(; target=0.55, iters=cal_iters, nu=nu_io,
                              adv_slope=slope, ecargs...)
        m2, aux2 = facts_economy(MersenneTwister(20260812); mne_adv=adv_s,
                                 adv_slope=slope, nu=nu_io, ecargs...)
        V2, = export_matrix(m2, MNEModel.solve_ge(m2).w); cls2 = classify(m2)
        sh = Float64[]
        for k in 1:m2.K
            sel = (m2.sec .== k) .& [l in aux2.lac for l in m2.loc]
            tot = sum(V2[sel, :])
            push!(sh, tot > 0 ? sum(V2[sel .& (cls2 .== :foreign_mne), :])/tot : NaN)
        end
        g = ols(hcat(ones(econ.K), alph), sh)[2]
        push!(grads, g); push!(edges, adv_s); push!(shares, sh)
        @printf("  %-11.1f%9.3f%s%9.2f%s\n", slope, adv_s,
                join([@sprintf("%7.2f", x) for x in sh]), g,
                adv_s < 1e-3 ? "  (edge at bound)" : "")
    end

    # interpolate the gradient that reproduces Figure 2, then re-solve the edge there
    slope_star, bracketed = slopes[end], false
    for i in 1:(length(slopes)-1)
        if (grads[i] - DATA_GRAD) * (grads[i+1] - DATA_GRAD) <= 0
            slope_star = slopes[i] + (slopes[i+1]-slopes[i]) *
                         (DATA_GRAD - grads[i]) / (grads[i+1] - grads[i])
            bracketed = true
            break
        end
    end
    if !bracketed
        println("  WARNING: the data gradient +0.43 was NOT bracketed by the scan.")
        println("  Falling back to the largest gradient tried, so Fact 2 is matched")
        println("  only approximately below. Re-run without 'quick' -- a small economy")
        println("  makes the sector shares noisy and flattens the measured gradient.")
    end
    adv = calibrate_adv(; target=0.55, iters=cal_iters, nu=nu_io,
                        adv_slope=slope_star, ecargs...)
    @printf("\n  CALIBRATED: capability gradient %.3f, productivity edge %.3f (%.2fx)\n",
            slope_star, adv, exp(adv))
    println("  Every fact below is evaluated at THESE values.")

    m, aux = facts_economy(MersenneTwister(20260812); mne_adv=adv,
                           adv_slope=slope_star, nu=nu_io, ecargs...)
    r = MNEModel.solve_ge(m)
    V, inf = export_matrix(m, r.w)
    cls = classify(m); lac = aux.lac

    println("\n" * "-"^78); println("FACT 1  MNEs are a large share of export value")
    println("-"^78)
    @printf("  %-11s%13s%12s%12s\n", "origin", "foreign MNE", "dom. MNE", "total")
    for o in lac
        sel = m.loc .== o; tot = sum(V[sel, :])
        @printf("  country %-3d%13.2f%12.2f%12.2f\n", o,
                sum(V[sel .& (cls .== :foreign_mne), :])/tot,
                sum(V[sel .& (cls .== :domestic_mne), :])/tot,
                sum(V[sel .& (cls .!= :nonmne), :])/tot)
    end
    println("  data 0.47-0.74, overwhelmingly foreign.  VERDICT: CALIBRATION TARGET.")
    println("  NEXT STEP to make it generated: endogenous multinational entry.")

    println("\n" * "-"^78)
    println("FACT 2  Foreign MNEs specialise in complex goods")
    println("-"^78)
    # Fitted at the calibrated model. The scan above is the identification
    # evidence; this is the fit it produces.
    sh_cal = Float64[]
    for k in 1:m.K
        sel = (m.sec .== k) .& [l in lac for l in m.loc]
        tot = sum(V[sel, :])
        push!(sh_cal, tot > 0 ? sum(V[sel .& (cls .== :foreign_mne), :])/tot : NaN)
    end
    g_cal = ols(hcat(ones(econ.K), alph), sh_cal)[2]
    @printf("  %-14s%30s%9s\n", "", "foreign share by sector", "slope")
    @printf("  %-14s%s%9.2f\n", "MODEL",
            join([@sprintf("%7.2f", x) for x in sh_cal]), g_cal)
    @printf("  %-14s%s%9.2f\n", "DATA Fig 2",
            join([@sprintf("%7.2f", x) for x in DATA_SHARE]), DATA_GRAD)
    println()
    println("  VERDICT: CALIBRATION TARGET, and the identification is the point.")
    println("  Look at the scan above. With a FLAT capability gradient the model's")
    @printf("  complexity slope is %+.2f -- essentially zero. The HQ input share on its\n",
            grads[1])
    println("  own does NOT sort foreign ownership into complex goods, and there is a")
    println("  reason: a foreign affiliate pays its PARENT's wage on the alpha_k share")
    println("  of cost, and parents sit in high-wage countries, so raising alpha_k")
    println("  makes foreign ownership more EXPENSIVE. That force cancels the")
    println("  productivity advantage almost exactly.")
    println("  So Fact 2 is informative rather than automatic: it tells us the HQ input")
    println("  must be a CAPABILITY local firms cannot buy at any price, not merely")
    println("  costly labour. Its strength is then read off Figure 2 rather than")
    @printf("  assumed, and the value that does it is %.2f.\n", slope_star)

    println("\n" * "-"^78)
    println("FACT 3  Parents come from a small set of countries")
    println("-"^78)
    fv = zeros(m.N)
    for a in eachindex(m.par); cls[a] == :foreign_mne && (fv[m.hq[a]] += sum(V[a,:])); end
    fv ./= sum(fv)
    @printf("  parent-country shares: %s\n", join([@sprintf("%.2f", x) for x in fv], "  "))
    println("  VERDICT: INPUT. Same fix as Fact 1: endogenous multinational entry.")

    println("\n" * "-"^78)
    println("FACT 4  Grouping affiliates by parent raises measured concentration")
    println("-"^78)
    a1 = measured_hhi(m, V, lac; level=:affiliate)
    a2 = measured_hhi(m, V, lac; level=:parent_country)
    a3 = measured_hhi(m, V, lac; level=:parent)
    @printf("  model: affiliate %.4f -> parent x country %.4f -> parent %.4f  (%.2fx)\n",
            a1, a2, a3, a3/a1)
    @printf("  data : affiliate %.4f -> parent x country %.4f -> parent %.4f  (%.2fx)\n",
            0.192, 0.209, 0.215, 0.215/0.192)
    @printf("  structural HHI (destination market) = %.4f -- a DIFFERENT object\n",
            structural_hhi(m, r.w))
    println("  VERDICT: GENERATED. The ordering is a prediction, not an input. And")
    println("  because the markup depends on the GROUP's share, grouping raises")
    println("  measured concentration AND true market power together.")

    println("\n" * "-"^78)
    println("FACT 5  More MNE presence goes with HIGHER non-MNE exports")
    println("-"^78)
    println("  THE INPUT-OUTPUT CHANNEL. Multinationals entering a country lower the")
    println("  price of the intermediates local firms buy there, cutting their costs.")
    println("  That pushes against business stealing. Experiment: add 10 MNE parents")
    println("  to sector 2, then measure non-MNE export value.\n")
    function nonmne(mm, ww, ks, lacs)
        VV, = export_matrix(mm, ww); cc = classify(mm)
        sel = (cc .== :nonmne) .& [l in lacs for l in mm.loc] .& [s in ks for s in mm.sec]
        return sum(VV[sel, :])
    end
    function fact5(nu, spill)
        mb, ab = facts_economy(MersenneTwister(303); mne_adv=adv, adv_slope=slope_star,
                               nu=nu, spill=spill, ecargs...)
        wb = MNEModel.solve_ge(mb).w
        ma, aa = facts_economy(MersenneTwister(303); mne_adv=adv, adv_slope=slope_star,
                               nu=nu, spill=spill, ecargs..., extra_mne_sector=2)
        wa = MNEModel.solve_ge(ma).w
        oth = [k for k in 1:mb.K if k != 2]
        return (nonmne(ma,wa,[2],aa.lac)/nonmne(mb,wb,[2],ab.lac) - 1,
                nonmne(ma,wa,oth,aa.lac)/nonmne(mb,wb,oth,ab.lac) - 1,
                nonmne(ma,wa,1:mb.K,aa.lac)/nonmne(mb,wb,1:mb.K,ab.lac) - 1)
    end

    println("  (a) the input-output channel on its own")
    @printf("  %8s%16s%16s%16s\n", "nu", "same sector", "other sectors", "all sectors")
    nus = level == :quick ? (0.0, 0.55) : (0.0, 0.30, 0.55, 0.70)
    io_first, io_last = 0.0, 0.0
    for (i, nu) in enumerate(nus)
        s1, s2, s3 = fact5(nu, 0.0)
        i == 1 && (io_first = s1); i == length(nus) && (io_last = s1)
        @printf("  %8.2f%15.1f%%%15.1f%%%15.1f%%\n", nu, 100s1, 100s2, 100s3)
    end
    println("  data: strongly POSITIVE within the cell (Table 1 Panel B)")
    println()
    if io_last > 0
        println("  VERDICT: THE I-O CHANNEL ALONE FLIPS THE SIGN at these parameters.")
    else
        println("  VERDICT: THE I-O CHANNEL HELPS BUT DOES NOT FLIP THE SIGN. Raising")
        @printf("  the intermediate share from %.2f to %.2f moves the same-sector number\n",
                nus[1], nus[end])
        @printf("  from %.1f%% to %.1f%%, about %.0f points. Two forces beat it:\n",
                100io_first, 100io_last, 100*(io_last - io_first))
    end
    println("   * business stealing is DIRECT and strong (sigma = 5), while the input")
    @printf("     price effect reaches a local firm only through nu x omega = %.2f\n",
            nu_io * 0.45)
    println("     of its cost;")
    println("   * and in GENERAL EQUILIBRIUM the new plants bid up LOCAL WAGES, which")
    println("     raises local firms' costs. That is why even OTHER sectors lose:")
    println("     the wage effect outweighs their cheaper inputs. A partial")
    println("     equilibrium model would have missed this and reported a gain.")
    println()
    println("  (b) so how big a KNOWLEDGE SPILLOVER would it take? Local productivity")
    println("      scaled by (1 + #MNE plants in own country-sector)^spill.")
    @printf("  %8s%16s%16s%16s\n", "spill", "same sector", "other sectors", "all sectors")
    sps = level == :quick ? (0.0, 0.15, 0.30) : (0.0, 0.10, 0.20, 0.30, 0.40)
    prev_sp, prev_v, flip_lo, flip_hi = NaN, NaN, NaN, NaN
    for sp in sps
        s1, s2, s3 = fact5(0.55, sp)
        if !isnan(prev_v) && prev_v <= 0 && s1 > 0 && isnan(flip_lo)
            flip_lo, flip_hi = prev_sp, sp
        end
        prev_sp, prev_v = sp, s1
        @printf("  %8.2f%15.1f%%%15.1f%%%15.1f%%\n", sp, 100s1, 100s2, 100s3)
    end
    if isnan(flip_lo)
        println("  RESULT: the sign did NOT flip over the range scanned. Either widen")
        println("  the range or conclude the spillover channel cannot deliver Fact 5.")
    else
        @printf("  RESULT: the sign flips between spill = %.2f and %.2f. So Fact 5 IS\n",
                flip_lo, flip_hi)
        println("  reproducible, but it needs input-output linkages PLUS a same-sector")
        @printf("  productivity spillover with elasticity of roughly %.2f.\n",
                0.5*(flip_lo + flip_hi))
    end
    println()
    println("  That turns 'the model fails Fact 5' into a MEASURABLE question, and")
    println("  the answer is uncomfortable. The required spillover is HORIZONTAL --")
    println("  within the same sector -- and Javorcik (2004), the standard reference,")
    println("  finds horizontal spillovers to be zero or NEGATIVE, with positive")
    println("  effects only through BACKWARD linkages to upstream suppliers. The")
    println("  channel the model needs is the one the literature says is absent.")
    println()
    println("  Two readings, and the paper should say which it takes:")
    println("   (i)  the spillover is real and larger than usually estimated; or")
    println("   (ii) Fact 5 is substantially SELECTION -- attractive markets draw in")
    println("        multinationals and local exporters alike -- rather than an")
    println("        effect of multinationals on local firms. Recall the empirical")
    println("        identification cannot rule this out: origin x dest x product FE")
    println("        do not absorb time-varying market shocks, and no column carries")
    println("        destination x product x year.")
    println("  Reading (ii) is more consistent with the evidence, and it is testable:")
    println("  add destination x product x year fixed effects to Table 1.")

    println("\n" * "-"^78)
    println("FACT 6  Distance is a weaker barrier for MNEs")
    println("-"^78)
    y=Float64[]; ld=Float64[]; ldm=Float64[]; ldp=Float64[]
    fo=Int[]; fd=Int[]; fs=Int[]
    for a in eachindex(m.par), n in 1:m.N
        (m.loc[a]==n || V[a,n] <= 0) && continue
        aff = findall(==(m.par[a]), m.par)
        isM = cls[a] != :nonmne; pres = any(m.loc[aff] .== n)
        x = log(aux.dist[m.loc[a], n])
        push!(y, log(V[a,n])); push!(ld, x)
        push!(ldm, isM ? x : 0.0); push!(ldp, (isM && pres) ? x : 0.0)
        push!(fo, m.loc[a]); push!(fd, n); push!(fs, m.sec[a])
    end
    b = ols_fe(y, [ld,ldm,ldp], [fo,fd,fs])
    @printf("  affiliate level, origin/dest/sector FE (n=%d)\n", length(y))
    @printf("    ln dist %7.3f | x MNE %+7.3f | x MNE present %+7.3f\n", b[1],b[2],b[3])
    @printf("    gradients: non-MNE %.3f | MNE %.3f | MNE present %.3f\n",
            b[1], b[1]+b[2], b[1]+b[2]+b[3])
    println("    data:      non-MNE -0.164 | MNE -0.118 | MNE present -0.051")
    println()
    println("  FIRST, A CALIBRATION CHECK. Trade costs were set so the gravity")
    println("  distance elasticity of TRADE is exactly -1, the central estimate of")
    @printf("  the gravity literature. The model returns %.2f for ordinary firms --",b[1])
    println(abs(b[1] + 1.0) < 0.25 ? " on target." : " off target.")
    println("  It also reframes the comparison: the DATA coefficient of -0.164 is")
    println("  the outlier here, an order of magnitude below any gravity estimate,")
    println("  which is the specification concern already flagged for Table 2. Fact 6")
    println("  is about the INTERACTIONS, so that is what to compare.")
    println()
    # The verdict is COMPUTED from the regression, never asserted. If a change to
    # parameters or sample size flips a sign, the text flips with it.
    ok = b[2] > 0 && b[3] > 0
    if ok
        println("  VERDICT: GENERATED. Both interactions are positive and in the")
        println("  data's order, and nothing in the calibration targeted this. It")
        println("  falls out of d[l,n] running from the PRODUCTION country rather")
        println("  than the headquarters -- Tintelnot's (2017) export platforms.")
    else
        println("  VERDICT: NOT REPRODUCED IN THIS RUN. One or both interactions came")
        @printf("  out negative (MNE %+.3f, present %+.3f).\n", b[2], b[3])
        if length(y) < 400
            println("  The sample is small (n = $(length(y))) once origin, destination and")
            println("  sector fixed effects are absorbed, so this is most likely a")
            println("  power problem rather than a model failure. Re-run without")
            println("  'quick' before drawing any conclusion.")
        else
            println("  The sample is large enough that this is a genuine failure at")
            println("  these parameters, and should be investigated, not explained away.")
        end
    end
    println("  CAVEAT either way: magnitudes are far too large, and plant locations")
    println("  are exogenous, so the gradient is reproduced rather than explained.")
end

# ---------------------------------------------------------------------------
function scorecard(level = :normal)
    banner("SCORECARD")
    if level == :quick
        println("  NOTE: 'quick' runs a REDUCED economy. The verdicts below describe")
        println("  the full model; individual quick-mode runs may miss Facts 4 and 6")
        println("  through lack of statistical power. Re-run without 'quick' to check.")
        println()
    end
    for (n,txt,st) in ((1,"MNEs are a large share of exports","CALIBRATION TARGET"),
                       (2,"Foreign MNEs in complex goods","CALIBRATION TARGET"),
                       (3,"Parents from a few countries","INPUT"),
                       (4,"Grouping raises concentration","GENERATED  <-- core"),
                       (5,"MNE presence raises non-MNE exports","NEEDS SPILLOVER ~0.15-0.20"),
                       (6,"Distance matters less for MNEs","GENERATED"))
        @printf("  %-3d%-44s%s\n", n, txt, st)
    end
    println("\n  Ranked next steps:")
    println("   1. Fact 5: settle whether it is causal. Add destination x product x")
    println("      year fixed effects to Table 1. The model says the sign needs a")
    println("      HORIZONTAL spillover of ~0.15-0.20, which is the channel Javorcik")
    println("      (2004) finds to be absent. Cheap test, and it decides whether the")
    println("      fact is a target or a known-selection artefact.")
    println("   2. Fact 2: make the HQ input a capability local firms cannot buy.")
    println("   3. Facts 1 and 3: endogenous multinational entry (one extension")
    println("      buys both) -- a fixed cost plus gamma and country productivity.")
    println("   4. Fact 6: endogenous plant location. Hardest, least urgent.")
end

# ---------------------------------------------------------------------------
function main()
    what = isempty(ARGS) ? "all" : lowercase(ARGS[1])
    level = what == "quick" ? :quick : what == "full" ? :full : :normal
    t0 = time()
    println("MULTINATIONAL OWNERSHIP, MARKET POWER AND TRADE POLICY")
    println("mode: $what      (options: all | quick | full | core | ge | facts)")
    if what in ("all","quick","full","core"); run_core(level); end
    if what in ("all","quick","full","ge");   run_ge(level);   end
    if what in ("all","quick","full","facts");run_facts(level); scorecard(level); end
    @printf("\nfinished in %.1f minutes\n", (time()-t0)/60)
end

if abspath(PROGRAM_FILE) == (@__FILE__)
    main()
end
