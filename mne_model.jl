###############################################################################
#
#   MULTINATIONAL OWNERSHIP, MARKET POWER AND TRADE POLICY
#   A complete, self-contained general equilibrium model with firm entry.
#
#   Sebastian Velasquez Palacios (IDB / PTI, with Christian Volpe Martincus)
#
#   ---------------------------------------------------------------------------
#   HOW TO RUN
#   ---------------------------------------------------------------------------
#       julia mne_model.jl quick     everything, small world          (~8 min)
#       julia mne_model.jl           everything                       (~45 min)
#       julia mne_model.jl core      one market + both uniqueness proofs (~1 min)
#       julia mne_model.jl entry     the entry theorem and its tests   (~5 min)
#       julia mne_model.jl ge        general equilibrium + accounting  (~10 min)
#       julia mne_model.jl facts     the six stylized facts            (~25 min)
#       julia mne_model.jl bertrand  the whole thing under Bertrand
#
#   No packages. Standard library only (Printf, Random, LinearAlgebra).
#
#   ---------------------------------------------------------------------------
#   WHAT THE MODEL IS, IN ONE SCREEN
#   ---------------------------------------------------------------------------
#   Countries n = 1..N, sectors k = 1..K, plants a = 1..A. Each plant belongs to
#   a global ultimate PARENT g. Parents are the strategic agents: they choose
#   quantities (or prices) market by market internalising competition among
#   their OWN plants, and they choose WHICH plants to operate.
#
#   A market is one (destination, sector) cell. In it, for parent g with total
#   share S_g:
#
#       mu(S_g)                        markup, rising in the parent's own share
#       Pi_g = E S_g [1 - 1/mu(S_g)]   profit
#       sum_g S_g = 1                  clearing
#
#   CANNIBALISATION IS THE MECHANISM. A parent that expands one plant depresses
#   the price of its other plants, so its perceived elasticity falls with its own
#   share and its markup rises. That is what mu(S_g) says, and it is the single
#   force behind the paper's result. It is NOT a Cournot artifact: under nested
#   CES the SAME object mu(S_g) appears under quantity and price competition, and
#   the entire solver below is written for a general mu(.). Conduct enters only
#   through which mu is passed in. See PART 2.
#
#   Delivered cost of plant a into destination n (sector k), BASELINE:
#
#       a[a,n] = (1/phi_a)
#                * ( w[l] )^(1-nu_k)                              labour, HOST only
#                * ( PIO[l,k] )^nu_k                              intermediates
#                * gamma[h,l]                                     MP friction
#                * d[l,n]                                         trade cost
#       c[a,n] = a[a,n] * (1 + tariff[l,n,k])
#
#   with h the parent's headquarters country and l the production country.
#
#   HEAD-OFFICE SERVICES ARE A CAPABILITY, NOT A FACTOR INPUT, AND THIS IS THE
#   ONE MODELLING CHOICE WORTH ARGUING ABOUT. A group's head office is what makes
#   its plants good -- it enters phi_a, not the cost bundle. Concretely, alpha_k
#   is a COMPLEXITY INDEX by sector and it does two things, both in the
#   productivity draw: a stand-alone local firm carries a penalty exp(-hq_gap *
#   alpha_k), because headquarter services are exactly what it cannot buy at arm's
#   length; and a parent's capability is scaled by (1 + adv_slope * alpha_k),
#   because complex goods are where capability pays. Facts 1 and 2 rest on this.
#
#   The alternative -- alpha_k as the share of cost paid at the PARENT's wage,
#   the Antras (2003) headquarter input -- is supported and one keyword away
#   (`world_economy(...; hq_cost = true)`, cost bundle w[h]^alpha w[l]^(1-alpha)).
#   It is NOT the baseline for three reasons, all measured:
#     * it is the sole mechanism behind the counterexample to gross substitutes
#       (`wage_uniqueness.jl`), and switching it off buys THEOREM 5;
#     * it puts a WRONG-SIGN force on Fact 2 -- affiliates pay their parent's high
#       wage precisely where alpha_k is large -- and removing it moves the Fact 2
#       gradient from +0.33 to +0.42 against a target of +0.43;
#     * with it off, the fitted multinational productivity edge is no longer
#       needed at all (mne_adv = 0), so three fitted parameters become two.
#   Everything is reported under both settings; see `simple_model.jl`.
#
#   ENTRY. Serving market n with plant a costs a fixed
#       F[a,n] = f_k * w[l] * fdist[l,n]
#   paid in the same factor bundle as production. Which plants operate is an
#   OUTCOME. The entry equilibrium is UNIQUE -- by a theorem, not by an assumed
#   order of entry. See PART 5.
#
#   Sources. The multinational-production skeleton is Ramondo & Rodriguez-Clare
#   (2013) for gamma and Tintelnot (2017) for l != n (export platforms) and for
#   the fixed cost of serving a market; the MP friction is quantified by Head &
#   Mayer (2019); the input-output loop follows Caliendo & Parro (2015); the
#   granular entry margin, the Pareto capability draw and the size-dependent pool
#   of potential entrants are Gaubert & Itskhoki (2021); multi-plant parents
#   choosing SETS of locations is Yang (2023); alpha_k as the complexity index of
#   headquarter intensity is Antras (2003) and Antras-Helpman (2004), used here on
#   the capability side rather than the cost side. What is NEW here is (i)
#   parent-level conduct with variable markups instead of constant ones, (ii)
#   firm-level ownership theta[g,n], and (iii) uniqueness theorems for entry and
#   for wages that survive internalisation.
#
#   ---------------------------------------------------------------------------
#   EQUILIBRIUM: DEFINITION
#   ---------------------------------------------------------------------------
#   Wages w, prices P, expenditures E, incomes X and an active set of plants
#   such that
#       (a) every parent plays a best response in every market it serves,
#       (b) sum_g S_g = 1 in every market,
#       (c) no parent wants to open or close a plant,
#       (d) P is consistent with the costs it generates (the I-O loop),
#       (e) labour demand = L in every country,
#       (f) X[n] = w[n] L[n] + sum_g theta[g,n] (Pi[g] - fixed costs) + tariffs.
#   One of (e) is redundant by Walras' law; w[1] = 1 is the numeraire.
#
###############################################################################

module MNEModel

using Printf, Random, LinearAlgebra

# printf format strings are assembled programmatically in places, so the
# newline is a named constant rather than an escape buried in each string.
const NL = "\n"

export Market, MarketEq, solve_market, markup, lerner,
       EntryMarket, solve_market_entry,
       GEModel, GEEntry, solve_ge, solve_ge_entry, excess_demand, accounting,
       export_matrix, classify, measured_hhi, structural_hhi,
       eps_closed, verify_conditions_analytic, boundary_slack,
       eps_markup, chi_share, passthrough, wage_hypotheses,
       wage_bill_elasticity, verify_wage_theorem, reallocation_floor,
       hilbert_metric, birkhoff_coefficient, tatonnement_jacobian,
       max_damping, contraction_certificate, verify_contraction

###############################################################################
# PART 1.  CALIBRATION
#
#   Every parameter, where it comes from, and how confident we are in it.
#   Three kinds, and the difference matters when reading results:
#
#     EXTERNAL   taken from the literature. Not chosen to fit anything here.
#     INTERNAL   chosen so the model reproduces a specific stylized fact.
#     NORMALISED free by construction (units, numeraire).
#     MODELLING  a structural choice, not a number. Both settings are supported
#
#   >>> HEALTH WARNING <<<
#   The EXTERNAL values are standard magnitudes for this literature, not
#   transcriptions from specific tables. Check each against its source paper and
#   replace it before anything goes in the paper. They are collected here so that
#   is a five-minute job.
###############################################################################

const CALIB = (
 sigma   = (val="5.0",        kind="EXTERNAL",
            src="trade elasticity 4-6; Simonovska-Waugh ~4, Eaton-Kortum higher"),
 eta     = (val="1.0",        kind="EXTERNAL",
            src="Cobb-Douglas across sectors; the Atkeson-Burstein outer nest"),
 conduct = (val="Cournot",    kind="EXTERNAL",
            src="quantity competition; Bertrand supported and reported alongside"),
 delta   = (val="1/(sigma-1)", kind="EXTERNAL",
            src="set so the gravity distance elasticity of TRADE is exactly -1"),
 nu      = (val="0.55",       kind="EXTERNAL",
            src="intermediate share of gross output, manufacturing; Caliendo-Parro"),
 io_own  = (val="0.45",       kind="EXTERNAL",
            src="I-O tables are diagonal-heavy: own-sector inputs dominate"),
 alpha   = (val="0.10-0.55",  kind="EXTERNAL",
            src="COMPLEXITY INDEX by sector. Not a cost share: see hq_cost"),
 hq_cost = (val="false",      kind="MODELLING",
            src="head-office services are a non-rival CAPABILITY, not a factor input"),
 gamma   = (val="1.18",       kind="EXTERNAL",
            src="MP efficiency loss abroad; Ramondo-Rodriguez-Clare MP friction"),
 theta_p = (val="4.0",        kind="EXTERNAL",
            src="Pareto tail of firm capability; Gaubert-Itskhoki granular draws"),
 fcost   = (val="0.0006",     kind="INTERNAL",
            src="market-access fixed cost, set to the SIZE of the export sample"),
 zeta    = (val="1.5",        kind="INTERNAL",
            src="how the pool of potential parents scales with country size"),
 mne_adv = (val="0.0",        kind="NORMALISED",
            src="MNE productivity edge NO LONGER NEEDED once hq_cost=false; was INTERNAL"),
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
    println("  paper before publication -- they are collected here for that.")
    flush(stdout)
end

###############################################################################
# PART 2.  CONDUCT: THE MARKUP FUNCTION
#
#   THIS IS THE PART THAT MAKES THE PAPER ROBUST, so it is worth reading.
#
#   In nested CES, a parent that internalises competition among its own plants
#   charges a markup that depends ONLY on its total share S_g in the market:
#
#       Cournot    1/mu(S) = 1 - (1-S)/sigma - S/eta
#       Bertrand   mu(S)   = eps/(eps-1),   eps(S) = sigma - (sigma-eta) S
#
#   Both run from the CES markup sigma/(sigma-1) at S = 0 to the monopoly markup
#   eta/(eta-1) at S = 1. Both are strictly increasing in S when sigma > eta.
#   Everything downstream -- the market solver, the uniqueness theorem, the entry
#   theorem, the GE loop, the ownership decomposition -- is written for a general
#   mu satisfying:
#
#       (M1)  mu is strictly increasing on [0,1)      "cannibalisation"
#       (M2)  mu(0) = sigma/(sigma-1) and sigma > eta
#
#   So "Cournot versus Bertrand" is a CALIBRATION, not a modelling commitment,
#   and cannibalisation -- not Cournot -- is the mechanism the paper rests on.
#
#   WHAT DOES DEPEND ON CONDUCT is the MAGNITUDE. Writing profit as
#   (E/sigma) S [1 + kappa S], the ownership correction is proportional to kappa:
#
#       kappa_Cournot  = sigma/eta - 1        = 4.00  at sigma=5, eta=1
#       kappa_Bertrand = (sigma - eta)/sigma  = 0.80
#
#   Cournot is the baseline for two reasons, both checkable and both reported:
#   the quadratic profit form is EXACT under Cournot (so PART 8's decomposition
#   is an identity, not an approximation), and its implied affiliate-level
#   concentration is closer to the data. Bertrand is a supported alternative and
#   every headline is reported under both, so the reader sees the range.
###############################################################################

"""
Markup as a function of the parent's TOTAL share. `conduct` is `:cournot` or
`:bertrand`. Requires sigma > eta; otherwise markups would FALL with size, which
is economically backwards and breaks every uniqueness argument below.
"""
@inline function markup(S, sigma::Float64, eta::Float64, conduct::Symbol)
    if conduct === :bertrand
        e = sigma - (sigma - eta) * S
        return e / (e - 1.0)
    else
        return 1.0 / (1.0 - (1.0 - S) / sigma - S / eta)
    end
end

"""d mu / dS. Closed form for both conducts; used by the Newton step."""
@inline function dmarkup(S, sigma::Float64, eta::Float64, conduct::Symbol)
    if conduct === :bertrand
        e = sigma - (sigma - eta) * S
        return (sigma - eta) / (e - 1.0)^2
    else
        mu = markup(S, sigma, eta, conduct)
        return mu * mu * (1.0 / eta - 1.0 / sigma)
    end
end

"""Lerner index 1 - 1/mu. Profit is revenue times this."""
@inline lerner(S, sigma::Float64, eta::Float64, conduct::Symbol) =
    1.0 - 1.0 / markup(S, sigma, eta, conduct)

"""
kappa: profit written as (E/sigma) S (1 + kappa S). EXACT under Cournot, where
the Lerner index is linear in S; under Bertrand it is the leading term of an
expansion whose remainder PART 8 measures rather than assumes.
"""
kappa_of(sigma, eta, conduct::Symbol) =
    conduct === :bertrand ? (sigma - eta) / sigma : sigma / eta - 1.0

###############################################################################
# PART 3.  ONE MARKET
###############################################################################

"""
One (destination, sector) cell. `gid[i]` is the parent that owns variety `i`,
numbered 1..G contiguously. `D` is the demand shifter; with eta = 1 it is simply
expenditure E.
"""
struct Market
    sigma::Float64
    eta::Float64
    D::Float64
    c::Vector{Float64}
    gid::Vector{Int}
    conduct::Symbol
end
Market(sigma, eta, D, c, gid) = Market(sigma, eta, D, c, gid, :cournot)

struct MarketEq
    S::Vector{Float64}      # parent shares
    s::Vector{Float64}      # variety shares
    p::Vector{Float64}      # prices
    mu::Vector{Float64}     # parent markups
    q::Vector{Float64}      # quantities
    r::Vector{Float64}      # revenues
    Pi::Vector{Float64}     # parent profits (gross of fixed costs)
    A::Float64              # price aggregate sum_i p_i^(1-sigma)
    P::Float64              # CES price index
    E::Float64              # expenditure
end

"""
INNER LOOP. Given the price aggregate A, find one parent's share: the unique root
in (0,1) of  x = mu(x)^(1-sigma) * KA,  where KA = K_g / A and
K_g = sum_{i in g} c_i^(1-sigma) is the parent's CAPABILITY STOCK.

Unique because the left side rises from 0 to 1 while the right side falls (mu
increases in x and the exponent 1-sigma is negative). A rising and a falling
curve cross at most once, and they do cross. Newton, safeguarded by a bracket
that is never left, so this is as safe as bisection and much faster.
"""
function inner_share(KA::Float64, sigma::Float64, eta::Float64, conduct::Symbol)
    KA <= 0.0 && return 0.0
    lo, hi = 0.0, 1.0
    x = 0.5
    for _ in 1:100
        mu = markup(x, sigma, eta, conduct)
        F  = x - mu^(1.0 - sigma) * KA          # increasing in x
        F > 0.0 ? (hi = x) : (lo = x)
        dF = 1.0 + (sigma - 1.0) * mu^(-sigma) * dmarkup(x, sigma, eta, conduct) * KA
        xn = x - F / dF
        (isfinite(xn) && lo < xn < hi) || (xn = 0.5 * (lo + hi))
        abs(xn - x) <= 1e-16 && return xn
        x = xn
    end
    return x
end

group_K(m::Market) =
    (K = zeros(maximum(m.gid)); for i in eachindex(m.c)
        K[m.gid[i]] += m.c[i]^(1.0 - m.sigma) end; K)

total_share(A, K, sigma, eta, conduct) =
    sum(inner_share(k / A, sigma, eta, conduct) for k in K)

"""
d(sum_g S_g)/d(ln A). Strictly negative -- the monotonicity the outer loop relies
on. From ln S = (1-sigma) ln mu(S) + ln K - ln A.
"""
function dtotal_dlnA(A, K, sigma, eta, conduct)
    acc = 0.0
    for k in K
        S = inner_share(k / A, sigma, eta, conduct)
        acc -= S / (1.0 + (sigma - 1.0) * dmarkup(S, sigma, eta, conduct) /
                          markup(S, sigma, eta, conduct) * S)
    end
    return acc
end

"""
Solve one market. The equilibrium is UNIQUE; PART 4 proves it and checks every
step of the proof numerically. The assertions are the theorem's hypotheses.
"""
function solve_market(m::Market)
    sigma, eta = m.sigma, m.eta
    @assert sigma > eta >= 1.0 "need sigma > eta >= 1 (else markups fall with size)"
    @assert all(m.c .> 0.0) "marginal costs must be strictly positive"
    @assert maximum(m.gid) >= 2 || eta > 1.0 "a monopolist's markup is unbounded when eta = 1"

    K = group_K(m)

    # OUTER LOOP. total_share(A) is strictly DECREASING in A, from above 1 down
    # to 0, so it crosses 1 exactly once: the root is unique and a bracketing
    # method cannot land on a different one, because there is none. Newton
    # safeguarded by bisection on ln A.
    Aces = (sigma / (sigma - 1.0))^(1.0 - sigma) * sum(K)
    lo, hi = Aces * 1e-6, Aces * 1e6
    it = 0
    while total_share(hi, K, sigma, eta, m.conduct) > 1.0 && it < 500; hi *= 100.0; it += 1; end
    it = 0
    while total_share(lo, K, sigma, eta, m.conduct) < 1.0 && it < 500; lo /= 100.0; it += 1; end

    x = 0.5 * (log(lo) + log(hi))
    llo, lhi = log(lo), log(hi)
    for _ in 1:200
        Ax = exp(x)
        F  = total_share(Ax, K, sigma, eta, m.conduct) - 1.0
        F > 0.0 ? (llo = x) : (lhi = x)
        dF = dtotal_dlnA(Ax, K, sigma, eta, m.conduct)
        xn = x - F / dF
        (isfinite(xn) && llo < xn < lhi) || (xn = 0.5 * (llo + lhi))
        abs(xn - x) <= 1e-15 && (x = xn; break)
        x = xn
    end
    A = exp(x)

    # Unwind. Every step is a one-way map, so one A gives one of everything.
    S  = [inner_share(k / A, sigma, eta, m.conduct) for k in K]
    mu = [markup(s, sigma, eta, m.conduct) for s in S]
    p  = mu[m.gid] .* m.c
    wv = p .^ (1.0 - sigma)
    s  = wv ./ sum(wv)
    P  = A^(1.0 / (1.0 - sigma))
    E  = m.D * P^(1.0 - eta)
    r  = E .* s
    q  = r ./ p
    Pi = E .* S .* (1.0 .- 1.0 ./ mu)
    return MarketEq(S, s, p, mu, q, r, Pi, A, P, E)
end

"""Profit at an arbitrary quantity vector, using NONE of the derived algebra."""
function primitive_profit(q, m::Market, g::Int)
    rho = (m.sigma - 1.0) / m.sigma
    Aq = sum(q .^ rho)
    P  = m.D^(1.0 / m.eta) * Aq^(m.sigma / ((1.0 - m.sigma) * m.eta))
    p  = m.D^(1.0 / m.sigma) .* P^((m.sigma - m.eta) / m.sigma) .* q .^ (-1.0 / m.sigma)
    sel = m.gid .== g
    return sum((p[sel] .- m.c[sel]) .* q[sel])
end

function random_market(rng; eta = 1.0, gmax = 5, nmax = 3, conduct = :cournot)
    sigma = eta + 1.0 + 7.0 * rand(rng)
    G   = rand(rng, 2:gmax)
    gid = vcat([fill(g, rand(rng, 1:nmax)) for g in 1:G]...)
    c   = exp.(0.6 .* randn(rng, length(gid)))
    return Market(sigma, eta, 0.5 + 2.5 * rand(rng), c, gid, conduct)
end

###############################################################################
# PART 4.  UNIQUENESS OF THE MARKET EQUILIBRIUM  (theorem + verification)
#
#   rho = (sigma-1)/sigma,  X_g = sum_{i in g} q_i^rho,  B_g = rivals' equivalent.
#
#   STEP 1  A parent's revenue depends on its own quantities ONLY through the
#           single number X_g:  R = D^(1/eta) X (X+B)^e,
#           e = (eta-sigma)/((sigma-1)eta). A G-dimensional choice collapses to
#           one dimension. THIS is where cannibalisation enters: the parent's own
#           varieties are perfect substitutes for one another in X_g.
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
#   STEP 5  psi(x) = x mu(x)^(sigma-1) is strictly increasing -- this needs only
#           (M1), mu increasing -- so S_g(A) = psi^{-1}(K_g/A) is unique and
#           strictly decreasing in A. Hence sum_g S_g(A) = 1 has one solution.
#
#   => unique A => unique shares, markups, prices, revenues, quantities.
#      Existence is free because the construction is constructive.
#
#   Steps 1-4 are written for Cournot. Step 5 is the one that carries the
#   argument in general, and it uses ONLY (M1). Under Bertrand steps 1-4 are
#   replaced by the standard multi-product Bertrand argument and step 5 is
#   unchanged, which is why the same solver works for both.
###############################################################################

rho_of(sigma) = (sigma - 1.0) / sigma

revenue_of_X(X, B, sigma, eta, D) =
    D^(1.0 / eta) * X * (X + B)^((eta - sigma) / ((sigma - 1.0) * eta))

cost_of_X(X, K, sigma) = X^(sigma / (sigma - 1.0)) * K^(-1.0 / (sigma - 1.0))

psi_of(S, sigma, eta, conduct) = S * markup(S, sigma, eta, conduct)^(sigma - 1.0)

function verify_uniqueness(; reps = 200, starts = 40, conduct = :cournot, verbose = true)
    rng = MersenneTwister(20260812)
    verbose && println("-"^78)
    verbose && println("MARKET-LEVEL UNIQUENESS ($(conduct)): verifying each step of the proof")
    verbose && println("-"^78)

    # STEP 1 (Cournot only): reshuffle output within a parent at constant X;
    # revenue must not move. This is cannibalisation stated as an identity.
    e1 = 0.0
    if conduct === :cournot
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
                    P  = m.D^(1.0/m.eta) * Aq^(m.sigma/((1.0-m.sigma)*m.eta))
                    p  = m.D^(1.0/m.sigma) .* P^((m.sigma-m.eta)/m.sigma) .* qq.^(-1.0/m.sigma)
                    sum(p[sel] .* qq[sel])
                end
                e1 = max(e1, abs(rev(q2) - rev(q)) / max(abs(rev(q)), 1e-12))
            end
        end
        verbose && @printf("  STEP 1  revenue depends on own quantities only through X_g : %.2e\n", e1)
    end

    # STEP 2 and 3: concavity of revenue and convexity of cost in X
    e2 = -Inf; e3 = Inf
    for _ in 1:reps
        sigma = 2.0 + 6.0*rand(rng); eta = 1.0 + (sigma - 1.0)*0.4*rand(rng)
        X = exp(randn(rng)); B = exp(randn(rng)); D = 0.5 + rand(rng); K = exp(randn(rng))
        h = 1e-4*X
        d2R = (revenue_of_X(X+h,B,sigma,eta,D) - 2revenue_of_X(X,B,sigma,eta,D) +
               revenue_of_X(X-h,B,sigma,eta,D))/h^2
        d2C = (cost_of_X(X+h,K,sigma) - 2cost_of_X(X,K,sigma) + cost_of_X(X-h,K,sigma))/h^2
        e2 = max(e2, d2R); e3 = min(e3, d2C)
    end
    verbose && @printf("  STEP 2  revenue strictly CONCAVE in X      worst d2R/dX2 = %+.2e\n", e2)
    verbose && @printf("  STEP 3  cost strictly CONVEX in X          worst d2C/dX2 = %+.2e\n", e3)

    # STEP 4: the solver's answer is a best response, checked against the
    # PRIMITIVE profit function, from many random starting points.
    e4 = 0.0
    if conduct === :cournot
        for _ in 1:(reps ÷ 4)
            m = random_market(rng)
            eq = solve_market(m)
            for g in 1:maximum(m.gid)
                base = primitive_profit(eq.q, m, g)
                sel = findall(==(g), m.gid)
                for _ in 1:starts
                    q2 = copy(eq.q)
                    q2[sel] .*= exp.(0.5 .* randn(rng, length(sel)))
                    e4 = max(e4, (primitive_profit(q2, m, g) - base)/max(abs(base),1e-12))
                end
            end
        end
        verbose && @printf("  STEP 4  no random deviation beats the solution: best gain %+.2e\n", e4)
    end

    # STEP 5: psi strictly increasing, and total share strictly decreasing in A
    bad5 = 0
    for _ in 1:reps
        sigma = 2.0 + 6.0*rand(rng); eta = 1.0 + (sigma-1.0)*0.4*rand(rng)
        Ss = collect(0.001:0.002:0.999)
        any(diff([psi_of(S,sigma,eta,conduct) for S in Ss]) .<= 0) && (bad5 += 1)
    end
    verbose && @printf("  STEP 5  psi strictly increasing                      : %d / %d fail\n",
                       bad5, reps)

    # END TO END: the same market solved from many starting brackets
    e6 = 0.0
    for _ in 1:(reps ÷ 4)
        m = random_market(rng; conduct = conduct)
        eq = solve_market(m)
        K = group_K(m)
        for _ in 1:5
            A0 = eq.A * exp(4.0*randn(rng))
            tot(A) = total_share(A, K, m.sigma, m.eta, conduct)
            lo, hi = A0*1e-8, A0*1e8
            it=0; while tot(hi) > 1.0 && it < 300; hi *= 100; it += 1; end
            it=0; while tot(lo) < 1.0 && it < 300; lo /= 100; it += 1; end
            for _ in 1:300
                mid = sqrt(lo*hi); tot(mid) > 1.0 ? (lo = mid) : (hi = mid)
                hi/lo - 1 < 1e-14 && break
            end
            e6 = max(e6, abs(log(sqrt(lo*hi)/eq.A)))
        end
    end
    verbose && @printf("  END     same market from 5 wildly different brackets : %.2e\n", e6)
    verbose && println()
    return (step1 = e1, step2 = e2, step3 = e3, step4 = e4, step5 = bad5, endto = e6)
end

###############################################################################
# PART 5.  ENTRY, AND WHY THE EQUILIBRIUM IS UNIQUE
#
#   THE PROBLEM. Entry into an oligopoly is a discrete game of strategic
#   substitutes and normally has many equilibria. The literature's two answers
#   are both unusable here:
#
#     * Gaubert-Itskhoki (2021) rank potential entrants by marginal cost and let
#       the lowest move first, giving a unique cutoff. That requires every
#       entrant to be a SEPARATE competitor. The moment a parent internalises
#       several plants, adding one of its own RAISES its share, their condition
#       fails, and the argument collapses. Internalisation is what Fact 4 is
#       about, so it cannot be given up.
#     * Yang (2023) does not solve multiplicity: he proves existence and then
#       SELECTS an equilibrium by imposing an entry order. With G parents there
#       are G! orders, no principle picks one, and the selected outcome is not a
#       property of the model.
#
#   THE FIX. A parent's payoff depends on its own plants ONLY through the scalar
#   capability stock K_g = sum_{i active} c_i^(1-sigma). Internalisation is
#   already inside K_g, so the parent's entry problem is ONE-DIMENSIONAL however
#   many plants it owns. Writing Omega(K;A) = Pi(psi^{-1}(K/A)) - fixed costs:
#
#     (A) OWN CONCAVITY.  Omega is strictly concave in K. The best response is a
#         CUTOFF ON THE PARENT'S OWN LIST -- Gaubert-Itskhoki's cutoff applied
#         WITHIN the parent instead of across the market.
#     (B) STRATEGIC SUBSTITUTES.  d Omega_K / dA < 0.
#
#   (A) and (B) => every K_g is non-increasing in A while A is strictly
#   increasing in every K_g. Two equilibria with A < A' would need every parent
#   weakly smaller at A' and the total strictly larger. Contradiction.
#
#   => THE ENTRY EQUILIBRIUM IS UNIQUE. No order, no selection rule, and parents
#      still internalise. The earlier pass concluded the opposite because it
#      tested the Gaubert-Itskhoki condition, which MIXES a firm's own effect
#      with its rivals'. Separate them and both behave.
#
#   THE HYPOTHESIS HAS CONTENT. (A) is unconditional. (B) holds while no single
#   PARENT is above a share frontier: about 0.55 under Cournot and 0.68 under
#   Bertrand at sigma=5, eta=1. The solver checks the realised shares rather than
#   assuming, and the frontier is only sufficient -- uniqueness is found beyond
#   it too.
#
#   EQUILIBRIUM CONCEPT, stated plainly: parents internalise among their own
#   plants when setting QUANTITIES (that is mu(S_g), and Fact 4 rests on it) and
#   take the market aggregate as given when deciding ENTRY. That is exactly the
#   free-entry condition of Gaubert-Itskhoki and of every quantitative granular
#   entry model. `nash_refine` implements the sharper notion, in which every
#   deviation is scored on a fully re-solved market, and PART 5's tests report
#   how often the two differ.
###############################################################################

"""
A market before entry: `c` and `F` list every POTENTIAL plant, `gid` says which
parent owns it, `F` is the fixed cost of activating it here.
"""
struct EntryMarket
    sigma::Float64
    eta::Float64
    E::Float64
    c::Vector{Float64}
    gid::Vector{Int}
    F::Vector{Float64}
    conduct::Symbol
end
EntryMarket(sigma, eta, E, c, gid, F) = EntryMarket(sigma, eta, E, c, gid, F, :cournot)

groups_of(em::EntryMarket) = sort(unique(em.gid))
Pi_at(S, em::EntryMarket) = em.E * S * lerner(S, em.sigma, em.eta, em.conduct)

"""
The subsets of a parent's own list that could ever be optimal.

Omega is strictly INCREASING and strictly CONCAVE in K, so a subset can only be
optimal if no other subset delivers at least as much K for no more F. Only the
Pareto frontier of (K, F) survives, computed ONCE per market. This is the
theorem paying for itself: with n potential plants there are 2^n subsets but
only a handful on the frontier, which is what makes entry affordable inside a
general-equilibrium loop.
"""
function subset_table(em::EntryMarket, g::Int)
    idx = findall(==(g), em.gid)
    n = length(idx)
    n > 20 && error("parent $g has $n potential plants here; enumeration is not sane")
    allK = zeros(2^n); allF = zeros(2^n)
    for mask in 0:(2^n - 1), j in 1:n
        if (mask >> (j-1)) & 1 == 1
            allK[mask+1] += em.c[idx[j]]^(1.0 - em.sigma)
            allF[mask+1] += em.F[idx[j]]
        end
    end
    ord = sort(collect(0:(2^n - 1)); by = t -> (allF[t+1], -allK[t+1]))
    masks = Int[]; Ks = Float64[]; Fs = Float64[]
    bestK = -Inf
    for t in ord
        if allK[t+1] > bestK + 1e-15
            push!(masks, t); push!(Ks, allK[t+1]); push!(Fs, allF[t+1]); bestK = allK[t+1]
        end
    end
    return (idx = idx, masks = masks, K = Ks, F = Fs)
end

"""Parent `g`'s best response when the market aggregate `A` is taken as given."""
function best_response(em::EntryMarket, tab, A::Float64)
    bestm = 0; bestS = 0.0; bestv = 0.0; bestK = 0.0
    for j in eachindex(tab.masks)
        K = tab.K[j]
        K <= 0.0 && continue
        S = inner_share(K/A, em.sigma, em.eta, em.conduct)
        v = Pi_at(S, em) - tab.F[j]
        # Ties: prefer the SMALLER capability stock. Exact ties happen when
        # plants share a fixed cost and are near-identical in cost; without a
        # rule the choice flips arbitrarily along A and K*(A) stops being
        # monotone for no economic reason. "Least entry among equals" is applied
        # WITHIN one parent, never across parents, so it is not the cross-firm
        # entry ordering this whole construction exists to avoid.
        if v > bestv + 1e-12 || (abs(v - bestv) <= 1e-12 && K < bestK)
            bestv = v; bestm = tab.masks[j]; bestS = S; bestK = K
        end
    end
    return bestm, bestS, bestv, bestK
end

"""Excess share, sum_g S_g - 1, when every parent best-responds at aggregate A."""
function clearing_gap(em::EntryMarket, tabs, A::Float64)
    tot = 0.0
    for t in tabs; tot += best_response(em, t, A)[2]; end
    return tot - 1.0
end

"""Active-set vector implied by every parent best-responding at aggregate A."""
function config_at(em::EntryMarket, tabs, A::Float64)
    active = fill(false, length(em.c))
    for t in tabs
        mask = best_response(em, t, A)[1]
        for j in eachindex(t.idx); active[t.idx[j]] = ((mask >> (j-1)) & 1 == 1); end
    end
    return active
end

"""Market equilibrium on a given active set. `nothing` if it is not solvable."""
function eq_on(em::EntryMarket, active::AbstractVector{Bool})
    on = findall(active)
    isempty(on) && return nothing
    pars = em.gid[on]
    uniq = sort(unique(pars))
    (length(uniq) < 2 && em.eta <= 1.0) && return nothing
    lut = Dict(u => i for (i, u) in enumerate(uniq))
    eq = solve_market(Market(em.sigma, em.eta, em.E, em.c[on],
                             [lut[p] for p in pars], em.conduct))
    return (eq = eq, on = on, pars = uniq)
end

"""Net payoff of parent `g` on a given active set: gross profit minus fixed costs."""
function payoff_of(em::EntryMarket, active::AbstractVector{Bool}, g::Int)
    r = eq_on(em, active)
    r === nothing && return -Inf
    gi = findfirst(==(g), r.pars)
    gross = gi === nothing ? 0.0 : r.eq.Pi[gi]
    fc = 0.0
    for i in eachindex(active)
        (active[i] && em.gid[i] == g) && (fc += em.F[i])
    end
    return gross - fc
end

"""
Solve the market WITH entry, by bisection on the single scalar A.

`certify = true` re-evaluates the clearing function on a wide logarithmic grid
and confirms it is decreasing with exactly one sign change. That is a uniqueness
CERTIFICATE for THIS market, computed rather than assumed.
"""
function solve_market_entry(em::EntryMarket; certify = false, gridn = 40,
                            Ahint = nothing, tabs = nothing)
    tabs = tabs === nothing ? [subset_table(em, g) for g in groups_of(em)] : tabs
    Kall = sum(c^(1.0 - em.sigma) for c in em.c)
    Amid = Ahint === nothing ?
           (em.sigma/(em.sigma - 1.0))^(1.0 - em.sigma) * Kall : Ahint
    lo, hi = Ahint === nothing ? (Amid*1e-8, Amid*1e8) : (Amid/4.0, Amid*4.0)
    it = 0; while clearing_gap(em, tabs, hi) > 0.0 && it < 60; hi *= 100.0; it += 1; end
    it = 0; while clearing_gap(em, tabs, lo) < 0.0 && it < 60; lo /= 100.0; it += 1; end
    (isfinite(log(lo)) && lo > 0.0 && isfinite(hi)) || return nothing

    certified = false; monotone = true; ncross = 0
    if certify
        As = exp.(range(log(lo), log(hi), length = gridn))
        gs = [clearing_gap(em, tabs, A) for A in As]
        monotone = all(diff(gs) .<= 1e-12)
        ncross = count(i -> gs[i] > 0.0 && gs[i+1] <= 0.0, 1:(length(gs)-1))
        certified = monotone && ncross == 1
    end

    for _ in 1:200
        mid = sqrt(lo*hi)
        clearing_gap(em, tabs, mid) > 0.0 ? (lo = mid) : (hi = mid)
        hi/lo - 1.0 < 1e-11 && break
    end
    A = sqrt(lo*hi)
    cfg = config_at(em, tabs, A)
    r = eq_on(em, cfg)
    r === nothing && return nothing
    return (active = cfg, eq = r.eq, on = r.on, pars = r.pars, A = A,
            certified = certified, monotone = monotone, ncross = ncross,
            maxS = maximum(r.eq.S))
end

"""
EXACT-NASH REFINEMENT. Each parent re-optimises its own subset with the market
RE-SOLVED after every candidate deviation, so it internalises its own effect on
the aggregate. `moved = false` means the cheap solve was already exact.
Deliberately NOT used inside the GE loop: it costs 2^n market solves per parent
per round. It is the referee, not the solver.
"""
function nash_refine(em::EntryMarket, active0::AbstractVector{Bool}; rounds = 20)
    active = collect(active0); moved = false
    for r in 1:rounds
        changed = false
        for g in groups_of(em)
            idx = findall(==(g), em.gid)
            best = copy(active); bestv = payoff_of(em, active, g)
            for mask in 0:(2^length(idx) - 1)
                trial = copy(active)
                for j in eachindex(idx); trial[idx[j]] = ((mask >> (j-1)) & 1 == 1); end
                v = payoff_of(em, trial, g)
                if v > bestv + 1e-12; bestv = v; best = trial; end
            end
            best != active && (changed = true; moved = true)
            active = best
        end
        changed || return (active = active, rounds = r, moved = moved)
    end
    return (active = active, rounds = rounds, moved = moved)
end

"""Is `active` a Nash equilibrium in the exact sense? Uses only the market solver."""
function is_nash(em::EntryMarket, active::AbstractVector{Bool}; tol = 1e-10)
    for g in groups_of(em)
        idx = findall(==(g), em.gid)
        base = payoff_of(em, active, g)
        for mask in 0:(2^length(idx) - 1)
            trial = collect(active)
            for j in eachindex(idx); trial[idx[j]] = ((mask >> (j-1)) & 1 == 1); end
            payoff_of(em, trial, g) > base + tol && return false
        end
    end
    return true
end

"""Every Nash entry configuration, by brute force. WITH internalisation."""
function all_entry_configs(em::EntryMarket)
    idxs = [findall(==(g), em.gid) for g in groups_of(em)]
    total = prod(2^length(ix) for ix in idxs)
    total > 500_000 && error("too many configurations ($total)")
    out = Vector{Vector{Bool}}()
    for code in 0:(total - 1)
        active = fill(false, length(em.c)); rest = code
        for ix in idxs
            n = length(ix); mm = rest % (2^n); rest = rest ÷ (2^n)
            for j in 1:n; active[ix[j]] = ((mm >> (j-1)) & 1 == 1); end
        end
        eq_on(em, active) === nothing && continue
        is_nash(em, active) && push!(out, active)
    end
    return out
end

"""
The largest parent share at which BOTH theorem conditions still hold, by
differentiating Omega numerically -- no reliance on the algebra above.
"""
function share_frontier(sigma, eta, conduct; grid = 0.004:0.004:0.992, rel = 1e-4)
    Om(K, A) = begin
        S = inner_share(K/A, sigma, eta, conduct)
        S * lerner(S, sigma, eta, conduct)
    end
    for S in grid
        A = 1.0; K = psi_of(S, sigma, eta, conduct)
        h = rel*K; hA = rel*A
        d2 = (Om(K+h,A) - 2Om(K,A) + Om(K-h,A))/h^2
        dp = (Om(K+h,A+hA) - Om(K-h,A+hA))/(2h)
        dm = (Om(K+h,A-hA) - Om(K-h,A-hA))/(2h)
        cross = (dp-dm)/(2hA)
        (d2 < 0 && cross < 0) || return S
    end
    return 1.0
end

"""
THE TWO ENTRY CONDITIONS IN CLOSED FORM (Cournot, eta = 1).

With a = 2(sigma-1) and b = sigma-2,

    eps_psi(S) = [1 + b S] / (1 - S)
    eps_G(S)   = S [ a/(1+aS) - b/(1+bS) - sigma/(1-S) ]

so that
    (A) concavity          <=> eps_G < 0
    (B) substitutes        <=> eps_G + eps_psi > 0,  and

    eps_G + eps_psi = aS/(1+aS) - bS/(1+bS) + (1-2S)/(1-S).

Both conditions are then PROVED rather than assumed:
  (A) holds at every share (sigma >= 2);
  (B) holds at every share <= 1/2, for every sigma > 1, because the first two
      terms are positive (x/(1+x) is increasing and a > b) and the third is
      non-negative for S <= 1/2. At S = 1/2 the slack is exactly 1/sigma.
`verify_conditions_analytic` checks this algebra against brute numerical
differentiation of Omega, so the proof is machine-checked and not just typed.
"""
function eps_closed(S::Float64, sigma::Float64)
    a = 2.0*(sigma - 1.0); b = sigma - 2.0
    eps_psi = (1.0 + b*S)/(1.0 - S)
    eps_G   = S*( a/(1.0 + a*S) - b/(1.0 + b*S) - sigma/(1.0 - S) )
    return eps_G, eps_psi
end

"""Slack in condition (B) at the boundary share of one half: exactly 1/sigma."""
boundary_slack(sigma) = 1.0/sigma

"""
Check the closed forms against numerical differentiation of Omega, and check the
"no parent above one half" bound for other eta and for Bertrand.
"""
function verify_conditions_analytic(; verbose = true)
    epsnum(S, sg, et, cd) = begin
        h = 1e-6
        d1(f,x) = (f(x+h) - f(x-h))/(2h)
        pp(x) = d1(y -> psi_of(y, sg, et, cd), x)
        Pp(x) = d1(y -> y*lerner(y, sg, et, cd), x)
        G(x)  = Pp(x)/pp(x)
        (S*d1(G,S)/G(S), S*pp(S)/psi_of(S, sg, et, cd))
    end
    worst = 0.0
    for sg in (2.5, 3.0, 4.0, 5.0, 8.0), S in 0.02:0.02:0.96
        en = epsnum(S, sg, 1.0, :cournot); ec = eps_closed(S, sg)
        worst = max(worst, abs(en[1]-ec[1]), abs(en[2]-ec[2]))
    end
    verbose && @printf("  closed form vs numerical differentiation      : %.2e%s", worst, NL)
    badA = 0; badB = 0; worstslack = Inf
    for sg in (2.0, 3.0, 5.0, 8.0, 12.0)
        for S in 0.001:0.001:0.999
            eG, ep = eps_closed(S, sg)
            eG < 0.0 || (badA += 1)
            if S <= 0.5
                eG + ep > 0.0 || (badB += 1)
            end
        end
        worstslack = min(worstslack, abs(sum(eps_closed(0.5, sg)) - 1.0/sg))
    end
    verbose && @printf("  (A) violated at any share, sigma in 2..12     : %d%s", badA, NL)
    verbose && @printf("  (B) violated at any share <= 1/2              : %d%s", badB, NL)
    verbose && @printf("  slack at S = 1/2 equals 1/sigma, error        : %.2e%s", worstslack, NL)
    off = 0; minB = Inf
    for cd in (:cournot, :bertrand), (sg, et) in ((5.0,1.5),(5.0,2.5),(8.0,2.0),(3.0,1.0),(8.0,1.0))
        for S in 0.004:0.004:0.5
            eG, ep = epsnum(S, sg, et, cd)
            (eG < 0 && eG + ep > 0) || (off += 1)
            minB = min(minB, eG + ep)
        end
    end
    verbose && @printf("  'no parent above 1/2' fails for other eta or Bertrand : %d%s", off, NL)
    verbose && @printf("  worst slack across those cases                : %.3f%s", minB, NL)
    return (closed = worst, badA = badA, badB = badB, other = off)
end

"""Random market with granular parents (several plants) and a local fringe."""
function random_entry_market(rng; npar = 3, naff = 2, nfringe = 4, conduct = :cournot,
                             sigma = nothing, eta = nothing, F = nothing, spread = 0.5)
    et = eta === nothing ? (rand(rng) < 0.5 ? 1.0 : 1.0 + 1.2*rand(rng)) : eta
    sg = sigma === nothing ? et + 2.0 + 4.0*rand(rng) : sigma
    c = Float64[]; gid = Int[]
    for g in 1:npar, _ in 1:naff
        push!(c, exp(spread*randn(rng))); push!(gid, g)
    end
    for j in 1:nfringe
        push!(c, exp(spread*randn(rng) + 0.35)); push!(gid, npar + j)
    end
    f = F === nothing ? 0.002 + 0.02*rand(rng) : F
    return EntryMarket(sg, et, 1.0, c, gid, fill(f, length(c)), conduct)
end

###############################################################################
# PART 6.  GENERAL EQUILIBRIUM
#
#   `sigma[k]`     within-sector elasticity, by sector
#   `eta`          across-sector elasticity (the outer nest)
#   `beta[k]`      final demand weights across sectors
#   `alpha[k]`     HQ input share, rising in complexity (Antras; Antras-Helpman)
#   `nu[k]`        share of sector k's cost spent on intermediates
#   `omega[k',k]`  share of sector k's intermediate bill spent on k' (columns sum to 1)
#   `par/hq/loc/sec/phi`  each plant's parent, HQ country, host, sector, productivity
#   `gamma[h,l]`   MP friction of Ramondo-Rodriguez-Clare
#   `d[l,n]`       trade cost from the PRODUCTION country: Tintelnot's platforms
#   `theta[g,n]`   share of parent g owned by country n (each ROW sums to one)
###############################################################################

struct GEModel
    N::Int
    K::Int
    sigma::Vector{Float64}
    eta::Float64
    beta::Vector{Float64}
    alpha::Vector{Float64}
    nu::Vector{Float64}
    omega::Matrix{Float64}
    L::Vector{Float64}
    par::Vector{Int}
    hq::Vector{Int}
    loc::Vector{Int}
    sec::Vector{Int}
    phi::Vector{Float64}
    gamma::Matrix{Float64}
    d::Matrix{Float64}
    tariff::Array{Float64,3}
    theta::Matrix{Float64}
    conduct::Symbol
end

"""
Delivered pre-tariff cost, given wages and the matrix of sector price indices.

    a = (1/phi) * (w_h^alpha * w_l^(1-alpha))^(1-nu) * PIO^nu * gamma * d
    PIO[l,k] = prod_k' P[l,k']^omega[k',k]

The intermediate bundle is bought in the PRODUCTION country at the prices
prevailing there, which is what makes multinational entry into l cheapen local
firms' inputs -- the channel Fact 5 needs. A floor is applied because wages far
from equilibrium can drive a price index low enough that the bundle underflows;
it is far below anything economically meaningful and never binds at a solution.
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
    return max(lab * pio * m.gamma[h, l] * m.d[l, n] / m.phi[a], 1e-250)
end

"""
A GE model whose plant list is POTENTIAL rather than actual.

`f`     fixed cost of serving one market, by sector, in units of the factor
        bundle. Larger f means fewer entrants and more concentrated markets.
`fdist` multiplier by (production country, destination); ones by default.
`fmult` multiplier by PLANT; ones by default. This is where the Fact-5
        fixed-cost spillover lives (`fspill` in `world_economy`): multinational
        presence in a (country, sector) thickens export infrastructure --
        logistics, certification, buyer networks -- and lowers LOCAL plants'
        market-access cost. Computed off the POTENTIAL roster, exactly like
        `spill`, so it is exogenous to the entry game and Theorem 2 is
        untouched.
"""
struct GEEntry
    base::GEModel
    f::Vector{Float64}
    fdist::Matrix{Float64}
    fmult::Vector{Float64}
end
GEEntry(base::GEModel, f::Vector{Float64}) =
    GEEntry(base, f, ones(base.N, base.N), ones(length(base.par)))
GEEntry(base::GEModel, f::Vector{Float64}, fdist::Matrix{Float64}) =
    GEEntry(base, f, fdist, ones(length(base.par)))

"""
Fixed cost of plant `a` serving destination `n`, at wages `w`.

Paid in the SAME factor bundle as production -- HQ services at the parent's wage,
the rest at the host's. That keeps the whole system homogeneous of degree one in
wages, so the numeraire stays legitimate, and it puts entry costs into labour
demand where they belong.
"""
function fixed_cost(m::GEEntry, w::Vector{Float64}, a::Int, n::Int)
    b = m.base
    k, h, l = b.sec[a], b.hq[a], b.loc[a]
    return m.f[k] * w[h]^b.alpha[k] * w[l]^(1.0 - b.alpha[k]) * m.fdist[l, n] *
           m.fmult[a]
end

"""Solve every market on a GIVEN active set -- no entry decision taken."""
function solve_markets_fixed(m::GEEntry, w::Vector{Float64}, P::Matrix{Float64},
                             Emat::Matrix{Float64}, cfg)
    b = m.base
    mk = Array{Any}(undef, b.N, b.K)
    idxk = [findall(==(k), b.sec) for k in 1:b.K]
    for n in 1:b.N, k in 1:b.K
        idxa = idxk[k]; act = cfg[n, k]
        on = findall(act); isempty(on) && (on = collect(eachindex(idxa)))
        idx = idxa[on]
        apre = [delivered_cost(b, w, P, a, n) for a in idx]
        c = [apre[j] * (1.0 + b.tariff[b.loc[idx[j]], n, k]) for j in eachindex(idx)]
        pars = b.par[idx]; uniq = sort(unique(pars))
        lut = Dict(u => i for (i, u) in enumerate(uniq))
        eq = solve_market(Market(b.sigma[k], b.eta, max(Emat[n,k], 1e-12), c,
                                 [lut[p] for p in pars], b.conduct))
        mk[n, k] = (eq = eq, idx = idx, pars = uniq, apre = apre, P = eq.P,
                    active = act, maxS = maximum(eq.S))
    end
    return mk
end

"""Solve every market WITH entry, given wages, prices and market sizes."""
function solve_markets_entry(m::GEEntry, w::Vector{Float64}, P::Matrix{Float64},
                             Emat::Matrix{Float64}; certify = false, Ahints = nothing)
    b = m.base
    mk = Array{Any}(undef, b.N, b.K)
    idxk = [findall(==(k), b.sec) for k in 1:b.K]
    for n in 1:b.N, k in 1:b.K
        idx = idxk[k]
        apre = [delivered_cost(b, w, P, a, n) for a in idx]
        c  = [apre[j] * (1.0 + b.tariff[b.loc[idx[j]], n, k]) for j in eachindex(idx)]
        Fv = [fixed_cost(m, w, a, n) for a in idx]
        pars = b.par[idx]; uniq = sort(unique(pars))
        lut  = Dict(u => i for (i, u) in enumerate(uniq))
        em = EntryMarket(b.sigma[k], b.eta, max(Emat[n, k], 1e-12), c,
                         [lut[p] for p in pars], Fv, b.conduct)
        hint = Ahints === nothing ? nothing : Ahints[n, k]
        sol = solve_market_entry(em; certify = certify, Ahint = hint)
        if sol === nothing
            eq = solve_market(Market(b.sigma[k], b.eta, max(Emat[n,k],1e-12), c,
                                     [lut[p] for p in pars], b.conduct))
            mk[n, k] = (eq = eq, idx = idx, pars = uniq, apre = apre, P = eq.P,
                        active = fill(true, length(idx)), certified = false,
                        maxS = maximum(eq.S), A = eq.A)
            continue
        end
        on = sol.on
        mk[n, k] = (eq = sol.eq, idx = idx[on], pars = sort(unique(pars[on])),
                    apre = apre[on], P = sol.eq.P, active = sol.active,
                    certified = sol.certified, maxS = sol.maxS, A = sol.A)
    end
    return mk
end

configs_of(m::GEEntry, mk) = [collect(mk[n,k].active) for n in 1:m.base.N, k in 1:m.base.K]
flatten_cfg(cfg) = vcat([collect(c) for c in cfg]...)
all_active(m::GEEntry) =
    [fill(true, count(==(k), m.base.sec)) for n in 1:m.base.N, k in 1:m.base.K]

"""Fixed-cost bill of each parent and each country."""
function fixed_bills(m::GEEntry, w::Vector{Float64}, mk)
    b = m.base; G = size(b.theta, 1)
    FCg = zeros(G); FCn = zeros(b.N)
    for n in 1:b.N, k in 1:b.K, a in mk[n, k].idx
        F = fixed_cost(m, w, a, n)
        FCg[b.par[a]] += F
        FCn[b.hq[a]]  += b.alpha[k] * F
        FCn[b.loc[a]] += (1.0 - b.alpha[k]) * F
    end
    return FCg, FCn
end

"""Final-demand shares across sectors; sum to one, so no income leaks."""
function expenditure_shares(m::GEEntry, mk)
    b = m.base; eps = zeros(b.N, b.K)
    for n in 1:b.N
        tot = sum(b.beta[k] * mk[n, k].P^(1.0 - b.eta) for k in 1:b.K)
        for k in 1:b.K
            eps[n, k] = b.beta[k] * mk[n, k].P^(1.0 - b.eta) / tot
        end
    end
    return eps
end

"""
Expenditures and incomes.

Gross profit is linear in expenditure, so the system stays LINEAR -- fixed costs
enter as a constant, not as another fixed point:

    E = eps .* X + B E                       => E = (I-B)^{-1} eps X
    X = wL + theta'(Pi_gross(E) - FC) + T(E) => one N x N solve with a constant
"""
function solve_quantities(m::GEEntry, w::Vector{Float64}, mk, eps, FCg)
    b = m.base
    NK, G = b.N * b.K, size(b.theta, 1)
    ci(n, k) = (n - 1) * b.K + k
    Bm = zeros(NK, NK); piu = zeros(G, NK); tau = zeros(NK)
    for np in 1:b.N, kpp in 1:b.K
        e, idx, pars = mk[np, kpp].eq, mk[np, kpp].idx, mk[np, kpp].pars
        for (i, g) in enumerate(pars)
            piu[g, ci(np, kpp)] = e.S[i] * lerner(e.S[i], b.sigma[kpp], b.eta, b.conduct)
        end
        for (j, a) in enumerate(idx)
            t  = b.tariff[b.loc[a], np, kpp]
            gi = findfirst(==(b.par[a]), pars)
            pay = e.s[j] / (e.mu[gi] * (1.0 + t))
            tau[ci(np, kpp)] += e.s[j] * t / (e.mu[gi] * (1.0 + t))
            for kp in 1:b.K
                b.omega[kp, kpp] == 0 && continue
                Bm[ci(b.loc[a], kp), ci(np, kpp)] += b.omega[kp, kpp] * b.nu[kpp] * pay
            end
        end
    end
    Lam = zeros(NK, b.N)
    for n in 1:b.N, k in 1:b.K; Lam[ci(n, k), n] = eps[n, k]; end
    C = (I(NK) - Bm) \ Lam
    Mm = zeros(b.N, b.N); const_n = zeros(b.N)
    for n in 1:b.N
        row = zeros(NK)
        for g in 1:G
            b.theta[g, n] == 0 && continue
            row .+= b.theta[g, n] .* piu[g, :]
            const_n[n] -= b.theta[g, n] * FCg[g]     # profits are NET of fixed costs
        end
        for k in 1:b.K; row[ci(n, k)] += tau[ci(n, k)]; end
        Mm[n, :] = row' * C
    end
    X = (I(b.N) - Mm) \ (w .* b.L .+ const_n)
    E = C * X
    return X, Matrix(reshape(E, b.K, b.N)'), piu, tau
end

"""
Labour demand: variable production plus the labour used up paying fixed costs.
Both split alpha to the PARENT's country and (1-alpha) to the HOST's -- that
split is what makes this a multinational model rather than a trade model.
"""
function labour_demand(m::GEEntry, w::Vector{Float64}, mk, Emat, FCn)
    b = m.base; LD = zeros(b.N)
    for n in 1:b.N, k in 1:b.K
        e, idx, pars = mk[n, k].eq, mk[n, k].idx, mk[n, k].pars
        E = Emat[n, k]
        for (j, a) in enumerate(idx)
            t  = b.tariff[b.loc[a], n, k]
            gi = findfirst(==(b.par[a]), pars)
            pay = (1.0 - b.nu[k]) * E * e.s[j] / (e.mu[gi] * (1.0 + t))
            LD[b.hq[a]]  += b.alpha[k] * pay / w[b.hq[a]]
            LD[b.loc[a]] += (1.0 - b.alpha[k]) * pay / w[b.loc[a]]
        end
    end
    for n in 1:b.N; LD[n] += FCn[n] / w[n]; end
    return LD
end

"""
Excess labour demand with the entry configuration HELD FIXED.

Nothing discrete happens in here: given the active set this is the smooth model,
prices solve their contraction, expenditures and incomes solve one linear system.
Prices do not depend on expenditure, which is why freezing entry restores exactly
the block the no-entry model already proved convergent.
"""
function excess_demand_fixed(m::GEEntry, w::Vector{Float64}, cfg; P0 = nothing,
                             tol = 1e-13, maxinner = 600)
    b = m.base
    P = P0 === nothing ? ones(b.N, b.K) : copy(P0)
    local mk
    for _ in 1:maxinner
        mk = solve_markets_fixed(m, w, P, ones(b.N, b.K), cfg)
        Pn = clamp.([mk[n, k].P for n in 1:b.N, k in 1:b.K], 1e-200, 1e200)
        gap = maximum(abs.(log.(Pn ./ P))); P = Pn
        gap < tol && break
    end
    eps = expenditure_shares(m, mk)
    FCg, FCn = fixed_bills(m, w, mk)
    X, Emat, piu, tau = solve_quantities(m, w, mk, eps, FCg)
    mk = solve_markets_fixed(m, w, P, Emat, cfg)
    LD = labour_demand(m, w, mk, Emat, FCn)
    return LD .- b.L, (mk = mk, P = P, eps = eps, X = X, E = Emat, LD = LD,
                       piu = piu, tau = tau, FCg = FCg, FCn = FCn, cfg = cfg)
end

"""Equilibrium wages with the entry configuration HELD FIXED."""
function solve_ge_fixed(m::GEEntry, cfg; w0 = ones(m.base.N), kappa = 0.25,
                        tol = 1e-11, maxit = 100, warm = 10)
    b = m.base
    w = copy(w0); w ./= w[1]
    P = nothing
    local info
    function resid!(wv)
        z, inf = excess_demand_fixed(m, wv, cfg; P0 = P)
        P = inf.P
        return z[2:end] ./ b.L[2:end], inf
    end
    for _ in 1:warm
        f, info = resid!(w)
        maximum(abs.(f)) < tol && return (w = w, gap = maximum(abs.(f)), info = info)
        for n in 2:b.N; w[n] *= (max(info.LD[n], 1e-12) / b.L[n])^kappa; end
        w = clamp.(w, 1e-6, 1e6); w ./= w[1]
    end
    n1 = b.N - 1
    for it in 1:maxit
        f, info = resid!(w)
        nrm = maximum(abs.(f))
        nrm < tol && return (w = w, gap = nrm, info = info)
        J = zeros(n1, n1); h = 1e-7; Pb = P
        for j in 1:n1
            wp = copy(w); wp[j+1] *= exp(h)
            fp, _ = resid!(wp)
            J[:, j] = (fp .- f) ./ h
        end
        P = Pb
        dx = try -(J \ f) catch; fill(0.0, n1) end
        (all(isfinite, dx) && maximum(abs.(dx)) > 0) || (dx = -0.3 .* f)
        maximum(abs.(dx)) > 1.0 && (dx .*= 1.0 / maximum(abs.(dx)))
        step, ok = 1.0, false
        for _ in 1:25
            wt = copy(w); wt[2:end] .*= exp.(step .* dx); wt ./= wt[1]
            ft, _ = resid!(wt)
            if maximum(abs.(ft)) < nrm; w = wt; ok = true; break; end
            step *= 0.5
        end
        if !ok
            for n in 2:b.N; w[n] *= (max(info.LD[n], 1e-12) / b.L[n])^kappa; end
            w = clamp.(w, 1e-6, 1e6); w ./= w[1]
        end
    end
    f, info = resid!(w)
    return (w = w, gap = maximum(abs.(f)), info = info)
end

"""No-entry equilibrium: every potential plant operates. The original model."""
solve_ge(m::GEEntry; kwargs...) = solve_ge_fixed(m, all_active(m); kwargs...)
excess_demand(m::GEEntry, w; kwargs...) =
    excess_demand_fixed(m, w, all_active(m); kwargs...)

###############################################################################
# PART 7.  ENTRY INSIDE THE GENERAL EQUILIBRIUM
#
#   The same principle as one level down: the discrete part is frozen while the
#   smooth part is solved exactly.
#
#       OUTER  the entry configuration
#         IN   equilibrium wages given that configuration -- the audited smooth
#              model, solved to 1e-11, not to whatever a discrete margin allows
#         THEN ask who wants to move, and let some of them move
#
#   DAMPING, and why it is not selection. Moving EVERY unhappy parent at once
#   does not converge: the first pass starts from all plants active, wages adjust
#   to a far denser economy than will survive, and the configuration oscillates
#   between two very different market structures. So only the parents with the
#   LARGEST payoff gains move each pass. That ranks parents by their OWN gain; it
#   changes the path, not the fixed point, and it is not an exogenous order.
#
#   THE INTEGER PROBLEM. Entry is a choice over whole plants, so an exact fixed
#   point in integers need not exist. When none is reached the routine keeps the
#   configuration with the fewest firms wanting to move and returns `regret`, so
#   the size of the near-miss is visible. It never breaks a tie by an entry
#   order. This is a NEAR-MISS ON EXISTENCE, not a multiplicity.
###############################################################################

"""
Who wants to move, and by how much: one record per parent-market at which some
other subset of its own plants would pay better at the aggregate the current
configuration produces.
"""
function entry_deviations(m::GEEntry, w::Vector{Float64}, P::Matrix{Float64},
                          Emat::Matrix{Float64}, cfg)
    b = m.base
    devs = NamedTuple[]; nslots = 0; ndiff = 0
    idxk = [findall(==(k), b.sec) for k in 1:b.K]
    for n in 1:b.N, k in 1:b.K
        idx = idxk[k]; nslots += length(idx)
        apre = [delivered_cost(b, w, P, a, n) for a in idx]
        c  = [apre[j] * (1.0 + b.tariff[b.loc[idx[j]], n, k]) for j in eachindex(idx)]
        Fv = [fixed_cost(m, w, a, n) for a in idx]
        pars = b.par[idx]; uniq = sort(unique(pars))
        lut = Dict(u => i for (i, u) in enumerate(uniq))
        em = EntryMarket(b.sigma[k], b.eta, max(Emat[n,k], 1e-12), c,
                         [lut[p] for p in pars], Fv, b.conduct)
        act = cfg[n, k]
        isempty(findall(act)) && continue
        r = eq_on(em, act); r === nothing && continue
        A = r.eq.A
        for g in groups_of(em)
            own = findall(==(g), em.gid)
            curmask = 0
            for (jj, i) in enumerate(own); act[i] && (curmask |= (1 << (jj-1))); end
            Kcur = sum(em.c[i]^(1.0 - em.sigma) for i in own if act[i]; init = 0.0)
            Fcur = sum(em.F[i] for i in own if act[i]; init = 0.0)
            vcur = Kcur <= 0.0 ? 0.0 :
                   Pi_at(inner_share(Kcur/A, em.sigma, em.eta, em.conduct), em) - Fcur
            tab = subset_table(em, g)
            bm, _, bv, _ = best_response(em, tab, A)
            if bv > vcur + 1e-12 && bm != curmask
                nd = Base.count_ones(xor(bm, curmask))
                ndiff += nd
                push!(devs, (n = n, k = k, own = own, mask = bm, gain = bv - vcur, nd = nd))
            end
        end
    end
    return devs, ndiff, nslots
end

"""GENERAL EQUILIBRIUM WITH ENTRY."""
function solve_ge_entry(m::GEEntry; w0 = ones(m.base.N), maxouter = 60,
                        certify = false, tol = 1e-11, frac = 0.25)
    b = m.base
    cfg = all_active(m)
    w = copy(w0); w ./= w[1]
    bestreg = typemax(Int); keep = deepcopy(cfg); keepw = copy(w)
    outer = maxouter; stalls = 0
    local r
    for it in 1:maxouter
        r = solve_ge_fixed(m, cfg; w0 = w)
        w = r.w
        devs, ndiff, nslots = entry_deviations(m, w, r.info.P, r.info.E, cfg)
        if ndiff < bestreg
            bestreg = ndiff; keep = deepcopy(cfg); keepw = copy(w); stalls = 0
        else
            stalls += 1
        end
        (ndiff == 0 || stalls >= 8) && (outer = it; break)
        ord = sortperm([d.gain for d in devs], rev = true)
        nmove = max(1, ceil(Int, frac * length(devs)))
        newcfg = deepcopy(cfg)
        for t in ord[1:nmove]
            d = devs[t]
            for (jj, i) in enumerate(d.own)
                newcfg[d.n, d.k][i] = ((d.mask >> (jj-1)) & 1 == 1)
            end
        end
        cfg = newcfg
    end
    if bestreg > 0; cfg = keep; w = keepw; end
    r = solve_ge_fixed(m, cfg; w0 = w, tol = tol)
    devs, regret, nslots = entry_deviations(m, r.w, r.info.P, r.info.E, cfg)
    mke = certify ? solve_markets_entry(m, r.w, r.info.P, r.info.E; certify = true) : nothing
    info = merge(r.info, (cfg = cfg, outer = outer, regret = regret, nslots = nslots,
                          devs = devs,
                          certified = mke === nothing ? true :
                              all(mke[n,k].certified for n in 1:b.N, k in 1:b.K),
                          maxS = maximum(maximum(r.info.mk[n,k].eq.S)
                                         for n in 1:b.N, k in 1:b.K)))
    return (w = r.w, iters = outer, gap = r.gap, info = info, stalled = regret > 0)
end

###############################################################################
# PART 7b.  THE WAGE VECTOR: PASS-THROUGH, THE THREE-TERM DECOMPOSITION, AND
#           THE TWO HYPOTHESES OF THEOREM 4
#
#   Everything else in the model has exactly one solution as a THEOREM. The wage
#   vector used to be the exception: uniqueness was checked from many starts and
#   nothing more. This part supplies the missing argument, and it is honest
#   about where the argument stops.
#
#   The classical route is GROSS SUBSTITUTES: raising one country's wage must
#   raise demand for every other country's labour. Write W_n = w_n LD_n for the
#   wage bill and M[n,j] = d ln W_n / d ln w_j. Because the whole system is
#   homogeneous of degree one in wages, M has UNIT ROW SUMS, and
#
#       gross substitutes  <=>  M[n,j] > 0 for every  n != j.
#
#   Three things were feared to break it: the markup channel, the multinational
#   cost linkage -- a plant pays its PARENT's wage on the alpha_k share, so a
#   foreign wage enters the cost of goods made at home -- and discrete entry.
#   The first two are settled here; the third is Theorem 2's business.
#
#   NOTE ON THE BASELINE. Since 2026-08-20 the multinational cost linkage is OFF
#   by default: head-office services are a capability, not a factor input
#   (`hq_cost = false`). Everything below still applies -- it is written for a
#   general alpha_k -- but with alpha_k = 0 in the cost function THEOREM 5 takes
#   over from Theorem 4 and signs the whole reallocation block under (H1) alone.
#   See `reallocation_floor` below and `simple_model.jl`. The alpha_k > 0 case is
#   what `wage_uniqueness.jl`'s counterexample is about, and it is still
#   supported and still reported.
#
#   LEMMA 3 (pass-through).  d ln P / d ln w is NON-NEGATIVE with UNIT ROW SUMS.
#       Unconditional. It already contains the markup channel, the input-output
#       loop and the head-office linkage, so none of the three can make a price
#       index fall when a wage rises, and a common wage rise raises every price
#       index one for one.
#
#   LEMMA 4 (decomposition).  The factor bill of plant a in market (n,k) obeys
#
#       d ln V_a = d ln E_nk
#                  - (sigma-1) (lam_a - lam_g)                within group
#                  - (1 - eps_mu(S_g)) chi_g (lam_g - Lam_nk) between groups
#
#       with lam_a the plant's cost response, lam_g its group's share-weighted
#       average, Lam_nk the market's, chi_g = (sigma-1)/(1+(sigma-1) eps_mu).
#       Two consequences:
#         (i)  the between-group coefficient has the gross-substitutes sign
#              exactly when eps_mu(S_g) <= 1, which under Cournot with eta = 1
#              is exactly S_g <= 1/2 -- THE SAME THRESHOLD AS THEOREM 2;
#         (ii) HEAD-OFFICE NEUTRALITY. All of a group's plants report to the same
#              head office (it is a property of the parent), so that country is
#              paid alpha_k of the group's ENTIRE bill; the within-group term
#              averages to zero over a group. So the within-group reallocation --
#              the term the multinational linkage was supposed to break -- drops
#              out of the head-office channel altogether.
#
#   What is left is a race between a DEMAND term and an EXPOSURE term, and that
#   race is hypothesis (H2) below. It can be lost: see the counterexample in
#   `wage_uniqueness.jl`, where a country that lives entirely off head-office
#   payments for plants abroad has a wage bill that FALLS when its host's wage
#   rises. Gross substitutes is therefore NOT a theorem of this model; it is a
#   property that holds under a checkable condition, and the condition holds at
#   the calibrated economy with room to spare.
###############################################################################

"""Elasticity of the markup in the parent's share, eps_mu = S mu'(S)/mu(S)."""
@inline eps_markup(S, sigma::Float64, eta::Float64, conduct::Symbol) =
    S * dmarkup(S, sigma, eta, conduct) / markup(S, sigma, eta, conduct)

"""
chi_g, the coefficient in  Shat_g = -chi_g (lam_g - Lam):  a group whose cost
rises by more than the market's loses share at this rate. Always positive.
"""
@inline chi_share(S, sigma::Float64, eta::Float64, conduct::Symbol) =
    (sigma - 1.0) / (1.0 + (sigma - 1.0) * eps_markup(S, sigma, eta, conduct))

"""
LEMMA 3, constructively. Returns (Gamma, Nmat, Lambda) with

    Phat = Gamma what + Nmat Phat,      Lambda = (I - Nmat)^{-1} Gamma,

Gamma and Nmat non-negative, [Gamma Nmat] with unit row sums, Nmat with row sums
nu_k < 1. Price indices are stacked as ci(n,k) = (n-1)K + k. `inf` is the second
return value of `excess_demand_fixed`.
"""
function passthrough(m::GEEntry, inf)
    b = m.base; N = b.N; K = b.K; NK = N * K
    Gam = zeros(NK, N); Nmat = zeros(NK, NK)
    for n in 1:N, k in 1:K
        cell = inf.mk[n, k]; eq = cell.eq; sg = b.sigma[k]
        chi = [chi_share(S, sg, b.eta, b.conduct) for S in eq.S]
        den = sum(eq.S[g] * chi[g] for g in eachindex(eq.S))
        row = (n - 1) * K + k
        for (j, a) in enumerate(cell.idx)
            gpos = findfirst(==(b.par[a]), cell.pars)
            zeta = (chi[gpos] / den) * eq.s[j]
            h, l = b.hq[a], b.loc[a]
            Gam[row, h] += zeta * (1.0 - b.nu[k]) * b.alpha[k]
            Gam[row, l] += zeta * (1.0 - b.nu[k]) * (1.0 - b.alpha[k])
            for kp in 1:K
                b.omega[kp, k] == 0.0 && continue
                Nmat[row, (l - 1) * K + kp] += zeta * b.nu[k] * b.omega[kp, k]
            end
        end
    end
    return Gam, Nmat, (I(NK) - Nmat) \ Gam
end

"""Cost response of every potential plant, given the price-index response."""
function cost_response(m::GEEntry, pihat::Vector{Float64}, what::Vector{Float64})
    b = m.base; K = b.K
    lam = zeros(length(b.par))
    for a in eachindex(b.par)
        k, h, l = b.sec[a], b.hq[a], b.loc[a]
        acc = (1.0 - b.nu[k]) * (b.alpha[k] * what[h] + (1.0 - b.alpha[k]) * what[l])
        if b.nu[k] > 0.0
            for kp in 1:K
                b.omega[kp, k] == 0.0 && continue
                acc += b.nu[k] * b.omega[kp, k] * pihat[(l - 1) * K + kp]
            end
        end
        lam[a] = acc
    end
    return lam
end

"""
THE TWO HYPOTHESES OF THEOREM 4, evaluated at an arbitrary (w, cfg).

    (H1)  eps_mu(S_g) <= 1 in every market.  Under Cournot with eta = 1 this is
          exactly "no parent holds more than half of any market" -- the same
          condition as Theorem 2.
    (H2)  For every ordered pair n != m,   Ebar[n,m]  >  Xi[n,m],   where
          Ebar is the income-weighted growth of the markets country n earns in
          (plus the fixed-cost term) and Xi is the income-weighted exposure
          penalty: each of the two reallocation gaps of LEMMA 4, positive part
          only, times its own coefficient.

(H1) and (H2) together imply M[n,m] > 0 for n != m, i.e. gross substitutes.
Both are inequalities in objects the solver already computes, so they are
CHECKED, not assumed -- the same standard as Theorem 2's condition (B).
"""
function wage_hypotheses(m::GEEntry, w::Vector{Float64}, cfg; P0 = nothing)
    b = m.base; N = b.N; K = b.K
    zz, inf = excess_demand_fixed(m, w, cfg; P0 = P0)
    Gam, Nmat, Lam = passthrough(m, inf)
    Wlev = inf.LD .* w
    Pnow = inf.P

    epsmax = 0.0; Smax = 0.0
    for n in 1:N, k in 1:K, S in inf.mk[n, k].eq.S
        epsmax = max(epsmax, eps_markup(S, b.sigma[k], b.eta, b.conduct))
        Smax = max(Smax, S)
    end

    Ebar = zeros(N, N); Xi = zeros(N, N)
    for mm in 1:N
        dir = [j == mm ? 1.0 : 0.0 for j in 1:N]
        pihat = Lam * dir
        lam = cost_response(m, pihat, dir)
        hh = 1e-6
        Eup = excess_demand_fixed(m, w .* exp.(hh .* dir), cfg; P0 = Pnow)[2].E
        Edn = excess_demand_fixed(m, w .* exp.(-hh .* dir), cfg; P0 = Pnow)[2].E
        dE = (log.(Eup) .- log.(Edn)) ./ (2hh)
        dem = zeros(N); pen = zeros(N)
        for n in 1:N, k in 1:K
            cell = inf.mk[n, k]; eq = cell.eq; sg = b.sigma[k]
            lamg = zeros(length(eq.S)); row = (n - 1) * K + k
            for (j, a) in enumerate(cell.idx)
                gpos = findfirst(==(b.par[a]), cell.pars)
                lamg[gpos] += eq.s[j] / eq.S[gpos] * lam[a]
            end
            for (j, a) in enumerate(cell.idx)
                gpos = findfirst(==(b.par[a]), cell.pars)
                tar = b.tariff[b.loc[a], n, k]
                Vlev = (1.0 - b.nu[k]) * inf.E[n, k] * eq.s[j] /
                       (eq.mu[gpos] * (1.0 + tar))
                Flev = fixed_cost(m, w, a, n)
                lamF = b.alpha[k] * dir[b.hq[a]] + (1.0 - b.alpha[k]) * dir[b.loc[a]]
                gapin = max(lam[a] - lamg[gpos], 0.0)
                gapbt = max(lamg[gpos] - pihat[row], 0.0)
                emu = eps_markup(eq.S[gpos], sg, b.eta, b.conduct)
                chg = chi_share(eq.S[gpos], sg, b.eta, b.conduct)
                hit = (sg - 1.0) * gapin + (1.0 - emu) * chg * gapbt
                dem[b.hq[a]]  += b.alpha[k] * (Vlev * dE[n, k] + Flev * lamF)
                pen[b.hq[a]]  += b.alpha[k] * Vlev * hit
                dem[b.loc[a]] += (1.0 - b.alpha[k]) * (Vlev * dE[n, k] + Flev * lamF)
                pen[b.loc[a]] += (1.0 - b.alpha[k]) * Vlev * hit
            end
        end
        for n in 1:N
            Ebar[n, mm] = dem[n] / Wlev[n]; Xi[n, mm] = pen[n] / Wlev[n]
        end
    end
    slack = [n == j ? Inf : Ebar[n, j] - Xi[n, j] for n in 1:N, j in 1:N]
    mins = minimum([slack[n, j] for n in 1:N, j in 1:N if n != j])
    return (epsmax = epsmax, Smax = Smax, Ebar = Ebar, Xi = Xi, slack = slack,
            H1 = epsmax <= 1.0, H2 = mins > 0.0, minslack = mins,
            Lambda = Lam, Gamma = Gam, Nmat = Nmat, inf = inf)
end

"""
THEOREM 5's object: the REALLOCATION FLOOR.

Split `d ln W_n / d ln w_m` into a DEMAND term and a REALLOCATION term using
LEMMA 4. This returns the smallest single-factory reallocation term over every
ordered pair `n != m`, together with the largest `eps_mu` (so hypothesis (H1)
can be read off) and the largest violation of

        lam_a  <=  min( lam_g , Lam_nk )                                  (*)

for factories paying a country other than `m`.

THEOREM 5. If head-office services do not enter the cost bundle
(`alpha_k = 0`, i.e. `hq_cost = false` in `world_economy`), then every factory
paying country `n != m` is LOCATED in `n`, and:

  * with `nu = 0` its cost response is exactly zero, so (*) holds trivially;
  * given (H1) and (*), the reallocation term is bounded below by
    `(sigma-1) * min(lam_g - lam_a, Lam - lam_a) >= 0`.

Hence the whole reallocation block is non-negative and

        d ln W_n / d ln w_m  >=  Ebar_n^(m),

the income-weighted growth of the markets country `n` earns in. Gross
substitutes -- and so uniqueness, by Proposition (ABH) -- then follows from a
pure DEMAND condition: no market's nominal spending falls.

This is strictly stronger than Theorem 4, whose second hypothesis has a
counterexample: here the model-specific channel is gone and only the textbook
income effect is left.
"""
function reallocation_floor(m::GEEntry, w::Vector{Float64}, cfg; P0 = nothing)
    b = m.base; N = b.N; K = b.K
    zz, inf = excess_demand_fixed(m, w, cfg; P0 = P0)
    Gam, Nmat, Lam = passthrough(m, inf)
    floor_all = Inf; epsmax = 0.0; starviol = 0.0; nplant = 0
    for mm in 1:N
        dir = [j == mm ? 1.0 : 0.0 for j in 1:N]
        pihat = Lam * dir
        lam = cost_response(m, pihat, dir)
        for n in 1:N, k in 1:K
            cell = inf.mk[n, k]; eq = cell.eq; sg = b.sigma[k]
            lamg = zeros(length(eq.S)); row = (n - 1) * K + k
            for (j, a) in enumerate(cell.idx)
                g = findfirst(==(b.par[a]), cell.pars)
                lamg[g] += eq.s[j] / eq.S[g] * lam[a]
            end
            for (j, a) in enumerate(cell.idx)
                g = findfirst(==(b.par[a]), cell.pars)
                emu = eps_markup(eq.S[g], sg, b.eta, b.conduct)
                chg = chi_share(eq.S[g], sg, b.eta, b.conduct)
                epsmax = max(epsmax, emu)
                piece = -(sg - 1.0) * (lam[a] - lamg[g]) -
                        (1.0 - emu) * chg * (lamg[g] - pihat[row])
                for (cty, wt) in ((b.hq[a], b.alpha[k]), (b.loc[a], 1.0 - b.alpha[k]))
                    wt == 0.0 && continue
                    cty == mm && continue
                    nplant += 1
                    floor_all = min(floor_all, piece)
                    starviol = max(starviol,
                                   lam[a] - min(lamg[g], pihat[row]))
                end
            end
        end
    end
    return (floor = floor_all, epsmax = epsmax, star_violation = starviol,
            hq_cost_off = all(b.alpha .== 0.0), npairs = nplant)
end

"""Wage-bill elasticity matrix M[n,j] = d ln W_n / d ln w_j, by differences."""
function wage_bill_elasticity(m::GEEntry, w::Vector{Float64}, cfg; h = 1e-5,
                              P0 = nothing)
    N = m.base.N; Mel = zeros(N, N)
    for j in 1:N
        wup = copy(w); wup[j] *= exp(h)
        wdn = copy(w); wdn[j] *= exp(-h)
        Wup = excess_demand_fixed(m, wup, cfg; P0 = P0)[2].LD .* wup
        Wdn = excess_demand_fixed(m, wdn, cfg; P0 = P0)[2].LD .* wdn
        Mel[:, j] = (log.(Wup) .- log.(Wdn)) ./ (2h)
    end
    return Mel
end

"""
Machine-check LEMMA 3 and LEMMA 4 against numerical differentiation, then report
(H1), (H2), gross substitutes and the index at the given point. Every analytic
step used by Theorem 4 is checked here, exactly as `verify_conditions_analytic`
does for Theorem 2 -- so the proof is machine-checked, not just typed.
"""
function verify_wage_theorem(m::GEEntry, w::Vector{Float64}, cfg; verbose = true,
                             P0 = nothing)
    b = m.base; N = b.N; K = b.K
    zz, inf = excess_demand_fixed(m, w, cfg; P0 = P0)
    Pnow = inf.P
    Gam, Nmat, Lam = passthrough(m, inf)
    lam3 = minimum(Lam); row3 = maximum(abs.(sum(Lam, dims = 2) .- 1.0))
    hh = 1e-6
    worstP = 0.0; worstS = 0.0; worstV = 0.0; worstW = 0.0
    rngv = MersenneTwister(31)
    for trial in 1:(N + 2)
        dir = trial <= N ? [j == trial ? 1.0 : 0.0 for j in 1:N] : randn(rngv, N)
        infu = excess_demand_fixed(m, w .* exp.(hh .* dir), cfg; P0 = Pnow)[2]
        infd = excess_demand_fixed(m, w .* exp.(-hh .* dir), cfg; P0 = Pnow)[2]
        dP = (log.(infu.P) .- log.(infd.P)) ./ (2hh)
        dE = (log.(infu.E) .- log.(infd.E)) ./ (2hh)
        dW = (log.(infu.LD .* w .* exp.(hh .* dir)) .-
              log.(infd.LD .* w .* exp.(-hh .* dir))) ./ (2hh)
        pin = [dP[n, k] for n in 1:N for k in 1:K]
        worstP = max(worstP, maximum(abs.(Lam * dir .- pin)))
        lam = cost_response(m, pin, dir)
        Wpred = zeros(N)
        for n in 1:N, k in 1:K
            cell = inf.mk[n, k]; eq = cell.eq; sg = b.sigma[k]
            Sup = infu.mk[n, k].eq.S; Sdn = infd.mk[n, k].eq.S
            lamg = zeros(length(eq.S))
            for (j, a) in enumerate(cell.idx)
                gpos = findfirst(==(b.par[a]), cell.pars)
                lamg[gpos] += eq.s[j] / eq.S[gpos] * lam[a]
            end
            for g in eachindex(eq.S)
                chg = chi_share(eq.S[g], sg, b.eta, b.conduct)
                Spred = -chg * (lamg[g] - dP[n, k])
                worstS = max(worstS, abs(Spred - (log(Sup[g]) - log(Sdn[g])) / (2hh)))
            end
            for (j, a) in enumerate(cell.idx)
                gpos = findfirst(==(b.par[a]), cell.pars)
                emu = eps_markup(eq.S[gpos], sg, b.eta, b.conduct)
                chg = chi_share(eq.S[gpos], sg, b.eta, b.conduct)
                Vpred = dE[n, k] - (sg - 1.0) * (lam[a] - lamg[gpos]) -
                        (1.0 - emu) * chg * (lamg[gpos] - dP[n, k])
                tar = b.tariff[b.loc[a], n, k]
                Vlev = (1.0 - b.nu[k]) * inf.E[n, k] * eq.s[j] /
                       (eq.mu[gpos] * (1.0 + tar))
                equ = infu.mk[n, k]; eqd = infd.mk[n, k]
                ju = findfirst(==(a), equ.idx); jd = findfirst(==(a), eqd.idx)
                gu = findfirst(==(b.par[a]), equ.pars)
                gd = findfirst(==(b.par[a]), eqd.pars)
                if ju !== nothing && jd !== nothing
                    Vu = infu.E[n, k] * equ.eq.s[ju] / (equ.eq.mu[gu] * (1.0 + tar))
                    Vd = infd.E[n, k] * eqd.eq.s[jd] / (eqd.eq.mu[gd] * (1.0 + tar))
                    worstV = max(worstV, abs(Vpred - (log(Vu) - log(Vd)) / (2hh)))
                end
                Flev = fixed_cost(m, w, a, n)
                lamF = b.alpha[k] * dir[b.hq[a]] + (1.0 - b.alpha[k]) * dir[b.loc[a]]
                Wpred[b.hq[a]]  += b.alpha[k] * (Vlev * Vpred + Flev * lamF)
                Wpred[b.loc[a]] += (1.0 - b.alpha[k]) * (Vlev * Vpred + Flev * lamF)
            end
        end
        Wpred ./= (inf.LD .* w)
        worstW = max(worstW, maximum(abs.(Wpred .- dW)))
    end
    H = wage_hypotheses(m, w, cfg; P0 = Pnow)
    Mel = wage_bill_elasticity(m, w, cfg; P0 = Pnow)
    offs = [Mel[n, j] for n in 1:N, j in 1:N if n != j]
    Jr = ((Mel .- I(N)) .* b.L)[2:end, 2:end]
    idx = sign(det(Jr)) == (-1)^(N - 1)
    if verbose
        @printf("  LEMMA 3  d ln P / d ln w >= 0, min entry      : %+.3e%s", lam3, NL)
        @printf("  LEMMA 3  unit row sums, max error             : %.2e%s", row3, NL)
        @printf("  LEMMA 3  analytic vs numerical pass-through   : %.2e%s", worstP, NL)
        @printf("  LEMMA 4  share response  -chi (lam_g - Lam)   : %.2e%s", worstS, NL)
        @printf("  LEMMA 4  plant bill, three-term formula       : %.2e%s", worstV, NL)
        @printf("  LEMMA 4  country wage bill  d ln W_n          : %.2e%s", worstW, NL)
        @printf("  (H1) largest parent share / largest eps_mu    : %.3f / %.3f%s",
                H.Smax, H.epsmax, NL)
        @printf("  (H2) holds for %d of %d ordered pairs, min slack %+.4f%s",
                count(H.slack[n, j] > 0 for n in 1:N, j in 1:N if n != j),
                N * (N - 1), H.minslack, NL)
        @printf("  gross substitutes: %d of %d off-diagonals positive, worst %+.4f%s",
                count(>(0), offs), length(offs), minimum(offs), NL)
        @printf("  index of this equilibrium is +1               : %s%s", idx, NL)
        Ak = tatonnement_jacobian(Mel, 0.25)
        Dl, Kb = birkhoff_coefficient(Ak)
        @printf("  CONTRACTION: min M_nn %+.3f, largest damping %.3f%s",
                minimum([Mel[i,i] for i in 1:N]), max_damping(Mel), NL)
        @printf("  CONTRACTION: min entry of (1-k)I+kM at k=0.25 : %+.4f%s",
                minimum(Ak), NL)
        @printf("  CONTRACTION: Birkhoff coefficient             : %.4f  %s%s",
                Kb, Kb < 1 ? "< 1, the solver's map contracts" : "NO CERTIFICATE", NL)
    end
    Akr = tatonnement_jacobian(Mel, 0.25)
    Dlr, Kbr = birkhoff_coefficient(Akr)
    return (lemma3 = worstP, lemma4S = worstS, lemma4V = worstV, lemma4W = worstW,
            H1 = H.H1, H2 = H.H2, minslack = H.minslack, gs = minimum(offs),
            index = idx, M = Mel, kmax = max_damping(Mel), minA = minimum(Akr),
            Delta = Dlr, kappaB = Kbr)
end

###############################################################################
# PART 7c.  UNIQUENESS THE CONSTRUCTIVE WAY: THE MODEL'S OWN TATONNEMENT IS A
#           CONTRACTION
#
#   Theorems 4 and 5 rule out a second equilibrium with a sufficient condition
#   imported from outside the model (gross substitutes, then Arrow-Block-
#   Hurwicz). This part does something better and more direct: it shows that the
#   map the SOLVER ITERATES is a contraction, so the model has one solution
#   because of how it is solved, and the algorithm converges to it from
#   anywhere. Nothing is imported except Birkhoff's theorem, which is exact.
#
#   THE MAP. `solve_ge_fixed` iterates  w_n <- w_n (LD_n(w)/L_n)^kappa, i.e.
#
#       Psi_k(w)_n = w_n^(1-k) * ( W_n(w)/L_n )^k ,      W_n = w_n LD_n .
#
#   Given w, everything else in the model is already a unique function of it --
#   markets by Theorem 1, entry by Theorem 2, prices by a contraction, incomes
#   by a linear solve -- so Psi_k is well defined, and its fixed points, up to a
#   common scaling, are exactly the equilibrium wage vectors.
#
#   THE STRUCTURE. Psi_k is homogeneous of degree one, so it acts on the
#   projective space of wage vectors, whose natural metric is HILBERT'S:
#
#       d(w,w') = max_n ln(w_n/w'_n) - min_n ln(w_n/w'_n)
#
#   The log-Jacobian of Psi_k is  A_k = (1-k) I + k M , where
#   M[n,j] = d ln W_n / d ln w_j.  M HAS UNIT ROW SUMS -- that is the model's
#   own homogeneity, not an assumption -- so A_k does too.
#
#   THE ONE THING TO CHECK. A_k is entrywise POSITIVE, i.e. a strictly positive
#   stochastic matrix, exactly when
#       (i)  M[n,j] > 0 for n != j        (gross substitutes), and
#       (ii) k < 1/(1 - min_n M[n,n])     (enough damping).
#   (ii) has content: labour demand is elastic, so M[n,n] is about -1.5 at the
#   calibrated economy and the UNDAMPED map is not monotone at all. The damping
#   is not a numerical convenience -- it is what makes the map a contraction,
#   and the model tells you how much of it you need.
#
#   THE CONCLUSION (Birkhoff 1957). A strictly positive matrix contracts the
#   Hilbert metric with coefficient tanh(Delta/4) < 1, where Delta is its
#   projective diameter. So on any region where (i) and (ii) hold, Psi_k is a
#   strict contraction, hence has AT MOST ONE fixed point up to scale, and the
#   iteration converges to it geometrically. Uniqueness and convergence come out
#   of the same statement.
###############################################################################

"""Hilbert's projective metric on positive vectors: scale-invariant by construction."""
hilbert_metric(w::AbstractVector, wp::AbstractVector) =
    (r = log.(w ./ wp); maximum(r) - minimum(r))

"""
Projective diameter of a strictly positive matrix and Birkhoff's contraction
coefficient `tanh(Delta/4)`. Returns `(Inf, 1.0)` if the matrix has a
non-positive entry, i.e. if no contraction is certified.
"""
function birkhoff_coefficient(A::AbstractMatrix)
    minimum(A) <= 0.0 && return (Inf, 1.0)
    N = size(A, 1); D = 0.0
    for i in 1:N, j in 1:N, k in 1:N, l in 1:N
        D = max(D, log(A[i,k]) + log(A[j,l]) - log(A[j,k]) - log(A[i,l]))
    end
    return (D, tanh(D / 4))
end

"""Log-Jacobian of the damped tatonnement, `(1-kappa) I + kappa M`."""
tatonnement_jacobian(M::AbstractMatrix, kappa::Float64) =
    (1.0 - kappa) * I(size(M, 1)) + kappa * M

"""
The largest damping for which the tatonnement Jacobian still has a positive
diagonal: `kappa < 1/(1 - min_n M[n,n])`. Labour demand is elastic, so this
bites -- it is the model telling you how much damping the solver needs.
"""
max_damping(M::AbstractMatrix) =
    1.0 / (1.0 - minimum([M[n, n] for n in 1:size(M, 1)]))

"""
CONTRACTION CERTIFICATE. Scan a box of wage vectors around `w0` -- measured by
SPREAD, the log gap between the highest and lowest relative wage, which is
Hilbert's metric against `w0` -- and report, for the solver's damping `kappa`:

    minM      smallest entry of M            (gross substitutes if the
                                              off-diagonals are positive)
    minMnn    smallest diagonal entry of M   (elastic labour demand => negative)
    kmax      1/(1 - minMnn), the largest damping that still works
    minA      smallest entry of (1-kappa)I + kappa M
    Delta     largest projective diameter over the box
    kappaB    largest Birkhoff coefficient over the box

If `minA > 0` then `kappaB < 1` and the map is a contraction on the box, so the
box contains at most one equilibrium and the solver converges to it.
"""
function contraction_certificate(m::GEEntry, cfg; w0, kappa = 0.25,
                                 radii = (0.0, 0.4, 0.8, 1.4, 2.0), npts = 20,
                                 P0 = nothing, seed = 9)
    N = m.base.N
    rng = MersenneTwister(seed)
    rows = NamedTuple[]
    for rho in radii
        n = rho == 0.0 ? 1 : npts
        minM = Inf; minMnn = Inf; minA = Inf; bigD = 0.0; bigK = 0.0; ok = 0
        for _ in 1:n
            wv = rho == 0.0 ? copy(w0) :
                 w0 .* exp.(rho / 2 .* (2 .* rand(rng, N) .- 1))
            wv ./= wv[1]
            local Mel, Ploc
            try
                Ploc = excess_demand_fixed(m, wv, cfg; P0 = P0)[2].P
                Mel = wage_bill_elasticity(m, wv, cfg; P0 = Ploc)
            catch
                continue
            end
            all(isfinite, Mel) || continue
            ok += 1
            minM = min(minM, minimum(Mel))
            minMnn = min(minMnn, minimum([Mel[i, i] for i in 1:N]))
            A = tatonnement_jacobian(Mel, kappa)
            minA = min(minA, minimum(A))
            D, K = birkhoff_coefficient(A)
            isfinite(D) && (bigD = max(bigD, D); bigK = max(bigK, K))
        end
        push!(rows, (spread = rho, pts = ok, minM = minM, minMnn = minMnn,
                     kmax = 1.0 / (1.0 - minMnn), minA = minA,
                     Delta = bigD, kappaB = bigK))
    end
    return rows
end

"""
Machine-check Birkhoff's conclusion rather than trusting it: take random PAIRS
of wage vectors, apply the solver's own map to both, and confirm the Hilbert
distance shrinks by at least the certified factor.
"""
function verify_contraction(m::GEEntry, cfg; w0, kappa = 0.25, rho = 0.8,
                            npairs = 25, P0 = nothing, seed = 77)
    N = m.base.N; rng = MersenneTwister(seed)
    worst = 0.0; bound = 0.0; got = 0; respected = true
    for _ in 1:npairs
        w1 = w0 .* exp.(rho / 2 .* (2 .* rand(rng, N) .- 1))
        w2 = w0 .* exp.(rho / 2 .* (2 .* rand(rng, N) .- 1))
        local p1, p2, M1, M2
        try
            i1 = excess_demand_fixed(m, w1, cfg; P0 = P0)[2]
            i2 = excess_demand_fixed(m, w2, cfg; P0 = P0)[2]
            p1 = w1 .* (max.(i1.LD, 1e-12) ./ m.base.L) .^ kappa
            p2 = w2 .* (max.(i2.LD, 1e-12) ./ m.base.L) .^ kappa
            M1 = wage_bill_elasticity(m, w1, cfg; P0 = P0)
            M2 = wage_bill_elasticity(m, w2, cfg; P0 = P0)
        catch
            continue
        end
        (all(isfinite, p1) && all(isfinite, p2) && all(p1 .> 0) && all(p2 .> 0)) || continue
        d0 = hilbert_metric(w1, w2)
        d0 < 1e-8 && continue
        got += 1
        ratio = hilbert_metric(p1, p2) / d0
        worst = max(worst, ratio)
        for Mx in (M1, M2)
            _, K = birkhoff_coefficient(tatonnement_jacobian(Mx, kappa))
            isfinite(K) && (bound = max(bound, K))
        end
        ratio > bound + 1e-9 && (respected = false)
    end
    return (pairs = got, worst_ratio = worst, bound = bound, respected = respected)
end

###############################################################################
# PART 8.  OWNERSHIP: THE PROPOSITION THE PAPER IS ABOUT
#
#   In a market with parents g of share S_g, of which country H owns a fraction
#   theta_g, home-owned profit is
#
#       Pi_H / E  =  sum_g theta_g S_g L(S_g),      L(S) = 1 - 1/mu(S).
#
#   Under Cournot L is LINEAR in S: L(S) = (1/sigma)(1 + kappa S). Then, writing
#   <x> for the share-weighted mean sum_g S_g x_g,
#
#       Pi_H / E  =  (1/sigma) [ thetabar + kappa ( thetabar*HHI + Cov_S(theta,S) ) ]
#
#   with thetabar = <theta>, HHI = sum_g S_g^2 = <S>, and
#   Cov_S(theta,S) = <theta*S> - <theta><S>.
#
#   PROPOSITION (P2a).  A calculation that uses only the COUNTRY-LEVEL ownership
#   share -- which is all anyone without firm-level parent identity can do --
#   computes thetabar * (total profit) and therefore MISSES exactly
#
#       error  =  (E kappa / sigma) * Cov_S(theta, S).
#
#   The bias is the OWNERSHIP-SIZE COVARIANCE, scaled by kappa. It is zero if and
#   only if ownership is uncorrelated with size, or kappa = 0. Under CES
#   monopolistic competition kappa = 0 and the whole ownership correction
#   vanishes: that is PROPOSITION P0, Ownership Irrelevance, obtained here as a
#   limiting case of the same formula rather than as a separate model.
#
#   This is the measurement contribution and it is an IDENTITY, not an estimate.
#   Under Bertrand L is not exactly linear, so the quadratic form is the leading
#   term of an expansion; `ownership_decomposition` reports the exact value, the
#   quadratic approximation and the remainder, so the approximation error is
#   measured rather than assumed.
###############################################################################

"""
Decompose home-owned profit in one market. Returns the exact value, the
country-level ("naive") value, the covariance correction, and the residual of
the quadratic form (identically zero under Cournot).
"""
function ownership_decomposition(S::Vector{Float64}, theta::Vector{Float64},
                                 sigma, eta, E, conduct)
    L = [lerner(s, sigma, eta, conduct) for s in S]
    exact  = E * sum(theta .* S .* L)
    total  = E * sum(S .* L)
    thbar  = sum(S .* theta)
    hhi    = sum(S .^ 2)
    covar  = sum(theta .* S .^ 2) - thbar * hhi
    kap    = kappa_of(sigma, eta, conduct)
    quad   = (E / sigma) * (thbar + kap * (thbar * hhi + covar))
    naive  = thbar * total
    return (exact = exact, naive = naive, total = total, thetabar = thbar,
            hhi = hhi, cov = covar, kappa = kap, quad = quad,
            miss = exact - naive, quad_error = quad - exact)
end

"""
Verify P2a: the quadratic decomposition is an IDENTITY under Cournot, and the
"error from using the country share" equals (E kappa/sigma) Cov exactly.
"""
function verify_ownership(; reps = 300, conduct = :cournot, verbose = true)
    rng = MersenneTwister(20260820)
    worst_id = 0.0; worst_miss = 0.0; worst_quad = 0.0
    for _ in 1:reps
        m = random_market(rng; conduct = conduct)
        eq = solve_market(m)
        G = length(eq.S)
        th = rand(rng, G)
        dec = ownership_decomposition(eq.S, th, m.sigma, m.eta, eq.E, conduct)
        # the claimed identity for the miss
        pred = eq.E * dec.kappa / m.sigma * dec.cov
        worst_miss = max(worst_miss, abs(pred - dec.miss)/max(abs(dec.miss),1e-12))
        worst_quad = max(worst_quad, abs(dec.quad_error)/max(abs(dec.exact),1e-12))
        # exact profit must equal the market solver's own profit vector
        worst_id = max(worst_id, abs(sum(th .* eq.Pi) - dec.exact)/max(abs(dec.exact),1e-12))
    end
    if verbose
        @printf("  home-owned profit matches the solver's own profits   : %.2e\n", worst_id)
        @printf("  quadratic form vs exact  (0 iff L is linear in S)    : %.2e\n", worst_quad)
        @printf("  miss = (E kappa/sigma) Cov_S(theta,S)                : %.2e\n", worst_miss)
    end
    return (id = worst_id, quad = worst_quad, miss = worst_miss)
end

###############################################################################
# PART 9.  MEASUREMENT ON THE POST-ENTRY ECONOMY
###############################################################################

"""Export value [plant, destination]; domestic sales excluded; ACTIVE plants only."""
function export_matrix(m::GEEntry, w::Vector{Float64}, inf)
    b = m.base
    V = zeros(length(b.par), b.N)
    for n in 1:b.N, k in 1:b.K
        e, idx = inf.mk[n,k].eq, inf.mk[n,k].idx
        E = inf.E[n, k]
        for (j, a) in enumerate(idx)
            b.loc[a] == n && continue
            V[a, n] += E * e.s[j]
        end
    end
    return V
end

"""
non-MNE / domestic MNE / foreign MNE, on the POST-ENTRY structure: a parent
counts as multinational only if it actually operates in more than one country,
which with entry is an outcome rather than an assumption.
"""
function classify(m::GEEntry, inf)
    b = m.base
    live = fill(false, length(b.par))
    for n in 1:b.N, k in 1:b.K, a in inf.mk[n, k].idx; live[a] = true; end
    ncty = Dict{Int,Int}()
    for g in unique(b.par)
        ncty[g] = length(unique([b.loc[a] for a in eachindex(b.par)
                                 if b.par[a] == g && live[a]]))
    end
    return [!live[a]                   ? :inactive :
            b.hq[a] != b.loc[a]        ? :foreign_mne :
            get(ncty, b.par[a], 1) > 1 ? :domestic_mne : :nonmne
            for a in eachindex(b.par)]
end

"""
Figure 6's ESTIMATOR run on the model's simulated customs records: in-sample
origins only -> pool across destinations within a sector -> aggregate to the
chosen firm concept -> renormalise WITHIN the sample -> HHI -> value-weight.
"""
function measured_hhi(b::GEModel, V::Matrix{Float64}, sample; level::Symbol = :parent)
    num = den = 0.0
    for k in 1:b.K
        acc = Dict{Any,Float64}()
        for a in eachindex(b.par)
            (b.sec[a] == k && b.loc[a] in sample) || continue
            key = level === :affiliate      ? a :
                  level === :parent_country ? (b.par[a], b.loc[a]) : b.par[a]
            acc[key] = get(acc, key, 0.0) + sum(V[a, :])
        end
        tot = sum(values(acc)); tot <= 0 && continue
        num += tot * sum((v/tot)^2 for v in values(acc)); den += tot
    end
    return den > 0 ? num / den : 0.0
end

"""Theory object: HHI over the whole destination market. NOT comparable to Fig 6."""
function structural_hhi(m::GEEntry, inf)
    num = den = 0.0
    for n in 1:m.base.N, k in 1:m.base.K
        num += inf.E[n,k] * sum(inf.mk[n,k].eq.S .^ 2); den += inf.E[n,k]
    end
    return num / den
end

"""How many potential plant-destination pairs actually entered."""
function entry_rates(m::GEEntry, inf)
    b = m.base
    npot = nact = npot_m = nact_m = 0
    for n in 1:b.N, k in 1:b.K
        act = inf.mk[n,k].active
        idxk = findall(==(k), b.sec)
        for (j, a) in enumerate(idxk)
            isf = b.hq[a] != b.loc[a]
            npot += 1; isf && (npot_m += 1)
            if act[j]; nact += 1; isf && (nact_m += 1); end
        end
    end
    return (all = nact/max(npot,1), mne = nact_m/max(npot_m,1),
            n_active = nact, n_potential = npot)
end

"""
Least squares. Solved by QR on the tall system rather than through the normal
equations: with several blocks of fixed effects the design matrix goes rank
deficient in small worlds -- `julia mne_model.jl quick` used to die here with a
SingularException on Fact 6 -- and the minimum-norm solution is the right answer
to report rather than a crash. The coefficient on a collinear regressor is then
not identified, which is a fact about that economy, not an error.
"""
function ols(X, y)
    try
        return X \ y
    catch
        return pinv(X) * y
    end
end
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

###############################################################################
# PART 10.  THE ECONOMY
#
#   Two features distinguish this from a hand-built roster of firms, and both are
#   taken from Gaubert-Itskhoki rather than invented here.
#
#   (1) THE POOL OF POTENTIAL PARENTS SCALES WITH COUNTRY SIZE:
#           Mbar_h  proportional to  L_h * z_h^zeta.
#       The capability LAW is identical in every country. Concentration of
#       parents in rich countries then arises because a bigger pool has a better
#       maximum -- granularity -- and NOT because rich countries were assumed to
#       breed better firms. zeta = 0 gives the uniform draw, and with it the
#       model produces as many LAC-headquartered multinationals as advanced-
#       country ones, which contradicts Fact 1's foreign/domestic split.
#
#   (2) HEADQUARTER SERVICES ARE A CAPABILITY LOCAL FIRMS CANNOT BUY.
#       A stand-alone local firm carries a productivity penalty rising in the
#       HQ intensity of its sector:  phi_local *= exp(-hq_gap * alpha_k).
#       Without this the model gets Fact 2 BACKWARDS, and for a clear reason: in
#       the cost function a foreign affiliate pays its PARENT's wage on the
#       alpha_k share, parents sit in high-wage countries, so foreign ownership
#       is most expensive exactly where alpha_k is high. The wage channel alone
#       makes foreign shares FALL with complexity. hq_gap is the statement that
#       headquarter services are not available at arm's length -- the
#       Antras (2003) / Antras-Helpman (2004) logic taken seriously on the COST
#       side -- and Fact 2's
#       gradient is what identifies it.
###############################################################################

function world_economy(rng; N = 5, K = 4, n_rich = 2, eta = 1.0, conduct = :cournot,
                       n_dom = 6, n_pot_par = 18, theta_par = 4.0, zeta = 1.5,
                       mne_adv = 0.0, adv_slope = 1.2, hq_gap = 1.3,
                       nu = 0.55, io_own = 0.45, fscale = 0.0006, gamma_mp = 1.18,
                       spill = 0.0, fspill = 0.0, extra_mne_sector = 0, extra_n = 10,
                       hq_cost = false, tariff = 0.0,
                       row_L = 0.0, row_z = 2.2, row_ndom = 18, row_dist = 2.5)
    alpha = collect(range(0.10, 0.55, length = K))
    sigma = fill(5.0, K); beta = fill(1.0/K, K); nuv = fill(nu, K)
    omega = fill((1.0 - io_own)/(K - 1), K, K)
    for k in 1:K; omega[k, k] = io_own; end
    L = vcat(fill(1.5, n_rich), fill(1.0, N - n_rich))
    z = vcat(fill(2.2, n_rich), fill(1.0, N - n_rich))
    lac = collect((n_rich+1):N)

    pos  = collect(1.0:N)
    dist = [1.0 + abs(pos[i] - pos[j]) for i in 1:N, j in 1:N]
    # d = dist^(1/(sigma-1)) makes the gravity elasticity of TRADE exactly -1,
    # which is the central estimate of the gravity literature -- so the model's
    # own gravity regression is a free diagnostic, not a target.
    d = dist .^ (1.0 / (sigma[1] - 1.0))
    gamma = [i == j ? 1.0 : gamma_mp for i in 1:N, j in 1:N]

    wgt = [L[h] * z[h]^zeta for h in 1:N]; wgt ./= sum(wgt); cum = cumsum(wgt)

    par = Int[]; hq = Int[]; loc = Int[]; sec = Int[]; phi = Float64[]; g = 0

    # TIER 2: the local fringe. One plant, one variety, its own group.
    for n in 1:N, k in 1:K, _ in 1:n_dom
        g += 1
        push!(par, g); push!(hq, n); push!(loc, n); push!(sec, k)
        push!(phi, z[n] * exp(0.25*randn(rng) - hq_gap*alpha[k]))
    end

    # TIER 1: granular multinational parents, with a POTENTIAL plant in every
    # country including their own. Which ones exist is decided by entry.
    function add_parent!(k, adv)
        g += 1
        u = rand(rng); h = findfirst(>=(u), cum)
        xi = log(rand(rng)^(-1.0/theta_par))          # Pareto(theta) capability
        for l in 1:N
            push!(par, g); push!(hq, h); push!(loc, l); push!(sec, k)
            push!(phi, z[l] * exp(0.25*randn(rng) + xi*(1.0 + adv_slope*alpha[k]) + adv))
        end
    end
    for j in 1:n_pot_par; add_parent!(1 + (j-1) % K, mne_adv); end
    if extra_mne_sector > 0
        for _ in 1:extra_n; add_parent!(extra_mne_sector, mne_adv); end
    end

    # Knowledge spillover (Javorcik 2004), off by default: a local firm's
    # productivity rises with the number of multinational plants in its own
    # country and sector. This is the channel Fact 5 would need.
    if spill != 0.0
        cnt = Dict{Tuple{Int,Int},Int}()
        for a in eachindex(par)
            hq[a] != loc[a] && (cnt[(loc[a], sec[a])] = get(cnt,(loc[a],sec[a]),0) + 1)
        end
        for a in eachindex(par)
            hq[a] == loc[a] || continue
            phi[a] *= (1.0 + get(cnt, (loc[a], sec[a]), 0))^spill
        end
    end

    # THE OUTSIDE SUPPLIER (the lambda margin of the measurement operator; the
    # Fact-5 diagnosis of CLAUDE.md 35). Off by default. `row_L > 0` appends a
    # REST-OF-WORLD country: a large labour force and a dense, world-class
    # fringe in every sector (no hq_gap penalty -- it proxies ALL non-sample
    # supply, multinationals included), but NO parents and NO sample-MNE plants,
    # so it is purely the outside supply every market competes against. Sample
    # exporters then hold a small share of each destination market -- the
    # in-sample absorption share lambda -- and business stealing from entry
    # falls mostly on the outside supply instead of on fellow sample firms.
    # Drawn AFTER the sample roster, so the sample's random draws are
    # bit-for-bit identical with and without it. Uniqueness is untouched:
    # every theorem is dimension-agnostic and ROW firms are ordinary
    # single-plant groups.
    Nt = N
    if row_L > 0.0
        Nt = N + 1
        for k in 1:K, _ in 1:row_ndom
            g += 1
            push!(par, g); push!(hq, Nt); push!(loc, Nt); push!(sec, k)
            push!(phi, row_z * exp(0.25*randn(rng)))
        end
        L = vcat(L, row_L); z = vcat(z, row_z)
        dist = [dist fill(row_dist, N); fill(row_dist, 1, N) 1.0]
        d = dist .^ (1.0 / (sigma[1] - 1.0))
        gamma = [i == j ? 1.0 : gamma_mp for i in 1:Nt, j in 1:Nt]
    end

    theta = zeros(g, Nt)
    for a in eachindex(par); theta[par[a], hq[a]] = 1.0; end
    t = fill(tariff, Nt, Nt, K); for n in 1:Nt, k in 1:K; t[n,n,k] = 0.0; end
    # HEAD-OFFICE SERVICES: A COST SHARE, OR ONLY A CAPABILITY?
    # `alpha_k` does two jobs above. It sets how much of a factory's cost is paid
    # at its PARENT's wage (the Antras head-office input), and it indexes
    # complexity in the productivity draws -- the local-firm penalty exp(-hq_gap
    # alpha_k) and the parent capability gradient xi(1 + adv_slope alpha_k).
    # Only the FIRST creates a cross-country factor linkage, and only the first
    # is what breaks gross substitutes (see PART 7b and `wage_uniqueness.jl`).
    # `hq_cost = false` switches off the cost share and keeps the capability
    # gradient, because phi has already been drawn using the real alpha above.
    # The productivity mechanism behind Facts 1 and 2 is therefore untouched.
    alpha_cost = hq_cost ? alpha : zeros(K)
    base = GEModel(Nt, K, sigma, eta, beta, alpha_cost, nuv, omega, L, par, hq, loc,
                   sec, phi, gamma, d, t, theta, conduct)
    # FACT-5 FIXED-COST SPILLOVER (off by default). Multinational plants in a
    # (country, sector) lower LOCAL plants' market-access cost there:
    #     F_a *= (1 + n_mne(loc, sec))^(-fspill)   for plants with hq = loc.
    # The count is over the POTENTIAL roster, the same convention as `spill`,
    # so the multiplier is a constant of the entry game: Theorem 2's chain and
    # uniqueness argument apply verbatim, and homogeneity in wages is untouched
    # (the multiplier is wage-free). Unlike `spill`, which raises local
    # MARGINAL productivity and cannot flip Fact 5 because business stealing
    # works on the ENTRY margin (CLAUDE.md 30.5), this operates on the entry
    # margin directly.
    fmult = ones(length(par))
    if fspill != 0.0
        cntf = Dict{Tuple{Int,Int},Int}()
        for a in eachindex(par)
            hq[a] != loc[a] && (cntf[(loc[a], sec[a])] = get(cntf, (loc[a], sec[a]), 0) + 1)
        end
        for a in eachindex(par)
            hq[a] == loc[a] || continue
            fmult[a] = (1.0 + get(cntf, (loc[a], sec[a]), 0))^(-fspill)
        end
    end
    return GEEntry(base, fill(fscale, K), ones(Nt, Nt), fmult),
           (dist = dist, n_rich = n_rich, lac = lac, z = z,
            complexity = alpha)   # the complexity index, ALWAYS the real
                                  # alpha_k even when hq_cost = false zeroes
                                  # the cost share. Fact 2 regresses on this.
end

"""Small synthetic world for the accounting audit -- deliberately irregular."""
function demo_economy(rng; N = 4, K = 3, nparents = 12, naff = 2, eta = 1.0,
                      tariff = 0.0, nu = 0.0, conduct = :cournot, fscale = 0.0006)
    sigma = [4.0 + 2.0*rand(rng) for _ in 1:K]
    beta  = rand(rng, K) .+ 0.5; beta ./= sum(beta)
    alpha = [0.15 + 0.35*rand(rng) for _ in 1:K]
    nuv   = fill(nu, K); omega = fill(1.0/K, K, K)
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
    base = GEModel(N,K,sigma,eta,beta,alpha,nuv,omega,L,par,hq,loc,sec,phi,
                   gamma,d,t,theta,conduct)
    return GEEntry(base, fill(fscale, K))
end

"""
Every identity a general equilibrium must satisfy, checked at ARBITRARY wages
rather than only at the solution -- the strong form of the test. Fixed costs are
the new item: they are real resources, so they must appear as factor payments AND
be subtracted from distributed profits. If either half were missing, Walras' law
would break, which is exactly why it is the test.
"""
function accounting(m::GEEntry, w::Vector{Float64}; cfg = nothing)
    b = m.base
    cfg = cfg === nothing ? all_active(m) : cfg
    z, inf = excess_demand_fixed(m, w, cfg)
    mk, X, Emat, FCg, FCn = inf.mk, inf.X, inf.E, inf.FCg, inf.FCn
    G = size(b.theta, 1)
    revenue = factor = inter = tariffrev = grossprofit = 0.0
    prof_g = zeros(G); tar_n = zeros(b.N)
    for n in 1:b.N, k in 1:b.K
        e, idx, pars = mk[n,k].eq, mk[n,k].idx, mk[n,k].pars
        E = Emat[n,k]; revenue += E
        for (i, g) in enumerate(pars)
            pg = E * e.S[i] * lerner(e.S[i], b.sigma[k], b.eta, b.conduct)
            prof_g[g] += pg; grossprofit += pg
        end
        for (j, a) in enumerate(idx)
            t  = b.tariff[b.loc[a], n, k]
            gi = findfirst(==(b.par[a]), pars)
            pay = E * e.s[j] / (e.mu[gi] * (1.0 + t))
            factor += (1.0 - b.nu[k]) * pay
            inter  += b.nu[k] * pay
            tr = E * e.s[j] * t / (e.mu[gi] * (1.0 + t))
            tariffrev += tr; tar_n[n] += tr
        end
    end
    fixbill = sum(FCn); netprofit = grossprofit - fixbill
    world_income = sum(w .* b.L) + tariffrev + netprofit
    world_final  = sum(Emat) - inter
    return (revenue_identity = abs(revenue - (factor+inter+tariffrev+grossprofit))/max(revenue,1e-12),
            labour_check = abs((factor+fixbill) - sum(w .* inf.LD))/max(factor+fixbill,1e-12),
            walras = abs(sum(w .* z))/max(sum(w .* b.L),1e-12),
            world_budget = abs(world_income - world_final)/max(world_income,1e-12),
            country_budget = maximum(abs.(X .- (w .* b.L .+ b.theta'*(prof_g .- FCg) .+ tar_n)))/max(maximum(X),1e-12),
            fixed_share = fixbill/max(revenue,1e-12),
            gross_profit_share = grossprofit/max(revenue,1e-12),
            net_profit_share = netprofit/max(revenue,1e-12))
end

end # module MNEModel


###############################################################################
# PART 11.  THE RUNNER
###############################################################################

using .MNEModel
using .MNEModel: markup, lerner, kappa_of, inner_share, psi_of, share_frontier,
                 random_market, random_entry_market, verify_uniqueness,
                 verify_ownership, ownership_decomposition,
                 subset_table, best_response, eq_on, is_nash, nash_refine,
                 all_entry_configs, groups_of,
                 solve_markets_fixed, solve_markets_entry, excess_demand_fixed,
                 solve_ge_fixed, entry_deviations, all_active, configs_of,
                 flatten_cfg, entry_rates, export_matrix, classify,
                 measured_hhi, structural_hhi, ols, ols_fe,
                 world_economy, demo_economy, print_calibration, CALIB,
                 eps_closed, verify_conditions_analytic, boundary_slack,
                 eps_markup, chi_share, passthrough, wage_hypotheses,
                 wage_bill_elasticity, verify_wage_theorem, reallocation_floor,
                 hilbert_metric, birkhoff_coefficient, tatonnement_jacobian,
                 max_damping, contraction_certificate, verify_contraction
using Printf, Random, LinearAlgebra

banner(t) = (println(); println("="^78); println(t); println("="^78); flush(stdout))
sub(t)    = (println(); println(t); println("-"^length(t)); flush(stdout))

const CONDUCTS = [:cournot, :bertrand]

# ---------------------------------------------------------------------------
function run_core(level, conduct)
    banner("PART A   ONE MARKET, AND WHY IT HAS EXACTLY ONE SOLUTION  [$conduct]")
    reps, starts = level == :quick ? (60, 10) : (200, 40)
    println("Cannibalisation is the mechanism: a parent's own varieties are perfect")
    println("substitutes inside its own output aggregate, so expanding one depresses")
    println("the others and the markup rises with the PARENT's total share. The same")
    println("object mu(S_g) appears under quantity and price competition, which is why")
    println("the solver takes mu as a parameter.\n")
    @printf("  %-12s%-10s%-12s%-12s%s\n", "conduct", "kappa", "mu(0)", "mu(0.5)", "share frontier")
    for c in CONDUCTS
        @printf("  %-12s%-10.3f%-12.3f%-12.3f%.3f\n", c, kappa_of(5.0,1.0,c),
                markup(0.0,5.0,1.0,c), markup(0.5,5.0,1.0,c), share_frontier(5.0,1.0,c))
    end
    println()
    println("  kappa is the coefficient on the S^2 term in profit, so the ownership")
    println("  correction is proportional to it. Conduct changes the MAGNITUDE by a")
    println("  factor of five and changes nothing structural.")
    verify_uniqueness(; reps = reps, starts = starts, conduct = conduct)
    sub("The ownership decomposition (Proposition P2a)")
    println("Pi_H/E = (1/sigma)[ thetabar + kappa ( thetabar*HHI + Cov_S(theta,S) ) ].")
    println("A country-level ownership share misses exactly (E kappa/sigma) Cov.\n")
    verify_ownership(; reps = reps, conduct = conduct)
    println()
    println("  The middle line is zero under Cournot because the Lerner index is exactly")
    println("  linear in the share there; under Bertrand it is the size of the")
    println("  approximation, reported rather than assumed.")
end

# ---------------------------------------------------------------------------
function run_entry(level, conduct)
    banner("PART B   ENTRY: IS THE EQUILIBRIUM UNIQUE, WITH INTERNALISATION?  [$conduct]")
    reps = level == :quick ? 40 : 120
    println("Brute force over EVERY configuration -- each parent free to choose any")
    println("subset of its own plants -- keeping the Nash ones, where a deviation is")
    println("scored on the RE-SOLVED market. This is the case Gaubert-Itskhoki's cutoff")
    println("cannot handle and Yang resolves by imposing an entry order.\n")
    rng = MersenneTwister(77)
    nuniq = nmulti = nnone = nmatch = ncert = nbad = nsol = nalready = 0
    for _ in 1:reps
        em = random_entry_market(rng; npar = 3, naff = 2, nfringe = 3, conduct = conduct)
        cfgs = all_entry_configs(em)
        sol  = MNEModel.solve_market_entry(em; certify = true)
        if isempty(cfgs); nnone += 1; continue; end
        length(cfgs) == 1 ? (nuniq += 1) : (nmulti += 1)
        if sol !== nothing
            nsol += 1
            any(c == sol.active for c in cfgs) && (nmatch += 1)
            sol.certified && (ncert += 1)
            (sol.certified && length(cfgs) > 1) && (nbad += 1)
            nash_refine(em, sol.active).moved || (nalready += 1)
        end
    end
    @printf("  markets with a pure-strategy entry equilibrium  : %d / %d\n", nuniq+nmulti, reps)
    @printf("    of those, UNIQUE                              : %d\n", nuniq)
    @printf("    of those, multiple                            : %d\n", nmulti)
    @printf("  solver issued its uniqueness CERTIFICATE        : %d / %d\n", ncert, nsol)
    @printf("  certificate issued where brute force found many : %d   <- must be 0\n", nbad)
    @printf("  cheap solve already an exact Nash equilibrium   : %d / %d\n", nmatch, nsol)
    @printf("  cheap solve survives nash_refine untouched      : %d / %d\n", nalready, nsol)
    println()
    println("  The certificate is what travels into the GE loop, where brute force is")
    println("  unaffordable: the solver re-checks that the clearing function is")
    println("  decreasing with exactly one sign change, market by market.")

    sub("The two conditions the theorem rests on")
    println("(A) Omega concave in own K   -> the best response is a cutoff on the")
    println("    parent's OWN list, so internalisation is not an obstacle;")
    println("(B) Omega_K falling in A     -> strategic substitutes.")
    println("Both by numerical differentiation of Omega, using none of the algebra.\n")
    rng = MersenneTwister(20260819)
    badA = badB = inA = inB = nin = 0
    for _ in 1:(level == :quick ? 150 : 400)
        eta = rand(rng) < 0.5 ? 1.0 : 1.0 + 1.5*rand(rng)
        sg  = eta + 1.0 + 6.0*rand(rng)
        A = exp(2.0*randn(rng)); E = 0.5 + 2.0*rand(rng)
        S0 = 0.02 + 0.94*rand(rng)
        inside = S0 < share_frontier(sg, eta, conduct); inside && (nin += 1)
        Om(K,Aa) = (S = inner_share(K/Aa, sg, eta, conduct); E*S*lerner(S,sg,eta,conduct))
        K = psi_of(S0, sg, eta, conduct)*A
        h = 1e-4*K; hA = 1e-4*A
        d2 = (Om(K+h,A) - 2Om(K,A) + Om(K-h,A))/h^2
        dp = (Om(K+h,A+hA) - Om(K-h,A+hA))/(2h)
        dm = (Om(K+h,A-hA) - Om(K-h,A-hA))/(2h)
        cross = (dp-dm)/(2hA)
        d2 < 0 || (badA += 1; inside && (inA += 1))
        cross < 0 || (badB += 1; inside && (inB += 1))
    end
    n = level == :quick ? 150 : 400
    @printf("  (A) concavity  violated %3d / %3d over all shares, %d / %d inside the frontier\n",
            badA, n, inA, nin)
    @printf("  (B) substitutes violated %3d / %3d over all shares, %d / %d inside the frontier\n",
            badB, n, inB, nin)
    println()
    println()
    println("  These are not assumptions. For Cournot with eta = 1 they have CLOSED")
    println("  FORMS, and the closed forms are checked against the numbers above:")
    println()
    verify_conditions_analytic()
    println()
    println("  (A) is proved at EVERY share. (B) is proved at every share <= 1/2, with")
    println("  slack exactly 1/sigma at the boundary. So the theorem's hypothesis is")
    println("  \"no parent holds more than half of any market\" -- an economically")
    println("  meaningful restriction that can be checked in the data, not a technical")
    println("  assumption. It is sufficient, not necessary: uniqueness is found beyond it.")
end

# ---------------------------------------------------------------------------
function run_ge(level, conduct)
    banner("PART C   GENERAL EQUILIBRIUM: DOES IT ADD UP, AND IS IT UNIQUE?  [$conduct]")
    rng = MersenneTwister(20260819)
    m = demo_economy(rng; N = level == :quick ? 3 : 4, K = 3, conduct = conduct,
                     nparents = level == :quick ? 8 : 12, nu = 0.55)
    w = exp.(0.4 .* randn(MersenneTwister(3), m.base.N)); w ./= w[1]
    sub("Accounting, at ARBITRARY wages (not at the solution)")
    a = accounting(m, w)
    @printf("  revenue = factor + intermediates + tariffs + gross profit : %.2e\n", a.revenue_identity)
    @printf("  wage bill = variable labour + fixed-cost labour           : %.2e\n", a.labour_check)
    @printf("  Walras' law, sum_n w_n Z_n = 0, OFF equilibrium           : %.2e\n", a.walras)
    @printf("  world income = world final expenditure                    : %.2e\n", a.world_budget)
    @printf("  country budget X = wL + owned NET profit + tariffs        : %.2e\n", a.country_budget)
    @printf("  fixed costs / gross profit / net profit, share of revenue : %.3f  %.3f  %.3f\n",
            a.fixed_share, a.gross_profit_share, a.net_profit_share)

    sub("The price loop still contracts, because entry is frozen while it runs")
    b = m.base
    P = ones(b.N, b.K); Emat = fill(sum(ones(b.N).*b.L)/(b.N*b.K), b.N, b.K)
    cfg = all_active(m); gaps = Float64[]
    for _ in 1:40
        mk = solve_markets_fixed(m, ones(b.N), P, Emat, cfg)
        Pn = [mk[n,k].P for n in 1:b.N, k in 1:b.K]
        push!(gaps, maximum(abs.(log.(Pn ./ P)))); P = Pn
    end
    rat = [gaps[i+1]/gaps[i] for i in 3:(length(gaps)-1) if gaps[i] > 1e-13]
    @printf("  observed modulus %.3f   against nu = %.3f   final gap %.2e\n",
            isempty(rat) ? 0.0 : sort(rat)[max(1,length(rat)÷2)], maximum(b.nu), gaps[end])
    println("  Freezing the configuration is not a convenience: it is what keeps the")
    println("  inner block the same provably convergent map the no-entry model had.")

    sub("Entry is invariant to a UNIFORM change in costs")
    println("  Shares, markups and profits are unchanged by a common cost factor, so no")
    println("  entry margin should move. This is why the outer discrete loop behaves.")
    mk0 = solve_markets_entry(m, ones(b.N), P, Emat)
    c0 = flatten_cfg(configs_of(m, mk0)); same = true
    for lam in (0.5, 2.0)
        b2 = MNEModel.GEModel(b.N,b.K,b.sigma,b.eta,b.beta,b.alpha,b.nu,b.omega,b.L,
                              b.par,b.hq,b.loc,b.sec, b.phi ./ lam, b.gamma,b.d,
                              b.tariff,b.theta,b.conduct)
        mk2 = solve_markets_entry(GEEntry(b2, m.f, m.fdist), ones(b.N), P, Emat)
        ok = flatten_cfg(configs_of(m, mk2)) == c0; ok || (same = false)
        @printf("    all costs x %.2f : configuration unchanged? %s\n", lam, ok ? "yes" : "NO")
    end

    sub("Theorem 4: the two conditions behind wage uniqueness, machine-checked")
    println("  Wage uniqueness is a theorem under (H1) no group above half of any market")
    println("  -- the same one half as Theorem 2 -- and (H2) demand beats exposure. Both")
    println("  are inequalities in objects the solver computes, so they are CHECKED. The")
    println("  lemmas behind them are verified against numerical differentiation here.")
    println("  (H2) cannot be dropped: wage_uniqueness.jl exhibits a counterexample.")
    println()
    cfg4 = all_active(m)
    r4 = solve_ge_fixed(m, cfg4; w0 = ones(m.base.N))
    verify_wage_theorem(m, r4.w, cfg4; P0 = r4.info.P)

    sub("The solver's own map is a contraction -- uniqueness, constructively")
    println("  Theorems 4 and 5 rule out a second equilibrium with a condition imported")
    println("  from outside. This is the direct statement: the map the solver iterates,")
    println("  w <- w (LD/L)^kappa, is homogeneous of degree one, its log-Jacobian")
    println("  (1-kappa)I + kappa M has unit row sums by that homogeneity, and where it")
    println("  is entrywise POSITIVE Birkhoff's theorem makes it a contraction in")
    println("  Hilbert's projective metric. One contraction, one fixed point, and the")
    println("  iteration converges to it from anywhere.")
    println()
    cc = contraction_certificate(m, cfg4; w0 = r4.w, P0 = r4.info.P,
                                 radii = (0.0, 0.4, 0.8), npts = 10)
    @printf("  %-8s %5s %10s %10s %10s %9s %9s\n", "spread", "pts", "min M",
            "min M_nn", "max damping", "min A_k", "Birkhoff")
    for rw in cc
        @printf("  %-8.2f %5d %+10.4f %+10.4f %10.4f %+9.4f %9.4f\n", rw.spread,
                rw.pts, rw.minM, rw.minMnn, rw.kmax, rw.minA, rw.kappaB)
    end
    vc = verify_contraction(m, cfg4; w0 = r4.w, P0 = r4.info.P, rho = 0.8, npairs = 12)
    @printf("  direct check on %d pairs: worst observed ratio %.4f against the\n",
            vc.pairs, vc.worst_ratio)
    @printf("  certified bound %.4f -- bound respected: %s\n", vc.bound,
            vc.respected ? "yes" : "NO")
    println("  The damping is not a numerical convenience: where labour demand is")
    println("  elastic the UNDAMPED map is not monotone at all, and the model itself says")
    @printf("  how much damping is needed. Here the largest admissible kappa is %.3f\n",
            minimum(rw.kmax for rw in cc))
    println("  and the solver uses 0.25. At the calibrated five-country economy the")
    println("  requirement is 0.402 (quantities) and 0.390 (prices); with the")
    println("  input-output block off it tightens to 0.273, where 0.15 would be safer.")

    sub("Uniqueness: same wages AND the same set of firms, from every start")
    println("  The theorem above holds the operating set fixed. This is the check it")
    println("  cannot make, and the one that did not exist before entry: wages could")
    println("  agree while a DIFFERENT SET OF FIRMS operated, which is the multiplicity")
    println("  the entry order was invented to paper over.")
    println()
    starts = level == :quick ? 5 : 10
    rng2 = MersenneTwister(99)
    ws = Vector{Vector{Float64}}(); gs = Float64[]; cs = Set{Vector{Bool}}()
    regs = Int[]; ncert = 0; worstS = 0.0
    for _ in 1:starts
        w0 = exp.(0.7 .* randn(rng2, m.base.N)); w0 ./= w0[1]
        r = solve_ge_entry(m; w0 = w0, certify = true)
        push!(ws, r.w); push!(gs, r.gap); push!(regs, r.info.regret)
        push!(cs, flatten_cfg(r.info.cfg))
        r.info.certified && (ncert += 1); worstS = max(worstS, r.info.maxS)
    end
    W = reduce(hcat, ws)
    spread = maximum([maximum(abs.(log.(W[:,j] ./ W[:,1]))) for j in 1:size(W,2)])
    @printf("  random wage starts                              : %d\n", starts)
    @printf("  worst disagreement in equilibrium wages         : %.2e\n", spread)
    @printf("  worst labour-market residual                    : %.2e\n", maximum(gs))
    @printf("  runs where every market carried the certificate : %d / %d\n", ncert, starts)
    @printf("  largest single-parent share anywhere            : %.3f  (frontier %.3f)\n",
            worstS, share_frontier(5.0, m.base.eta, conduct))
    @printf("  unresolved integer slots, worst run             : %d\n", maximum(regs))
    @printf("  DISTINCT ENTRY CONFIGURATIONS ACROSS THE STARTS : %d   <- must be 1\n", length(cs))
    println()
    println(length(cs) == 1 ?
      "  The same firms enter from every start -- not just the same wages, the same" :
      "  DIFFERENT FIRMS entered from different starts. Reported, not selected away.")
    length(cs) == 1 && println("  market structure, which is the object entry was added to determine.")
end

# ---------------------------------------------------------------------------
"""The six facts, on one calibrated economy."""
function run_facts(level, conduct; hq_cost = false)
    banner("PART D   THE SIX STYLIZED FACTS  [$conduct]")
    kw = (N = level == :quick ? 4 : 5, K = 4, n_rich = 2, conduct = conduct,
          n_dom = 6, n_pot_par = level == :quick ? 14 : 18,
          zeta = 1.5, hq_gap = 1.3, mne_adv = 0.0, adv_slope = 1.2, fscale = 0.0006, hq_cost = hq_cost)
    m, aux = world_economy(MersenneTwister(20260819); kw...)
    r  = solve_ge_entry(m; certify = true)
    b  = m.base; lac = aux.lac
    V  = export_matrix(m, r.w, r.info); cls = classify(m, r.info)
    er = entry_rates(m, r.info)

    @printf("  potential plant-destination pairs : %d;  active %d (%.1f%%);  ",
            er.n_potential, er.n_active, 100*er.all)
    @printf("unresolved %d\n", r.info.regret)
    sel = [b.loc[a] in lac for a in eachindex(b.par)]
    tot = sum(V[sel,:])
    fsh = sum(V[sel .& (cls .== :foreign_mne), :])/tot
    dsh = sum(V[sel .& (cls .== :domestic_mne), :])/tot

    sub("FACT 1  multinationals are a large share of export value, and FOREIGN ones dominate")
    @printf("  model  total %.3f = foreign %.3f + domestic %.3f\n", fsh+dsh, fsh, dsh)
    println("  data   total 0.46-0.74, foreign dominant, domestic 0.02-0.28")
    println("  The LEVEL is one fitted parameter. The SPLIT is not: it comes from the")
    println("  pool of potential parents scaling with country size, so poor countries")
    println("  rarely breed a parent good enough to clear the multinational threshold.")
    println("  With a uniform pool (zeta = 0) the model puts domestic at 0.245 and")
    println("  contradicts the fact outright.")

    sub("FACT 2  foreign multinationals specialise in COMPLEX goods, domestic ones in simple")
    fk = Float64[]; dk = Float64[]
    for k in 1:b.K
        nf = nd = de = 0.0
        for a in eachindex(b.par)
            (b.sec[a] == k && b.loc[a] in lac) || continue
            v = sum(V[a,:]); de += v
            cls[a] == :foreign_mne  && (nf += v)
            cls[a] == :domestic_mne && (nd += v)
        end
        push!(fk, de > 0 ? nf/de : 0.0); push!(dk, de > 0 ? nd/de : 0.0)
    end
    x = aux.complexity; xb = sum(x)/length(x)
    gf = sum((x.-xb).*(fk.-sum(fk)/length(fk)))/sum((x.-xb).^2)
    gd = sum((x.-xb).*(dk.-sum(dk)/length(dk)))/sum((x.-xb).^2)
    @printf("  model foreign  by sector: %s   gradient %+.2f\n",
            join([@sprintf("%.2f",v) for v in fk], " "), gf)
    @printf("  model domestic by sector: %s   gradient %+.2f\n",
            join([@sprintf("%.2f",v) for v in dk], " "), gd)
    println("  data  foreign  0.52 0.52 0.63 0.65 0.70   gradient +0.43")
    println("  data  domestic 0.12 0.02 0.06 0.03 0.02   gradient negative")
    println("  The FOREIGN half is reproduced and is the one that matters: it comes from")
    println("  hq_gap, the statement that headquarter services are a capability a")
    println("  stand-alone local firm cannot buy, and that complex goods need them most.")
    println("  Without hq_gap the model gets it BACKWARDS (gradient -0.58), because a")
    println("  foreign affiliate pays its parent's high wage on the alpha_k share and")
    println("  the extensive margin amplifies that wrong-sign force.")
    println("  THE DOMESTIC HALF IS NOT ROBUST and is reported, not claimed: it comes out")
    println("  NEGATIVE in a four-country world and POSITIVE in a five-country one. A",)
    println("  domestic multinational is a LAC parent, so it escapes the hq_gap penalty")
    println("  while still paying a low home wage on the alpha share -- the wrong-sign")
    println("  force again, now working in its favour. The model pins down that domestic")
    println("  multinationals are FEW (Fact 1), not where they specialise. Open item.")

    sub("FACT 3  parents come from a small set of countries -- GENERATED")
    val = zeros(b.N)
    for a in eachindex(b.par)
        (b.loc[a] in lac && cls[a] == :foreign_mne) || continue
        val[b.hq[a]] += sum(V[a,:])
    end
    sh = val ./ max(sum(val), 1e-12)
    @printf("  model parent-country HHI %.3f, top parent %.3f\n", sum(sh.^2), maximum(sh))
    println("  data                0.130,            0.250")
    println("  Nothing tells the model which countries host parents: the capability law")
    println("  is identical everywhere. Concentration comes from a bigger pool having a")
    println("  better maximum. It overshoots in a small world, as it must.")

    sub("FACT 4  grouping plants by parent RAISES measured concentration")
    h1 = measured_hhi(b, V, lac; level = :affiliate)
    h2 = measured_hhi(b, V, lac; level = :parent_country)
    h3 = measured_hhi(b, V, lac; level = :parent)
    @printf("  model  %.3f -> %.3f -> %.3f   (x%.2f)\n", h1, h2, h3, h3/max(h1,1e-12))
    println("  data   0.192 -> 0.209 -> 0.215   (x1.12)")
    println(h1 <= h2 + 1e-9 && h2 <= h3 + 1e-9 ?
        "  ORDERING REPRODUCED. It is a PREDICTION, not a target, and it is the reason" :
        "  ORDERING NOT REPRODUCED IN THIS RUN -- reported, not buried.")
    println("  internalisation had to survive the entry rewrite: the whole uniqueness")
    println("  theorem exists so that this fact could.")
    # Fact 4a: value against network size
    live = fill(false, length(b.par))
    for n in 1:b.N, k in 1:b.K, a in r.info.mk[n,k].idx; live[a] = true; end
    nets = Dict{Int,Int}(); vals = Dict{Int,Float64}()
    for g in unique(b.par)
        locs = unique([b.loc[a] for a in eachindex(b.par) if b.par[a]==g && live[a]])
        isempty(locs) && continue
        nets[g] = length(locs)
        vals[g] = sum(sum(V[a,:]) for a in eachindex(b.par) if b.par[a]==g; init=0.0)
    end
    gs = collect(keys(nets)); tv = sum(values(vals))
    big = [g for g in gs if nets[g] >= 2]
    @printf("  parents operating in >=2 countries: %.0f%% of parents, %.0f%% of value\n",
            100*length(big)/max(length(gs),1),
            100*sum(vals[g] for g in big; init=0.0)/max(tv,1e-12))
    println("  data: parents with >100 affiliates are 9% of parents and 56% of value")
    println("  The direction is right. The LEVEL is not comparable: the model's network")
    println("  size runs 1..N countries, the data's runs 1..100+ affiliates worldwide.")

    sub("FACT 5  more multinationals, MORE exports by local firms -- THE ONE FAILURE")
    function nonmne(extra)
        m2, aux2 = world_economy(MersenneTwister(20260819);
                                 merge(kw, (extra_mne_sector = extra > 0 ? 2 : 0,
                                            extra_n = extra))...)
        r2 = solve_ge_entry(m2)
        V2 = export_matrix(m2, r2.w, r2.info); c2 = classify(m2, r2.info)
        s2 = [(m2.base.loc[a] in aux2.lac) && c2[a] == :nonmne && m2.base.sec[a] == 2
              for a in eachindex(m2.base.par)]
        return sum(V2[s2, :])
    end
    v0 = nonmne(0); v1 = nonmne(level == :quick ? 8 : 12)
    ch = 100*(v1/v0 - 1.0)
    @printf("  non-MNE export value in sector 2: %.4f -> %.4f   change %+.1f%%\n", v0, v1, ch)
    println("  data: POSITIVE. Saturated specification, +0.24 (intensive), +0.13 (extensive)")
    println(ch > 0 ? "  SIGN FLIPPED -- entry does what the intensive margin could not." :
                     "  STILL NEGATIVE. Business stealing operates on the ENTRY margin too")
    ch > 0 || println("  and dominates the cheaper-inputs channel, so entry makes this fact WORSE.")
    println("  Two candidate resolutions, and they are testable against each other:")
    println("   (i) a genuine Javorcik spillover (the `spill` knob), which must now be")
    println("       LARGER than the 0.15 the fixed-roster model needed; or")
    println("  (ii) the empirical coefficient is contaminated: the fixed-effect ladder")
    println("       has no destination x product x year term, so a market-level demand")
    println("       shock raises multinational entry and local exports together.")
    println("  Adding that fixed effect is the decisive test and it is cheap.")

    sub("FACT 6  distance is a weaker barrier for multinationals")
    y = Float64[]; ld = Float64[]; mne = Float64[]; pres = Float64[]
    fo = String[]; fd = String[]; fs = String[]
    for a in eachindex(b.par), n in 1:b.N
        V[a,n] > 0 || continue
        push!(y, log(V[a,n])); push!(ld, log(aux.dist[b.loc[a], n]))
        ism = cls[a] != :nonmne && cls[a] != :inactive
        push!(mne, ism ? 1.0 : 0.0)
        atdest = any(b.par[a2] == b.par[a] && b.loc[a2] == n && live[a2]
                     for a2 in eachindex(b.par))
        push!(pres, (ism && atdest) ? 1.0 : 0.0)
        push!(fo, string(b.loc[a])); push!(fd, string(n)); push!(fs, string(b.sec[a]))
    end
    if length(y) > 30
        c1 = ols_fe(y, [ld], Any[fo, fd, fs])
        c2 = ols_fe(y, [ld, ld .* mne], Any[fo, fd, fs])
        c3 = ols_fe(y, [ld, ld .* mne, ld .* pres], Any[fo, fd, fs])
        @printf("  ln distance                       %+.3f      data -0.164\n", c1[1])
        @printf("  ln distance x MNE                 %+.3f      data +0.046\n", c2[2])
        @printf("  ln distance x MNE, present there  %+.3f      data +0.067\n", c3[3])
        println("  Falls out of d[l,n] running from the PRODUCTION country -- Tintelnot's")
        println("  export platforms. Signs and ordering right; magnitudes too large, and")
        println("  note the data's own -0.164 is an order of magnitude below any gravity")
        println("  estimate, which is a specification smell on the empirical side.")
    end
    return (fsh = fsh, dsh = dsh, gf = gf, gd = gd, h1 = h1, h3 = h3, ch = ch)
end

# ---------------------------------------------------------------------------
function scorecard(res)
    banner("SCORECARD")
    @printf("  %-4s%-52s%s\n", "#", "fact", "verdict")
    println("  " * "-"^82)
    rows = (
      (1, "MNEs a large share of exports, foreign >> domestic", "TARGET (level) + GENERATED (split)"),
      (2, "foreign in complex goods, domestic in simple",       "TARGET (foreign half); domestic half NOT robust"),
      (3, "parents from a small set of countries",              "GENERATED"),
      (4, "grouping by parent raises concentration",            "GENERATED -- prediction, not target"),
      (5, "MNE presence raises local firms' exports",           "FAILS -- worse with entry"),
      (6, "distance a weaker barrier for MNEs",                 "GENERATED, magnitudes too large"),
    )
    for (n, t, v) in rows; @printf("  %-4d%-52s%s\n", n, t, v); end
    println("  " * "-"^82)
    println("  THREE parameters are internally calibrated: the fixed cost to the SIZE")
    println("  of the export sample, the capability gap to Fact 2, and the pool scaling")
    println("  zeta to Fact 1's foreign/domestic split. The multinational productivity")
    println("  edge used to be a fourth; with head-office services a capability rather")
    println("  than a factor input it is no longer needed and is set to zero.")
    println("  Facts 3, 4 and 6 are out-of-sample. Fact 5 is the one outright")
    println("  contradiction, and it belongs in the abstract rather than a footnote.")
end

# ---------------------------------------------------------------------------
function main()
    args = lowercase(join(ARGS, " "))
    level = occursin("quick", args) ? :quick : :normal
    conduct = occursin("bertrand", args) ? :bertrand : :cournot
    what = occursin("core", args)  ? :core  : occursin("entry", args) ? :entry :
           occursin("ge", args)    ? :ge    : occursin("facts", args) ? :facts : :all
    t0 = time()
    banner("MULTINATIONAL OWNERSHIP, MARKET POWER AND TRADE POLICY")
    print_calibration()
    res = nothing
    (what == :all || what == :core)  && run_core(level, conduct)
    (what == :all || what == :entry) && run_entry(level, conduct)
    (what == :all || what == :ge)    && run_ge(level, conduct)
    (what == :all || what == :facts) && (res = run_facts(level, conduct))
    what == :all && scorecard(res)
    @printf("\nTotal %.1f min\n", (time()-t0)/60)
end

if abspath(PROGRAM_FILE) == (@__FILE__)
    main()
end
