###############################################################################
# A RIGOROUS CERTIFICATE ON THE CONTINUUM: GROSS SUBSTITUTES AND THE BIRKHOFF
# CONTRACTION, ENCLOSED OVER A BOX OF WAGE VECTORS BY INTERVAL ARITHMETIC
#
#   WHAT THIS CLOSES. Section 26.7 route 4 of CLAUDE.md: every certificate so
#   far evaluates the hypotheses at FINITELY MANY points of a wage box (grids,
#   random draws). This script encloses them over the WHOLE box -- every wage
#   vector in it at once -- with outward-rounded interval arithmetic, so the
#   conclusion is a theorem about the continuum, not a report about a sample.
#
#   THE MODEL VERSION. The nu = 0 capability baseline (hq_cost = false), the
#   version CLAUDE.md 30.6 designates as "the case where the theorem certifies
#   rather than merely verifies": eta = 1, alpha_k = 0 in the cost function,
#   no intermediates, zero tariffs, Cournot. There the model is CLOSED FORM in
#   wages given the market shares:
#
#     cost      c_a,n   = w_loc(a) * gamma[h,l] * d[l,n] / phi_a
#     market    S_g     solves  S = mu(S)^(1-sigma) K_g / A,  sum_g S_g = 1,
#                       K_g = sum_l w_l^(1-sigma) C_gl,  C_gl exact constants
#     incomes   X_n     = w_n L_n + sum_g theta[g,n] (Pi_g - FC_g)
#                       with Pi_g = sum_{n',k} beta_k X_n' S_g lerner(S_g)
#     wage bill W_n     = sum_{mkt} sum_g E S_g kappa_gn / mu_g + FC paid in n
#
#   and every derivative the certificate needs has a closed form through the
#   comparative statics of Section 4.6 (first order) and their derivatives
#   (second order, listed in PART 4b below).
#
#   THE METHOD: A CENTERED FORM WITH SUBDIVISION. A naive interval evaluation
#   of M over a box blows up by spread 0.01 (dependency), while the sampled
#   evidence shows M in truth barely moves out to spread 2.0. So each subbox is
#   enclosed as
#
#       M over subbox  <=  M(center)  +  sum_q  dM/dlnw_q(subbox) * [-r, r]
#
#   -- the exact value at the subbox CENTER (a point evaluation, error = pure
#   rounding) plus an interval GRADIENT bound times the subbox radius. The
#   gradient is itself interval-evaluated over the subbox, so its own
#   dependency error is multiplied by r and the total overestimation shrinks
#   QUADRATICALLY with the subdivision. The box certificate is the entrywise
#   hull over subboxes, valid because the mean-value Jacobian between any two
#   points of the box is an average of per-point values, each inside its
#   subbox's bounds.
#
#   Supporting tightenings: every sum over plants is collapsed to (group,
#   location) with exact constants so each wage occurs once per expression;
#   shares are corner-solved using the model's own monotonicity (S_g rises in
#   own K, falls in rivals'; Lambda > 0, 0 < omega < 1 at every interior
#   share); dlnA and its derivative use the convex-combination structure
#   (weights omega sum to one; their derivatives sum to zero).
#
#   THE CONCLUSION. A_k = (1-k) I + k M over the box; if min A_k > 0 then the
#   mean-value Jacobian of the log-tatonnement between ANY two points of the
#   box is a positive stochastic matrix inside the interval hull, so the map
#   contracts the Hilbert metric (= the oscillation seminorm in log
#   coordinates) at rate min(Dobrushin, Birkhoff) < 1, and the box holds AT
#   MOST ONE equilibrium -- given the frozen entry configuration, exactly as
#   Theorem 6 is stated; entry itself is Theorem 2's business.
#
#   HONESTY. (a) IEEE +,-,*,/ are correctly rounded; exp and log are
#   faithfully rounded in Julia's runtime -- we take a 4-ulp cushion on both,
#   far beyond their documented error. (b) The box is a log-box around the
#   solved wages with country 1 the numeraire; "spread rho" means each ln w_n,
#   n >= 2, ranges over +- rho/2. (c) The certificate freezes the entry
#   configuration, exactly as Theorem 6 does.
#
#   VALIDATION BEFORE TRUST. At the solution the interval M must reproduce
#   wage_bill_elasticity's finite differences; the interval X and W must
#   contain the solver's own values; and the ANALYTIC GRADIENT of M must
#   reproduce finite differences of the point evaluation. The script refuses
#   to print a certificate unless all of it passes.
#
#   Run:  julia interval_certificate.jl        (committed output:
#         run_interval_certificate.txt; caches the one-off nu = 0 GE solve in
#         interval_cache.txt)
###############################################################################

# -- load the model module WITHOUT its driver ---------------------------------
let src = read(joinpath(@__DIR__, "mne_model.jl"), String)
    stop = findfirst("end # module MNEModel", src)
    include_string(Main, src[1:last(stop)], "mne_model.jl(module)")
end
using .MNEModel
const MM = MNEModel
using Printf, Random, LinearAlgebra

say(x...) = (println(x...); flush(stdout))
hdr(x) = (println(); println("="^78); println(x); println("="^78); flush(stdout))

###############################################################################
# PART 1.  OUTWARD-ROUNDED INTERVAL ARITHMETIC (base Julia, no packages)
###############################################################################

@inline up4(x::Float64) = nextfloat(nextfloat(nextfloat(nextfloat(x))))
@inline dn4(x::Float64) = prevfloat(prevfloat(prevfloat(prevfloat(x))))
@inline up1(x::Float64) = nextfloat(x)
@inline dn1(x::Float64) = prevfloat(x)

struct Iv
    lo::Float64
    hi::Float64
    function Iv(lo::Float64, hi::Float64)
        @assert lo <= hi "empty interval [$lo, $hi]"
        new(lo, hi)
    end
end
Iv(x::Real) = Iv(Float64(x), Float64(x))
const IZERO = Iv(0.0); const IONE = Iv(1.0)

mid(a::Iv) = 0.5 * (a.lo + a.hi)
mag(a::Iv) = max(abs(a.lo), abs(a.hi))
wid(a::Iv) = a.hi - a.lo

import Base: +, -, *, /, show
show(io::IO, a::Iv) = @printf(io, "[% .6g,% .6g]", a.lo, a.hi)

+(a::Iv, b::Iv) = Iv(dn1(a.lo + b.lo), up1(a.hi + b.hi))
-(a::Iv, b::Iv) = Iv(dn1(a.lo - b.hi), up1(a.hi - b.lo))
-(a::Iv)        = Iv(-a.hi, -a.lo)
function *(a::Iv, b::Iv)
    p = (a.lo*b.lo, a.lo*b.hi, a.hi*b.lo, a.hi*b.hi)
    Iv(dn1(minimum(p)), up1(maximum(p)))
end
function /(a::Iv, b::Iv)
    @assert b.lo > 0.0 || b.hi < 0.0 "division by an interval containing zero"
    p = (a.lo/b.lo, a.lo/b.hi, a.hi/b.lo, a.hi/b.hi)
    Iv(dn1(minimum(p)), up1(maximum(p)))
end
+(a::Iv, b::Real) = a + Iv(b);  +(b::Real, a::Iv) = a + Iv(b)
-(a::Iv, b::Real) = a - Iv(b);  -(b::Real, a::Iv) = Iv(b) - a
*(a::Iv, b::Real) = a * Iv(b);  *(b::Real, a::Iv) = a * Iv(b)
/(a::Iv, b::Real) = a / Iv(b);  /(b::Real, a::Iv) = Iv(b) / a

iexp(a::Iv) = Iv(max(dn4(exp(a.lo)), 0.0), up4(exp(a.hi)))
ilog(a::Iv) = (@assert a.lo > 0.0; Iv(dn4(log(a.lo)), up4(log(a.hi))))
function ipow(a::Iv, p::Float64)
    @assert a.lo > 0.0 "ipow needs a positive interval"
    iexp(ilog(a) * Iv(p))
end

"a/(a+c) for a, c >= 0 -- monotone up in a, down in c (tight)."
function ifrac(a::Iv, c::Iv)
    @assert a.lo >= 0.0 && c.lo >= 0.0
    lo = a.lo + c.hi <= 0.0 ? 0.0 : dn1(a.lo / up1(a.lo + c.hi))
    hi = a.hi + c.lo <= 0.0 ? 1.0 : up1(a.hi / dn1(a.hi + c.lo))
    Iv(max(lo, 0.0), min(hi, 1.0))
end
ihull(a::Iv, b::Iv) = Iv(min(a.lo, b.lo), max(a.hi, b.hi))
iisect(a::Iv, b::Iv) = Iv(max(a.lo, b.lo), min(a.hi, b.hi))

###############################################################################
# PART 2.  RIGOROUS MARKET ENCLOSURES (Cournot, eta = 1)
#
#   mu(S) = sigma/((sigma-1)(1-S))       eps_mu(S) = S/(1-S)
#   Lambda(S) = (1-S)/(1+(sigma-2)S)     lerner(S) = 1 - 1/mu(S)
#   eps_L = eps_mu/(mu-1)                (elasticity of lerner in S)
#   eps_Lam = -(sigma-1) S / ((1-S)(1+(sigma-2)S))   (elasticity of Lambda in S)
###############################################################################

imu(S::Iv, sigma::Float64)     = Iv(sigma) / (Iv(sigma - 1.0) * (IONE - S))
iepsmu(S::Iv)                  = S / (IONE - S)
ilambda(S::Iv, sigma::Float64) = (IONE - S) / (IONE + Iv(sigma - 2.0) * S)
ilerner(S::Iv, sigma::Float64) = IONE - IONE / imu(S, sigma)
iepsL(S::Iv, sigma::Float64)   = iepsmu(S) / (imu(S, sigma) - IONE)
iepsLam(S::Iv, sigma::Float64) =
    -(Iv(sigma - 1.0) * S) / ((IONE - S) * (IONE + Iv(sigma - 2.0) * S))

"F(x; KA) = x - mu(x)^(1-sigma) KA at POINT x, rigorous. Increasing in x."
function inner_F(x::Float64, KA::Float64, sigma::Float64)
    Xi = Iv(x)
    Xi - ipow(imu(Xi, sigma), 1.0 - sigma) * Iv(KA)
end

"Certified bracket of the inner root S(KA)."
function inner_root(KA::Float64, sigma::Float64; tol = 1e-12)
    KA <= 0.0 && return Iv(0.0)
    lo, hi = 0.0, 1.0 - 1e-12
    for _ in 1:100
        hi - lo < tol && break
        x = 0.5 * (lo + hi)
        F = inner_F(x, KA, sigma)
        if F.lo > 0.0;     hi = x
        elseif F.hi < 0.0; lo = x
        else break end
    end
    Iv(dn1(lo), up1(hi))
end

"total_share(A; K) - 1 at POINT A with POINT K, as a certified interval."
function TS_minus1(A::Float64, K::Vector{Float64}, sigma::Float64)
    acc = IZERO
    for k in K
        acc = acc + inner_root(k / A, sigma)
    end
    acc - IONE
end

"Certified bracket of the outer root A*(K) for a POINT capability vector."
function outer_root(K::Vector{Float64}, sigma::Float64; tol = 1e-11)
    Aces = (sigma / (sigma - 1.0))^(1.0 - sigma) * sum(K)
    lo, hi = Aces * 1e-8, Aces * 1e8
    it = 0
    while TS_minus1(hi, K, sigma).lo > 0.0 && it < 200; hi *= 100.0; it += 1; end
    it = 0
    while TS_minus1(lo, K, sigma).hi < 0.0 && it < 200; lo /= 100.0; it += 1; end
    for _ in 1:120
        (hi - lo) < tol * lo && break
        x = sqrt(lo * hi)
        F = TS_minus1(x, K, sigma)
        if F.lo > 0.0;     lo = x
        elseif F.hi < 0.0; hi = x
        else break end
    end
    Iv(dn1(lo), up1(hi))
end

"""
One market over one wage box (T1 group x location collapse + T2 corner-solved
shares). `Cgl[g, l]` = exact constants sum_{a in g, loc = l} (gamma d/phi)^(1-sigma);
`wpow[l]` = interval of w_l^(1-sigma).
"""
struct MarketEnc
    A::Iv
    K::Vector{Iv}
    S::Vector{Iv}
    mu::Vector{Iv}
    lam::Vector{Iv}
    emu::Vector{Iv}
    ler::Vector{Iv}
    epsl::Vector{Iv}
    epslam::Vector{Iv}
    kap::Matrix{Iv}
end

function enclose_market(Cgl::Matrix{Float64}, wpow::Vector{Iv}, sigma::Float64)
    G, N = size(Cgl)
    K = Vector{Iv}(undef, G)
    for g in 1:G
        acc = IZERO
        for l in 1:N
            Cgl[g, l] == 0.0 && continue
            acc = acc + wpow[l] * Iv(Cgl[g, l])
        end
        K[g] = acc
    end
    thin = all(wid(k) <= 1e-12 * abs(k.hi) for k in K)
    S = Vector{Iv}(undef, G)
    local A
    if thin
        # a point evaluation: one outer root, shares straight from it
        A = outer_root([k.hi for k in K], sigma)
        for g in 1:G
            slo = inner_root(K[g].lo / A.hi, sigma)
            shi = inner_root(K[g].hi / A.lo, sigma)
            S[g] = Iv(max(slo.lo, 0.0), min(shi.hi, 1.0 - 1e-12))
        end
    else
        Alo = outer_root([k.lo for k in K], sigma)
        Ahi = outer_root([k.hi for k in K], sigma)
        A = Iv(Alo.lo, Ahi.hi)
        for g in 1:G
            Kup = [h == g ? K[h].hi : K[h].lo for h in 1:G]
            Kdn = [h == g ? K[h].lo : K[h].hi for h in 1:G]
            Aup = outer_root(Kup, sigma)
            Adn = outer_root(Kdn, sigma)
            shi = inner_root(Kup[g] / Aup.lo, sigma)
            slo = inner_root(Kdn[g] / Adn.hi, sigma)
            S[g] = Iv(max(slo.lo, 0.0), min(shi.hi, 1.0 - 1e-12))
        end
    end
    MarketEnc(A, K, S,
              [imu(s, sigma) for s in S], [ilambda(s, sigma) for s in S],
              [iepsmu(s) for s in S], [ilerner(s, sigma) for s in S],
              [iepsL(s, sigma) for s in S], [iepsLam(s, sigma) for s in S],
              [begin
                   if Cgl[g, m] == 0.0
                       IZERO
                   else
                       a_own = wpow[m] * Iv(Cgl[g, m])
                       c_rest = IZERO
                       for l in 1:N
                           (l == m || Cgl[g, l] == 0.0) && continue
                           c_rest = c_rest + wpow[l] * Iv(Cgl[g, l])
                       end
                       ifrac(a_own, c_rest)
                   end
               end for g in 1:G, m in 1:N])
end

###############################################################################
# PART 3.  ECONOMY-WIDE LEVELS OVER ONE (SUB)BOX
###############################################################################

struct CfgStruct
    pars::Matrix{Vector{Int}}
    Cgl::Matrix{Matrix{Float64}}
    nFC_gl::Matrix{Float64}
end

function build_struct(m, cfg)
    b = m.base; N, Kn = b.N, b.K
    pars = Matrix{Vector{Int}}(undef, N, Kn)
    Cgl = Matrix{Matrix{Float64}}(undef, N, Kn)
    Gtot = size(b.theta, 1)
    nFC = zeros(Gtot, N)
    idxk = [findall(==(k), b.sec) for k in 1:Kn]
    for n in 1:N, k in 1:Kn
        idxa = idxk[k]; act = cfg[n, k]
        on = findall(act); isempty(on) && (on = collect(eachindex(idxa)))
        idx = idxa[on]
        uniq = sort(unique(b.par[idx]))
        lut = Dict(u => i for (i, u) in enumerate(uniq))
        C = zeros(length(uniq), N)
        for a in idx
            l, h = b.loc[a], b.hq[a]
            C[lut[b.par[a]], l] += (b.gamma[h, l] * b.d[l, n] / b.phi[a])^(1.0 - b.sigma[k])
            nFC[b.par[a], l] += m.f[k]
        end
        pars[n, k] = uniq; Cgl[n, k] = C
    end
    CfgStruct(pars, Cgl, nFC)
end

"Rigorous enclosure of (I - Q) x = b, 0 <= Q with column sums < 1 (Neumann)."
function ilinsolve(Q::Matrix{Iv}, b::Vector{Iv})
    N = length(b)
    qbar = maximum(sum(up1(Q[i, j].hi) for i in 1:N) for j in 1:N)
    @assert qbar < 1.0 "Neumann bound needs column sums < 1 (got $qbar)"
    Qm = [mid(Q[i, j]) for i in 1:N, j in 1:N]
    bm = [mid(b[i]) for i in 1:N]
    xt = (I(N) - Qm) \ bm
    r = Vector{Iv}(undef, N)
    for i in 1:N
        acc = b[i] - Iv(xt[i])
        for j in 1:N
            acc = acc + Q[i, j] * Iv(xt[j])
        end
        r[i] = acc
    end
    rho = up1(sum(mag.(r)) / (1.0 - qbar))
    x = [Iv(dn1(xt[i] - rho), up1(xt[i] + rho)) for i in 1:N]
    for _ in 1:5
        xn = Vector{Iv}(undef, N)
        for i in 1:N
            acc = b[i]
            for j in 1:N
                acc = acc + Q[i, j] * x[j]
            end
            xn[i] = iisect(acc, x[i])
        end
        x = xn
    end
    return x
end

struct EconEnc
    N::Int
    K::Int
    mkts::Matrix{MarketEnc}
    wbox::Vector{Iv}
    FCg::Vector{Iv}
    FCn::Vector{Iv}
    FCgm::Matrix{Iv}
    Q::Matrix{Iv}
    X::Vector{Iv}
    W::Vector{Iv}
end

function enclose_economy(m, cs::CfgStruct, wbox::Vector{Iv})
    b = m.base; N, Kn = b.N, b.K
    @assert b.eta == 1.0 && b.conduct == :cournot
    @assert all(b.nu .== 0.0) && all(b.alpha .== 0.0)
    @assert all(b.tariff .== 0.0) && all(m.fdist .== 1.0)
    @assert all(m.fmult .== 1.0) "certificate assumes no fixed-cost spillover"
    Gtot = size(b.theta, 1)

    mkts = Matrix{MarketEnc}(undef, N, Kn)
    for k in 1:Kn
        wpow = [ipow(wbox[l], 1.0 - b.sigma[k]) for l in 1:N]
        for n in 1:N
            mkts[n, k] = enclose_market(cs.Cgl[n, k], wpow, b.sigma[k])
        end
    end

    FCg = [IZERO for _ in 1:Gtot]
    FCn = [IZERO for _ in 1:N]
    FCgm = [IZERO for _ in 1:Gtot, _ in 1:N]
    for g in 1:Gtot, l in 1:N
        cs.nFC_gl[g, l] == 0.0 && continue
        F = Iv(cs.nFC_gl[g, l]) * wbox[l]
        FCg[g] = FCg[g] + F
        FCn[l] = FCn[l] + F
        FCgm[g, l] = FCgm[g, l] + F
    end

    Q = [IZERO for _ in 1:N, _ in 1:N]
    for np in 1:N, k in 1:Kn
        e = mkts[np, k]
        for (gi, g) in enumerate(cs.pars[np, k])
            SL = e.S[gi] * e.ler[gi]
            for n in 1:N
                b.theta[g, n] == 0.0 && continue
                Q[n, np] = Q[n, np] + Iv(b.theta[g, n] * b.beta[k]) * SL
            end
        end
    end
    bx = Vector{Iv}(undef, N)
    for n in 1:N
        acc = wbox[n] * Iv(b.L[n])
        for g in 1:Gtot
            b.theta[g, n] == 0.0 && continue
            acc = acc - Iv(b.theta[g, n]) * FCg[g]
        end
        bx[n] = acc
    end
    X = ilinsolve(Q, bx)
    @assert all(x.lo > 0.0 for x in X) "income enclosure not positive -- subdivide"

    W = [IZERO for _ in 1:N]
    for np in 1:N, k in 1:Kn
        e = mkts[np, k]
        Emkt = Iv(b.beta[k]) * X[np]
        for (gi, g) in enumerate(cs.pars[np, k])
            for n in 1:N
                (e.kap[gi, n].hi == 0.0) && continue
                W[n] = W[n] + Emkt * e.S[gi] * e.kap[gi, n] / e.mu[gi]
            end
        end
    end
    for n in 1:N
        W[n] = W[n] + FCn[n]
    end

    EconEnc(N, Kn, mkts, wbox, FCg, FCn, FCgm, Q, X, W)
end

###############################################################################
# PART 4.  FIRST ORDER: dlnK, dlnA, dlnS, dlnmu PER MARKET AND DIRECTION,
#          THE INCOME RESPONSE, AND M ITSELF
#
#   dlnK_g^v = (1-sigma) kappa_gv
#   dlnA^v   = sum_g omega_g dlnK_g^v,  omega_g = S_g Lam_g / sum (in (0,1),
#              summing to one, so dlnA^v also lies in the hull of the dlnK^v)
#   dlnS_g^v = Lam_g (dlnK_g^v - dlnA^v)
#   dlnmu_g^v = eps_mu(S_g) dlnS_g^v
###############################################################################

struct MktResp
    om::Vector{Iv}                    # omega_g
    dK::Matrix{Iv}                    # [g, v]
    dA::Vector{Iv}                    # [v]
    dS::Matrix{Iv}                    # [g, v]
    dmu::Matrix{Iv}                   # [g, v]
end

function market_response(e::MarketEnc, sigma::Float64, N::Int)
    G = length(e.S)
    slam = [e.S[g] * e.lam[g] for g in 1:G]
    om = Vector{Iv}(undef, G)
    for g in 1:G
        rest = IZERO
        for h in 1:G
            h == g || (rest = rest + slam[h])
        end
        om[g] = ifrac(slam[g], rest)
    end
    dK = Matrix{Iv}(undef, G, N)
    dA = Vector{Iv}(undef, N)
    dS = Matrix{Iv}(undef, G, N)
    dmu = Matrix{Iv}(undef, G, N)
    for v in 1:N
        for g in 1:G
            dK[g, v] = Iv(1.0 - sigma) * e.kap[g, v]
        end
        acc = IZERO
        for g in 1:G
            acc = acc + om[g] * dK[g, v]
        end
        hullK = Iv(minimum(dK[g, v].lo for g in 1:G),
                   maximum(dK[g, v].hi for g in 1:G))
        dA[v] = iisect(acc, hullK)
        for g in 1:G
            dS[g, v] = e.lam[g] * (dK[g, v] - dA[v])
            dmu[g, v] = e.emu[g] * dS[g, v]
        end
    end
    MktResp(om, dK, dA, dS, dmu)
end

struct FirstOrder
    resp::Matrix{MktResp}             # per market
    Qd::Array{Iv,3}                   # dQ[n, n', v]
    bhat::Matrix{Iv}                  # rhs of the response system [n, v]
    Xd::Matrix{Iv}                    # Xdot [n, v]
    Wd::Matrix{Iv}                    # Wdot [n, v]  (level derivative of W_n)
    M::Matrix{Iv}                     # M[n, v] = Wdot/W
end

function first_order(m, cs::CfgStruct, enc::EconEnc)
    b = m.base; N, Kn = enc.N, enc.K
    Gtot = size(b.theta, 1)
    resp = Matrix{MktResp}(undef, N, Kn)
    for n in 1:N, k in 1:Kn
        resp[n, k] = market_response(enc.mkts[n, k], b.sigma[k], N)
    end

    Qd = Array{Iv,3}(undef, N, N, N)
    for n in 1:N, np in 1:N, v in 1:N
        Qd[n, np, v] = IZERO
    end
    for np in 1:N, k in 1:Kn
        e = enc.mkts[np, k]; r = resp[np, k]
        for (gi, g) in enumerate(cs.pars[np, k])
            SL = e.S[gi] * e.ler[gi]
            one_epsl = IONE + e.epsl[gi]
            for n in 1:N
                b.theta[g, n] == 0.0 && continue
                th = Iv(b.theta[g, n] * b.beta[k])
                for v in 1:N
                    Qd[n, np, v] = Qd[n, np, v] + th * SL * one_epsl * r.dS[gi, v]
                end
            end
        end
    end

    bhat = Matrix{Iv}(undef, N, N)
    Xd = Matrix{Iv}(undef, N, N)
    for v in 1:N
        for n in 1:N
            acc = v == n ? enc.wbox[n] * Iv(b.L[n]) : IZERO
            for np in 1:N
                acc = acc + Qd[n, np, v] * enc.X[np]
            end
            for g in 1:Gtot
                b.theta[g, n] == 0.0 && continue
                acc = acc - Iv(b.theta[g, n]) * enc.FCgm[g, v]
            end
            bhat[n, v] = acc
        end
        xv = ilinsolve_signed(enc.Q, bhat[:, v])
        for n in 1:N
            Xd[n, v] = xv[n]
        end
    end

    Wd = Matrix{Iv}(undef, N, N)
    for n in 1:N, v in 1:N
        Wd[n, v] = IZERO
    end
    for np in 1:N, k in 1:Kn
        e = enc.mkts[np, k]; r = resp[np, k]
        Emkt = Iv(b.beta[k]) * enc.X[np]
        sgk = b.sigma[k]
        for (gi, g) in enumerate(cs.pars[np, k])
            for n in 1:N
                (e.kap[gi, n].hi == 0.0) && continue
                V = Emkt * e.S[gi] * e.kap[gi, n] / e.mu[gi]
                for v in 1:N
                    dlnE = Xd[np, v] / enc.X[np]
                    D = dlnE + Iv(n == v ? 1.0 - sgk : 0.0) - r.dA[v] -
                        Iv(sgk) * r.dmu[gi, v]
                    Wd[n, v] = Wd[n, v] + V * D
                end
            end
        end
    end
    M = Matrix{Iv}(undef, N, N)
    for n in 1:N, v in 1:N
        n == v && (Wd[n, v] = Wd[n, v] + enc.FCn[n])
        M[n, v] = Wd[n, v] / enc.W[n]
    end
    FirstOrder(resp, Qd, bhat, Xd, Wd, M)
end

"ilinsolve for a signed right-hand side (same Neumann argument)."
function ilinsolve_signed(Q::Matrix{Iv}, b::Vector{Iv})
    N = length(b)
    qbar = maximum(sum(up1(Q[i, j].hi) for i in 1:N) for j in 1:N)
    @assert qbar < 1.0
    Qm = [mid(Q[i, j]) for i in 1:N, j in 1:N]
    bm = [mid(b[i]) for i in 1:N]
    xt = (I(N) - Qm) \ bm
    r = Vector{Iv}(undef, N)
    for i in 1:N
        acc = b[i] - Iv(xt[i])
        for j in 1:N
            acc = acc + Q[i, j] * Iv(xt[j])
        end
        r[i] = acc
    end
    rho = up1(sum(mag.(r)) / (1.0 - qbar))
    x = [Iv(dn1(xt[i] - rho), up1(xt[i] + rho)) for i in 1:N]
    for _ in 1:5
        xn = Vector{Iv}(undef, N)
        for i in 1:N
            acc = b[i]
            for j in 1:N
                acc = acc + Q[i, j] * x[j]
            end
            xn[i] = iisect(acc, x[i])
        end
        x = xn
    end
    return x
end

###############################################################################
# PART 4b.  SECOND ORDER: THE GRADIENT OF M, CLOSED FORM
#
#   With d_q = d/d ln w_q and the market objects above (direction m):
#
#     d_q kappa_gm  = (1-sigma) kappa_gm (delta_qm - kappa_gq)
#     d_q dlnK_g^m  = (1-sigma) d_q kappa_gm
#     d_q Lambda_g  = Lambda_g eps_Lam(S_g) dlnS_g^q
#     d_q omega_g   = omega_g [ t_g^q - sum_h omega_h t_h^q ],
#                     t_g^q = (1 + eps_Lam(S_g)) dlnS_g^q     (sums to zero)
#     d_q dlnA^m    = sum_g d_q omega_g (dlnK_g^m - c)  +  sum_g omega_g d_q dlnK_g^m
#                     (any constant c may be subtracted since the d_q omega sum
#                      to zero; c = the midpoint of the dlnK^m hull)
#     d_q dlnS_g^m  = Lambda_g [ eps_Lam dlnS_g^q (dlnK_g^m - dlnA^m)
#                                + d_q dlnK_g^m - d_q dlnA^m ]
#     d_q eps_mu    = S dlnS^q / (1-S)^2
#     d_q dlnmu^m   = d_q eps_mu * dlnS^m + eps_mu * d_q dlnS^m
#
#   Incomes: differentiate (I-Q) Xdot^m = bhat^m:
#     (I-Q) d_q Xdot^m = d_q bhat^m + (d_q Q) Xdot^m
#   with d_q of every bhat^m piece taken by the product rule (d_q(S ler) =
#   S ler (1+eps_L) dlnS^q; d_q eps_L = [d_q eps_mu (mu-1) - eps_mu mu dlnmu^q]/(mu-1)^2).
#
#   Wage bills: with D^v = dlnE^v + (1-sigma) delta_nv - dlnA^v - sigma dlnmu^v
#   and V = E S kappa / mu the level,
#     d_q V = V D^q
#     d_q dlnE^m = d_q Xdot^m / X - dlnE^m dlnE^q
#     d_q Wdot^m_n = sum V [ D^q D^m + d_q D^m ] + delta_nm delta_qn FCn_n
#     d_q M[n,m] = d_q Wdot^m_n / W_n - M[n,m] M[n,q]
#   the last step using d_q W_n = Wdot^q_n.
###############################################################################

"""
Second-order objects over the box: for every market the arrays d_q dlnS^m,
d_q dlnmu^m, d_q dlnA^m, and economy-wide d_q Xdot^m. Everything raw-interval;
the centred forms are assembled in `grad_M`.
"""
struct SecondOrder
    dqdS::Matrix{Array{Iv,3}}
    dqdmu::Matrix{Array{Iv,3}}
    dqdA::Matrix{Matrix{Iv}}
    Xd2::Array{Iv,3}
end

function second_order(m, cs::CfgStruct, enc::EconEnc, fo::FirstOrder)
    b = m.base; N, Kn = enc.N, enc.K
    Gtot = size(b.theta, 1)

    # per-market second-order objects
    dqdS = Matrix{Array{Iv,3}}(undef, N, Kn)      # [g, m, q]
    dqdmu = Matrix{Array{Iv,3}}(undef, N, Kn)
    dqdA = Matrix{Matrix{Iv}}(undef, N, Kn)       # [m, q]
    for n in 1:N, k in 1:Kn
        e = enc.mkts[n, k]; r = fo.resp[n, k]
        sg = b.sigma[k]; G = length(e.S)
        dS2 = Array{Iv,3}(undef, G, N, N)
        dmu2 = Array{Iv,3}(undef, G, N, N)
        dA2 = Matrix{Iv}(undef, N, N)
        # t_g^q = (1 + eps_Lam) dlnS^q; centred omega derivative
        for q in 1:N
            tq = [(IONE + e.epslam[g]) * r.dS[g, q] for g in 1:G]
            tbar = IZERO
            for g in 1:G
                tbar = tbar + r.om[g] * tq[g]
            end
            domq = [r.om[g] * (tq[g] - tbar) for g in 1:G]
            for mm in 1:N
                # d_q dlnK^m
                dqdK = [Iv((1.0 - sg)^2) * e.kap[g, mm] *
                        (Iv(q == mm ? 1.0 : 0.0) - e.kap[g, q]) for g in 1:G]
                # d_q dlnA^m: centred first piece + convex-combination second
                cmid = 0.5 * (minimum(r.dK[g, mm].lo for g in 1:G) +
                              maximum(r.dK[g, mm].hi for g in 1:G))
                acc1 = IZERO
                for g in 1:G
                    acc1 = acc1 + domq[g] * (r.dK[g, mm] - Iv(cmid))
                end
                acc2 = IZERO
                for g in 1:G
                    acc2 = acc2 + r.om[g] * dqdK[g]
                end
                hull2 = Iv(minimum(dqdK[g].lo for g in 1:G),
                           maximum(dqdK[g].hi for g in 1:G))
                dA2[mm, q] = acc1 + iisect(acc2, hull2)
                for g in 1:G
                    dS2[g, mm, q] = e.lam[g] *
                        (e.epslam[g] * r.dS[g, q] * (r.dK[g, mm] - r.dA[mm]) +
                         dqdK[g] - dA2[mm, q])
                    dqem = e.S[g] * r.dS[g, q] / ((IONE - e.S[g]) * (IONE - e.S[g]))
                    dmu2[g, mm, q] = dqem * r.dS[g, mm] + e.emu[g] * dS2[g, mm, q]
                end
            end
        end
        dqdS[n, k] = dS2; dqdmu[n, k] = dmu2; dqdA[n, k] = dA2
    end

    # income second order: (I-Q) d_q Xdot^m = d_q bhat^m + (d_q Q) Xdot^m
    Xd2 = Array{Iv,3}(undef, N, N, N)             # [n, m, q]
    for q in 1:N
        for mm in 1:N
            rhs = Vector{Iv}(undef, N)
            for n in 1:N
                # d_q of bhat^m
                acc = (mm == n && q == n) ? enc.wbox[n] * Iv(b.L[n]) : IZERO
                for np in 1:N, k in 1:Kn
                    e = enc.mkts[np, k]; r = fo.resp[np, k]
                    for (gi, g) in enumerate(cs.pars[np, k])
                        b.theta[g, n] == 0.0 && continue
                        th = Iv(b.theta[g, n] * b.beta[k])
                        SL = e.S[gi] * e.ler[gi]
                        oe = IONE + e.epsl[gi]
                        # d_q eps_L
                        mu = e.mu[gi]; emu = e.emu[gi]
                        dqem = e.S[gi] * r.dS[gi, q] /
                               ((IONE - e.S[gi]) * (IONE - e.S[gi]))
                        dqel = (dqem * (mu - IONE) - emu * mu * r.dmu[gi, q]) /
                               ((mu - IONE) * (mu - IONE))
                        # d_q [ X SL (1+eps_L) dS^m ]  (X level included)
                        term = fo.Xd[np, q] * SL * oe * r.dS[gi, mm] +
                               enc.X[np] * SL * (oe * r.dS[gi, q] * oe * r.dS[gi, mm] +
                                                 dqel * r.dS[gi, mm] +
                                                 oe * dqdS[np, k][gi, mm, q])
                        acc = acc + th * term
                    end
                end
                for g in 1:Gtot
                    b.theta[g, n] == 0.0 && continue
                    q == mm && (acc = acc - Iv(b.theta[g, n]) * enc.FCgm[g, mm])
                end
                # + (d_q Q) Xdot^m
                for np in 1:N
                    acc = acc + fo.Qd[n, np, q] * fo.Xd[np, mm]
                end
                rhs[n] = acc
            end
            xv = ilinsolve_signed(enc.Q, rhs)
            for n in 1:N
                Xd2[n, mm, q] = xv[n]
            end
        end
    end
    SecondOrder(dqdS, dqdmu, dqdA, Xd2)
end

"centred first-order object: value at the center + second-order term times radius."
function centred1(c::Iv, d2::AbstractVector{Iv}, r::Vector{Float64})
    acc = c
    for q in eachindex(r)
        r[q] == 0.0 && continue
        acc = acc + d2[q] * Iv(-r[q], r[q])
    end
    acc
end

"""
Gradient of M over the box: G[n, m, q] = d M[n,m] / d ln w_q, as intervals.

First-order objects entering PRODUCTS (the D terms, dlnE, and M itself) are
CENTRED: exact center value + second-order term * radius, so the gradient's
dependency error is second order in the subbox width. `Mbox` is a valid
enclosure of M over the box used for the M*M term (two-pass bootstrap).
"""
function grad_M(m, cs::CfgStruct, enc::EconEnc, fo::FirstOrder,
                so::SecondOrder, foc::FirstOrder, r::Vector{Float64},
                Mbox::Matrix{Iv})
    b = m.base; N, Kn = enc.N, enc.K
    # centred Xdot and dlnE
    Xdc = Matrix{Iv}(undef, N, N)
    for n in 1:N, v in 1:N
        Xdc[n, v] = iisect(centred1(foc.Xd[n, v], so.Xd2[n, v, :], r), fo.Xd[n, v])
    end
    Gm = Array{Iv,3}(undef, N, N, N)
    Wd2 = Array{Iv,3}(undef, N, N, N)
    for n in 1:N, mm in 1:N, q in 1:N
        Wd2[n, mm, q] = IZERO
    end
    for np in 1:N, k in 1:Kn
        e = enc.mkts[np, k]; rr = fo.resp[np, k]; rc = foc.resp[np, k]
        Emkt = Iv(b.beta[k]) * enc.X[np]
        sgk = b.sigma[k]
        # centred market first-order objects
        G = length(e.S)
        dAc = [iisect(centred1(rc.dA[v], so.dqdA[np, k][v, :], r), rr.dA[v])
               for v in 1:N]
        dmuc = [iisect(centred1(rc.dmu[g, v], so.dqdmu[np, k][g, v, :], r),
                       rr.dmu[g, v]) for g in 1:G, v in 1:N]
        for (gi, g) in enumerate(cs.pars[np, k])
            for n in 1:N
                (e.kap[gi, n].hi == 0.0) && continue
                V = Emkt * e.S[gi] * e.kap[gi, n] / e.mu[gi]
                for mm in 1:N, q in 1:N
                    dlnEm = Xdc[np, mm] / enc.X[np]
                    dlnEq = Xdc[np, q] / enc.X[np]
                    Dm = dlnEm + Iv(n == mm ? 1.0 - sgk : 0.0) - dAc[mm] -
                         Iv(sgk) * dmuc[gi, mm]
                    Dq = dlnEq + Iv(n == q ? 1.0 - sgk : 0.0) - dAc[q] -
                         Iv(sgk) * dmuc[gi, q]
                    dqdlnEm = so.Xd2[np, mm, q] / enc.X[np] - dlnEm * dlnEq
                    dqDm = dqdlnEm - so.dqdA[np, k][mm, q] -
                           Iv(sgk) * so.dqdmu[np, k][gi, mm, q]
                    Wd2[n, mm, q] = Wd2[n, mm, q] + V * (Dq * Dm + dqDm)
                end
            end
        end
    end
    for n in 1:N, mm in 1:N, q in 1:N
        (n == mm && q == n) && (Wd2[n, mm, q] = Wd2[n, mm, q] + enc.FCn[n])
        Gm[n, mm, q] = Wd2[n, mm, q] / enc.W[n] - Mbox[n, mm] * Mbox[n, q]
    end
    return Gm
end

###############################################################################
# PART 5.  THE CENTERED-FORM CERTIFICATE WITH SUBDIVISION
###############################################################################

function certificate_from_hull(A::Matrix{Iv}, M::Matrix{Iv})
    N = size(M, 1)
    minA = minimum(A[i, j].lo for i in 1:N, j in 1:N)
    minGS = minimum(M[i, j].lo for i in 1:N, j in 1:N if i != j)
    minMnn = minimum(M[i, i].lo for i in 1:N)
    dob = 1.0 - sum(minimum(A[i, j].lo for i in 1:N) for j in 1:N)
    birk = 1.0
    if minA > 0.0
        D = 0.0
        for i in 1:N, j in 1:N, k in 1:N, l in 1:N
            D = max(D, log(A[i, k].hi) + log(A[j, l].hi) -
                       log(A[j, k].lo) - log(A[i, l].lo))
        end
        birk = tanh(D / 4)
    end
    (minA = minA, minGS = minGS, minMnn = minMnn, dobrushin = dob,
     birkhoff = birk, tau = min(dob, birk),
     certified = minA > 0.0 && min(dob, birk) < 1.0)
end

"M over one subbox by the centered form (two-pass bootstrap for the M*M term)."
function M_over_subbox(m, cs::CfgStruct, wc::Vector{Float64}, r::Vector{Float64})
    N = length(wc)
    # exact evaluation at the center (thin box: error = rounding only)
    encc = enclose_economy(m, cs, [Iv(w) for w in wc])
    foc = first_order(m, cs, encc)
    Mc = foc.M
    all(r .== 0.0) && return Mc
    # wide (raw) enclosure over the subbox
    wbox = [r[n] == 0.0 ? Iv(wc[n]) :
            Iv(dn1(wc[n] * exp(-r[n])), up1(wc[n] * exp(r[n]))) for n in 1:N]
    enc = enclose_economy(m, cs, wbox)
    fo = first_order(m, cs, enc)
    so = second_order(m, cs, enc, fo)
    Mbox = copy(fo.M)                          # pass 0: the raw enclosure
    for _ in 1:2                               # two centred-form passes
        G = grad_M(m, cs, enc, fo, so, foc, r, Mbox)
        Mn = Matrix{Iv}(undef, N, N)
        for n in 1:N, mm in 1:N
            acc = Mc[n, mm]
            for q in 1:N
                r[q] == 0.0 && continue
                acc = acc + G[n, mm, q] * Iv(-r[q], r[q])
            end
            Mn[n, mm] = iisect(acc, Mbox[n, mm])
        end
        Mbox = Mn
    end
    return Mbox
end

function certify_box(m, cs::CfgStruct, w0::Vector{Float64}, rho::Float64,
                     s::Int; kappa = 0.15)
    b = m.base; N = b.N
    if rho == 0.0
        M = M_over_subbox(m, cs, w0, zeros(N))
        A = [(i == j ? Iv(1.0 - kappa) : IZERO) + Iv(kappa) * M[i, j]
             for i in 1:N, j in 1:N]
        return certificate_from_hull(A, M), 1
    end
    half = rho / (2.0 * s)                      # subbox radius in ln w
    Ah = Matrix{Union{Nothing,Iv}}(nothing, N, N)
    Mh = Matrix{Union{Nothing,Iv}}(nothing, N, N)
    idx = ones(Int, N - 1)
    nbox = 0
    while true
        wc = copy(w0); rv = zeros(N)
        for n in 2:N
            c = -rho / 2 + (2 * idx[n - 1] - 1) * half
            wc[n] = w0[n] * exp(c); rv[n] = half
        end
        M = M_over_subbox(m, cs, wc, rv)
        nbox += 1
        for i in 1:N, j in 1:N
            Aij = (i == j ? Iv(1.0 - kappa) : IZERO) + Iv(kappa) * M[i, j]
            Ah[i, j] = Ah[i, j] === nothing ? Aij : ihull(Ah[i, j], Aij)
            Mh[i, j] = Mh[i, j] === nothing ? M[i, j] : ihull(Mh[i, j], M[i, j])
        end
        p = 1
        while p <= N - 1
            idx[p] += 1
            idx[p] <= s && break
            idx[p] = 1; p += 1
        end
        p > N - 1 && break
    end
    A = Matrix{Iv}(undef, N, N); M = Matrix{Iv}(undef, N, N)
    for i in 1:N, j in 1:N
        A[i, j] = Ah[i, j]; M[i, j] = Mh[i, j]
    end
    return certificate_from_hull(A, M), nbox
end

###############################################################################
# PART 6.  DRIVER: SOLVE, VALIDATE, CERTIFY
###############################################################################

const CACHE = joinpath(@__DIR__, "interval_cache.txt")

function calibrated_nu0(conduct = :cournot)
    MM.world_economy(MersenneTwister(20260819); N = 5, K = 4, n_rich = 2,
                     n_dom = 6, n_pot_par = 18, zeta = 1.5, hq_gap = 1.3,
                     mne_adv = 0.0, adv_slope = 1.2, fscale = 0.0006,
                     conduct = conduct, hq_cost = false, nu = 0.0)[1]
end

function load_or_solve(m)
    b = m.base
    if isfile(CACHE)
        lines = readlines(CACHE)
        w = parse.(Float64, split(lines[1]))
        flat = parse.(Bool, split(lines[2]))
        cfg = MM.all_active(m)
        pos = 1
        for n in 1:b.N, k in 1:b.K
            for i in eachindex(cfg[n, k])
                cfg[n, k][i] = flat[pos]; pos += 1
            end
        end
        z, inf = MM.excess_demand_fixed(m, w, cfg)
        gap = maximum(abs.(z ./ b.L))
        if gap < 1e-8
            say(@sprintf("  cache hit: wages loaded, labour-market gap %.2e", gap))
            return w, cfg, inf
        end
        say("  cache stale; re-solving")
    end
    say("  solving the nu = 0 GE with entry (one-off; result is cached)...")
    t0 = time()
    r = MM.solve_ge_entry(m)
    say(@sprintf("  solved in %.1f min, gap %.2e", (time() - t0) / 60, r.gap))
    w, cfg = r.w, r.info.cfg
    open(CACHE, "w") do io
        println(io, join(w, " "))
        println(io, join(MM.flatten_cfg(cfg), " "))
    end
    return w, cfg, r.info
end

function main()
    hdr("A CONTINUUM CERTIFICATE FOR WAGE UNIQUENESS -- INTERVAL ARITHMETIC")
    say("  Model: nu = 0 capability baseline (hq_cost = false), Cournot, eta = 1,")
    say("  entry configuration frozen at the solution, country 1 the numeraire.")
    say("  Method: centered form -- exact value at each subbox center plus an")
    say("  interval gradient bound times the radius -- so the enclosure error")
    say("  shrinks QUADRATICALLY with the subdivision.")

    m = calibrated_nu0()
    w0, cfg, inf0 = load_or_solve(m)
    b = m.base; N = b.N
    cs = build_struct(m, cfg)

    # ---------------- validation -------------------------------------------
    hdr("VALIDATION: the interval machinery against the model's own numbers")
    enc0 = enclose_economy(m, cs, [Iv(w) for w in w0])
    fo0 = first_order(m, cs, enc0)
    Wtrue = inf0.LD .* w0
    inX = all(enc0.X[n].lo <= inf0.X[n] <= enc0.X[n].hi for n in 1:N)
    inW = all(enc0.W[n].lo <= Wtrue[n] <= enc0.W[n].hi for n in 1:N)
    say(@sprintf("  income X contained: %s   wage bill W contained: %s", inX, inW))
    Mfd = MM.wage_bill_elasticity(m, w0, cfg; P0 = inf0.P)
    Merr = maximum(abs(mid(fo0.M[i, j]) - Mfd[i, j]) for i in 1:N, j in 1:N)
    rs = [sum(mid(fo0.M[i, j]) for j in 1:N) for i in 1:N]
    say(@sprintf("  M vs the model's finite differences: max gap %.2e", Merr))
    say(@sprintf("  M row sums (homogeneity): %s",
                 join([@sprintf("%.6f", r) for r in rs], "  ")))

    # analytic gradient vs finite differences of the point evaluation
    so0 = second_order(m, cs, enc0, fo0)
    G0 = grad_M(m, cs, enc0, fo0, so0, fo0, zeros(N), fo0.M)
    hh = 1e-5; Gerr = 0.0
    for q in 2:N
        wup = copy(w0); wup[q] *= exp(hh)
        wdn = copy(w0); wdn[q] *= exp(-hh)
        Mup = first_order(m, cs, enclose_economy(m, cs, [Iv(w) for w in wup])).M
        Mdn = first_order(m, cs, enclose_economy(m, cs, [Iv(w) for w in wdn])).M
        for n in 1:N, mm in 1:N
            fd = (mid(Mup[n, mm]) - mid(Mdn[n, mm])) / (2hh)
            Gerr = max(Gerr, abs(mid(G0[n, mm, q]) - fd))
        end
    end
    say(@sprintf("  analytic gradient of M vs finite differences: max gap %.2e", Gerr))
    ok = inX && inW && Merr < 1e-3 && Gerr < 1e-3
    say(ok ? "  VALIDATION PASSED" : "  VALIDATION FAILED -- no certificate is printed")
    ok || return

    # ---------------- the certificate --------------------------------------
    hdr("THE CERTIFICATE: uniform bounds over the whole box, by spread")
    say("  spread rho: every ln w_n (n >= 2) ranges over +- rho/2 around ln w0_n;")
    say("  s^4 subboxes, centered form per subbox, bounds hulled.")
    say("  minGS = lower bound of every off-diagonal of M over the box;")
    say("  minA  = lower bound of every entry of (1-k)I + kM at k = 0.25;")
    say("  tau   = certified contraction factor, min(Dobrushin, Birkhoff).")
    say("  A row with tau < 1 is a THEOREM about the continuum: that box holds")
    say("  at most one equilibrium with this entry configuration.")
    say("")
    say("  The damping used is kappa = 0.15, the value Section 31.2 of CLAUDE.md")
    say("  itself recommends at nu = 0 (the fixed points of Psi_kappa are the same")
    say("  equilibria for every kappa, so the uniqueness conclusion is unchanged).")
    say("")
    say(@sprintf("  %-8s %4s %6s %10s %10s %10s %10s %10s   %s", "spread", "s",
                 "boxes", "minGS", "minMnn", "minA", "Dobrushin", "Birkhoff",
                 "certified"))
    for (rho, s) in ((0.0, 1), (0.02, 1), (0.05, 2), (0.10, 3), (0.15, 4),
                     (0.20, 5), (0.30, 6), (0.40, 8))
        local c, nb
        try
            t0 = time()
            c, nb = certify_box(m, cs, w0, rho, s)
            say(@sprintf("  %-8.2f %4d %6d %+10.4f %+10.4f %+10.4f %10.4f %10.4f   %s  (%.1f s)",
                         rho, s, nb, c.minGS, c.minMnn, c.minA, c.dobrushin,
                         c.birkhoff, c.certified ? "YES" : "no", time() - t0))
        catch err
            say(@sprintf("  %-8.2f %4d      -  enclosure failed inside a subbox: %s",
                         rho, s, sprint(showerror, err)))
        end
        flush(stdout)
    end
    say("")
    say("  Reading. A certified row at spread rho means: the continuum of wage")
    say("  vectors within +-rho/2 (log) of the solution contains EXACTLY the one")
    say("  equilibrium the solver found -- 'at most one' from the contraction,")
    say("  'at least one' because the solved wages sit inside the box. The grid")
    say("  and random-pair checks in run_model_full.txt reach further but prove")
    say("  nothing between their points; this table has no between-points gap.")
end

if abspath(PROGRAM_FILE) == (@__FILE__)
    main()
end
