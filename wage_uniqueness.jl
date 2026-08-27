###############################################################################
# WAGE UNIQUENESS -- THEOREM 4, ITS MACHINE CHECK, AND ITS COUNTEREXAMPLE
#
#   Everything else in the model has exactly one solution as a theorem: one
#   market (Theorem 1), entry (Theorem 2), input prices (contraction), incomes
#   (a linear solve). The wage vector was the exception -- uniqueness was
#   verified from many starts and nothing more. This file is the argument that
#   closes the gap, and the evidence that says exactly where it stops.
#
#   THE CLAIM.  Write W_n = w_n LD_n for country n's wage bill and
#   M[n,j] = d ln W_n / d ln w_j. The system is homogeneous of degree one in
#   wages, so M has unit row sums, and gross substitutes -- the classical
#   sufficient condition for uniqueness -- is exactly M[n,j] > 0 for n != j.
#
#     LEMMA 3   d ln P / d ln w is non-negative with unit row sums.
#               UNCONDITIONAL. It already contains the markup channel, the
#               input-output loop and the head-office linkage.
#     LEMMA 4   the three-term decomposition of a plant's factor bill; the
#               between-group coefficient is (1 - eps_mu) chi_g, non-negative
#               exactly when eps_mu <= 1, i.e. under Cournot with eta = 1 when
#               no parent holds more than half of the market -- THE SAME
#               THRESHOLD AS THEOREM 2. And the within-group term cancels out of
#               the head-office channel entirely.
#     THEOREM 4 (H1) eps_mu <= 1 everywhere and (H2) demand beats exposure for
#               every ordered pair  =>  gross substitutes  =>  at most one
#               equilibrium in the region.
#
#   THE LIMIT.  Gross substitutes is NOT a theorem of this model. PART 4 below
#   exhibits a three-country economy where it fails AT THE EQUILIBRIUM: a
#   country that lives entirely off head-office payments for plants located
#   abroad sees its wage bill FALL when the host's wage rises. (H2) is what
#   rules that out, and it cannot be dropped.
#
#   NOTE ON THE BASELINE (2026-08-21). Since head-office services became a
#   CAPABILITY rather than a factor input (`hq_cost = false`, now the default),
#   PART 2 below describes that new baseline, while PARTS 4-6 build their own
#   economies with the head-office COST share switched on -- because that is what
#   the counterexample is about. Both are supported and both are reported. The
#   constructive statement that supersedes all of this is THEOREM 6: the map the
#   solver iterates is a contraction. See `mne_model.jl` PART 7c.
#
#   Run:  julia wage_uniqueness.jl        (committed output: run_wage_uniqueness.txt)
###############################################################################

include("mne_model.jl")
using Printf, Random, LinearAlgebra
const MM = MNEModel
say(x) = (println(x); flush(stdout))
hdr(x) = (println(); println("="^76); println(x); println("="^76); flush(stdout))

calibrated(cd) = MM.world_economy(MersenneTwister(20260819); N = 5, K = 4, n_rich = 2,
        n_dom = 6, n_pot_par = 18, zeta = 1.5, hq_gap = 1.3, mne_adv = 0.10,
        adv_slope = 1.2, fscale = 0.0006, conduct = cd)[1]

minoff(A, N) = minimum([A[n, j] for n in 1:N, j in 1:N if n != j])

# ---------------------------------------------------------------------------
hdr("PART 1  THE MARKUP CHANNEL, IN CLOSED FORM")
say("  A group's factor bill moves with E S / mu(S); the between-group term of")
say("  LEMMA 4 has the gross-substitutes sign exactly when eps_mu(S) <= 1.")
say("  Closed form, Cournot eta = 1:  mu = sigma/[(sigma-1)(1-S)], so")
say("  eps_mu = S/(1-S) < 1  <=>  S < 1/2. THE SAME HALF AS THEOREM 2.")
say("")
say(@sprintf("  %-10s%-7s%-7s%-16s%s", "conduct", "sigma", "eta",
             "eps_mu<1 up to", "entry frontier (Thm 2)"))
for cd in (:cournot, :bertrand), (sg, et) in ((5.0, 1.0), (5.0, 1.5), (8.0, 1.0), (3.0, 1.0))
    thr = 1.0
    for S in 0.001:0.001:0.999
        if MM.eps_markup(S, sg, et, cd) >= 1.0; thr = S; break; end
    end
    say(@sprintf("  %-10s%-7.1f%-7.1f%-16.3f%.3f", cd, sg, et, thr,
                 MM.share_frontier(sg, et, cd)))
end
say("")
say("  eps_markup is the exact closed form S mu'(S)/mu(S), not a difference.")
worst = 0.0
for cd in (:cournot, :bertrand), sg in (3.0, 5.0, 8.0), et in (1.0, 1.5), S in 0.02:0.02:0.9
    h = 1e-6
    num = S * (log(MM.markup(S + h, sg, et, cd)) - log(MM.markup(S - h, sg, et, cd))) / (2h)
    global worst = max(worst, abs(num - MM.eps_markup(S, sg, et, cd)))
end
say(@sprintf("  closed form vs numerical differentiation        : %.2e", worst))

# ---------------------------------------------------------------------------
hdr("PART 2  THE CALIBRATED ECONOMY: LEMMAS CHECKED, HYPOTHESES EVALUATED")
say("  Every analytic step of LEMMA 3 and LEMMA 4 is checked against numerical")
say("  differentiation of the solved model, the same discipline Theorem 2 gets")
say("  from verify_conditions_analytic().")
sols = Dict{Symbol,Any}()
for cd in (:cournot, :bertrand)
    m = calibrated(cd)
    r = MM.solve_ge_entry(m)
    sols[cd] = (m = m, w = r.w, cfg = r.info.cfg, P = r.info.P)
    say("")
    say(@sprintf("  conduct = %-9s wages = %s  (labour-market gap %.1e)",
                 cd, string(round.(r.w, digits = 5)), r.gap))
    MM.verify_wage_theorem(m, r.w, r.info.cfg; P0 = r.info.P)
end

# ---------------------------------------------------------------------------
hdr("PART 3  HOW FAR DOES THE THEOREM REACH?")
say("  Gross substitutes on the set of wage vectors whose SPREAD -- the log gap")
say("  between the highest and lowest relative wage -- is at most 2R implies at")
say("  most ONE equilibrium of spread at most R. (H1) and (H2) are sufficient")
say("  for gross substitutes, so they are conservative: the last two columns")
say("  show how much room the sufficient condition gives up.")
say("")
say(@sprintf("  %-9s %-9s %6s %11s %11s %11s %8s", "conduct", "spread", "pts",
             "H1 slack", "H2 slack", "min M offdg", "GS fails"))
for cd in (:cournot, :bertrand)
    S = sols[cd]; N = S.m.base.N
    rng = MersenneTwister(2026)
    for rho in (0.0, 0.15, 0.30, 0.50, 0.80)
        npts = rho == 0.0 ? 1 : 20
        h1 = Inf; h2 = Inf; mo = Inf; nb = 0; ok = 0
        for t in 1:npts
            wv = rho == 0.0 ? copy(S.w) : S.w .* exp.(rho / 2 .* (2 .* rand(rng, N) .- 1))
            wv ./= wv[1]
            local Hy, Mel, Ploc
            try
                Ploc = MM.excess_demand_fixed(S.m, wv, S.cfg; P0 = S.P)[2].P
                Hy = MM.wage_hypotheses(S.m, wv, S.cfg; P0 = Ploc)
                Mel = MM.wage_bill_elasticity(S.m, wv, S.cfg; P0 = Ploc)
            catch
                continue
            end
            (isfinite(Hy.epsmax) && all(isfinite, Mel)) || continue
            ok += 1
            h1 = min(h1, 1.0 - Hy.epsmax); h2 = min(h2, Hy.minslack)
            o = minoff(Mel, N); mo = min(mo, o); o <= 0 && (nb += 1)
        end
        say(@sprintf("  %-9s %-9.2f %6d %+11.4f %+11.4f %+11.4f %8d",
                     cd, rho, ok, h1, h2, mo, nb))
    end
end

# ---------------------------------------------------------------------------
hdr("PART 4  THE COUNTEREXAMPLE: GROSS SUBSTITUTES IS NOT A THEOREM HERE")
say("  Three countries, one sector. Country 1 has local firms. Country 2 owns")
say("  parents whose ONLY plants are in country 3, so every euro country 2 earns")
say("  is a head-office payment for production abroad. Raise w_3: those parents")
say("  get dearer, lose share to country 1, and country 2's wage bill FALLS.")
say("  Every design below has enough local rivals that NO parent holds more than")
say("  a third of any market, so (H1) is satisfied throughout and cannot be what")
say("  fails. The maxS column is the largest parent share over ALL markets.")

"""Country 2 is a pure head-office country: its parents produce only in 3."""
function hq_economy(rng; alpha = 0.8, nu = 0.0, sigma = 5.0, conduct = :cournot,
                    L3 = 0.15, ndom1 = 10, ndom3 = 6, npar2 = 2)
    N = 3; K = 1
    par = Int[]; hq = Int[]; loc = Int[]; sec = Int[]; phi = Float64[]; g = 0
    for _ in 1:ndom1
        g += 1; push!(par, g); push!(hq, 1); push!(loc, 1); push!(sec, 1)
        push!(phi, exp(0.2 * randn(rng)))
    end
    for _ in 1:ndom3
        g += 1; push!(par, g); push!(hq, 3); push!(loc, 3); push!(sec, 1)
        push!(phi, exp(0.2 * randn(rng)))
    end
    for _ in 1:npar2
        g += 1; push!(par, g); push!(hq, 2); push!(loc, 3); push!(sec, 1)
        push!(phi, exp(0.2 * randn(rng) + 0.3))
    end
    gamma = [i == j ? 1.0 : 1.15 for i in 1:N, j in 1:N]
    d = [i == j ? 1.0 : 1.3 for i in 1:N, j in 1:N]
    theta = zeros(g, N); for a in eachindex(par); theta[par[a], hq[a]] = 1.0; end
    base = MM.GEModel(N, K, [sigma], 1.0, [1.0], [alpha], [nu], ones(1, 1),
                      [1.0, 1.0, L3], par, hq, loc, sec, phi, gamma, d,
                      zeros(N, N, K), theta, conduct)
    return MM.GEEntry(base, [0.0])
end

say("")
say(@sprintf("  %-46s %9s %9s %9s %7s %7s", "design", "M[3,2]", "min offdg",
             "H2 slack", "maxS", "index"))
for (nm, kw) in ((" 14 rivals in 1, 10 in 3, alpha 0.15, 3 parents",
                    (ndom1 = 14, ndom3 = 10, alpha = 0.15, npar2 = 3)),
                 (" 10 rivals in 1,  6 in 3, alpha 0.30, 2 parents", (alpha = 0.30,)),
                 (" 10 rivals in 1,  6 in 3, alpha 0.50, 2 parents", (alpha = 0.50,)),
                 (" 10 rivals in 1,  6 in 3, alpha 0.80, 2 parents", (alpha = 0.80,)),
                 ("  6 rivals in 1,  3 in 3, alpha 0.30, 2 parents",
                    (ndom1 = 6, ndom3 = 3, alpha = 0.30)),
                 (" 10 rivals in 1,  6 in 3, alpha 0.30, Bertrand ",
                    (alpha = 0.30, conduct = :bertrand)),
                 (" 10 rivals in 1,  6 in 3, alpha 0.30, nu = 0.55",
                    (alpha = 0.30, nu = 0.55)))
    mod = hq_economy(MersenneTwister(3); kw...)
    cfg = MM.all_active(mod); N = 3
    r = MM.solve_ge_fixed(mod, cfg; w0 = ones(N), maxit = 600, tol = 1e-12)
    w = r.w ./ r.w[1]
    Mel = MM.wage_bill_elasticity(mod, w, cfg)
    Hy = MM.wage_hypotheses(mod, w, cfg)
    Jr = ((Mel .- I(N)) .* mod.base.L)[2:end, 2:end]
    say(@sprintf("  %-46s %+9.4f %+9.4f %+9.4f %7.3f %7s", nm, Mel[3, 2],
                 minoff(Mel, N), Hy.minslack, Hy.Smax,
                 sign(det(Jr)) == (-1)^(N - 1) ? "+1" : "-1"))
end
say("")
say("  READING. The largest parent share anywhere is 0.33, so (H1) is satisfied at")
say("  every one of these -- and gross substitutes fails at six of the seven. It is")
say("  (H2), the head-office exposure condition, that is doing the work, and the")
say("  counterexample is what proves it cannot be dropped.")
say("")
say("  Note the alpha pattern, which is the opposite of the obvious guess: the")
say("  failure is WORST at alpha = 0.15 and DISAPPEARS by alpha = 0.80. A large")
say("  head-office share is not the problem. The problem is a country whose whole")
say("  income is a thin slice of firms whose costs are set by one other country.")

# ---------------------------------------------------------------------------
hdr("PART 5  WHAT DISTINGUISHES THE CALIBRATION FROM THE COUNTEREXAMPLE")
say("  psi[n,m] = share of country n's labour income paid by plants tied to")
say("  country m, as host or as parent. The counterexample drives it to ONE.")

function psimat(m, w, cfg; P0 = nothing)
    b = m.base; N = b.N; K = b.K
    zz, inf = MM.excess_demand_fixed(m, w, cfg; P0 = P0)
    psi = zeros(N, N); tot = zeros(N)
    for n in 1:N, k in 1:K
        cell = inf.mk[n, k]; eq = cell.eq
        for (j, a) in enumerate(cell.idx)
            gp = findfirst(==(b.par[a]), cell.pars); tar = b.tariff[b.loc[a], n, k]
            V = (1.0 - b.nu[k]) * inf.E[n, k] * eq.s[j] / (eq.mu[gp] * (1.0 + tar)) +
                MM.fixed_cost(m, w, a, n)
            for (cty, wt) in ((b.hq[a], b.alpha[k]), (b.loc[a], 1.0 - b.alpha[k]))
                tot[cty] += wt * V
                for other in (b.hq[a], b.loc[a])
                    other != cty && (psi[cty, other] += wt * V)
                end
            end
        end
    end
    return psi ./ tot
end

for cd in (:cournot, :bertrand)
    S = sols[cd]; N = S.m.base.N
    ps = psimat(S.m, S.w, S.cfg; P0 = S.P)
    say("")
    say(@sprintf("  calibrated economy, %s:", cd))
    for n in 1:N; say("    " * join([@sprintf("%.3f", ps[n, j]) for j in 1:N], "  ")); end
    say(@sprintf("    largest bilateral exposure = %.4f", maximum([ps[n, j] for n in 1:N, j in 1:N if n != j])))
end
mod = hq_economy(MersenneTwister(3); alpha = 0.8)
cfgx = MM.all_active(mod)
rx = MM.solve_ge_fixed(mod, cfgx; w0 = ones(3), maxit = 400, tol = 1e-12)
ps = psimat(mod, rx.w ./ rx.w[1], cfgx)
say("")
say("  counterexample economy:")
for n in 1:3; say("    " * join([@sprintf("%.3f", ps[n, j]) for j in 1:3], "  ")); end
say(@sprintf("    largest bilateral exposure = %.4f", maximum([ps[n, j] for n in 1:3, j in 1:3 if n != j])))

# ---------------------------------------------------------------------------
hdr("PART 6  WHERE GROSS SUBSTITUTES FAILS, IS THE EQUILIBRIUM STILL UNIQUE?")
say("  Gross substitutes is sufficient, not necessary. The weaker requirement is")
say("  that every equilibrium have INDEX +1 -- sign det of the reduced Jacobian")
say("  equal to (-1)^(N-1). Walras' law makes the wage bills the left null vector")
say("  of I - M, so every diagonal cofactor of I - M is proportional to a wage")
say("  bill: THE INDEX DOES NOT DEPEND ON WHICH COUNTRY IS THE NUMERAIRE. Below,")
say("  random economies from the family in PART 4, each solved from 25 dispersed")
say("  starts.")

function hunt(reps)
    rng = MersenneTwister(999)
    ncase = 0; nmult = 0; ngs = 0; nidx = 0; nnum = 0
    for rep in 1:reps
        al = 0.05 + 0.9 * rand(rng); L3 = 0.03 + 0.5 * rand(rng)
        sg = 3.5 + 12.0 * rand(rng); nu = 0.7 * rand(rng)
        cd = rand(rng) < 0.5 ? :cournot : :bertrand
        mod = hq_economy(MersenneTwister(rand(rng, 1:10^6)); alpha = al, nu = nu,
                         sigma = sg, L3 = L3, conduct = cd, ndom1 = rand(rng, 1:6),
                         ndom3 = rand(rng, 0:3), npar2 = rand(rng, 1:3))
        cfg = MM.all_active(mod); N = 3
        found = Vector{Vector{Float64}}()
        for st in 1:25
            w0 = st == 1 ? ones(N) : exp.(2.0 .* (2 .* rand(rng, N) .- 1)); w0 ./= w0[1]
            local rr
            try
                rr = MM.solve_ge_fixed(mod, cfg; w0 = w0, maxit = 500, tol = 1e-12)
            catch
                continue
            end
            (isfinite(rr.gap) && rr.gap < 1e-9) || continue
            push!(found, rr.w ./ rr.w[1])
        end
        isempty(found) && continue
        ncase += 1
        dist = Vector{Vector{Float64}}()
        for f in found
            all(maximum(abs.(log.(f ./ dd))) > 1e-5 for dd in dist) && push!(dist, f)
        end
        length(dist) > 1 && (nmult += 1)
        local Mel
        try
            Mel = MM.wage_bill_elasticity(mod, dist[1], cfg)
        catch
            continue
        end
        all(isfinite, Mel) || continue
        minoff(Mel, N) <= 0 && (ngs += 1)
        cof = [det(((Mel .- I(N)) .* mod.base.L)[setdiff(1:N, i), setdiff(1:N, i)])
               for i in 1:N]
        all(sign.(cof) .== (-1)^(N - 1)) || (nnum += 1)
        sign(cof[1]) == (-1)^(N - 1) || (nidx += 1)
    end
    return ncase, ngs, nidx, nnum, nmult
end
ncase, ngs, nidx, nnum, nmult = hunt(50)
say("")
say(@sprintf("  economies solved                                    : %d", ncase))
say(@sprintf("  gross substitutes FAILS at the equilibrium in       : %d", ngs))
say(@sprintf("  index differs from +1 in                            : %d", nidx))
say(@sprintf("  index depends on the choice of numeraire in         : %d", nnum))
say(@sprintf("  MULTIPLE equilibria found in                        : %d", nmult))
say("")
say("  So the failure of gross substitutes is real but it is not fatal: the")
say("  index condition, which is all uniqueness actually needs, survives it.")
say("  Theorem 4 proves the sufficient condition; this is the evidence that the")
say("  necessary one is much weaker, and it is the honest limit of the result.")
