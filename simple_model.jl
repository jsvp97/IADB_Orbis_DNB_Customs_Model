###############################################################################
# A VERSION OF THE MODEL WHOSE WAGE EQUILIBRIUM IS UNIQUE FOR A BETTER REASON
#
#   `wage_uniqueness.jl` proves Theorem 4 -- gross substitutes, hence a unique
#   wage vector, under (H1) no group above half of any market and (H2) demand
#   beats exposure -- and then exhibits a counterexample showing (H2) cannot be
#   dropped. The counterexample runs on ONE modelling choice: that a factory
#   pays its PARENT's wage on the head-office share alpha_k of its costs.
#
#   That choice does two different jobs, and only one of them is needed.
#
#     (i)  alpha_k as a COST SHARE   -- creates the cross-country factor linkage
#                                       and is the whole content of the
#                                       counterexample;
#     (ii) alpha_k as a COMPLEXITY INDEX in the productivity draws -- the
#          local-firm penalty exp(-hq_gap alpha_k) and the parent capability
#          gradient xi(1 + adv_slope alpha_k). This is what Facts 1 and 2 are
#          built on, and it does not touch wages at all.
#
#   `world_economy(...; hq_cost = false)` keeps (ii) and drops (i). The
#   productivity draws are bit-for-bit identical, so nothing behind Facts 1-2
#   moves for a mechanical reason.
#
#   THEOREM 5. With head-office services a capability, nu = 0, eta = 1 and (H1):
#   every factory paying country n != m is LOCATED in n, so its cost response to
#   w_m is exactly zero, and its whole reallocation term is bounded below by
#   (sigma-1) min{lam_g, Lam} >= 0. Hence
#
#           d ln W_n / d ln w_m  >=  Ebar_n^(m)
#
#   and gross substitutes follows from a pure DEMAND condition -- no market's
#   nominal spending falls. The model-specific obstacle is gone; what is left is
#   the textbook income effect, which is not peculiar to multinationals.
#
#   Run:  julia simple_model.jl      (committed output: run_simple_model.txt)
###############################################################################

include("mne_model.jl")
using Printf, Random, LinearAlgebra
const MM = MNEModel
say(x) = (println(x); flush(stdout))
hdr(x) = (println(); println("="^78); println(x); println("="^78); flush(stdout))
minoff(A, N) = minimum([A[n, j] for n in 1:N, j in 1:N if n != j])

calib(cd, hq, nu) = MM.world_economy(MersenneTwister(20260819); N = 5, K = 4,
        n_rich = 2, n_dom = 6, n_pot_par = 18, zeta = 1.5, hq_gap = 1.3,
        mne_adv = 0.10, adv_slope = 1.2, fscale = 0.0006, conduct = cd,
        hq_cost = hq, nu = nu)[1]

# ---------------------------------------------------------------------------
hdr("PART 1  THE CALIBRATED ECONOMY, THREE WAYS")
say("  floor   = smallest single-factory reallocation term. Theorem 5 says it is")
say("            non-negative once the head-office COST share is gone and nu = 0.")
say("  (*)viol = largest violation of lam_a <= min(lam_g, Lam), the extra condition")
say("            the input-output block needs. Identically zero when nu = 0.")
say("")
say(@sprintf("  %-34s %8s %8s %9s %9s %9s %6s", "model", "maxS", "max eps",
             "floor", "(*)viol", "min GS", "index"))
sols = Dict{String,Any}()
for (nm, hq, nu) in (("baseline: HQ cost, nu = 0.55", true,  0.55),
                     ("HQ a capability, nu = 0.55",   false, 0.55),
                     ("HQ a capability, nu = 0",      false, 0.0))
    m = calib(:cournot, hq, nu)
    r = MM.solve_ge_entry(m)
    fl = MM.reallocation_floor(m, r.w, r.info.cfg; P0 = r.info.P)
    Hy = MM.wage_hypotheses(m, r.w, r.info.cfg; P0 = r.info.P)
    Mel = MM.wage_bill_elasticity(m, r.w, r.info.cfg; P0 = r.info.P)
    N = m.base.N
    Jr = ((Mel .- I(N)) .* m.base.L)[2:end, 2:end]
    sols[nm] = (m = m, w = r.w, cfg = r.info.cfg, P = r.info.P)
    say(@sprintf("  %-34s %8.3f %8.3f %+9.4f %9.1e %+9.4f %6s", nm, Hy.Smax,
                 Hy.epsmax, fl.floor, fl.star_violation, minoff(Mel, N),
                 sign(det(Jr)) == (-1)^(N - 1) ? "+1" : "-1"))
end

# ---------------------------------------------------------------------------
hdr("PART 2  HOW WIDE IS THE REGION ON WHICH GROSS SUBSTITUTES HOLDS?")
say("  Gross substitutes on every wage vector of spread <= 2R gives at most ONE")
say("  equilibrium of spread <= R. The question is how big R can be made.")
say("")
say(@sprintf("  %-34s %7s %6s %11s %11s %8s", "model", "spread", "pts",
             "min GS", "min floor", "GS fails"))
for nm in ("baseline: HQ cost, nu = 0.55", "HQ a capability, nu = 0.55",
           "HQ a capability, nu = 0")
    S = sols[nm]; N = S.m.base.N
    rng = MersenneTwister(31415)
    for rho in (0.0, 0.4, 0.8, 1.4, 2.0)
        npts = rho == 0.0 ? 1 : 25
        mo = Inf; mf = Inf; nb = 0; ok = 0
        for t in 1:npts
            wv = rho == 0.0 ? copy(S.w) :
                 S.w .* exp.(rho / 2 .* (2 .* rand(rng, N) .- 1))
            wv ./= wv[1]
            local Mel, fl, Ploc
            try
                Ploc = MM.excess_demand_fixed(S.m, wv, S.cfg; P0 = S.P)[2].P
                Mel = MM.wage_bill_elasticity(S.m, wv, S.cfg; P0 = Ploc)
                fl = MM.reallocation_floor(S.m, wv, S.cfg; P0 = Ploc)
            catch
                continue
            end
            all(isfinite, Mel) || continue
            ok += 1
            o = minoff(Mel, N); mo = min(mo, o); o <= 0 && (nb += 1)
            mf = min(mf, fl.floor)
        end
        say(@sprintf("  %-34s %7.2f %6d %+11.4f %+11.4f %8d", nm, rho, ok, mo, mf, nb))
    end
end

# ---------------------------------------------------------------------------
hdr("PART 3  STRESS TEST ON DELIBERATELY NASTY ECONOMIES")
say("  Every feature that could feed the one channel the simplification does NOT")
say("  remove -- profits earned abroad and repatriated -- is switched on:")
say("  scrambled cross-ownership, countries with no local production, extreme")
say("  size asymmetry, extreme elasticities, tariffs.")

function nasty(rng; alpha = 0.0, nu = 0.0, N = 3, K = 1, sigma = 5.0,
               conduct = :cournot, crossown = true, holes = true, tariff = 0.0)
    sig = fill(sigma, K); beta = fill(1.0 / K, K); nuv = fill(nu, K)
    omega = fill(1.0 / K, K, K); alph = fill(alpha, K)
    L = exp.(1.5 .* randn(rng, N)); L ./= maximum(L) / 2
    par = Int[]; hq = Int[]; loc = Int[]; sec = Int[]; phi = Float64[]; g = 0
    hosts = holes ? [n for n in 1:N if rand(rng) < 0.7] : collect(1:N)
    isempty(hosts) && (hosts = [1])
    for n in hosts, k in 1:K, _ in 1:rand(rng, 1:4)
        g += 1; push!(par, g); push!(hq, n); push!(loc, n); push!(sec, k)
        push!(phi, exp(0.4 * randn(rng)))
    end
    for j in 1:rand(rng, 2:5)
        g += 1; h = rand(rng, 1:N); k = rand(rng, 1:K); placed = false
        for l in 1:N
            (l == h || rand(rng) < 0.6) || continue
            l in hosts || rand(rng) < 0.5 || continue
            push!(par, g); push!(hq, h); push!(loc, l); push!(sec, k)
            push!(phi, exp(0.4 * randn(rng) + 0.5)); placed = true
        end
        if !placed
            l = rand(rng, 1:N)
            push!(par, g); push!(hq, h); push!(loc, l); push!(sec, k)
            push!(phi, exp(0.4 * randn(rng) + 0.5))
        end
    end
    for k in 1:K
        while length(unique(par[sec .== k])) < 2
            j = rand(rng, 1:length(sec)); sec[j] = k
        end
    end
    gamma = [i == j ? 1.0 : 1.0 + 0.5 * rand(rng) for i in 1:N, j in 1:N]
    d = [i == j ? 1.0 : 1.0 + 0.8 * rand(rng) for i in 1:N, j in 1:N]
    t = fill(tariff, N, N, K); for n in 1:N, k in 1:K; t[n, n, k] = 0.0; end
    theta = zeros(g, N)
    if crossown
        for gg in 1:g; v = rand(rng, N) .^ 3; v ./= sum(v); theta[gg, :] = v; end
    else
        for a in eachindex(par); theta[par[a], hq[a]] = 1.0; end
    end
    base = MM.GEModel(N, K, sig, 1.0, beta, alph, nuv, omega, L, par, hq, loc,
                      sec, phi, gamma, d, t, theta, conduct)
    return MM.GEEntry(base, fill(0.0, K))
end

say("")
say("")
say("  Split by whether (H1) -- no group above half of any market -- holds AT THE")
say("  POINT. That is the same condition Theorem 2 already needs, and it is what")
say("  Theorem 5 assumes; a stress point that violates it is outside both theorems.")
say("")
say(@sprintf("  %-34s %5s | %5s %7s %9s | %5s %7s | %9s", "family", "pts",
             "H1 ok", "GSfail", "worst GS", "H1 bad", "GSfail", "floor|H1"))
function stress(nm; reps = 40, starts = 5, kw...)
    rng = MersenneTwister(20260820)
    tot = 0; nok = 0; nbad = 0; fok = 0; fbad = 0; wok = Inf; flok = Inf
    for r in 1:reps
        local mod
        try; mod = nasty(MersenneTwister(9000 + r); kw...); catch; continue; end
        cfg = MM.all_active(mod); N = mod.base.N
        for t in 1:starts
            wv = t == 1 ? ones(N) : exp.(1.5 .* (2 .* rand(rng, N) .- 1)); wv ./= wv[1]
            local Mel, fl, Ploc
            try
                Ploc = MM.excess_demand_fixed(mod, wv, cfg)[2].P
                Mel = MM.wage_bill_elasticity(mod, wv, cfg; P0 = Ploc)
                fl = MM.reallocation_floor(mod, wv, cfg; P0 = Ploc)
            catch
                continue
            end
            all(isfinite, Mel) || continue
            tot += 1
            o = minoff(Mel, N)
            if fl.epsmax <= 1.0
                nok += 1; o <= 0 && (fok += 1)
                wok = min(wok, o); flok = min(flok, fl.floor)
            else
                nbad += 1; o <= 0 && (fbad += 1)
            end
        end
    end
    say(@sprintf("  %-34s %5d | %5d %7d %+9.4f | %5d %7d | %+9.4f", nm, tot,
                 nok, fok, wok == Inf ? 0.0 : wok, nbad, fbad,
                 flok == Inf ? 0.0 : flok))
end
stress("BASELINE  HQ cost  nu .55  N=3";     alpha = 0.55, nu = 0.55, N = 3)
stress("BASELINE  HQ cost  nu .55  N=4 K=2"; alpha = 0.55, nu = 0.55, N = 4, K = 2)
stress("SIMPLE    capability nu .55 N=3";    alpha = 0.0,  nu = 0.55, N = 3)
stress("SIMPLE    capability nu 0   N=3";    alpha = 0.0,  nu = 0.0,  N = 3)
stress("SIMPLE    capability nu 0   N=4 K=2";alpha = 0.0,  nu = 0.0,  N = 4, K = 2)
stress("SIMPLE    capability nu 0   sigma 12";alpha = 0.0, nu = 0.0,  N = 4, K = 2, sigma = 12.0)
stress("SIMPLE    capability nu 0   Bertrand";alpha = 0.0, nu = 0.0,  N = 4, K = 2, conduct = :bertrand)
stress("SIMPLE    capability nu 0   tariff .3";alpha = 0.0, nu = 0.0, N = 4, K = 2, tariff = 0.3)

# ---------------------------------------------------------------------------
hdr("PART 4  AT THE EQUILIBRIUM: IS THERE EVER MORE THAN ONE?")
say("  Gross substitutes off equilibrium is a means. What matters is whether two")
say("  wage vectors can both clear every labour market. Each economy is solved")
say("  from 20 starts spread over e^4.")
say("")
say(@sprintf("  %-38s %6s %7s %8s %8s %10s", "family", "econ", "MULTI",
             "GS fail", "idx!=+1", "worst GS"))
function equil(nm; reps = 15, starts = 8, kw...)
    rng = MersenneTwister(555)
    ns = 0; nm_ = 0; ngs = 0; nidx = 0; wo = Inf
    for r in 1:reps
        local mod
        try; mod = nasty(MersenneTwister(4000 + r); kw...); catch; continue; end
        cfg = MM.all_active(mod); N = mod.base.N
        found = Vector{Vector{Float64}}()
        for s in 1:starts
            w0 = s == 1 ? ones(N) : exp.(2.0 .* (2 .* rand(rng, N) .- 1)); w0 ./= w0[1]
            local rr
            try
                rr = MM.solve_ge_fixed(mod, cfg; w0 = w0, maxit = 120, tol = 1e-11)
            catch
                continue
            end
            (isfinite(rr.gap) && rr.gap < 1e-9) || continue
            push!(found, rr.w ./ rr.w[1])
        end
        isempty(found) && continue
        ns += 1
        dist = Vector{Vector{Float64}}()
        for f in found
            all(maximum(abs.(log.(f ./ d))) > 1e-5 for d in dist) && push!(dist, f)
        end
        length(dist) > 1 && (nm_ += 1)
        local Mel
        try; Mel = MM.wage_bill_elasticity(mod, dist[1], cfg); catch; continue; end
        all(isfinite, Mel) || continue
        o = minoff(Mel, N); wo = min(wo, o); o <= 0 && (ngs += 1)
        Jr = ((Mel .- I(N)) .* mod.base.L)[2:end, 2:end]
        sign(det(Jr)) == (-1)^(N - 1) || (nidx += 1)
    end
    say(@sprintf("  %-38s %6d %7d %8d %8d %+10.4f", nm, ns, nm_, ngs, nidx, wo))
end
equil("BASELINE  HQ cost  nu .55  N=3";      alpha = 0.55, nu = 0.55, N = 3)
equil("SIMPLE    capability nu 0   N=3";     alpha = 0.0,  nu = 0.0,  N = 3)
equil("SIMPLE    capability nu 0   N=4 K=2"; alpha = 0.0,  nu = 0.0,  N = 4, K = 2)
# (the nu = 0.55 nasty family is dropped here: its price contraction is very slow
#  far from equilibrium, and PART 3 already reports it)

# ---------------------------------------------------------------------------
hdr("PART 5  THE CERTIFICATE: BOTH OF THEOREM 5'S INPUTS, ON A BOX")
say("  Theorem 5 needs (H1) and then one demand condition: Ebar >= 0. If BOTH hold")
say("  on every wage vector of spread <= 2R, gross substitutes holds there and there")
say("  is AT MOST ONE equilibrium of spread <= R. This is that check.")
say("")
say(@sprintf("  %-6s %-8s %5s %10s %10s %10s %10s %8s", "nu", "spread", "pts",
             "min Ebar", "min floor", "min GS", "H1 slack", "GS fails"))
for nu in (0.0, 0.55)
    mC = MM.world_economy(MersenneTwister(20260819); N = 5, K = 4, n_rich = 2,
         n_dom = 6, n_pot_par = 18, zeta = 1.5, hq_gap = 1.3, mne_adv = 0.0,
         adv_slope = 1.2, fscale = 0.0006, conduct = :cournot, nu = nu)[1]
    rC = MM.solve_ge_entry(mC); cfgC = rC.info.cfg; wC = rC.w; PC = rC.info.P
    N = mC.base.N; rng = MersenneTwister(2718)
    for rho in (0.0, 0.4, 0.8, 1.4)
        npts = rho == 0.0 ? 1 : 20
        mE = Inf; mF = Inf; mG = Inf; mH = Inf; nb = 0; ok = 0
        for t in 1:npts
            wv = rho == 0.0 ? copy(wC) : wC .* exp.(rho / 2 .* (2 .* rand(rng, N) .- 1))
            wv ./= wv[1]
            local Hy, fl, Mel, Pl
            try
                Pl = MM.excess_demand_fixed(mC, wv, cfgC; P0 = PC)[2].P
                Hy = MM.wage_hypotheses(mC, wv, cfgC; P0 = Pl)
                fl = MM.reallocation_floor(mC, wv, cfgC; P0 = Pl)
                Mel = MM.wage_bill_elasticity(mC, wv, cfgC; P0 = Pl)
            catch
                continue
            end
            all(isfinite, Mel) || continue
            ok += 1
            for a in 1:N, b2 in 1:N
                a != b2 && (mE = min(mE, Hy.Ebar[a, b2]))
            end
            mF = min(mF, fl.floor); mH = min(mH, 1.0 - Hy.epsmax)
            o = minoff(Mel, N); mG = min(mG, o); o <= 0 && (nb += 1)
        end
        say(@sprintf("  %-6.2f %-8.2f %5d %+10.4f %+10.4f %+10.4f %+10.4f %8d",
                     nu, rho, ok, mE, mF, mG, mH, nb))
    end
end

say("")
say("  READING. The simplification removes the channel that is peculiar to this")
say("  model and provably signs the reallocation block. What survives is the")
say("  ordinary income effect through repatriated profits, which is common to")
say("  every model with non-labour income. It is measured, not assumed away.")
