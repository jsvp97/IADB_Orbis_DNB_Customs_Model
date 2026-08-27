###############################################################################
# experiments.jl -- THE ONE-OFF GRIDS BEHIND SECTIONS 29-31 OF CLAUDE.md
#
#   `mne_model.jl`, `wage_uniqueness.jl` and `simple_model.jl` are the standing
#   programs and their outputs are committed. This file holds the parameter
#   SCANS that were run once to settle a question, so that the numbers quoted in
#   CLAUDE.md can be reproduced rather than taken on trust. Each is slow (every
#   row is one or two full GE solves with entry), so pick one:
#
#       julia experiments.jl spillover   Fact 5: how big a PRODUCTIVITY spillover?
#       julia experiments.jl fspill      Fact 5: the FIXED-COST spillover mechanism
#       julia experiments.jl rowfact5    Fact 5: the LAMBDA-MARGIN outside supplier
#       julia experiments.jl tariff      Layer 4 REDONE WITH ENTRY: tariff x ownership
#       julia experiments.jl nu          what does dropping the I-O block cost?
#       julia experiments.jl edge        re-fit the MNE productivity edge
#       julia experiments.jl slope       re-fit the capability gradient at nu = 0
#       julia experiments.jl contract    the contraction certificate, calibrated N=5
#       julia experiments.jl all         everything, and go for a walk
#
#   Expected answers are in CLAUDE.md sections 29.4, 30.5, 30.6, 31, 34 and 35.
###############################################################################

include("mne_model.jl")
using Printf, Random, LinearAlgebra
const MM = MNEModel
say(x) = (println(x); flush(stdout))
hdr(x) = (println(); println("="^76); println(x); println("="^76); flush(stdout))

BASE = (N = 5, K = 4, n_rich = 2, n_dom = 6, n_pot_par = 18, zeta = 1.5,
        hq_gap = 1.3, mne_adv = 0.0, adv_slope = 1.2, fscale = 0.0006,
        conduct = :cournot)

"""Facts 1-4 and the uniqueness diagnostics at one parameter vector."""
function snapshot(; kw...)
    m, aux = MM.world_economy(MersenneTwister(20260819); merge(BASE, kw)...)
    r = MM.solve_ge_entry(m)
    b = m.base; lac = aux.lac
    V = MM.export_matrix(m, r.w, r.info); cls = MM.classify(m, r.info)
    sel = [b.loc[a] in lac for a in eachindex(b.par)]
    tot = sum(V[sel, :])
    f = sum(V[sel .& (cls .== :foreign_mne), :]) / tot
    d = sum(V[sel .& (cls .== :domestic_mne), :]) / tot
    fk = Float64[]
    for k in 1:b.K
        nf = de = 0.0
        for a in eachindex(b.par)
            (b.sec[a] == k && b.loc[a] in lac) || continue
            v = sum(V[a, :]); de += v
            cls[a] == :foreign_mne && (nf += v)
        end
        push!(fk, de > 0 ? nf / de : 0.0)
    end
    x = aux.complexity; xb = sum(x) / length(x)
    g2 = sum((x .- xb) .* (fk .- sum(fk) / length(fk))) / sum((x .- xb) .^ 2)
    val = zeros(b.N)
    for a in eachindex(b.par)
        (b.loc[a] in lac && cls[a] == :foreign_mne) || continue
        val[b.hq[a]] += sum(V[a, :])
    end
    sh = val ./ max(sum(val), 1e-12)
    h1 = MM.measured_hhi(b, V, lac; level = :affiliate)
    h3 = MM.measured_hhi(b, V, lac; level = :parent)
    fl = MM.reallocation_floor(m, r.w, r.info.cfg; P0 = r.info.P)
    Mel = MM.wage_bill_elasticity(m, r.w, r.info.cfg; P0 = r.info.P)
    N = b.N
    gs = minimum([Mel[i, j] for i in 1:N, j in 1:N if i != j])
    return (tot = f + d, f = f, d = d, g2 = g2, hhi = sum(sh .^ 2),
            top = maximum(sh), fact4 = h3 / max(h1, 1e-12), floor = fl.floor,
            star = fl.star_violation, gs = gs, regret = r.info.regret,
            m = m, r = r)
end

# ---------------------------------------------------------------------------
function run_spillover()
    hdr("FACT 5: HOW BIG A SPILLOVER WOULD IT TAKE?   (expect: too big -- CLAUDE.md 30.5)")
    kw = (N = 4, K = 4, n_rich = 2, n_dom = 6, n_pot_par = 14, zeta = 1.5,
          hq_gap = 1.3, mne_adv = 0.0, adv_slope = 1.2, fscale = 0.0006,
          conduct = :cournot)
    function fact5(sp; hq_cost = false)
        function nonmne(extra)
            m2, aux2 = MM.world_economy(MersenneTwister(20260819);
                merge(kw, (spill = sp, hq_cost = hq_cost,
                           extra_mne_sector = extra > 0 ? 2 : 0, extra_n = extra))...)
            r2 = MM.solve_ge_entry(m2)
            V2 = MM.export_matrix(m2, r2.w, r2.info); c2 = MM.classify(m2, r2.info)
            s2 = [(m2.base.loc[a] in aux2.lac) && c2[a] == :nonmne &&
                  m2.base.sec[a] == 2 for a in eachindex(m2.base.par)]
            return sum(V2[s2, :])
        end
        v0 = nonmne(0); v1 = nonmne(8)
        return v0, v1, 100 * (v1 / v0 - 1.0)
    end
    say("  data: POSITIVE, +0.24 intensive / +0.13 extensive")
    say(@sprintf("  %-8s %12s %12s %10s", "spill", "no extra", "with extra", "change %"))
    for sp in (0.0, 0.05, 0.10, 0.15, 0.25, 0.40)
        local v0, v1, ch
        try; v0, v1, ch = fact5(sp); catch; say("  $sp failed"); continue; end
        say(@sprintf("  %-8.2f %12.5f %12.5f %+10.1f", sp, v0, v1, ch))
    end
    say("  head-office-cost variant at spill = 0, for comparison:")
    v0, v1, ch = fact5(0.0; hq_cost = true)
    say(@sprintf("  %-8.2f %12.5f %12.5f %+10.1f", 0.0, v0, v1, ch))
end

# ---------------------------------------------------------------------------
function run_fspill()
    hdr("FACT 5: THE FIXED-COST SPILLOVER  (the mechanism the 2026-08-26 test demands)")
    say("  The saturated empirical test says Fact 5 is REAL at half size (+0.087")
    say("  intensive). The productivity spillover cannot flip the model's sign because")
    say("  business stealing works on the ENTRY margin (30.5). This one works on that")
    say("  margin directly: multinational presence in a (country, sector) lowers LOCAL")
    say("  plants' market-access fixed cost,  F *= (1 + n_mne)^(-fspill).")
    say("")
    kw = (N = 4, K = 4, n_rich = 2, n_dom = 6, n_pot_par = 14, zeta = 1.5,
          hq_gap = 1.3, mne_adv = 0.0, adv_slope = 1.2, fscale = 0.0006,
          conduct = :cournot)
    function fact5(fs, sp)
        function nonmne(extra)
            m2, aux2 = MM.world_economy(MersenneTwister(20260819);
                merge(kw, (spill = sp, fspill = fs,
                           extra_mne_sector = extra > 0 ? 2 : 0, extra_n = extra))...)
            r2 = MM.solve_ge_entry(m2)
            V2 = MM.export_matrix(m2, r2.w, r2.info); c2 = MM.classify(m2, r2.info)
            s2 = [(m2.base.loc[a] in aux2.lac) && c2[a] == :nonmne &&
                  m2.base.sec[a] == 2 for a in eachindex(m2.base.par)]
            npair = 0
            for (i, a) in enumerate(eachindex(m2.base.par))
                s2[i] || continue
                npair += count(>(1e-12), V2[a, :])
            end
            return sum(V2[s2, :]), npair
        end
        (v0, p0) = nonmne(0); (v1, p1) = nonmne(8)
        return v0, v1, 100 * (v1 / v0 - 1.0), p0, p1
    end
    say("  data target: POSITIVE (+0.087 intensive / +0.061 extensive, saturated FE)")
    say(@sprintf("  %-8s %-8s %12s %12s %10s %8s %8s", "fspill", "spill",
                 "no extra", "with extra", "change %", "pairs0", "pairs1"))
    for (fs, sp) in ((0.0, 0.0), (0.25, 0.0), (0.5, 0.0), (0.75, 0.0), (1.0, 0.0),
                     (1.5, 0.0), (0.5, 0.15), (1.0, 0.15))
        local v0, v1, ch, p0, p1
        try; v0, v1, ch, p0, p1 = fact5(fs, sp)
        catch e; say("  fspill=$fs spill=$sp failed: $e"); continue; end
        say(@sprintf("  %-8.2f %-8.2f %12.5f %12.5f %+10.1f %8d %8d",
                     fs, sp, v0, v1, ch, p0, p1))
    end
    say("")
    say("  If a row turns the change positive, re-check Facts 1-4 and the uniqueness")
    say("  diagnostics at that point before adopting anything:")
    for fs in (0.5, 1.0)
        local r
        try; r = snapshot(; fspill = fs); catch e; say("  snapshot fspill=$fs failed"); continue; end
        say(@sprintf("  fspill=%-5.2f Fact1 %5.3f (f %5.3f) grad2 %+5.2f HHI %5.3f Fact4 %5.2f | minGS %+7.4f resid %d",
                     fs, r.tot, r.f, r.g2, r.hhi, r.fact4, r.gs, r.regret))
    end
end

# ---------------------------------------------------------------------------
function run_rowfact5()
    hdr("FACT 5: THE LAMBDA-MARGIN OUTSIDE SUPPLIER  (the CLAUDE.md 35 diagnosis, tested)")
    say("  Cost-side spillovers cannot flip Fact 5 because with fixed market spending")
    say("  entry is zero-sum in value (35). This adds the missing margin: a large")
    say("  rest-of-world supplier in every market (row_L > 0), so sample exporters hold")
    say("  only the in-sample absorption share lambda and business stealing from entry")
    say("  falls mostly on the outside supply. Spillovers to locals then have someone")
    say("  to take share FROM. Sample draws are bit-for-bit identical with ROW on.")
    say("")
    kw = (N = 4, K = 4, n_rich = 2, n_dom = 6, n_pot_par = 14, zeta = 1.5,
          hq_gap = 1.3, mne_adv = 0.0, adv_slope = 1.2, fscale = 0.0006,
          conduct = :cournot)
    function fact5(rowL, sp, fs)
        function nonmne(extra)
            m2, aux2 = MM.world_economy(MersenneTwister(20260819);
                merge(kw, (row_L = rowL, spill = sp, fspill = fs,
                           extra_mne_sector = extra > 0 ? 2 : 0, extra_n = extra))...)
            r2 = MM.solve_ge_entry(m2)
            V2 = MM.export_matrix(m2, r2.w, r2.info); c2 = MM.classify(m2, r2.info)
            s2 = [(m2.base.loc[a] in aux2.lac) && c2[a] == :nonmne &&
                  m2.base.sec[a] == 2 for a in eachindex(m2.base.par)]
            npair = 0
            for (i, a) in enumerate(eachindex(m2.base.par))
                s2[i] || continue
                npair += count(>(1e-12), V2[a, :])
            end
            return sum(V2[s2, :]), npair
        end
        (v0, p0) = nonmne(0); (v1, p1) = nonmne(8)
        return v0, v1, 100 * (v1 / v0 - 1.0), p0, p1
    end
    say("  data target: POSITIVE (+0.087 intensive / +0.061 extensive, saturated FE)")
    say(@sprintf("  %-7s %-7s %-7s %12s %12s %10s %7s %7s", "row_L", "spill",
                 "fspill", "no extra", "with extra", "change %", "pairs0", "pairs1"))
    for (rl, sp, fs) in ((0.0, 0.0, 0.0), (12.0, 0.0, 0.0), (12.0, 0.15, 0.0),
                         (12.0, 0.0, 0.5), (12.0, 0.15, 0.5), (12.0, 0.15, 1.0),
                         (12.0, 0.30, 0.5), (24.0, 0.15, 0.5), (48.0, 0.15, 0.5),
                         (96.0, 0.15, 0.5))
        local v0, v1, ch, p0, p1
        try; v0, v1, ch, p0, p1 = fact5(rl, sp, fs)
        catch e; say("  row=$rl spill=$sp fspill=$fs failed: $e"); continue; end
        say(@sprintf("  %-7.1f %-7.2f %-7.2f %12.5f %12.5f %+10.1f %7d %7d",
                     rl, sp, fs, v0, v1, ch, p0, p1))
        flush(stdout)
    end
    say("")
    say("  Facts and uniqueness diagnostics with the outside supplier on:")
    for (rl, sp, fs) in ((12.0, 0.15, 0.5),)
        local r
        try; r = snapshot(; row_L = rl, spill = sp, fspill = fs)
        catch e; say("  snapshot failed: $e"); continue; end
        say(@sprintf("  row=%-5.1f Fact1 %5.3f (f %5.3f) grad2 %+5.2f HHI %5.3f top %5.3f Fact4 %5.2f | minGS %+7.4f resid %d",
                     rl, r.tot, r.f, r.g2, r.hhi, r.top, r.fact4, r.gs, r.regret))
    end
end

# ---------------------------------------------------------------------------
function run_tariff()
    hdr("LAYER 4 REDONE WITH ENTRY: TARIFF x OWNERSHIP  (the 12.3 exercise, current model)")
    say("  Country 1 (rich) taxes ALL its imports at rate t; country 1 owns a share")
    say("  lambda of every foreign-headquartered parent. Welfare = X_1 / P_1 with")
    say("  P_1 = prod_k P_[1,k]^beta_k (eta = 1). With entry, a tariff can now REMOVE")
    say("  affiliates from the market -- the channel the fixed-roster table missed.")
    say("  nu = 0 grid first (fast, the certified variant), then nu = 0.55 spot checks.")
    say("")
    function economy(nu, tar, lam)
        m, aux = MM.world_economy(MersenneTwister(20260819); merge(BASE, (nu = nu,))...)
        b = m.base
        t = copy(b.tariff)
        for l in 2:b.N, k in 1:b.K; t[l, 1, k] = tar; end
        th = copy(b.theta)
        G = size(th, 1)
        for g in 1:G
            h = b.hq[findfirst(==(g), b.par)]
            h == 1 && continue
            th[g, :] .= 0.0
            th[g, 1] = lam; th[g, h] = 1.0 - lam
        end
        b2 = MM.GEModel(b.N, b.K, b.sigma, b.eta, b.beta, b.alpha, b.nu, b.omega,
                        b.L, b.par, b.hq, b.loc, b.sec, b.phi, b.gamma, b.d,
                        t, th, b.conduct)
        MM.GEEntry(b2, m.f, m.fdist, m.fmult), aux
    end
    function outcome(nu, tar, lam; w0 = nothing)
        m, aux = economy(nu, tar, lam)
        r = w0 === nothing ? MM.solve_ge_entry(m) : MM.solve_ge_entry(m; w0 = w0)
        b = m.base
        P1 = prod(r.info.P[1, k]^b.beta[k] for k in 1:b.K)
        X1 = r.info.X[1]
        # foreign affiliates actively selling INTO country 1
        act = 0
        for k in 1:b.K
            cell = r.info.mk[1, k]
            for a in cell.idx
                b.hq[a] != 1 && b.loc[a] != 1 && (act += 1)
            end
        end
        (W = X1 / P1, act = act, w = r.w, gap = r.gap)
    end
    for nu in (0.0, 0.55)
        say(@sprintf("  nu = %.2f", nu))
        say(@sprintf("  %-8s %-8s %14s %10s %10s %8s", "lambda", "t", "welfare_1",
                     "dW% vs t=0", "for.aff@1", "gap"))
        lams = nu == 0.0 ? (0.0, 0.25, 0.5, 0.75, 1.0) : (0.0, 0.5, 1.0)
        tars = nu == 0.0 ? (0.0, 0.05, 0.10, 0.20, 0.30) : (0.0, 0.10, 0.20)
        for lam in lams
            local base_res
            wprev = nothing
            for (i, tar) in enumerate(tars)
                local o
                try
                    o = outcome(nu, tar, lam; w0 = wprev)
                catch e
                    say(@sprintf("  %-8.2f %-8.2f  failed: %s", lam, tar,
                                 sprint(showerror, e))); continue
                end
                wprev = o.w
                i == 1 && (base_res = o)
                say(@sprintf("  %-8.2f %-8.2f %14.6f %+10.3f %10d %8.1e",
                             lam, tar, o.W, 100 * (o.W / base_res.W - 1.0),
                             o.act, o.gap))
            end
            say("")
        end
    end
    say("  Read the dW% column within each lambda block: if the tariff that maximises")
    say("  welfare RISES with lambda, the 12.3 reversal survives entry; the for.aff@1")
    say("  column shows the new channel -- affiliates driven out entirely.")
end

# ---------------------------------------------------------------------------
function run_nu()
    hdr("WHAT DOES DROPPING THE INPUT-OUTPUT BLOCK COST?   (CLAUDE.md 30.6)")
    say(@sprintf("  %-6s %7s %7s %7s %7s %7s %7s | %9s %8s %8s %6s", "nu",
                 "Fact1", "foreign", "grad2", "HHI", "top", "Fact4",
                 "floor", "(*)viol", "min GS", "resid"))
    for nu in (0.55, 0.30, 0.0)
        local r
        try; r = snapshot(; nu = nu); catch e; say("  nu=$nu failed"); continue; end
        say(@sprintf("  %-6.2f %7.3f %7.3f %+7.2f %7.3f %7.3f %7.2f | %+9.4f %8.1e %+8.4f %6d",
                     nu, r.tot, r.f, r.g2, r.hhi, r.top, r.fact4, r.floor,
                     r.star, r.gs, r.regret))
    end
    say("  data   0.46-0.74          +0.43   0.130   0.250    1.12")
end

# ---------------------------------------------------------------------------
function run_edge()
    hdr("RE-FIT THE MNE PRODUCTIVITY EDGE   (expect: 0.0 works -- CLAUDE.md 29.4)")
    say(@sprintf("  %-9s %8s %8s %8s %8s %8s %8s", "mne_adv", "Fact1", "foreign",
                 "domestic", "grad2", "HHI", "top"))
    for adv in (0.10, 0.06, 0.03, 0.00)
        local r
        try; r = snapshot(; mne_adv = adv); catch; say("  $adv failed"); continue; end
        say(@sprintf("  %-9.2f %8.3f %8.3f %8.3f %+8.2f %8.3f %8.3f",
                     adv, r.tot, r.f, r.d, r.g2, r.hhi, r.top))
    end
end

# ---------------------------------------------------------------------------
function run_slope()
    hdr("RE-FIT THE CAPABILITY GRADIENT AT nu = 0   (expect: it stops identifying)")
    say("  At nu = 0 the Fact 2 gradient is pinned by hq_gap, not adv_slope: it sits")
    say("  at +0.46 to +0.47 whatever adv_slope does. CLAUDE.md 30.6.")
    say(@sprintf("  %-11s %8s %8s %8s %7s", "adv_slope", "Fact1", "foreign", "grad2", "resid"))
    for sl in (1.2, 1.0, 0.9, 0.8)
        local r
        try; r = snapshot(; nu = 0.0, adv_slope = sl); catch; say("  $sl failed"); continue; end
        say(@sprintf("  %-11.2f %8.3f %8.3f %+8.2f %7d", sl, r.tot, r.f, r.g2, r.regret))
    end
end

# ---------------------------------------------------------------------------
function run_contract()
    hdr("THE CONTRACTION CERTIFICATE AT THE CALIBRATED ECONOMY   (CLAUDE.md 31)")
    say("  A_k = (1-k)I + kM with the solver's k = 0.25. Entrywise positive means")
    say("  the tatonnement is a Birkhoff contraction: one fixed point, and the")
    say("  iteration converges to it. Expect Birkhoff ~0.82 at the solution.")
    say("")
    say(@sprintf("  %-22s %-7s %5s %9s %9s %9s %9s", "model", "spread", "pts",
                 "min M_nn", "k_max", "min A_k", "Birkhoff"))
    for (nm, nu, cd) in (("nu=.55 cournot", 0.55, :cournot),
                         ("nu=.55 bertrand", 0.55, :bertrand),
                         ("nu=0   cournot", 0.0, :cournot))
        m = MM.world_economy(MersenneTwister(20260819);
                             merge(BASE, (nu = nu, conduct = cd))...)[1]
        r = MM.solve_ge_entry(m)
        rows = MM.contraction_certificate(m, r.info.cfg; w0 = r.w, P0 = r.info.P,
                    radii = (0.0, 0.4, 0.8, 1.4, 2.0), npts = 20)
        for rw in rows
            say(@sprintf("  %-22s %-7.2f %5d %+9.4f %9.4f %+9.4f %9.4f", nm,
                         rw.spread, rw.pts, rw.minMnn, rw.kmax, rw.minA, rw.kappaB))
        end
        vc = MM.verify_contraction(m, r.info.cfg; w0 = r.w, P0 = r.info.P,
                                   rho = 0.8, npairs = 25)
        say(@sprintf("    direct check: %d pairs, worst ratio %.4f, bound %.4f, respected %s",
                     vc.pairs, vc.worst_ratio, vc.bound, vc.respected))
    end
end

# ---------------------------------------------------------------------------
function main()
    what = lowercase(join(ARGS, " "))
    isempty(what) && (what = "help")
    if occursin("help", what)
        println("julia experiments.jl [spillover|nu|edge|slope|contract|all]")
        return
    end
    t0 = time()
    (occursin("all", what) || occursin("spillover", what)) && run_spillover()
    (occursin("all", what) || occursin("fspill", what))    && run_fspill()
    (occursin("all", what) || occursin("rowfact5", what))  && run_rowfact5()
    (occursin("all", what) || occursin("tariff", what))    && run_tariff()
    (occursin("all", what) || occursin("nu", what))        && run_nu()
    (occursin("all", what) || occursin("edge", what))      && run_edge()
    (occursin("all", what) || occursin("slope", what))     && run_slope()
    (occursin("all", what) || occursin("contract", what))  && run_contract()
    @printf("\nTotal %.1f min\n", (time() - t0) / 60)
end

main()
