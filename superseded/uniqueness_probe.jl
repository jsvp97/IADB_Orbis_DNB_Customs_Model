include("mne_model.jl")
using Printf, Random, LinearAlgebra
say(x)=(println(x);flush(stdout))

# ---------------------------------------------------------------------------
# PART 1. The markup elasticity, and when a group's WAGE BILL rises with its share.
#
#   A group's labour bill is proportional to  E * S / mu(S).
#   d ln(S/mu) / d ln S = 1 - eps_mu(S),  eps_mu = S mu'/mu.
#   So the bill rises with the share iff eps_mu < 1. That is the monotonicity a
#   gross-substitutes argument would need.
# ---------------------------------------------------------------------------
eps_mu(S,sg,et,cd) = begin
    h=1e-7
    S*(log(MNEModel.markup(S+h,sg,et,cd)) - log(MNEModel.markup(S-h,sg,et,cd)))/(2h)
end
thresh(sg,et,cd) = begin
    t = 1.0
    for S in 0.001:0.001:0.999
        if eps_mu(S,sg,et,cd) >= 1.0; t = S; break; end
    end
    t
end
say("="^74)
say("PART 1  WHEN DOES A GROUP'S WAGE BILL RISE WITH ITS MARKET SHARE?")
say("="^74)
say("  Bill proportional to S/mu(S); rises iff eps_mu(S) < 1.")
say(@sprintf("  %-11s%-7s%-7s%-14s%s","conduct","sigma","eta","threshold","entry frontier (Thm 2)"))
for cd in (:cournot,:bertrand), (sg,et) in ((5.0,1.0),(5.0,1.5),(8.0,1.0),(3.0,1.0))
    say(@sprintf("  %-11s%-7.1f%-7.1f%-14.3f%.3f", cd, sg, et, thresh(sg,et,cd),
                 MNEModel.share_frontier(sg,et,cd)))
end
say("")
say("  Closed form, Cournot eta=1: mu = sigma/[(sigma-1)(1-S)], so eps_mu = S/(1-S),")
say("  which is < 1 exactly when S < 1/2. THE SAME ONE-HALF THRESHOLD AS THEOREM 2.")

# ---------------------------------------------------------------------------
# PART 2. Does GROSS SUBSTITUTES actually hold at the calibrated equilibrium?
#   z_n(w) = labour demand - supply.  Gross substitutes: d z_n / d ln w_m > 0
#   for n != m. If it holds at every equilibrium, index theory delivers uniqueness.
# ---------------------------------------------------------------------------
say("")
say("="^74)
say("PART 2  IS GROSS SUBSTITUTES SATISFIED AT THE SOLVED EQUILIBRIUM?")
say("="^74)
for cd in (:cournot, :bertrand)
    m, aux = MNEModel.world_economy(MersenneTwister(20260819); N=5,K=4,n_rich=2,
                 n_dom=6,n_pot_par=18,zeta=1.5,hq_gap=1.3,mne_adv=0.10,
                 adv_slope=1.2,fscale=0.0006, conduct=cd)
    r = MNEModel.solve_ge_entry(m)
    b = m.base; cfg = r.info.cfg; w = r.w; N = b.N
    ld(wv) = MNEModel.excess_demand_fixed(m, wv, cfg)[2].LD
    base = ld(w); h = 1e-5
    J = zeros(N,N)
    for mm in 1:N
        wp = copy(w); wp[mm] *= exp(h)
        wq = copy(w); wq[mm] *= exp(-h)
        J[:,mm] = (ld(wp) .- ld(wq)) ./ (2h)     # d LD_n / d ln w_m
    end
    offs = [J[n,mm] for n in 1:N, mm in 1:N if n != mm]
    diag = [J[n,n] for n in 1:N]
    say(@sprintf("  conduct = %s", cd))
    say(@sprintf("    off-diagonals d LD_n/d ln w_m (n != m): %d of %d are POSITIVE",
                 count(>(0), offs), length(offs)))
    say(@sprintf("    most negative off-diagonal            : %+.4f", minimum(offs)))
    say(@sprintf("    diagonals d LD_n/d ln w_n             : %d of %d NEGATIVE",
                 count(<(0), diag), N))
    # homogeneity: rows of J times ones should be ~0 (scaling all wages does nothing)
    say(@sprintf("    homogeneity check, max |row sum|      : %.2e", maximum(abs.(sum(J,dims=2)))))
    # index condition on the reduced (non-numeraire) Jacobian
    Jr = J[2:end,2:end]
    ev = eigvals(Jr)
    say(@sprintf("    reduced Jacobian: det = %+.4e, eigenvalues all negative-real? %s",
                 det(Jr), string(all(real.(ev) .< 0))))
    say("")
end
say("  READING. Gross substitutes means every off-diagonal is POSITIVE. Where that")
say("  holds at every equilibrium, uniqueness follows by a standard index argument.")
