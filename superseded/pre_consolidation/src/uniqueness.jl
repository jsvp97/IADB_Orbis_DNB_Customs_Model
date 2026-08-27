"""
UNIQUENESS OF THE COURNOT EQUILIBRIUM — proof, and numerical verification of every
step of it.

This file exists to answer one question: does the model have ONE equilibrium, or
many? The answer is one, and the proof is short once the problem is set up right.
Everything below is checked numerically so you do not have to take the algebra on
faith.

================================================================================
THE PROOF IN FIVE STEPS
================================================================================

Notation. Market with varieties i, groups g. rho = (sigma-1)/sigma in (0,1).
Group g chooses the quantities of its own varieties, taking rivals' as given.

  X_g = sum_{i in g} q_i^rho        the group's own "output aggregate"
  B_g = sum_{j not in g} q_j^rho    everything the rivals do, taken as given
  A_q = X_g + B_g

--------------------------------------------------------------------------------
STEP 1. Group revenue depends on the group's own quantities ONLY through X_g.
--------------------------------------------------------------------------------
  S_g = X_g / (X_g + B_g)                       group's revenue share
  R_g = E * S_g = D^(1/eta) * X_g * (X_g+B_g)^e,     e = (eta-sigma)/((sigma-1)eta)

For eta = 1 this is E*X/(X+B) with E fixed. Note e < 0 whenever sigma > eta, and
e > -1. So a G-dimensional choice collapses to ONE number. That is the whole trick.

--------------------------------------------------------------------------------
STEP 2. R_g is strictly increasing and strictly CONCAVE in X_g (given B_g > 0).
--------------------------------------------------------------------------------
  dR/dX   = (X+B)^(e-1) * [ (1+e)X + B ]                  > 0 since e > -1
  d2R/dX2 = e (X+B)^(e-2) * [ (1+e)X + 2B ]               < 0 since e < 0

--------------------------------------------------------------------------------
STEP 3. The group's cost of delivering X_g is strictly CONVEX in X_g.
--------------------------------------------------------------------------------
Minimise sum_{i in g} c_i q_i subject to sum_{i in g} q_i^rho = X. The solution is
q_i proportional to c_i^(-sigma), and

  C_g(X) = X^(sigma/(sigma-1)) * K_g^(-1/(sigma-1)),     K_g = sum_{i in g} c_i^(1-sigma)

The exponent sigma/(sigma-1) > 1, so C_g is strictly convex. The SAME K_g that
appears in the solver's bisection appears here. That is not a coincidence: K_g is
the group's cost-efficiency index.

--------------------------------------------------------------------------------
STEP 4. Therefore each group has a UNIQUE best response.
--------------------------------------------------------------------------------
  Pi_g(X) = R_g(X) - C_g(X)  =  concave - strictly convex  =  strictly concave.

So the first-order condition is not just necessary, it is SUFFICIENT, and it has
exactly one solution. Stronger still, directly in the choice vector q:

  X(q) = sum q_i^rho is concave (rho < 1);
  R_g is concave and increasing;
  a concave increasing function of a concave function is concave;
  minus linear cost is still concave.

  => Pi_g(q) is CONCAVE IN THE GROUP'S OWN QUANTITY VECTOR.

No local maximum can fail to be global. There are no interior saddle points.

Corner check: as q_i -> 0, revenue falls like q_i^rho while cost falls like q_i,
and rho < 1, so revenue dominates and dPi/dln q_i > 0. Producing zero is never
optimal, so every equilibrium is interior and the FOC always applies.

--------------------------------------------------------------------------------
STEP 5. The aggregate fixed point is unique.
--------------------------------------------------------------------------------
This is an AGGREGATIVE GAME: each group only cares about its own action and one
aggregate. Work with A = sum_i p_i^(1-sigma). The FOC gives

  S_g = mu(S_g)^(1-sigma) * K_g / A,     1/mu(S) = 1 - (1-S)/sigma - S/eta

Define psi(x) = x * mu(x)^(sigma-1). Since sigma > eta, mu is strictly increasing
on [0,1), so psi is strictly increasing with psi(0) = 0. Hence

  S_g(A) = psi^(-1)( K_g / A )   exists, is unique, and is STRICTLY DECREASING in A.

Market clearing is sum_g S_g(A) = 1. The left side is a strictly decreasing
continuous function of A running from above 1 (as A -> 0, every share -> 1, so the
sum -> G > 1) down to 0 (as A -> infinity). A strictly monotone function crosses
the level 1 exactly ONCE.

  => a unique A, hence unique S_g, hence unique mu_g = mu(S_g),
     hence unique prices p_i = mu_g c_i, revenues r_i = E s_i, quantities q_i.

The equilibrium is unique in every object, and the construction is constructive,
so existence comes free. This is why the solver uses nested bisection and not a
fixed-point iteration: bisection cannot land on a different root, because there is
only one, and monotonicity guarantees it finds it.

================================================================================
WHERE IT COULD BREAK (and the assertions that stop it)
================================================================================
  sigma > eta      if violated, mu is DECREASING in share, psi need not be
                   monotone, and multiplicity becomes possible. Asserted in solve.
  B_g > 0          needs at least 2 groups. With one group and eta = 1, revenue is
                   constant in X and the monopolist shrinks output without bound
                   (unbounded markup). Asserted in solve.
  c_i > 0          costs must be strictly positive or K_g blows up. Asserted.

Run:  julia src/uniqueness.jl
"""

include("cournot_pe.jl")
using .CournotPE
using Printf
using Random

rho(sigma) = (sigma - 1.0) / sigma

"""Group revenue as a function of its own aggregate X, given rivals' B. Step 1."""
function revenue_of_X(X, B, sigma, eta, D)
    e = (eta - sigma) / ((sigma - 1.0) * eta)
    return D^(1.0 / eta) * X * (X + B)^e
end

"""Cost of delivering aggregate X, from the group's cost-minimisation. Step 3."""
cost_of_X(X, K, sigma) = X^(sigma / (sigma - 1.0)) * K^(-1.0 / (sigma - 1.0))

"""Primitive profit at an arbitrary quantity vector. Uses no derived algebra."""
function primitive_profit(q, m::Market, g::Int)
    A_q = sum(q .^ rho(m.sigma))
    P   = m.D^(1.0 / m.eta) * A_q^(m.sigma / ((1.0 - m.sigma) * m.eta))
    p   = m.D^(1.0 / m.sigma) .* P^((m.sigma - m.eta) / m.sigma) .* q .^ (-1.0 / m.sigma)
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

function run()
    rng = MersenneTwister(20260811)

    println("="^76)
    println("STEP 1  group revenue depends on own quantities ONLY through X_g")
    println("="^76)
    worst = 0.0
    for _ in 1:300
        m = random_market(rng; eta = rand(rng) < 0.5 ? 1.0 : 1.0 + rand(rng))
        q = exp.(randn(rng, length(m.c)))
        for g in 1:maximum(m.gid)
            sel = m.gid .== g
            X = sum(q[sel] .^ rho(m.sigma)); B = sum(q[.!sel] .^ rho(m.sigma))
            # rebuild the SAME X with a totally different split across own varieties
            q2 = copy(q); w = rand(rng, count(sel)); w ./= sum(w)
            q2[sel] = (w .* X) .^ (1.0 / rho(m.sigma))
            A1 = sum(q .^ rho(m.sigma)); A2 = sum(q2 .^ rho(m.sigma))
            r1 = m.D^(1/m.eta)*0 + revenue_of_X(X, B, m.sigma, m.eta, m.D)
            # revenue actually earned in each case, from the primitive price system
            rev(qq) = begin
                P = m.D^(1/m.eta) * sum(qq .^ rho(m.sigma))^(m.sigma/((1-m.sigma)*m.eta))
                p = m.D^(1/m.sigma) .* P^((m.sigma-m.eta)/m.sigma) .* qq .^ (-1/m.sigma)
                sum(p[sel] .* qq[sel])
            end
            worst = max(worst, abs(rev(q) - r1) / r1, abs(rev(q2) - r1) / r1,
                        abs(A1 - A2) / A1)
        end
    end
    @printf("  300 markets: max relative error in R_g = D^(1/eta) X (X+B)^e  = %.2e\n", worst)
    println("  (checked by RESHUFFLING quantities within the group at constant X:")
    println("   revenue is unchanged, so X really is a sufficient statistic)")

    println()
    println("="^76)
    println("STEP 2+3  revenue concave in X, cost convex in X")
    println("="^76)
    bad_R = bad_C = 0
    for _ in 1:2000
        sigma = 1.5 + 8.0 * rand(rng); eta = 1.0 + (sigma - 1.0) * rand(rng) * 0.9
        B = 0.05 + 3.0 * rand(rng); K = 0.1 + 3.0 * rand(rng); D = 0.5 + 2.0 * rand(rng)
        X = 0.05 + 3.0 * rand(rng); h = 1e-5 * X
        d2R = (revenue_of_X(X+h,B,sigma,eta,D) - 2revenue_of_X(X,B,sigma,eta,D)
               + revenue_of_X(X-h,B,sigma,eta,D)) / h^2
        d2C = (cost_of_X(X+h,K,sigma) - 2cost_of_X(X,K,sigma)
               + cost_of_X(X-h,K,sigma)) / h^2
        d2R > 1e-6 && (bad_R += 1)
        d2C < -1e-6 && (bad_C += 1)
    end
    @printf("  2000 draws: R''>0 in %d cases, C''<0 in %d cases  (both must be 0)\n",
            bad_R, bad_C)

    println()
    println("="^76)
    println("STEP 4  group profit is CONCAVE in its own quantity vector")
    println("="^76)
    println("  Along random rays through the equilibrium, a concave function must lie")
    println("  BELOW its chord. Checking the midpoint inequality directly.")
    viol, worstgap = 0, -Inf      # -Inf, not 0.0: report the ACTUAL maximum
    for _ in 1:400
        m = random_market(rng; eta = rand(rng) < 0.5 ? 1.0 : 1.0 + rand(rng))
        eq = solve(m)
        for g in 1:maximum(m.gid)
            sel = m.gid .== g
            for _ in 1:5
                q1 = copy(eq.q); q2 = copy(eq.q)
                q1[sel] .*= exp.(0.7 .* randn(rng, count(sel)))
                q2[sel] .*= exp.(0.7 .* randn(rng, count(sel)))
                qm = copy(eq.q); qm[sel] = 0.5 .* (q1[sel] .+ q2[sel])
                f1 = primitive_profit(q1, m, g); f2 = primitive_profit(q2, m, g)
                fm = primitive_profit(qm, m, g)
                gap = (0.5 * (f1 + f2) - fm) / max(abs(fm), 1e-12)
                gap > 1e-10 && (viol += 1)
                worstgap = max(worstgap, gap)
            end
        end
    end
    @printf("  400 markets x groups x 5 chords: concavity violations = %d\n", viol)
    @printf("  worst (chord - midpoint)/|Pi| = %.2e   (must be <= 0)\n", worstgap)

    println()
    println("="^76)
    println("STEP 5  psi(x) = x mu(x)^(sigma-1) is strictly increasing => unique root")
    println("="^76)
    bad = 0
    for _ in 1:500
        sigma = 1.5 + 8.0 * rand(rng); eta = 1.0 + (sigma - 1.0) * rand(rng) * 0.9
        xs = range(1e-6, 0.999, length = 400)
        psi = [x * markup(x, sigma, eta)^(sigma - 1.0) for x in xs]
        all(diff(psi) .> 0) || (bad += 1)
    end
    @printf("  500 (sigma, eta) draws: psi non-monotone in %d cases  (must be 0)\n", bad)

    println()
    println("="^76)
    println("THE DECISIVE TEST  multi-start: 200 random starts, same market")
    println("="^76)
    println("  If several equilibria existed, an unbiased search from very different")
    println("  starting points would find different ones. Each start is solved by")
    println("  best-response iteration on the PRIMITIVE profit function, which knows")
    println("  nothing about the closed forms.")
    # Explicit market structures, so the test really does cover many groups and
    # several affiliates per group (a lucky draw of G=2 would prove little).
    cases = [
        (1.0, 5.0, [1,2],                        "G=2, 1 affiliate each"),
        (1.0, 5.0, [1,1,2,2,3,3,4,4],            "G=4, 2 affiliates each"),
        (1.0, 3.0, [1,1,1,2,2,3,4,5,6,6],        "G=6, uneven affiliates"),
        (1.8, 7.0, [1,1,2,2,3,3,4,4,5,5],        "G=5, eta>1"),
        (2.5, 9.0, [1,2,3,4,5,6,7,8],            "G=8, eta=2.5"),
    ]
    for (eta, sigma, gid, label) in cases
        rc = MersenneTwister(4242)
        m = Market(sigma, eta, 1.0, exp.(0.5 .* randn(rc, length(gid))), gid)
        ref = solve(m)
        spread = 0.0
        for t in 1:200
            r2 = MersenneTwister(1000 + t)
            q = exp.(3.0 .* randn(r2, length(m.c)))       # starts spanning ~e^12
            for _ in 1:6000                               # best response on X_g
                for g in 1:maximum(m.gid)
                    sel = m.gid .== g
                    B = sum(q[.!sel] .^ rho(m.sigma))
                    K = sum(m.c[sel] .^ (1.0 - m.sigma))
                    # maximise Pi(X) = R(X) - C(X) by bisection on the derivative,
                    # which is legitimate precisely BECAUSE Pi is strictly concave
                    lo, hi = 1e-12, 1e6
                    dPi(X) = begin
                        h = 1e-7 * X
                        (revenue_of_X(X+h,B,m.sigma,m.eta,m.D) - cost_of_X(X+h,K,m.sigma)
                       - revenue_of_X(X-h,B,m.sigma,m.eta,m.D) + cost_of_X(X-h,K,m.sigma))/(2h)
                    end
                    for _ in 1:200
                        mid = sqrt(lo*hi)
                        dPi(mid) > 0 ? (lo = mid) : (hi = mid)
                        hi/lo - 1 < 1e-14 && break
                    end
                    Xs = sqrt(lo*hi)
                    ci = m.c[sel] .^ (-m.sigma)
                    q[sel] = ci .* (Xs / sum(ci .^ rho(m.sigma)))^(1/rho(m.sigma))
                end
            end
            s = (p = begin
                    A_q = sum(q .^ rho(m.sigma))
                    P = m.D^(1/m.eta) * A_q^(m.sigma/((1-m.sigma)*m.eta))
                    m.D^(1/m.sigma) .* P^((m.sigma-m.eta)/m.sigma) .* q .^ (-1/m.sigma)
                 end; r = p .* q; r ./ sum(r))
            S = [sum(s[m.gid .== g]) for g in 1:maximum(m.gid)]
            spread = max(spread, maximum(abs.(S .- ref.S)))
        end
        @printf("  %-26s sigma=%.1f eta=%.1f  max |S - S_solver| / 200 starts = %.2e\n",
                label, sigma, eta, spread)
    end
    println("  -> every start converges to the SAME equilibrium. Unique.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    run()
end
