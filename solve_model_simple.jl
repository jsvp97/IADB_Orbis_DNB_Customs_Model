###############################################################################
# solve_model_simple.jl
#
# Solves the model of docs/model_section_simple.tex. The code follows the steps
# of docs/model_solution_simple.tex, and the comments name the step and the
# equation of that document that each piece implements.
#
#   julia solve_model_simple.jl          baseline: plants and markets given
#   julia solve_model_simple.jl entry    baseline, then the entry extension
#
# Only the Julia standard library is used (LinearAlgebra, Printf, Random).
#
# Contents
#   0. Primitives                     the Economy, input checks, an example
#   1-3. Demand, costs, one market    shares, markups and prices in a market
#   4. Market sizes and incomes       one linear system
#   5. The labor market               wage bills and excess demand
#   6. Wages                          damped tatonnement, then Newton
#   7. Checks                         the checks listed at the end of the
#                                     solution document, and Theorem 1
#   8. Extension: endogenous entry    ladders, best responses, outer loop
#   9. Main                           runs the example and prints the results
#
# Using it with your own economy
#   eco    = Economy(N, K, L, beta, sigma, d, t, hq, gsec, theta,
#                    pgroup, ploc, phi, fk, fd)     # fields described in Part 0
#   serves = trues(nplants(eco), eco.N)  # or any true/false matrix: who sells where
#   validate(eco, serves)                # the assumptions of the model section
#   sol    = solve_wages(eco, serves)    # sol.w wages (w_1 = 1), sol.r the rest:
#   sol.r.X, sol.r.E, sol.r.mk[n, k]     # incomes, market sizes, market (n, k)
#   check_equilibrium(eco, serves, sol.w)
###############################################################################

using LinearAlgebra, Printf, Random     # the identity is written LinearAlgebra.I below,
                                         # so a user variable named I cannot break it

###############################################################################
# 0. PRIMITIVES
#
# Countries n = 1..N, sectors k = 1..K, groups g = 1..G, plants i = 1..I.
# Every plant belongs to one group; the sector of a plant is its group's sector.
# Which markets a plant serves is a separate object, `serves`, an I x N matrix of
# true/false: serves[i, n] is true when plant i sells in country n. In the
# baseline it is given. In the entry extension it is what groups choose.
###############################################################################

struct Economy
    N::Int
    K::Int
    L::Vector{Float64}        # labor endowment L_n
    beta::Vector{Float64}     # spending share beta_k > 0, sums to one
    sigma::Vector{Float64}    # elasticity of substitution sigma_k > 1
    d::Matrix{Float64}        # iceberg trade cost d[l, n] >= 1, d[n, n] = 1
    t::Array{Float64,3}       # tariff t[l, n, k] >= 0 of country n on goods made in l
    hq::Vector{Int}           # headquarters country h(g)
    gsec::Vector{Int}         # sector k(g)
    theta::Matrix{Float64}    # ownership theta[g, n] >= 0, each row sums to one
    pgroup::Vector{Int}       # group g(i) of plant i
    ploc::Vector{Int}         # location l(i) of plant i
    phi::Vector{Float64}      # productivity phi_i > 0
    fk::Vector{Float64}       # entry only: sector component f_k of the fixed cost
    fd::Matrix{Float64}       # entry only: bilateral component f^d[l, n] >= 1
end

const LAMBDA = 0.25           # damping of the wage update (Step 6)

nplants(eco::Economy) = length(eco.phi)
ngroups(eco::Economy) = length(eco.hq)
plant_sector(eco::Economy, i) = eco.gsec[eco.pgroup[i]]

"A multinational: a group with at least one plant outside its headquarters country."
is_multinational(eco::Economy, g) =
    any(eco.ploc[i] != eco.hq[g] for i in 1:nplants(eco) if eco.pgroup[i] == g)

"Number of groups with at least one plant serving market (n, k)."
ngroups_in(eco::Economy, serves, n, k) =
    length(unique(eco.pgroup[i] for i in 1:nplants(eco) if serves[i, n] && plant_sector(eco, i) == k))

"""
    validate(eco, serves)

Checks the assumptions of the model section and stops with a message if one fails:
spending shares positive and summing to one, sigma_k > 1, d >= 1 with d[n, n] = 1,
tariffs >= 0 with no tariff at home, ownership shares non-negative and summing to
one for each group, positive productivities, and at least two groups in every
market. It also checks the condition of Proposition 4 (existence of wages): for
every non-empty proper subset Z of countries, some plant located in Z serves a
country outside Z. That condition only guarantees existence, so its failure gives
a warning, not an error.
"""
function validate(eco::Economy, serves::BitMatrix)
    N, K, G, nP = eco.N, eco.K, ngroups(eco), nplants(eco)
    require(ok, msg) = ok || error(msg)
    require(length(eco.L) == N && length(eco.beta) == K && length(eco.sigma) == K &&
            size(eco.d) == (N, N) && size(eco.t) == (N, N, K) && size(eco.theta) == (G, N) &&
            length(eco.gsec) == G && length(eco.pgroup) == nP && length(eco.ploc) == nP &&
            length(eco.fk) == K && size(eco.fd) == (N, N) && size(serves) == (nP, N),
            "array sizes do not match N, K, the number of groups and the number of plants")
    require(all(eco.L .> 0), "labor endowments must be positive")
    require(all(eco.beta .> 0) && isapprox(sum(eco.beta), 1; atol = 1e-12),
            "spending shares beta_k must be positive and sum to one")
    require(all(eco.sigma .> 1), "elasticities sigma_k must exceed one")
    require(all(eco.d .>= 1) && all(eco.d[n, n] == 1 for n in 1:N),
            "trade costs must satisfy d >= 1 and d[n, n] = 1")
    require(all(eco.t .>= 0) && all(eco.t[n, n, k] == 0 for n in 1:N, k in 1:K),
            "tariffs must satisfy t >= 0 and t[n, n, k] = 0")
    require(all(eco.theta .>= 0) && all(isapprox.(sum(eco.theta, dims = 2), 1; atol = 1e-12)),
            "ownership shares must be non-negative and sum to one for every group")
    require(all(eco.phi .> 0), "productivities must be positive")
    require(all(eco.fk .>= 0) && all(eco.fd .>= 1), "fixed costs: f_k >= 0 and f^d >= 1")
    for n in 1:N, k in 1:K
        require(ngroups_in(eco, serves, n, k) >= 2,
                "market ($n, $k) is served by fewer than two groups")
    end
    # Proposition 4's condition holds exactly when every country reaches every other
    # through links l -> n, where l -> n means that some plant located in l serves n.
    link = falses(N, N)
    for i in 1:nP, n in 1:N
        serves[i, n] && (link[eco.ploc[i], n] = true)
    end
    for start in 1:N
        reached = falses(N); reached[start] = true; frontier = [start]
        while !isempty(frontier)
            l = pop!(frontier)
            for n in 1:N
                if link[l, n] && !reached[n]
                    reached[n] = true; push!(frontier, n)
                end
            end
        end
        if !all(reached)
            @warn "Proposition 4's condition fails: no plant links country $start to countries $(findall(.!reached)), so wages may not exist"
            return false
        end
    end
    return true
end

"""
    example_economy(; N, K, seed, tariff, n_local, n_mne, fk)

A small economy to illustrate the solver. Productivity is a primitive in the
model, so it is simply drawn here; the quantification will give it a structure.

- countries sit on a line; trade costs rise with distance, d = 1 + 0.3|l - n|;
- in every sector, each country has `n_local` stand-alone firms (one plant each);
- in every sector there are `n_mne` multinationals, each with a plant at
  headquarters and plants in one or two other countries;
- every country levies the same tariff `tariff` on all imports;
- ownership is the benchmark theta[g, n] = 1{n = h(g)}; any theta can be used.
"""
function example_economy(; N = 3, K = 2, seed = 1, tariff = 0.05,
                         n_local = 2, n_mne = 3, fk = 0.0)
    rng = MersenneTwister(seed)
    L = ones(N); L[1] = 2.0
    beta = fill(1.0 / K, K)
    sigma = [K == 1 ? 5.0 : 4.0 + 2.0 * (k - 1) / (K - 1) for k in 1:K]
    d = [1.0 + 0.3 * abs(l - n) for l in 1:N, n in 1:N]
    t = [l == n ? 0.0 : tariff for l in 1:N, n in 1:N, k in 1:K]
    fd = [1.0 + abs(l - n) for l in 1:N, n in 1:N]

    hq = Int[]; gsec = Int[]; pgroup = Int[]; ploc = Int[]; phi = Float64[]
    for k in 1:K
        for n in 1:N, _ in 1:n_local              # stand-alone firms
            push!(hq, n); push!(gsec, k); g = length(hq)
            push!(pgroup, g); push!(ploc, n); push!(phi, exp(0.2 * randn(rng)))
        end
        for m in 1:n_mne                           # multinationals
            h = 1 + (m - 1 + k - 1) % N
            push!(hq, h); push!(gsec, k); g = length(hq)
            capability = exp(0.3 + 0.2 * randn(rng))   # shared by the group's plants
            others = shuffle(rng, setdiff(1:N, h))
            for l in vcat(h, others[1:min(length(others), rand(rng, 1:2))])
                abroad = l == h ? 1.0 : 0.9                # less productive away from home
                push!(pgroup, g); push!(ploc, l)
                push!(phi, capability * abroad * exp(0.1 * randn(rng)))
            end
        end
    end
    G = length(hq)
    theta = [hq[g] == n ? 1.0 : 0.0 for g in 1:G, n in 1:N]
    return Economy(N, K, L, beta, sigma, d, t, hq, gsec, theta,
                   pgroup, ploc, phi, fill(fk, K), fd)
end

###############################################################################
# 1-3. DEMAND, COSTS AND ONE MARKET
#
# Step 1 (demand) gives s_i = (p_i / P)^(1 - sigma) and E_nk = beta_k X_n.
# Step 2 (costs) gives the delivered cost c_i = w_l d_ln (1 + t_lnk) / phi_i.
# Step 3 solves one market, given delivered costs:
#     markup        mu(S) = sigma / ((sigma - 1)(1 - S))              (markup)
#     strength      K_g = sum over the group's plants of c_i^(1 - sigma)
#     share map     psi(S_g) = K_g / A                                 (sharemap)
#     clearing      sum_g S_g(A) = 1                                   (clearing)
# Shares do not depend on spending, so the market is solved before incomes.
###############################################################################

markup(S, s) = s / ((s - 1) * (1 - S))            # mu(S)
lerner(S, s) = (1 + (s - 1) * S) / s              # m(S) = 1 - 1/mu(S)
Lam(S, s)    = (1 - S) / (1 + (s - 2) * S)        # dS/dlnA = -S Lambda
lnpsi0(s)    = (s - 1) * log(s / (s - 1))         # log of the constant in psi

"""
    group_share(x, s)

The root S in (0, 1) of  ln S - (s - 1) ln(1 - S) = x   (equation shareeq).
The left side rises from -Inf to +Inf, so the root is unique. Newton's method,
with a bisection step whenever a Newton step would leave the bracket.
Accuracy is limited only when 1 - S is close to machine precision (a share of
0.9999999...), which equilibrium shares do not reach when two or more groups
serve the market.
"""
function group_share(x, s)
    lo, hi = 0.0, 1.0
    S = x < -30 ? exp(x) : x > 30 ? 1 - exp(-x / (s - 1)) : 1 / (1 + exp(-x))
    S = clamp(S, floatmin(), prevfloat(1.0))
    for _ in 1:200
        f = log(S) - (s - 1) * log1p(-S) - x
        abs(f) < 1e-14 && break
        f > 0 ? (hi = S) : (lo = S)
        Snew = S - f * S * (1 - S) / (1 + (s - 2) * S)   # derivative [1+(s-2)S]/[S(1-S)]
        S = lo < Snew < hi ? Snew : (lo + hi) / 2
        hi - lo <= 4 * eps(S) && break                   # bracket at machine precision
    end
    return S
end

"""
    clear_market(c, gl, s)

Solves one market given delivered costs `c` of the plants that serve it and the
local group label `gl[j]` (1..G) of each plant. Returns the group shares S, the
plant shares s, the group markups mu and ln A.
"""
function clear_market(c::Vector{Float64}, gl::Vector{Int}, s::Float64)
    G = maximum(gl)
    G >= 2 || error("a market needs at least two groups")
    K = zeros(G)
    for j in eachindex(c)
        K[gl[j]] += c[j]^(1 - s)                       # strength (strength)
    end
    shares(lnA) = [group_share(log(K[h]) - lnA - lnpsi0(s), s) for h in 1:G]

    # Bracket. At A_hi = mu(0)^(1-s) sum_g K_g shares add up to at most one;
    # halving A from there reaches a point where they add up to at least one.
    hi = (1 - s) * log(s / (s - 1)) + log(sum(K))
    lo = hi
    while sum(shares(lo)) < 1
        lo -= log(2)
    end
    # Newton on ln A, with derivative -sum_g S_g Lambda_g, safeguarded by bisection.
    lnA = (lo + hi) / 2
    S = shares(lnA)
    for _ in 1:200
        F = sum(S) - 1
        abs(F) < 1e-14 && break
        F > 0 ? (lo = lnA) : (hi = lnA)
        step = lnA + F / sum(S .* Lam.(S, s))
        lnA = lo < step < hi ? step : (lo + hi) / 2
        S = shares(lnA)
        hi - lo < 1e-15 && break
    end
    mu = markup.(S, s)
    splant = [S[gl[j]] * c[j]^(1 - s) / K[gl[j]] for j in eachindex(c)]  # (withingroup)
    return S, splant, mu, lnA
end

"Everything about one market (n, k) that the rest of the model needs."
struct MarketSolution
    plants::Vector{Int}     # plants that serve the market
    groups::Vector{Int}     # groups that serve the market
    gl::Vector{Int}         # local label of each plant's group (index into groups)
    c::Vector{Float64}      # delivered costs c_i
    tar::Vector{Float64}    # tariff t_i on each plant
    S::Vector{Float64}      # group shares S_g
    s::Vector{Float64}      # plant shares s_i
    mu::Vector{Float64}     # group markups mu_g
    lnA::Float64            # ln A, with A = P^(1 - sigma)
    P::Float64              # price index P
    v::Vector{Float64}      # labor cost per dollar spent, s_i / (mu_g (1 + t_i))
    tau::Float64            # tariff revenue per dollar spent, sum_i t_i v_i
end

"The delivered cost of plant i in country n at wages w (Step 2, equation delivered)."
function delivered_cost(eco::Economy, w, i, n)
    l = eco.ploc[i]
    return w[l] * eco.d[l, n] * (1 + eco.t[l, n, plant_sector(eco, i)]) / eco.phi[i]
end

"Solves market (n, k) at wages w for the plants that `serves` says sell there."
function solve_market(eco::Economy, serves::BitMatrix, w, n, k)
    plants = [i for i in 1:nplants(eco) if serves[i, n] && plant_sector(eco, i) == k]
    groups = unique(eco.pgroup[plants])
    gl = [findfirst(==(eco.pgroup[i]), groups) for i in plants]
    c = [delivered_cost(eco, w, i, n) for i in plants]
    tar = [eco.t[eco.ploc[i], n, k] for i in plants]
    S, s, mu, lnA = clear_market(c, gl, eco.sigma[k])
    v = [s[j] / (mu[gl[j]] * (1 + tar[j])) for j in eachindex(plants)]      # (perdollar)
    return MarketSolution(plants, groups, gl, c, tar, S, s, mu, lnA,
                          exp(lnA / (1 - eco.sigma[k])), v, sum(tar .* v))
end

###############################################################################
# 4. MARKET SIZES AND INCOMES        X = wL - theta'Phi + Gamma X     (XofW)
# 5. THE LABOR MARKET                W_n = sum v beta_k X_n' (+ fixed costs)
#
# Gamma[n, n'] is the income country n receives, as profit or tariff revenue,
# out of each dollar country n' spends (equation incomesys). Phi is each group's
# total fixed cost; it is zero in the baseline (fk = 0) and only matters for the
# entry extension, where it is paid to labor where the plant is located and is
# deducted from the group's profit.
###############################################################################

"Fixed cost F_in = f_k w_l f^d_ln of plant i in market n (equation fixed)."
fixed_cost(eco::Economy, w, i, n) =
    eco.fk[plant_sector(eco, i)] * w[eco.ploc[i]] * eco.fd[eco.ploc[i], n]

"""
    evaluate(eco, serves, w)

Steps 2 to 5 at the wage vector w: every market, incomes, market sizes, wage
bills and excess demand for labor. Everything in the model is a function of w.
"""
function evaluate(eco::Economy, serves::BitMatrix, w::Vector{Float64})
    N, K, G = eco.N, eco.K, ngroups(eco)
    mk = [solve_market(eco, serves, w, n, k) for n in 1:N, k in 1:K]

    # Step 4: the income matrix Gamma
    Gam = zeros(N, N)
    for np in 1:N, k in 1:K
        m = mk[np, k]
        for (h, g) in enumerate(m.groups), n in 1:N
            Gam[n, np] += eco.beta[k] * eco.theta[g, n] * m.S[h] * lerner(m.S[h], eco.sigma[k])
        end
        Gam[np, np] += eco.beta[k] * m.tau
    end

    # fixed costs (entry extension only): deducted from profit, paid to labor
    Phi = zeros(G); Wfix = zeros(N)
    for i in 1:nplants(eco), n in 1:N
        serves[i, n] || continue
        F = fixed_cost(eco, w, i, n)
        Phi[eco.pgroup[i]] += F
        Wfix[eco.ploc[i]] += F
    end

    X = (LinearAlgebra.I - Gam) \ (w .* eco.L .- eco.theta' * Phi)       # (XofW)
    E = X * eco.beta'                                       # E[n, k] = beta_k X_n  (spending)

    # Step 5: wage bills and excess demand
    W = copy(Wfix)
    for np in 1:N, k in 1:K
        m = mk[np, k]
        for (j, i) in enumerate(m.plants)
            W[eco.ploc[i]] += E[np, k] * m.v[j]             # (wagebill)
        end
    end
    z = W ./ w .- eco.L                                     # (excess)
    return (mk = mk, Gam = Gam, X = X, E = E, W = W, Phi = Phi, Wfix = Wfix, z = z)
end

###############################################################################
# 6. WAGES
#
# The model reduces to z_n(1, w_2, ..., w_N) = 0 for n = 2..N (equation reduced);
# z_1 = 0 then follows from Walras' law. Two stages:
#   (a) damped tatonnement  w <- w^(1-lambda) (W/L)^lambda, rescaled so w_1 = 1;
#   (b) Newton steps on F_n = ln W_n - ln w_n - ln L_n, whose derivative is
#       M[n, j] - 1{n = j}, with M[n, j] = d ln W_n / d ln w_j.
###############################################################################

maxgap(r, eco) = maximum(abs.(r.z) ./ eco.L)

"""
    log_responses(eco, serves, w; h)

By central differences, one country's wage at a time:
M[n, j] = d ln W_n / d ln w_j  and  dX[n, j] = d ln X_n / d ln w_j.
"""
function log_responses(eco::Economy, serves::BitMatrix, w; h = 1e-5)
    N = eco.N
    M = zeros(N, N); dX = zeros(N, N)
    for j in 1:N
        up = copy(w); up[j] *= exp(h)
        dn = copy(w); dn[j] *= exp(-h)
        ru, rd = evaluate(eco, serves, up), evaluate(eco, serves, dn)
        M[:, j]  = (log.(ru.W) .- log.(rd.W)) ./ (2h)
        dX[:, j] = (log.(ru.X) .- log.(rd.X)) ./ (2h)
    end
    return (M = M, dX = dX)
end

"""
    solve_wages(eco, serves; w0, lambda, tol)

Equilibrium wages with w_1 = 1, for given plant sets. Returns the wages `w`, the
evaluation `r` of the model at those wages, the largest |z_n|/L_n (`gap`), and
whether the tolerance was reached (`converged`).
"""
function solve_wages(eco::Economy, serves::BitMatrix; w0 = ones(eco.N),
                     lambda = LAMBDA, tol = 1e-12)
    w = w0 ./ w0[1]
    # (a) damped tatonnement, until close to the solution
    for _ in 1:5000
        r = evaluate(eco, serves, w)
        maxgap(r, eco) < 1e-6 && break
        w = w .^ (1 - lambda) .* (r.W ./ eco.L) .^ lambda
        w ./= w[1]
    end
    # (b) Newton steps on countries 2..N, halving the step if the gap does not fall
    r = evaluate(eco, serves, w)
    for _ in 1:50
        gap = maxgap(r, eco)
        gap < tol && return (w = w, r = r, gap = gap, converged = true)
        M = log_responses(eco, serves, w).M
        F = log.(r.W[2:end]) .- log.(w[2:end]) .- log.(eco.L[2:end])
        dlnw = -((M[2:end, 2:end] - LinearAlgebra.I) \ F)
        step, moved = 1.0, false
        for _ in 1:30
            wn = copy(w); wn[2:end] .*= exp.(step .* dlnw)
            rn = evaluate(eco, serves, wn)
            if maxgap(rn, eco) < gap
                w, r, moved = wn, rn, true
                break
            end
            step /= 2
        end
        moved || break
    end
    gap = maxgap(r, eco)
    return (w = w, r = r, gap = gap, converged = gap < tol)
end

###############################################################################
# 7. CHECKS
#
# check_equilibrium: the first four checks of the solution document, plus the
#   first-order conditions, computed from the primitives (prices, quantities,
#   costs, inverse demand) rather than from the closed forms the solver uses, so
#   they test the implementation.
# theorem1_check: the hypotheses of Theorem 1 at one wage vector. Theorem 1 needs
#   them at EVERY point of a set D; checking points of a box tests them there and
#   proves nothing between the points.
###############################################################################

"Consumer price index P_n = prod_k (P_nk / beta_k)^beta_k; welfare is X_n / P_n."
cpi(eco::Economy, r, n) = prod((r.mk[n, k].P / eco.beta[k])^eco.beta[k] for k in 1:eco.K)

const CHECKS = ["shares add to one", "price index", "spending = revenue",
                "first-order conditions", "revenue = labor + tariff + profit",
                "labor cost formula", "profit closed form", "income identity"]

function check_equilibrium(eco::Economy, serves::BitMatrix, w; seed = 7)
    N, K, G = eco.N, eco.K, ngroups(eco)
    r = evaluate(eco, serves, w)
    err = Dict(name => 0.0 for name in CHECKS)
    upd!(name, x) = (err[name] = max(err[name], abs(x)))
    profit = zeros(G); TR = zeros(N)
    for n in 1:N, k in 1:K
        m = r.mk[n, k]; s = eco.sigma[k]; E = r.E[n, k]; A = exp(m.lnA)
        upd!("shares add to one", sum(m.S) - 1)
        p = m.mu[m.gl] .* m.c                                   # p_i = mu_g c_i
        upd!("price index", (sum(p .^ (1 - s)) - A) / A)
        q = p .^ (-s) .* m.P^(s - 1) .* E                       # demand (demand)
        rev = p .* q
        upd!("spending = revenue", (sum(rev) - E) / E)

        # First-order conditions (Proposition 1), from the primitive inverse demand:
        # group revenue is E sum_own q^gam / sum_all q^gam, gam = (sigma-1)/sigma, and
        # the derivative of group profit in each own quantity, rivals fixed, is zero.
        gam = (s - 1) / s
        for j in eachindex(q)
            own = findall(==(m.gl[j]), m.gl)
            gprofit(qj) = begin
                qq = copy(q); qq[j] = qj
                E * sum(qq[own] .^ gam) / sum(qq .^ gam) - sum(m.c[own] .* qq[own])
            end
            bump = 1e-6 * q[j]
            upd!("first-order conditions",
                 (gprofit(q[j] + bump) - gprofit(q[j] - bump)) / (2bump * m.c[j]))
        end

        a = [w[eco.ploc[i]] / eco.phi[i] for i in m.plants]     # unit cost (unitcost)
        V = a .* [eco.d[eco.ploc[i], n] for i in m.plants] .* q # labor cost (V)
        tariff = m.tar .* V
        prof = (p .- m.c) .* q
        upd!("revenue = labor + tariff + profit", maximum(abs.(rev .- V .- tariff .- prof) ./ rev))
        upd!("labor cost formula", maximum(abs.(V .- E .* m.v) ./ V))
        for (h, g) in enumerate(m.groups)
            pg = sum(prof[m.gl .== h])
            upd!("profit closed form", (pg - E / s * (m.S[h] + (s - 1) * m.S[h]^2)) / pg)
            profit[g] += pg
        end
        TR[n] += sum(tariff)
    end
    Xcheck = w .* eco.L .+ eco.theta' * (profit .- r.Phi) .+ TR  # (income)
    upd!("income identity", maximum(abs.(Xcheck .- r.X) ./ r.X))

    # Walras' law at arbitrary wages, not only at the solution
    rng = MersenneTwister(seed); walras = 0.0
    for _ in 1:5
        wr = exp.(0.5 .* randn(rng, N)); wr ./= wr[1]
        rr = evaluate(eco, serves, wr)
        walras = max(walras, abs(sum(wr .* rr.z)) / sum(wr .* eco.L))
    end
    return (errors = err, walras = walras, labor = maxgap(r, eco),
            min_X_over_wL = minimum(r.X ./ (w .* eco.L)))
end

"""
    theorem1_check(eco, serves, w)

At wages w: the largest group share in any market (hypothesis (i): at most 1/2),
the demand term Ebar[n, j] (hypothesis (ii): positive for n != j), the elasticity
matrix M, and the bound M[n, j] >= Ebar[n, j] that Theorem 1 derives from them.

Ebar[n, j] = sum over n' of (share of W_n paid out of sales in n') * d ln X_n'/d ln w_j,
since E_n'k = beta_k X_n'. Fixed costs, in the entry extension, count in W_n with
a zero response.
"""
function theorem1_check(eco::Economy, serves::BitMatrix, w)
    N, K = eco.N, eco.K
    r = evaluate(eco, serves, w)
    maxS = maximum(maximum(r.mk[n, k].S) for n in 1:N, k in 1:K)
    b = zeros(N, N)                    # b[n, n'] = share of W_n paid out of sales in n'
    for np in 1:N, k in 1:K
        m = r.mk[np, k]
        for (j, i) in enumerate(m.plants)
            b[eco.ploc[i], np] += r.E[np, k] * m.v[j]
        end
    end
    b ./= r.W
    M, dX = log_responses(eco, serves, w)
    Ebar = b * dX                      # demand term (demandterm)
    off = [(n, j) for n in 1:N, j in 1:N if n != j]
    return (maxS = maxS,
            minEbar = minimum(Ebar[n, j] for (n, j) in off),
            minM = minimum(M[n, j] for (n, j) in off),
            minbound = minimum(M[n, j] - Ebar[n, j] for (n, j) in off),
            rowsum = maximum(abs.(sum(M, dims = 2) .- 1)),
            M = M, Ebar = Ebar)
end

###############################################################################
# 8. EXTENSION: ENDOGENOUS ENTRY
#
# Each group chooses, in every market of its sector, which of its plants serve it.
# A plant serving market n pays F_in = f_k w_l f^d_ln. Given the aggregate A,
# a set T is worth  U_g(T; A) = Omega(K(T); A) - Phi(T)                (payoff)
#     Omega(K; A) = (E/sigma)(S + kappa S^2)  at  S = psi^{-1}(K/A),  kappa = sigma - 1.
# The ladder keeps the sets no other set beats on both strength and fixed cost,
# ordered by strength. The best response is the rung with the highest payoff.
# An entry equilibrium of a market: every group's set is a best response at A,
# and A clears the market with those sets. (The properties of Theorem 2 hold for
# sigma >= 2; the algorithm below does not rely on them.)
#
# General equilibrium with entry (outer loop of the solution document):
#   1. start with every potential plant serving every market of its sector;
#   2. solve wages for the current plant sets;
#   3. in every market, compute each group's best response at the market's A;
#   4. stop if no group gains;
#   5. otherwise the groups with the largest gains switch (a quarter of those
#      that gain, at least one), never leaving a market with fewer than two
#      groups, and return to 2.
# The loop is not guaranteed to converge, and plant sets are whole plants, so an
# exact equilibrium need not exist. The loop keeps the configuration with the
# fewest groups that wish to move and reports how many there are.
###############################################################################

"Omega(K; A): profit of a group with strength K at ln A in a market of size E."
function Omega(K, lnA, E, s)
    K <= 0 && return 0.0
    S = group_share(log(K) - lnA - lnpsi0(s), s)
    return E / s * (S + (s - 1) * S^2)
end

"""
    ladder(c, F, s)

The efficient plant sets of one group in one market, ordered by strength.
`c` and `F` are the delivered and fixed costs of the group's potential plants.
Each rung is (K, Phi, mask): bit j of `mask` is 1 when the j-th plant is in the set.
All 2^p sets are enumerated, which is fine for groups with a handful of plants.
"""
function ladder(c::Vector{Float64}, F::Vector{Float64}, s)
    p = length(c)
    sets = map(0:(2^p - 1)) do mask
        K, Phi = 0.0, 0.0
        for j in 1:p
            if (mask >> (j - 1)) & 1 == 1
                K += c[j]^(1 - s); Phi += F[j]
            end
        end
        (K = K, Phi = Phi, mask = mask)
    end
    dominated(x) = any(y -> y.K >= x.K && y.Phi <= x.Phi && (y.K > x.K || y.Phi < x.Phi), sets)
    return sort([x for x in sets if !dominated(x)], by = x -> x.K)
end

"""
    entry_deviations(eco, serves, w, r)

At wages w (with r = evaluate(eco, serves, w)), every group that would gain by
changing its plant set in some market, given that market's aggregate A.
"""
function entry_deviations(eco::Economy, serves::BitMatrix, w, r; reltol = 1e-10)
    devs = []
    for n in 1:eco.N, k in 1:eco.K
        m = r.mk[n, k]; s = eco.sigma[k]; E = r.E[n, k]
        for g in findall(==(k), eco.gsec)
            own = findall(==(g), eco.pgroup)             # the group's potential plants
            c = [delivered_cost(eco, w, i, n) for i in own]
            F = [fixed_cost(eco, w, i, n) for i in own]
            lad = ladder(c, F, s)
            pay = [Omega(x.K, m.lnA, E, s) - x.Phi for x in lad]
            best = argmax(pay)
            act = [serves[i, n] for i in own]
            Unow = Omega(sum(c[act] .^ (1 - s); init = 0.0), m.lnA, E, s) - sum(F[act]; init = 0.0)
            gain = pay[best] - Unow
            if gain > reltol * E
                push!(devs, (n = n, k = k, g = g, own = own, mask = lad[best].mask, gain = gain))
            end
        end
    end
    return devs
end

function solve_with_entry(eco::Economy; frac = 0.25, maxouter = 60, maxstalls = 8)
    serves = trues(nplants(eco), eco.N)              # 1. every potential plant active
    w = ones(eco.N)
    best = (nwant = typemax(Int), serves = copy(serves), w = copy(w))
    stalls, iters = 0, 0
    for it in 1:maxouter
        iters = it
        sol = solve_wages(eco, serves; w0 = w)       # 2. wages for these plant sets
        w = sol.w
        devs = entry_deviations(eco, serves, w, sol.r)   # 3. best responses
        if length(devs) < best.nwant
            best = (nwant = length(devs), serves = copy(serves), w = copy(w)); stalls = 0
        else
            stalls += 1
        end
        (isempty(devs) || stalls >= maxstalls) && break  # 4. stop
        sort!(devs, by = x -> -x.gain)                   # 5. largest gains switch
        for x in devs[1:max(1, ceil(Int, frac * length(devs)))]
            old = serves[x.own, x.n]
            for (j, i) in enumerate(x.own)
                serves[i, x.n] = (x.mask >> (j - 1)) & 1 == 1
            end
            if ngroups_in(eco, serves, x.n, x.k) < 2     # would leave one group: undo
                serves[x.own, x.n] = old
            end
        end
    end
    serves = best.serves
    sol = solve_wages(eco, serves; w0 = best.w)
    devs = entry_deviations(eco, serves, sol.w, sol.r)
    return (w = sol.w, r = sol.r, gap = sol.gap, converged = sol.converged,
            serves = serves, nwant = length(devs), outer = iters)
end

###############################################################################
# 9. MAIN
###############################################################################

function report(eco::Economy, serves::BitMatrix, sol; title = "")
    r = sol.r
    println("\n", title)
    @printf("  labor-market gap max|z_n|/L_n = %.1e   converged: %s\n", sol.gap, sol.converged)
    println("  country      wage    income X   X/(wL)   welfare X/P")
    for n in 1:eco.N
        @printf("  %4d    %8.4f  %10.4f  %7.4f  %12.4f\n", n, sol.w[n], r.X[n],
                r.X[n] / (sol.w[n] * eco.L[n]), r.X[n] / cpi(eco, r, n))
    end
    println("  market (n,k)   groups   plants   largest group share   group HHI   price index")
    for n in 1:eco.N, k in 1:eco.K
        m = r.mk[n, k]
        @printf("  (%d,%d)          %4d     %4d          %6.3f            %6.3f     %8.4f\n",
                n, k, length(m.groups), length(m.plants), maximum(m.S), sum(m.S .^ 2), m.P)
    end
end

function report_checks(eco::Economy, serves::BitMatrix, w; box = 0.2, npts = 20, seed = 3)
    ch = check_equilibrium(eco, serves, w)
    println("\n  Checks at the solution (largest relative error):")
    for name in CHECKS
        @printf("    %-36s %.1e%s\n", name, ch.errors[name],
                name == "first-order conditions" ? "   (finite differences)" : "")
    end
    @printf("    %-36s %.1e\n", "labor markets clear", ch.labor)
    @printf("    %-36s %.1e   (5 random wage vectors)\n", "Walras' law off equilibrium", ch.walras)
    @printf("    %-36s %.4f   (Proposition 3: >= 1 without fixed costs)\n",
            "smallest X_n/(w_n L_n)", ch.min_X_over_wL)

    println("\n  Theorem 1 hypotheses at w* and at $npts random points with w_1 = 1 and")
    println("  |ln w_n - ln w*_n| <= $box for n >= 2:")
    println("    point          max share (<=1/2)   min Ebar (>0)   min M offdiag   min(M - Ebar)")
    rng = MersenneTwister(seed)
    worst = (maxS = 0.0, minEbar = Inf, minM = Inf, minbound = Inf)
    Mnn, rowsum = 0.0, 0.0
    for p in 0:npts
        wp = copy(w)
        p > 0 && (wp[2:end] .*= exp.(box .* (2 .* rand(rng, eco.N - 1) .- 1)))
        t1 = theorem1_check(eco, serves, wp)
        worst = (maxS = max(worst.maxS, t1.maxS), minEbar = min(worst.minEbar, t1.minEbar),
                 minM = min(worst.minM, t1.minM), minbound = min(worst.minbound, t1.minbound))
        if p == 0
            @printf("    w*                 %8.4f          %+8.4f        %+8.4f        %+8.4f\n",
                    t1.maxS, t1.minEbar, t1.minM, t1.minbound)
            Mnn = minimum(t1.M[n, n] for n in 1:eco.N)
            rowsum = t1.rowsum
        end
    end
    @printf("    worst of all %2d     %8.4f          %+8.4f        %+8.4f        %+8.4f\n",
            npts + 1, worst.maxS, worst.minEbar, worst.minM, worst.minbound)
    println("    (M - Ebar >= 0 is the bound of Theorem 1. M is computed by finite differences,")
    println("     so values of order -1e-6 would be numerical error. Checking points proves")
    println("     nothing between them.)")
    @printf("  Rows of M sum to one (homogeneity): largest error %.1e\n", rowsum)
    @printf("  Damping: min M_nn = %+.3f at w*, so the update needs lambda < %.3f; the solver uses %.2f\n",
            Mnn, min(1.0, 1 / (1 - Mnn)), LAMBDA)
end

function multistart(eco::Economy, serves::BitMatrix, w; nstarts = 5, seed = 11)
    rng = MersenneTwister(seed); gap = 0.0; failed = 0
    for _ in 1:nstarts
        w0 = exp.(1.0 .* randn(rng, eco.N))
        sol = solve_wages(eco, serves; w0 = w0)
        sol.converged || (failed += 1)
        gap = max(gap, maximum(abs.(log.(sol.w) .- log.(w))))
    end
    @printf("\n  Solved again from %d random starting wages (%d did not converge):\n", nstarts, failed)
    @printf("  largest difference in ln w from the solution above = %.1e\n", gap)
end

function main(args)
    # Baseline: plants and the markets they serve are given. Here every plant
    # serves every country of the world, as an export platform.
    eco = example_economy()
    serves = trues(nplants(eco), eco.N)
    validate(eco, serves)
    @printf("Example economy: %d countries, %d sectors, %d groups (%d multinationals), %d plants\n",
            eco.N, eco.K, ngroups(eco), count(g -> is_multinational(eco, g), 1:ngroups(eco)),
            nplants(eco))
    sol = solve_wages(eco, serves)
    report(eco, serves, sol; title = "BASELINE (plants given)")
    report_checks(eco, serves, sol.w)
    multistart(eco, serves, sol.w)

    if "entry" in args
        # With few firms per market each plant is large, so larger fixed costs
        # often leave one or two groups wishing to move: the integer problem.
        eco_e = example_economy(fk = 0.001)
        validate(eco_e, trues(nplants(eco_e), eco_e.N))
        sol_e = solve_with_entry(eco_e)
        report(eco_e, sol_e.serves, sol_e;
               title = "EXTENSION: ENDOGENOUS ENTRY (f_k = $(eco_e.fk[1]))")
        @printf("  active plant-market pairs: %d of %d potential   outer iterations: %d\n",
                count(sol_e.serves), length(sol_e.serves), sol_e.outer)
        @printf("  groups that wish to change their plant set: %d %s\n", sol_e.nwant,
                sol_e.nwant == 0 ? "(an entry equilibrium in every market)" :
                                   "(no exact equilibrium in whole plants was reached)")
        report_checks(eco_e, sol_e.serves, sol_e.w)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
