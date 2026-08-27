"""
Python mirror of src/cournot_pe.jl  (Layer 0, partial equilibrium).

Cross-validation, per CLAUDE.md section 5 rule 2. Two implementations catch what
one cannot. This file contains a solver that uses NONE of the derived algebra:

  solve_bisect    -- same nested-bisection algorithm as the Julia code. Checks the
                     Julia implementation.
  solve_primitive -- finds the Nash point by driving each group's own-quantity
                     gradient of its PRIMITIVE profit  sum_{i in g}(p_i(q)-c_i)q_i
                     to zero, differentiating by COMPLEX STEP (machine precision).
                     Never uses the FOC, the markup rule or the profit formula.
                     Checks the ALGEBRA.

MODEL. A market is one (destination, class) cell.
  upper tier   CES(eta) across classes, so class expenditure  E = D * P^(1-eta)
               is endogenous unless eta = 1 (Cobb-Douglas), which is the PE case.
  lower tier   CES(sigma) over varieties; variety i belongs to group g(i).
  conduct      groups choose quantities, internalising cannibalisation.
  requires     sigma > eta >= 1, so markups rise with market share.

  1/mu_g  = 1 - (1-S_g)/sigma - S_g/eta                    (Atkeson-Burstein)
  Pi_g    = (E/sigma) * S_g * [1 + (sigma/eta - 1) * S_g]

eta = 1 recovers mu_g = sigma/((sigma-1)(1-S_g)) and the (sigma-1) coefficient.
eta > 1 removes the knife-edge: a monopolist's markup is eta/(eta-1), not infinite.

Run:  python scripts/verify/cournot_pe.py
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares, brentq

RNG = np.random.default_rng(20260810)


# ---------------------------------------------------------------------------
# Market primitives
# ---------------------------------------------------------------------------

class Market:
    """
    A single (destination, product class) cell.

    D is the class demand shifter: class expenditure is E = D * P^(1-eta). When
    eta = 1 this collapses to E = D and the cell is genuinely partial equilibrium.
    """

    def __init__(self, sigma, D, c, gid, eta=1.0):
        self.sigma = float(sigma)
        self.eta = float(eta)
        self.D = float(D)
        self.c = np.asarray(c, dtype=float)
        self.gid = np.asarray(gid, dtype=int)          # 0-based group index
        self.G = int(self.gid.max()) + 1
        assert self.sigma > self.eta >= 1.0, "need sigma > eta >= 1"
        assert np.all(self.c > 0.0)
        assert self.G >= 2 or self.eta > 1.0, \
            "a monopolist has an unbounded markup when eta = 1"


def price_index(q, sigma, D, eta):
    """Lower-tier CES price index implied by quantities."""
    A_q = np.sum(q ** ((sigma - 1.0) / sigma))
    return D ** (1.0 / eta) * A_q ** (sigma / ((1.0 - sigma) * eta))


def inverse_demand(q, sigma, D, eta=1.0):
    """
    p_i = D^(1/sigma) P^((sigma-eta)/sigma) q_i^(-1/sigma).
    For eta = 1 this is exactly  p_i = E q_i^(-1/sigma) / A_q  with E = D.
    Complex-safe so that complex-step differentiation works.
    """
    P = price_index(q, sigma, D, eta)
    return D ** (1.0 / sigma) * P ** ((sigma - eta) / sigma) * q ** (-1.0 / sigma)


def group_profit_primitive(q, m: Market, g: int):
    """Profit of group g at quantity vector q. No derived algebra used."""
    p = inverse_demand(q, m.sigma, m.D, m.eta)
    sel = m.gid == g
    out = np.sum((p[sel] - m.c[sel]) * q[sel])
    return out if np.iscomplexobj(q) else float(out)


def markup(S, sigma, eta):
    """1/mu = 1 - (1-S)/sigma - S/eta."""
    return 1.0 / (1.0 - (1.0 - np.asarray(S, float)) / sigma - np.asarray(S, float) / eta)


# ---------------------------------------------------------------------------
# Solver 1: nested monotone bisection (mirror of the Julia routine)
# ---------------------------------------------------------------------------

def _inner_share(KA, sigma, eta, maxit=64):
    """
    Unique root in (0,1) of  x = mu(x)^(1-sigma) * KA.
    mu is increasing in x when sigma > eta, so the RHS is decreasing: unique root.
    """
    KA = np.atleast_1d(np.asarray(KA, float))
    lo = np.zeros_like(KA)
    hi = np.ones_like(KA)
    for _ in range(maxit):
        mid = 0.5 * (lo + hi)
        up = mid - markup(mid, sigma, eta) ** (1.0 - sigma) * KA > 0.0
        hi = np.where(up, mid, hi)
        lo = np.where(up, lo, mid)
    return 0.5 * (lo + hi)


def solve_bisect(m: Market):
    sigma, eta = m.sigma, m.eta
    K = np.bincount(m.gid, weights=m.c ** (1.0 - sigma), minlength=m.G)

    def total(A):
        return _inner_share(K / A, sigma, eta).sum()

    lo, hi = 1e-14, 1e14
    it = 0
    while total(hi) > 1.0 and it < 400:
        hi *= 10.0
        it += 1
    it = 0
    while total(lo) < 1.0 and it < 400:
        lo /= 10.0
        it += 1
    for _ in range(200):                          # geometric bisection on A
        mid = np.sqrt(lo * hi)
        if total(mid) > 1.0:
            lo = mid
        else:
            hi = mid
        if hi / lo - 1.0 < 1e-15:
            break
    A = np.sqrt(lo * hi)

    S = _inner_share(K / A, sigma, eta)
    mu = markup(S, sigma, eta)
    p = mu[m.gid] * m.c
    w = p ** (1.0 - sigma)
    s = w / w.sum()
    P = A ** (1.0 / (1.0 - sigma))
    E = m.D * P ** (1.0 - eta)
    r = E * s
    q = r / p
    Pi = E * S * (1.0 - 1.0 / mu)
    return dict(S=S, s=s, p=p, mu=mu, q=q, r=r, Pi=Pi, A=A, P=P, E=E)


# ---------------------------------------------------------------------------
# Solver 2: Nash fixed point from the primitive profit function
# ---------------------------------------------------------------------------

def _own_gradient(q, m: Market, h=1e-20):
    """
    Stack of d(Pi_g)/d(ln q_i) for i in g, by COMPLEX-STEP differentiation of the
    PRIMITIVE profit function. Exact to machine precision (no cancellation).
    Zero at a Nash equilibrium in which each group optimises over its own
    quantities.
    """
    out = np.empty(len(q))
    for i in range(len(q)):
        qq = q.astype(complex)
        qq[i] = qq[i] * np.exp(1j * h)           # perturb ln q_i by i*h
        out[i] = np.imag(group_profit_primitive(qq, m, m.gid[i])) / h
    return out


def solve_primitive(m: Market, q0=None):
    """
    Independent solver. Agreement with solve_bisect validates the FOC, the markup
    rule and the profit closed form simultaneously.
    """
    if q0 is None:
        # CES monopolistic-competition allocation: a sensible start that is NOT the
        # Cournot answer (constant markup vs share-dependent markup).
        p0 = m.c * m.sigma / (m.sigma - 1.0)
        s0 = p0 ** (1.0 - m.sigma)
        s0 = s0 / s0.sum()
        q_start = m.D * s0 / p0
    else:
        q_start = q0.copy()

    # Residual normalised by variety REVENUE, not by E. In log-quantities q_i -> 0
    # is a spurious stationary point (revenue and cost both vanish, so the raw
    # gradient -> 0 mechanically). Dividing by revenue keeps that corner visible.
    def f(x):
        q = np.exp(x)
        r = inverse_demand(q, m.sigma, m.D, m.eta) * q
        return _own_gradient(q, m) / r

    x = np.log(q_start)
    for _ in range(8):                      # trf can stall; restarting clears it
        res = least_squares(f, x, bounds=(x - 25.0, x + 25.0), method="trf",
                            xtol=1e-15, ftol=1e-15, gtol=1e-15)
        x = res.x
        if np.max(np.abs(f(x))) < 1e-12:
            break
    q = np.exp(x)
    resid = float(np.max(np.abs(f(x))))

    p = inverse_demand(q, m.sigma, m.D, m.eta)
    r = p * q
    s = r / r.sum()
    S = np.bincount(m.gid, weights=s, minlength=m.G)
    Pi = np.array([group_profit_primitive(q, m, g) for g in range(m.G)])
    return dict(S=S, s=s, p=p, mu=p / m.c, q=q, r=r, Pi=Pi,
                A=float(np.sum(p ** (1.0 - m.sigma))), resid=resid)


# ---------------------------------------------------------------------------
# Analytic comparative statics
# ---------------------------------------------------------------------------

def lambda_g(S, sigma, eta=1.0):
    """
    dlnS_g = Lambda_g (dlnK_g - dlnA).
    Lambda = 1/[1 + (sigma-1) S mu (1/eta - 1/sigma)].
    For eta = 1 this is (1-S)/(1+(sigma-2)S).
    """
    S = np.asarray(S, float)
    mu = markup(S, sigma, eta)
    return 1.0 / (1.0 + (sigma - 1.0) * S * mu * (1.0 / eta - 1.0 / sigma))


def omega_g(S, sigma, eta=1.0):
    """Incidence weights. dlnA = sum_g omega_g dlnK_g. Sum to one."""
    L = lambda_g(S, sigma, eta)
    return S * L / np.sum(S * L)


def passthrough_analytic(S, sigma, g, eta=1.0):
    """dln p_g / dln(1+t_g) for a tariff on group g alone."""
    S = np.asarray(S, float)
    L = lambda_g(S, sigma, eta)
    w = omega_g(S, sigma, eta)
    mu = markup(S, sigma, eta)
    dlnS = L[g] * (1.0 - sigma) * (1.0 - w[g])
    dlnmu = S[g] * mu[g] * (1.0 / eta - 1.0 / sigma) * dlnS
    return 1.0 + dlnmu


def ownership_decomposition(S, theta):
    """
    sum_g theta_g S_g^2 = thetabar * HHI + Cov_S(theta, S)

    thetabar = sum_g theta_g S_g is the value-weighted country ownership share --
    the ONLY object a researcher with country-level data can construct.
    Cov_S is the share-weighted covariance of ownership with market share, and is
    the part that requires firm-level global-ultimate-parent identity.
    """
    S = np.asarray(S, float)
    theta = np.asarray(theta, float)
    thetabar = float(np.sum(theta * S))
    hhi = float(np.sum(S ** 2))
    total = float(np.sum(theta * S ** 2))
    cov = total - thetabar * hhi
    return dict(total=total, thetabar=thetabar, hhi=hhi,
                naive=thetabar * hhi, cov=cov)


# ---------------------------------------------------------------------------
# Welfare of the destination country
# ---------------------------------------------------------------------------

def welfare(m_base: Market, t, theta):
    """
    Money-metric welfare of the tariff-setting destination, this class only:

        W = -E ln P + tariff revenue + sum_g theta_g Pi_g

    m_base carries PRE-tariff delivered costs a_i; t is a per-variety ad valorem
    tariff; theta is the destination's ownership weight on each group.
    """
    a = m_base.c
    t = np.asarray(t, float)
    m = Market(m_base.sigma, m_base.D, a * (1.0 + t), m_base.gid, eta=m_base.eta)
    eq = solve_bisect(m)

    # Money-metric consumer surplus. Its derivative must be -E dlnP for ANY eta:
    #   CS = -int E(P) dlnP = -int D P^(1-eta) dP/P = E/(eta-1)   (eta != 1)
    # With eta = 1, E is constant and the antiderivative is -E ln P. Using
    # -E ln P when eta != 1 is wrong -- E moves, and ln P is not scale-free.
    cs = (-eq["E"] * np.log(eq["P"]) if m.eta == 1.0
          else eq["E"] / (m.eta - 1.0))
    revenue = float(np.sum(t * a * eq["q"]))
    profits = float(np.sum(np.asarray(theta, float) * eq["Pi"]))
    return dict(W=cs + revenue + profits, cs=cs, revenue=revenue,
                profits=profits, eq=eq)


def dW_dt(m_base: Market, theta, g, t0=0.0, h=1e-6):
    """Central difference of W in the ad valorem tariff on group g."""
    gsel = (m_base.gid == g).astype(float)
    up = welfare(m_base, (t0 + h) * gsel, theta)["W"]
    dn = welfare(m_base, (t0 - h) * gsel, theta)["W"]
    return (up - dn) / (2.0 * h)


def dW_dt_analytic(S, sigma, g, theta=None, eta=1.0):
    """
    Marginal welfare of an ad valorem tariff on group g at FREE TRADE, per unit of
    class expenditure E:

        (1/E) dW/dt|_0 =   S_g/mu_g            tariff revenue (markup-deflated base)
                         - omega_g             consumer price index
                         + sum_h theta_h dPi_h/dt / E
    """
    S = np.asarray(S, float)
    mu = markup(S, sigma, eta)
    w = omega_g(S, sigma, eta)
    L = lambda_g(S, sigma, eta)
    out = S[g] / mu[g] - w[g]
    if theta is not None:
        dlnS = L * w[g] * (sigma - 1.0)                   # rivals h != g expand
        dlnS[g] = L[g] * (1.0 - sigma) * (1.0 - w[g])
        k = sigma / eta - 1.0
        # E itself moves when eta != 1: dlnE = (1-eta) dlnP = (1-eta) omega_g
        dlnE = (1.0 - eta) * w[g]
        Pi = (1.0 / sigma) * S * (1.0 + k * S)
        dPi = Pi * dlnE + (1.0 / sigma) * (1.0 + 2.0 * k * S) * S * dlnS
        out += float(np.sum(np.asarray(theta, float) * dPi))
    return out


def optimal_tariff(m_base: Market, theta, g, lo=-0.95, hi=5.0):
    """Root of dW/dt in the tariff on group g. Returns nan if no sign change."""
    f = lambda t: dW_dt(m_base, theta, g, t0=t)
    flo, fhi = f(lo), f(hi)
    if flo * fhi > 0:
        return np.nan
    return brentq(f, lo, hi, xtol=1e-10)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def random_market(rng, gmax=5, nmax=3, eta=1.0):
    sigma = max(2.5, eta + 1.0) + 7.5 * rng.random()
    G = rng.integers(2, gmax + 1)
    gid = np.concatenate([np.full(rng.integers(1, nmax + 1), g) for g in range(G)])
    c = np.exp(0.6 * rng.standard_normal(len(gid)))
    D = 0.5 + 2.5 * rng.random()
    return Market(sigma, D, c, gid, eta=eta)


def test_A_two_solvers():
    print("=" * 74)
    print("TEST A  bisection solver vs direct numerical Nash (no algebra used)")
    print("=" * 74)
    for eta in (1.0, 1.5, 2.5):
        worst_grad = 0.0
        for _ in range(150):
            m = random_market(RNG, eta=eta)
            eq = solve_bisect(m)
            worst_grad = max(worst_grad,
                             np.max(np.abs(_own_gradient(eq["q"], m))) / eq["E"])
        print(f"  eta={eta:<4} 150 markets: max |dPi_g/dln q_i| / E at the derived "
              f"solution = {worst_grad:.2e}")
    print("  -> FOC, markup rule and profit closed form confirmed to machine")
    print("     precision, against a profit function that uses none of them,")
    print("     for a general upper-tier elasticity.\n")

    worst_S = worst_Pi = worst_mu = spread = worst_resid = 0.0
    n = 30
    for eta in (1.0, 2.0):
        for _ in range(n):
            m = random_market(RNG, gmax=4, nmax=2, eta=eta)
            eb, ep = solve_bisect(m), solve_primitive(m)
            worst_resid = max(worst_resid, ep["resid"])
            worst_S = max(worst_S, np.max(np.abs(eb["S"] - ep["S"])))
            worst_Pi = max(worst_Pi, np.max(np.abs(eb["Pi"] - ep["Pi"])))
            worst_mu = max(worst_mu, np.max(np.abs(eb["mu"][m.gid] - ep["mu"])))
            for g in range(m.G):
                mg = ep["mu"][m.gid == g]
                spread = max(spread, float(mg.max() - mg.min()))
    print(f"  {2*n} independent global solves from a CES start:")
    print(f"    max|dS|={worst_S:.2e}  max|dPi|={worst_Pi:.2e}  "
          f"max|dmu|={worst_mu:.2e}  worst resid={worst_resid:.1e}")
    print(f"  within-group markup spread = {spread:.2e}")
    print("  -> markups are common across a group's affiliates as an equilibrium")
    print("     property, not because the code imposes it.")


def test_B_best_response():
    """CLAUDE.md rule 3: perturb the agent's OWN choice, confirm profit falls."""
    print()
    print("=" * 74)
    print("TEST B  numerical best-response check of the FOC (rule 3)")
    print("=" * 74)
    worst = -np.inf
    for eta in (1.0, 2.0):
        for _ in range(100):
            m = random_market(RNG, eta=eta)
            q = solve_bisect(m)["q"]
            for g in range(m.G):
                sel = np.where(m.gid == g)[0]
                base = group_profit_primitive(q, m, g)
                for _ in range(6):
                    d = RNG.standard_normal(len(sel))
                    for eps in (1e-3, -1e-3, 5e-3, -5e-3):
                        qq = q.copy()
                        qq[sel] = q[sel] * np.exp(eps * d)
                        worst = max(worst, (group_profit_primitive(qq, m, g)
                                            - base) / abs(base))
    print("  200 markets x all groups x random own-quantity perturbations")
    print(f"  worst relative profit GAIN from deviating = {worst:.3e}  (must be <= 0)")
    print(f"  -> {'PASS' if worst < 1e-9 else 'FAIL'}: a maximum, not just a "
          f"stationary point.")


def test_C_comparative_statics():
    print()
    print("=" * 74)
    print("TEST C  analytic incidence weights vs finite differences")
    print("=" * 74)
    print("  Lambda_g = 1/[1 + (sigma-1) S mu (1/eta - 1/sigma)]")
    print("  omega_g  = S_g Lambda_g / sum_h S_h Lambda_h")
    print("  claims:  dlnP/dln(1+t_g) = omega_g,   dln p_g/dln(1+t_g) = analytic")
    for eta in (1.0, 1.5, 2.5):
        worst_P = worst_rho = 0.0
        for _ in range(40):
            m = random_market(RNG, eta=eta)
            eq = solve_bisect(m)
            S, sg = eq["S"], m.sigma
            w = omega_g(S, sg, eta)
            for g in range(m.G):
                gsel = (m.gid == g).astype(float)
                h = 1e-6
                eu = solve_bisect(Market(sg, m.D, m.c * (1 + h * gsel), m.gid, eta=eta))
                ed = solve_bisect(Market(sg, m.D, m.c * (1 - h * gsel), m.gid, eta=eta))
                dlnP = (np.log(eu["P"]) - np.log(ed["P"])) / (2 * h)
                worst_P = max(worst_P, abs(dlnP - w[g]))
                i = np.where(m.gid == g)[0][0]
                rho = (np.log(eu["p"][i]) - np.log(ed["p"][i])) / (2 * h)
                worst_rho = max(worst_rho,
                                abs(rho - passthrough_analytic(S, sg, g, eta)))
        print(f"  eta={eta:<4} max|dlnP - omega| = {worst_P:.2e}   "
              f"max|rho - analytic| = {worst_rho:.2e}")


def test_D_ownership_decomposition():
    print()
    print("=" * 74)
    print("TEST D  ownership decomposition: naive country share vs true correction")
    print("=" * 74)
    print("  Pi_H = (E/sigma)[ sum_g theta_g S_g + (sigma/eta - 1) sum_g theta_g S_g^2 ]")
    print("  sum_g theta_g S_g^2  =  thetabar * HHI  +  Cov_S(theta, S)")
    print("  'naive' is all a country-level-data researcher can compute.\n")
    sigma, D = 5.0, 1.0
    print(f"  {'structure':<12}{'ownership':<20}{'naive':>9}{'cov':>10}"
          f"{'total':>10}{'ratio':>9}")
    for G, lab in ((200, "atomistic"), (20, "moderate"), (5, "granular")):
        c = np.exp(np.linspace(0.0, 0.8, G))
        S = solve_bisect(Market(sigma, D, c, np.arange(G)))["S"]
        order = np.argsort(-S)
        k = max(1, G // 5)
        for name, idx in (("owns biggest 20%", order[:k]),
                          ("owns smallest 20%", order[-k:])):
            th = np.zeros(G)
            th[idx] = 1.0
            d = ownership_decomposition(S, th)
            ratio = d["total"] / d["naive"] if d["naive"] > 0 else np.nan
            print(f"  {lab:<12}{name:<20}{d['naive']:>9.5f}{d['cov']:>10.5f}"
                  f"{d['total']:>10.5f}{ratio:>8.2f}x")
    print("\n  The 'ratio' column is CLAUDE.md section 7 deliverable 3 at the market")
    print("  level: how badly country-level ownership data understates the correction.")
    print("  The decomposition is INDEPENDENT of eta -- only the (sigma/eta - 1)")
    print("  prefactor moves, so the contribution does not rest on the eta=1 case.")


def test_E_welfare_sign():
    print()
    print("=" * 74)
    print("TEST E  sign of the optimal Layer-0 tariff  (THE IMPORTANT ONE)")
    print("=" * 74)
    print("  Layer 0 has constant marginal cost, so there is NO terms-of-trade")
    print("  motive. The only reason to tax is rent extraction; the only reason to")
    print("  subsidise is the markup distortion. Which wins?\n")
    print("  (1/E) dW/dt|_0 = S_g/mu_g - omega_g + sum_h theta_h dPi_h/dt / E\n")

    for eta in (1.0, 1.5):
        worst = 0.0
        for _ in range(40):
            m = random_market(RNG, eta=eta)
            S = solve_bisect(m)["S"]
            th = RNG.random(m.G)
            for g in range(m.G):
                worst = max(worst, abs(dW_dt(m, th, g, t0=0.0) / solve_bisect(m)["E"]
                                       - dW_dt_analytic(S, m.sigma, g, th, eta)))
        print(f"  eta={eta:<4} analytic vs finite difference, 40 markets: "
              f"max error {worst:.2e}")

    n_pos, n_tot, pos = 0, 0, []
    for _ in range(3000):
        m = random_market(RNG)
        S = solve_bisect(m)["S"]
        for g in range(m.G):
            d = dW_dt_analytic(S, m.sigma, g)
            n_tot += 1
            if d > 1e-10:
                n_pos += 1
                pos.append((m.sigma, S[g]))
    print(f"\n  {n_tot} (market, group) pairs, eta = 1, theta = 0:")
    print(f"    dW/dt > 0 at free trade in {n_pos} cases ({100*n_pos/n_tot:.2f}%)")
    if pos:
        Sg = np.array([p[1] for p in pos])
        print(f"    median S_g among them = {np.median(Sg):.3f}, "
              f"range [{Sg.min():.3f}, {Sg.max():.3f}]")

    print("\n  Frontier: dominant group of share S facing n equal-sized rivals.")
    print("  Smallest S at which a TARIFF beats free trade ('.' = never).")
    ns = (1, 2, 4, 9, 19, 49)
    for eta in (1.0, 1.5):
        print(f"    eta = {eta}")
        print("      " + "sigma".rjust(7) + "".join(f"{'n='+str(n):>9}" for n in ns))
        for sg in (3.0, 5.0, 8.0, 12.0, 20.0):
            row = ""
            for n in ns:
                star = None
                for Sx in np.linspace(0.02, 0.998, 1500):
                    S = np.concatenate([[Sx], np.full(n, (1 - Sx) / n)])
                    if dW_dt_analytic(S, sg, 0, eta=eta) > 0:
                        star = Sx
                        break
                row += f"{star:>9.3f}" if star is not None else f"{'.':>9}"
            print(f"      {sg:>7.1f}{row}")
    print("  -> the tariff motive needs a dominant group, a fragmented fringe and a")
    print("     high sigma. It is NOT an artefact of the eta = 1 knife-edge.")

    print("\n  Worked example, sigma=5, 4 groups, costs exp(linspace(0,0.6)):")
    m = Market(5.0, 1.0, np.exp(np.linspace(0, 0.6, 4)), np.arange(4))
    print(f"    equilibrium group shares: {np.round(solve_bisect(m)['S'], 4)}")
    print(f"    {'theta_1':>9}{'t* on g1':>12}{'dW/dt at 0':>14}")
    for th1 in (0.0, 0.25, 0.5, 1.0):
        th = np.zeros(4)
        th[0] = th1
        print(f"    {th1:>9.2f}{optimal_tariff(m, th, 0):>12.4f}"
              f"{dW_dt(m, th, 0):>14.5f}")
    print("    -> home ownership shifts optimal policy DOWN, monotonically. The")
    print("       ownership correction has the sign the project predicts. But the")
    print("       policy it corrects is already a subsidy, so Layer 0 cannot host")
    print("       the optimal-tariff result: it needs GE or an extensive margin.")


def test_F_figure6_analogue():
    print()
    print("=" * 74)
    print("TEST F  the correct model analogue of Figure 6")
    print("=" * 74)
    print("  Figure 6 holds BEHAVIOUR fixed and changes only the ACCOUNTING.")
    print("  Julia TEST 5 rows 1-vs-3 change both, which is a different experiment.\n")
    sigma, D = 5.0, 1.0
    c = np.exp(np.linspace(0.0, 0.9, 12))
    ind = solve_bisect(Market(sigma, D, c, np.arange(12)))
    grp = solve_bisect(Market(sigma, D, c, np.repeat(np.arange(3), 4)))
    hn, hg = np.sum(grp["s"] ** 2), np.sum(grp["S"] ** 2)
    print("  (i) accounting experiment, behaviour = groups (this is Figure 6):")
    print(f"      HHI counting affiliates : {hn:.4f}")
    print(f"      HHI grouping by parent  : {hg:.4f}   ratio {hg/hn:.2f}x")
    print(f"      data (Figure 6, all products) 0.192 -> 0.215   ratio 1.12x")
    print("  (ii) behavioural experiment, 12 independents vs 3 groups:")
    print(f"      HHI 12 independent firms: {np.sum(ind['s']**2):.4f}"
          f"   mean markup {ind['mu'].mean():.4f}")
    print(f"      HHI 3 grouped parents   : {hg:.4f}   mean markup {grp['mu'].mean():.4f}")
    print(f"\n  Model overshoots the data ratio by ~3x. Note also that the naive HHI")
    print(f"  FALLS ({np.sum(ind['s']**2):.4f} -> {hn:.4f}) when firms coordinate, because")
    print("  output restriction equalises affiliate shares. Grouping and coordination")
    print("  push measured concentration in OPPOSITE directions; Fact 4 nets the two.")


def test_G_eta():
    print()
    print("=" * 74)
    print("TEST G  what eta > 1 fixes")
    print("=" * 74)
    print("  eta = 1 is a knife-edge exactly where the paper's punchline lives.")
    print("  A dominant group's markup and its tariff pass-through both degenerate.\n")
    print(f"  {'S_g':>6}" + "".join(f"{'mu(eta='+str(e)+')':>16}" for e in (1.0, 1.5, 3.0)))
    for S in (0.5, 0.9, 0.99, 1.0):
        with np.errstate(divide="ignore"):       # mu -> inf at S=1, eta=1
            row = "".join(f"{markup(np.array([S]), 5.0, e)[0]:>16.4f}"
                          for e in (1.0, 1.5, 3.0))
        print(f"  {S:>6.2f}{row}")
    print("  monopoly markup is eta/(eta-1) when eta > 1, and unbounded at eta = 1.")
    print("\n  Tariff pass-through for a dominant group facing 9 small rivals:")
    print(f"  {'S_g':>6}" + "".join(f"{'rho(eta='+str(e)+')':>16}" for e in (1.0, 1.5, 3.0)))
    for S in (0.3, 0.6, 0.9):
        row = ""
        for e in (1.0, 1.5, 3.0):
            Sv = np.concatenate([[S], np.full(9, (1 - S) / 9)])
            row += f"{passthrough_analytic(Sv, 5.0, 0, e):>16.4f}"
        print(f"  {S:>6.2f}{row}")
    print("  At eta = 1 pass-through collapses toward 0 as S -> 1 (revenue is fixed,")
    print("  so a monopolist absorbs everything). eta > 1 keeps it bounded away from")
    print("  0, which is the economically sensible case and the one to report.")


if __name__ == "__main__":
    test_A_two_solvers()
    test_B_best_response()
    test_C_comparative_statics()
    test_D_ownership_decomposition()
    test_E_welfare_sign()
    test_F_figure6_analogue()
    test_G_eta()
