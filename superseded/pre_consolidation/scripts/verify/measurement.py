"""
Independent Python mirror of the DENOMINATOR FIX in src/layer1_markets.jl.

The claim being cross-checked is the load-bearing one:

    the Figure 6 estimator is (nearly) INVARIANT to the competitive fringe,
    while the structural HHI collapses as the fringe grows.

If true, Figure 6 cannot identify the fringe, and the fringe must instead be
identified by lambda = the observed share of destination absorption.

This file builds its own panel and its own accounting from scratch. It shares only
the Layer 0 solver with the Julia code, so agreement is informative.

Run:  python scripts/verify/measurement.py
"""

from __future__ import annotations

import numpy as np

from cournot_pe import Market, solve_bisect


def build_panel(rng, n_parents=30, n_origins=4, n_dest=6, n_prod=12,
                n_fringe=8, c_fringe=1.0, presence=0.25, multi_aff=0.35,
                sigma=5.0, eta=1.0):
    """Markets = destination x product. Varieties carry parent/affiliate/origin."""
    next_aff = 0
    sites = [[] for _ in range(n_parents)]          # (origin, affiliate id)
    for g in range(n_parents):
        for o in range(1, n_origins + 1):
            if rng.random() >= 0.45:
                continue
            next_aff += 1
            sites[g].append((o, next_aff))
            if rng.random() < multi_aff:
                next_aff += 1
                sites[g].append((o, next_aff))
        if not sites[g]:
            next_aff += 1
            sites[g].append((int(rng.integers(1, n_origins + 1)), next_aff))

    markets = []
    for d in range(n_dest):
        for k in range(n_prod):
            pr = float(np.clip(presence * np.exp(0.6 * rng.normal()), 0.05, 0.95))
            gid, aff, org, c = [], [], [], []
            for g in range(n_parents):
                if rng.random() >= pr:
                    continue
                for (o, a) in sites[g]:
                    if rng.random() >= 0.6:
                        continue
                    gid.append(g); aff.append(a); org.append(o)
                    c.append(float(np.exp(0.35 * rng.normal())))
            if len(set(gid)) < 2:
                continue
            for j in range(n_fringe):               # the fringe: unobserved, orig = 0
                gid.append(n_parents + j); aff.append(-1 - j); org.append(0)
                c.append(c_fringe)
            markets.append(dict(D=float(0.5 + 2.0 * rng.random()),
                                c=np.array(c), gid=np.array(gid),
                                aff=np.array(aff), org=np.array(org),
                                prod=k, dest=d))
    return dict(sigma=sigma, eta=eta, markets=markets)


def solve_panel(p):
    out = []
    for m in p["markets"]:
        uniq, local = np.unique(m["gid"], return_inverse=True)
        out.append(solve_bisect(Market(p["sigma"], m["D"], m["c"], local,
                                       eta=p["eta"])))
    return out


def structural_hhi(p, eqs):
    """Theory object: HHI over the whole destination market."""
    E = np.array([e["E"] for e in eqs])
    H = np.array([float(np.sum(e["S"] ** 2)) for e in eqs])
    return float(np.sum(E / E.sum() * H))


def sample_share(p, eqs):
    """lambda: observed share of destination absorption."""
    E = np.array([e["E"] for e in eqs])
    lam = np.array([float(np.sum(eqs[i]["s"][p["markets"][i]["org"] > 0]))
                    for i in range(len(eqs))])
    return float(np.sum(E / E.sum() * lam))


def measured_hhi(p, eqs, level="parent"):
    """
    The Figure 6 estimator run on simulated customs records:
    in-sample only -> pool across destinations within a product -> renormalise
    -> HHI -> value-weighted mean across products.
    """
    value = {}
    for i, m in enumerate(p["markets"]):
        d = value.setdefault(m["prod"], {})
        for j in range(len(m["c"])):
            if m["org"][j] == 0:
                continue
            key = (m["aff"][j] if level == "affiliate"
                   else (m["gid"][j], m["org"][j]) if level == "parent_country"
                   else m["gid"][j])
            d[key] = d.get(key, 0.0) + eqs[i]["r"][j]
    hhis, wts = [], []
    for d in value.values():
        tot = sum(d.values())
        if tot <= 0:
            continue
        hhis.append(sum((v / tot) ** 2 for v in d.values()))
        wts.append(tot)
    hhis, wts = np.array(hhis), np.array(wts)
    return float(np.sum(hhis * wts) / np.sum(wts))


if __name__ == "__main__":
    print("=" * 76)
    print("DENOMINATOR FIX  structural HHI vs the Figure 6 estimator")
    print("=" * 76)
    print("  Same seed on every row, so only the fringe size changes.\n")
    print(f"  {'n_fringe':>9}{'lambda':>8}{'structural':>12}{'meas:aff':>12}"
          f"{'meas:p-ctry':>13}{'meas:parent':>13}")
    rows = []
    for nf in (0, 4, 16, 64, 256):
        p = build_panel(np.random.default_rng(2718), n_fringe=nf)
        eqs = solve_panel(p)
        row = (nf, sample_share(p, eqs), structural_hhi(p, eqs),
               measured_hhi(p, eqs, "affiliate"),
               measured_hhi(p, eqs, "parent_country"),
               measured_hhi(p, eqs, "parent"))
        rows.append(row)
        print(f"  {row[0]:>9}{row[1]:>8.4f}{row[2]:>12.5f}{row[3]:>12.5f}"
              f"{row[4]:>13.5f}{row[5]:>13.5f}")

    struct = np.array([r[2] for r in rows])
    meas = np.array([r[5] for r in rows])
    print(f"\n  structural HHI falls by {struct[0]/struct[-1]:.1f}x across the range")
    print(f"  measured   HHI moves by {max(meas)/min(meas):.2f}x across the same range")
    print("  -> the Figure 6 estimator is near-invariant to the fringe, so it")
    print("     CANNOT identify the fringe. lambda must do that job instead.")

    print("\n  Ordering check (must be increasing, as in Figure 6):")
    p = build_panel(np.random.default_rng(2718), n_fringe=16)
    eqs = solve_panel(p)
    a = measured_hhi(p, eqs, "affiliate")
    b = measured_hhi(p, eqs, "parent_country")
    c = measured_hhi(p, eqs, "parent")
    print(f"    model  affiliate {a:.4f} -> parent-country {b:.4f} -> parent {c:.4f}")
    print(f"    data   affiliate 0.1920 -> parent-country 0.2090 -> parent 0.2150")
    print(f"    monotone increasing: {a < b < c}")
    print(f"    model grouping ratio {c/a:.2f}x vs data {0.215/0.192:.2f}x")
