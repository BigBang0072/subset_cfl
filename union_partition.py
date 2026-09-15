#!/usr/bin/env python3
"""Union partition vs. full interventional partition.

The UNION partition is the join of the two sub-interventional partitions:
two points of the X-space merge in it iff they are connected by a chain of
merges made by do(X^A) or do(X^B).  Since f(y|do(x)) coarsens each sub-
interventional partition, it also coarsens their join:

        union  <=  full interventional      (always, a.s.)

Question: is the union EQUAL to the full partition?  This script contrasts
two mechanisms on the same graph (a): U->X^A, U->X^B, X^A->X^B, differing
in a single CPT row f(x^B | u, x^A = 1):

  ALIGNED  (the example of cfl_partitions.py): do(X^A) happens to satisfy
           the same balance equation as the prior, merges {00,11}, and the
           union climbs all the way to the full partition.
  GENERIC  (one row changed): neither sub-intervention merges {00,11}, yet
           do(x) still does -- the union stops strictly below the top.

Why: with f(y=1|u,00) = (1/5, 4/5) and f(y=1|u,11) = (3/5, 2/5), a regime
merges {00,11} iff its residual U-weights solve the linear equation

        (3/5) w(00) + (1/5) w(11) = 2/5        [w = weight on u=0]

The prior w = 1/2 solves it by construction (equal prior means), and the
full intervention forces w = f(u) at every x, so the top always merges the
pair.  A sub-intervention leaves an x-dependent tilt on U; generically that
tilt misses the solution line (GENERIC), though a tuned mechanism can land
exactly on it (ALIGNED: w = (3/5, 1/5)).

In graphs (b) and (c) one sub-intervention already equals the full density,
so union = full automatically -- the gap needs U to confound BOTH components.
Everything below is exact rational arithmetic.
"""

from fractions import Fraction as F
from itertools import product
import json

U_VALS = (0, 1)
X_VALS = tuple(product((0, 1), (0, 1)))          # x = (x^A, x^B)

# ------------------------------------------------------- shared parameters

fU = {0: F(1, 2), 1: F(1, 2)}

fA_u = {(0, 0): F(3, 4), (0, 1): F(1, 4),        # f(x^A|u), key (u, a)
        (1, 0): F(1, 4), (1, 1): F(3, 4)}
fA_marg = {0: F(1, 2), 1: F(1, 2)}               # graph c

fB_a = {(0, 0): F(3, 5), (0, 1): F(2, 5),        # f(x^B|x^A), graph b
        (1, 0): F(2, 5), (1, 1): F(3, 5)}

fY1 = {(0, (0, 0)): F(1, 5),  (1, (0, 0)): F(4, 5),
       (0, (1, 1)): F(3, 5),  (1, (1, 1)): F(2, 5),
       (0, (0, 1)): F(1, 10), (1, (0, 1)): F(3, 10),
       (0, (1, 0)): F(3, 10), (1, (1, 0)): F(1, 10)}

# --------------------------------------------- the two f(x^B|u,x^A) tables

FB_ALIGNED = {(0, 0, 0): F(3, 5),  (0, 0, 1): F(2, 5),
              (1, 0, 0): F(2, 5),  (1, 0, 1): F(3, 5),
              (0, 1, 0): F(4, 5),  (0, 1, 1): F(1, 5),
              (1, 1, 0): F(1, 5),  (1, 1, 1): F(4, 5)}

FB_GENERIC = dict(FB_ALIGNED)                     # only the x^A = 1 row changes
FB_GENERIC.update({(0, 1, 0): F(3, 10), (0, 1, 1): F(7, 10),
                   (1, 1, 0): F(7, 10), (1, 1, 1): F(3, 10)})

VARIANTS = {"aligned": FB_ALIGNED, "generic": FB_GENERIC}

REGIMES = {
    "obs":  dict(keep_A=True,  keep_B=True),
    "doA":  dict(keep_A=False, keep_B=True),
    "doB":  dict(keep_A=True,  keep_B=False),
    "doAB": dict(keep_A=False, keep_B=False),
}
ORDER = ["obs", "doA", "doB", "doAB"]

# ------------------------------------------------------------- machinery


def make_graphs(fB_ua):
    return {
        "a": dict(desc="U->X^A, U->X^B, X^A->X^B",
                  fA=lambda u, a: fA_u[(u, a)],
                  fB=lambda u, a, b: fB_ua[(u, a, b)]),
        "b": dict(desc="U->X^A, X^A->X^B",
                  fA=lambda u, a: fA_u[(u, a)],
                  fB=lambda u, a, b: fB_a[(a, b)]),
        "c": dict(desc="X^A->X^B, U->X^B",
                  fA=lambda u, a: fA_marg[a],
                  fB=lambda u, a, b: fB_ua[(u, a, b)]),
    }


def p_y1(graph, regime, x):
    a, b = x
    w = {}
    for u in U_VALS:
        m = fU[u]
        if regime["keep_A"]:
            m *= graph["fA"](u, a)
        if regime["keep_B"]:
            m *= graph["fB"](u, a, b)
        w[u] = m
    z = sum(w.values())
    return sum(w[u] / z * fY1[(u, x)] for u in U_VALS)


def blocks(graph, regime):
    """Equivalence classes of the X-space as a sorted list of tuples."""
    by_p = {}
    for x in X_VALS:
        by_p.setdefault(p_y1(graph, regime, x), []).append(x)
    return sorted(tuple(xs) for xs in by_p.values())


def join(*partitions):
    """Finest common coarsening: merge whatever any input partition merges."""
    parent = {x: x for x in X_VALS}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for P in partitions:
        for blk in P:
            for x in blk[1:]:
                parent[find(x)] = find(blk[0])
    groups = {}
    for x in X_VALS:
        groups.setdefault(find(x), []).append(x)
    return sorted(tuple(sorted(g)) for g in groups.values())


def refines(P, Q):
    return all(any(set(p) <= set(q) for q in Q) for p in P)


def fmt(P):
    return "  ".join("{" + ",".join(f"{a}{b}" for a, b in blk) + "}" for blk in P)


# --------------------------------------------------------------- report

result = {}
for vname, fB_ua in VARIANTS.items():
    print(f"\n================ mechanism: {vname.upper()} ================")
    for gkey, graph in make_graphs(fB_ua).items():
        parts = {r: blocks(graph, REGIMES[r]) for r in ORDER}
        uni = join(parts["doA"], parts["doB"])
        full = parts["doAB"]

        # hierarchy: obs <= subs <= union <= full, on every edge
        assert refines(parts["obs"], parts["doA"]) and refines(parts["obs"], parts["doB"])
        assert refines(parts["doA"], uni) and refines(parts["doB"], uni)
        assert refines(uni, full), f"union not refined by full: {vname}/{gkey}"

        gap = [blk for blk in full if not any(set(blk) <= set(u) for u in uni)]
        verdict = "union == full" if uni == full else \
                  f"union STRICTLY finer -- full alone merges {fmt(gap)}"
        print(f"graph ({gkey}) {graph['desc']}")
        print(f"    obs   {fmt(parts['obs'])}")
        print(f"    doA   {fmt(parts['doA'])}")
        print(f"    doB   {fmt(parts['doB'])}")
        print(f"    union {fmt(uni)}")
        print(f"    full  {fmt(full)}")
        print(f"    -> {verdict}")

        if gkey == "a":
            result[vname] = {
                "pvals": {r: {f"{a}{b}": [str(p_y1(graph, REGIMES[r], (a, b))),
                                          float(p_y1(graph, REGIMES[r], (a, b)))]
                              for a, b in X_VALS} for r in ORDER},
                "union": [[f"{a}{b}" for a, b in blk] for blk in uni],
                "union_equals_full": uni == full,
            }

# the balance-equation table for the pair {00, 11}
print("\nBalance equation for merging {00,11}:  (3/5) w(00) + (1/5) w(11) = 2/5")
for vname, fB_ua in VARIANTS.items():
    graph = make_graphs(fB_ua)["a"]
    for r in ORDER:
        reg = REGIMES[r]
        w = {}
        for x in ((0, 0), (1, 1)):
            a, b = x
            m = {u: fU[u] * (graph["fA"](u, a) if reg["keep_A"] else 1)
                    * (graph["fB"](u, a, b) if reg["keep_B"] else 1) for u in U_VALS}
            w[x] = m[0] / (m[0] + m[1])
        lhs = F(3, 5) * w[(0, 0)] + F(1, 5) * w[(1, 1)]
        print(f"  {vname:8s} {r:5s} w(00)={w[(0,0)]!s:5s} w(11)={w[(1,1)]!s:5s} "
              f"LHS={lhs!s:6s} {'MERGE' if lhs == F(2,5) else '--'}")

with open("union_partitions.json", "w") as fh:
    json.dump(result, fh, indent=2)
print("\nAll refinements verified exactly. Wrote union_partitions.json")
