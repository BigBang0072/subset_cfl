#!/usr/bin/env python3
"""Minimal worked example for subset-CFL.

Setup: U -> Y, X -> Y, with X = (X^A, X^B). U is an unobserved confounder.
Three candidate graphs for the mechanism generating X:

    (a)  U -> X^A,  U -> X^B,  X^A -> X^B
    (b)  U -> X^A,             X^A -> X^B      (U hits only the root A)
    (c)             U -> X^B,  X^A -> X^B      (U hits only the leaf B)

All variables are binary, so the X-space has 4 points and every partition is
easy to draw as a colored 2x2 grid.

The four partitions of the X-space compared here:

  observational        f(y | x)                = sum_u f(u|x)              f(y|u,x)
  sub-int. on X^A      f(y | x^B, do(x^A))     = sum_u f(u | x^B, do(x^A)) f(y|u,x)
  sub-int. on X^B      f(y | x^A, do(x^B))     = sum_u f(u | x^A, do(x^B)) f(y|u,x)
  full interventional  f(y | do(x))            = sum_u f(u)                f(y|u,x)

Unifying view (truncated factorization): every regime is a reweighting of U,

    w(u | x)  propto  f(u) * [f(x^A|pa) if A not intervened] * [f(x^B|pa) if B not intervened]

i.e. intervening on a node simply DELETES that node's mechanism factor from
the U-posterior.  do(x) deletes everything and leaves the prior f(u).

The same f(u) and f(y | u, x^A, x^B) are used for all three graphs, so any
difference between the partitions is caused purely by the U-X graph structure.

Exact rational arithmetic (fractions) -> equivalence classes are exact, no
floating point tolerance games.
"""

from fractions import Fraction as F
from itertools import product
import json

U_VALS = (0, 1)
X_VALS = tuple(product((0, 1), (0, 1)))          # x = (x^A, x^B)

# ---------------------------------------------------------------- parameters

fU = {0: F(1, 2), 1: F(1, 2)}                    # f(u)

# f(x^A | u): used when U -> X^A exists (graphs a, b)
fA_u = {(0, 0): F(3, 4), (0, 1): F(1, 4),        # key: (u, a)
        (1, 0): F(1, 4), (1, 1): F(3, 4)}

# f(x^A): used when U is NOT a parent of X^A (graph c)
fA_marg = {0: F(1, 2), 1: F(1, 2)}

# f(x^B | u, x^A): used when U -> X^B exists (graphs a, c)
fB_ua = {(0, 0, 0): F(3, 5), (0, 0, 1): F(2, 5),  # key: (u, a, b)
         (1, 0, 0): F(2, 5), (1, 0, 1): F(3, 5),
         (0, 1, 0): F(4, 5), (0, 1, 1): F(1, 5),
         (1, 1, 0): F(1, 5), (1, 1, 1): F(4, 5)}

# f(x^B | x^A): used when U is NOT a parent of X^B (graph b)
fB_a = {(0, 0): F(3, 5), (0, 1): F(2, 5),        # key: (a, b)
        (1, 0): F(2, 5), (1, 1): F(3, 5)}

# f(y=1 | u, x^A, x^B) -- SHARED by all three graphs.
# Columns are chosen so that pairs average out under the U-prior:
#   {(0,0),(1,1)} both give 1/2 under do(x);  {(0,1),(1,0)} both give 1/5.
fY1 = {(0, (0, 0)): F(1, 5),  (1, (0, 0)): F(4, 5),
       (0, (1, 1)): F(3, 5),  (1, (1, 1)): F(2, 5),
       (0, (0, 1)): F(1, 10), (1, (0, 1)): F(3, 10),
       (0, (1, 0)): F(3, 10), (1, (1, 0)): F(1, 10)}

# ---------------------------------------------------------------- graphs

GRAPHS = {
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

# Regime = which mechanism factors survive in the U-posterior.
REGIMES = {
    "obs":  dict(label="observational f(y|x)",              keep_A=True,  keep_B=True),
    "doA":  dict(label="sub-interventional f(y|x^B,do(x^A))", keep_A=False, keep_B=True),
    "doB":  dict(label="sub-interventional f(y|x^A,do(x^B))", keep_A=True,  keep_B=False),
    "doAB": dict(label="full interventional f(y|do(x))",     keep_A=False, keep_B=False),
}

# ---------------------------------------------------------------- machinery


def u_weights(graph, regime, a, b):
    """w(u|x) under the given regime, via truncated factorization."""
    w = {}
    for u in U_VALS:
        m = fU[u]
        if regime["keep_A"]:
            m *= graph["fA"](u, a)
        if regime["keep_B"]:
            m *= graph["fB"](u, a, b)
        w[u] = m
    z = sum(w.values())
    assert z > 0, "positivity violated"
    return {u: w[u] / z for u in U_VALS}


def p_y1(graph, regime, x):
    """P(Y=1 | x) under the regime (binary Y => this pins down f(y|.))."""
    a, b = x
    w = u_weights(graph, regime, a, b)
    return sum(w[u] * fY1[(u, x)] for u in U_VALS)


def partition(graph, regime):
    """Group the 4 x-points by their exact f(y|.) -> equivalence classes."""
    classes = {}
    for x in X_VALS:
        classes.setdefault(p_y1(graph, regime, x), []).append(x)
    # sort classes by their representative point for stable output
    return sorted(classes.items(), key=lambda kv: kv[1][0])


def refines(P, Q):
    """True iff partition P is a refinement of Q (Q merges classes of P)."""
    blocksQ = [set(xs) for _, xs in Q]
    return all(any(set(xs) <= bq for bq in blocksQ) for _, xs in P)


def fmt_class(xs):
    return "{" + ", ".join(f"({a},{b})" for a, b in xs) + "}"


def fmt_partition(P):
    return "  |  ".join(f"{fmt_class(xs)} -> P(Y=1)={p} = {float(p):.4f}"
                        for p, xs in P)


# ---------------------------------------------------------------- report

ORDER = ["obs", "doA", "doB", "doAB"]
HASSE = [("obs", "doA"), ("obs", "doB"), ("doA", "doAB"), ("doB", "doAB")]

result = {}
for gkey, graph in GRAPHS.items():
    print(f"\n=== Graph ({gkey}): {graph['desc']} ===")
    parts = {r: partition(graph, REGIMES[r]) for r in ORDER}
    for r in ORDER:
        print(f"  {REGIMES[r]['label']:45s} {len(parts[r])} classes")
        print(f"      {fmt_partition(parts[r])}")

    # coarsening hierarchy: finer -> coarser must hold along every Hasse edge
    for lo, hi in HASSE:
        ok = refines(parts[lo], parts[hi])
        print(f"  refinement {lo:4s} <= {hi:4s}: {'OK' if ok else 'VIOLATED'}")
        assert ok, f"hierarchy violated in graph {gkey}: {lo} vs {hi}"

    # collapses: regimes that induce the SAME density everywhere
    dens = {r: tuple(p_y1(graph, REGIMES[r], x) for x in X_VALS) for r in ORDER}
    for i, r1 in enumerate(ORDER):
        for r2 in ORDER[i + 1:]:
            if dens[r1] == dens[r2]:
                print(f"  collapse: {r1} == {r2} (identical densities)")

    result[gkey] = {
        "desc": graph["desc"],
        "pvals": {r: {f"{a}{b}": [str(p_y1(graph, REGIMES[r], (a, b))),
                                  float(p_y1(graph, REGIMES[r], (a, b)))]
                      for a, b in X_VALS} for r in ORDER},
        "classes": {r: [[f"{a}{b}" for a, b in xs] for _, xs in parts[r]]
                    for r in ORDER},
    }

print("\nAll refinement relations verified exactly (rational arithmetic).")

with open("partitions.json", "w") as fh:
    json.dump(result, fh, indent=2)
print("Wrote partitions.json")
