#!/usr/bin/env python3
"""A strict ladder:  do(A), do(B)  <  union  <  full intervention.

Construction (graph (a): U->X^A, U->X^B, X^A->X^B; all binary, f(u)=1/2):

Every outcome column has the form f(y=1|u,x) = (c_x, 1-c_x), so under the
prior f(u) = (1/2, 1/2) every x gives f(y=1|do(x)) = 1/2: the full
interventional partition is ONE class -- X has no average causal effect on
Y at all.  The weaker regimes are fooled to strictly decreasing degrees:

  observational        4 classes (all singletons)
  do(X^A)              3 classes: merges {00, 11}
  do(X^B)              3 classes: merges {01, 11}
  union = join         2 classes: {00, 01, 11} and {10}
                       -- the pair 00 ~ 01 is EMERGENT: it appears in
                          neither experiment, only by chaining through 11
  full do(x)           1 class: everything (q = 1/2 everywhere)

The point (1,0) has the u-flat column (1/2, 1/2): it answers 1/2 in every
regime, yet no partial regime can certify that any OTHER point equals it,
because their densities deviate from 1/2 until all confounding is severed.

Everything below is exact rational arithmetic; strictness of every step in
the ladder is asserted.
"""

from fractions import Fraction as F
from itertools import combinations, product

XS = tuple(product((0, 1), (0, 1)))
PAIRS = tuple(combinations(range(4), 2))

fU = {0: F(1, 2), 1: F(1, 2)}
fA = {0: F(3, 4), 1: F(1, 4)}                      # f(x^A=0 | u)
fB = {(0, 0): F(7, 10), (1, 0): F(3, 10),          # f(x^B=0 | u, x^A)
      (0, 1): F(9, 10), (1, 1): F(1, 10)}
C = {(0, 0): F(1, 10), (0, 1): F(3, 10),           # f(y=1|u=0,x) = c_x
     (1, 0): F(1, 2),  (1, 1): F(7, 10)}           # f(y=1|u=1,x) = 1-c_x

REGIMES = {"obs": (True, True), "doA": (False, True),
           "doB": (True, False), "full": (False, False)}
ORDER = ["obs", "doA", "doB", "full"]


def p_y1(reg, x):
    keep_a, keep_b = REGIMES[reg]
    a, b = x
    m = {}
    for u in (0, 1):
        w = fU[u]
        if keep_a:
            w *= fA[u] if a == 0 else 1 - fA[u]
        if keep_b:
            w *= fB[(u, a)] if b == 0 else 1 - fB[(u, a)]
        m[u] = w
    z = m[0] + m[1]
    c = C[x]
    return (m[0] * c + m[1] * (1 - c)) / z


def blocks(reg):
    by = {}
    for x in XS:
        by.setdefault(p_y1(reg, x), []).append(x)
    return sorted(tuple(v) for v in by.values())


def join(P, Q):
    parent = {x: x for x in XS}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for R in (P, Q):
        for blk in R:
            for x in blk[1:]:
                parent[find(x)] = find(blk[0])
    g = {}
    for x in XS:
        g.setdefault(find(x), []).append(x)
    return sorted(tuple(sorted(v)) for v in g.values())


def refines(P, Q):
    return all(any(set(p) <= set(q) for q in Q) for p in P)


def strictly_coarser(Q, P):          # Q strictly coarser than P
    return refines(P, Q) and P != Q


def fmt(P):
    return "  ".join("{" + ",".join("%d%d" % x for x in blk) + "}" for blk in P)


parts = {r: blocks(r) for r in ORDER}
uni = join(parts["doA"], parts["doB"])

print("Densities P(Y=1 | regime, x):")
for r in ORDER:
    vals = "  ".join("%d%d:%-6s" % (x[0], x[1], p_y1(r, x)) for x in XS)
    print(f"  {r:5s} {vals}   -> {fmt(parts[r])}")
print(f"  union{'':43s}-> {fmt(uni)}")

# the ladder, every step strict
assert refines(parts["obs"], parts["doA"]) and refines(parts["obs"], parts["doB"])
assert strictly_coarser(parts["doA"], parts["obs"])
assert strictly_coarser(parts["doB"], parts["obs"])
assert strictly_coarser(uni, parts["doA"]), "union must strictly coarsen do(A)"
assert strictly_coarser(uni, parts["doB"]), "union must strictly coarsen do(B)"
assert strictly_coarser(parts["full"], uni), "full must strictly coarsen union"
assert parts["full"] == [tuple(sorted(XS))], "full must be the total merge"

# the emergent pair: 00 ~ 01 in the union but in NEITHER experiment
def same(P, x1, x2):
    return any(x1 in blk and x2 in blk for blk in P)

assert not same(parts["doA"], (0, 0), (0, 1))
assert not same(parts["doB"], (0, 0), (0, 1))
assert same(uni, (0, 0), (0, 1))
print("\nEmergent merge: 00 ~ 01 holds in the union only (chained through 11).")
print("Ladder verified strict at every step:  doA, doB  <  union  <  full.")
