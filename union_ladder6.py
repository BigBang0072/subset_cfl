#!/usr/bin/env python3
"""Strict ladder with a NON-TRIVIAL top:  |X^A| = 2, |X^B| = 3.

Graph (a): U -> X^A, U -> X^B, X^A -> X^B; U, X^A, Y binary, X^B ternary.

  observational        6 classes (all singletons)
  do(X^A)              5: merges {00, 11}
  do(X^B)              4: merges {01, 11} and {02, 12}
  union = join         3: {00, 01, 11} (emergent 00 ~ 01), {02, 12}, {10}
  full do(x)           2: kappa1 = {00, 01, 10, 11} at 1/2,
                          kappa2 = {02, 12} at 1/5      <- NOT trivial

Outcome columns: the four kappa1 points have mirrored columns (c, 1-c),
c = 1/10, 3/10, 1/2, 7/10 (prior mean 1/2); the two kappa2 points have the
columns (1/10, 3/10) and (3/10, 1/10) (prior mean 1/5).  The u-flat loner
(1,0) reads 1/2 in every regime but joins kappa1 only at the top; the
kappa2 pair, by contrast, is already visible to do(X^B) -- some causal
equivalences ARE discoverable by single-node experiments, others are not.

Exact rational arithmetic; every step of the ladder asserted strict.
"""

from fractions import Fraction as F
from itertools import combinations, product

A_VALS, B_VALS = (0, 1), (0, 1, 2)
XS = tuple(product(A_VALS, B_VALS))

fU = {0: F(1, 2), 1: F(1, 2)}
fA = {0: F(3, 4), 1: F(1, 4)}                     # f(x^A=0 | u)
fB = {(0, 0): (F(7, 10), F(3, 20), F(3, 20)),     # f(x^B=. | u, x^A)
      (1, 0): (F(3, 10), F(3, 10), F(2, 5)),
      (0, 1): (F(2, 5),  F(1, 10), F(1, 2)),
      (1, 1): (F(1, 20), F(9, 10), F(1, 20))}
fY1 = {(0, (0, 0)): F(1, 10), (1, (0, 0)): F(9, 10),
       (0, (0, 1)): F(3, 10), (1, (0, 1)): F(7, 10),
       (0, (0, 2)): F(1, 10), (1, (0, 2)): F(3, 10),
       (0, (1, 0)): F(1, 2),  (1, (1, 0)): F(1, 2),
       (0, (1, 1)): F(7, 10), (1, (1, 1)): F(3, 10),
       (0, (1, 2)): F(3, 10), (1, (1, 2)): F(1, 10)}

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
            w *= fB[(u, a)][b]
        m[u] = w
    return (m[0] * fY1[(0, x)] + m[1] * fY1[(1, x)]) / (m[0] + m[1])


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


def strictly_coarser(Q, P):
    return refines(P, Q) and P != Q


def same(P, x1, x2):
    return any(x1 in blk and x2 in blk for blk in P)


def fmt(P):
    return "  ".join("{" + ",".join("%d%d" % x for x in blk) + "}" for blk in P)


parts = {r: blocks(r) for r in ORDER}
uni = join(parts["doA"], parts["doB"])

print("P(Y=1 | regime, x):")
for r in ORDER:
    vals = "  ".join("%d%d:%-7s" % (x[0], x[1], p_y1(r, x)) for x in XS)
    print(f"  {r:5s} {vals}")
    print(f"        -> {len(parts[r])} classes: {fmt(parts[r])}")
print(f"  union -> {len(uni)} classes: {fmt(uni)}")

assert len(parts["obs"]) == 6, "obs must be all singletons"
assert parts["doA"] != parts["doB"]
assert strictly_coarser(parts["doA"], parts["obs"])
assert strictly_coarser(parts["doB"], parts["obs"])
assert strictly_coarser(uni, parts["doA"]), "union must strictly coarsen do(A)"
assert strictly_coarser(uni, parts["doB"]), "union must strictly coarsen do(B)"
assert strictly_coarser(parts["full"], uni), "full must strictly coarsen union"
assert len(parts["full"]) == 2, "full must be NON-trivial (2 classes)"

# the emergent chain merge and the two top-only merges
assert not same(parts["doA"], (0, 0), (0, 1)) and not same(parts["doB"], (0, 0), (0, 1))
assert same(uni, (0, 0), (0, 1))
assert not same(uni, (1, 0), (0, 0)) and same(parts["full"], (1, 0), (0, 0))
assert same(parts["doB"], (0, 2), (1, 2)), "kappa2 IS visible to do(X^B)"

print("\nLadder strict at every step: 6 -> 5, 4 -> 3 -> 2 classes.")
print("Emergent union merge 00 ~ 01; the loner (1,0) joins kappa1 only at the top;")
print("kappa2 = {02,12} is found by do(X^B) alone -- the top is NOT trivial.")
