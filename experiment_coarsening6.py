#!/usr/bin/env python3
"""Grid-resolution sweep at the ladder cardinality: |X^A| = 2, |X^B| = 3.

Same methodology as experiment_coarsening.py (exact rational arithmetic;
each CPT parameter sampled uniformly from the k/D grid on its simplex --
ternary rows are uniform positive compositions of D), for the three U-X
graphs.  Only the two headline curves are produced:

  1. fraction of sampled distributions violating the partial coarsening
     property on ANY edge (obs->doA, obs->doB, doA->full, doB->full,
     union->full), per grid resolution D;
  2. fraction where the union partition join(doA, doB) is STRICTLY less
     informative than the full interventional partition (the union gap).

Output: coarsening6_curves.json, consumed by make_figures.py.
"""

import argparse
import json
import time
import random
from fractions import Fraction as F
from itertools import combinations, product
from multiprocessing import Pool, cpu_count

A_VALS, B_VALS = (0, 1), (0, 1, 2)
XS = tuple(product(A_VALS, B_VALS))               # 6 points
PAIRS = tuple(combinations(range(6), 2))          # 15 pairs
EDGES = (("obs", "doA"), ("obs", "doB"), ("doA", "full"), ("doB", "full"))
REGIME_KEEP = {"obs": (True, True), "doA": (False, True),
               "doB": (True, False), "full": (False, False)}
GRAPH_KEYS = ("a", "b", "c")
D_LIST = (5, 10, 20, 40, 80)


def composition3(rng, D):
    """Uniform positive composition (k0, k1, k2) with k0+k1+k2 = D."""
    c1 = rng.randrange(1, D)
    c2 = rng.randrange(1, D - 1)
    if c2 >= c1:
        c2 += 1
    lo, hi = (c1, c2) if c1 < c2 else (c2, c1)
    return (lo, hi - lo, D - hi)


def draw(rng, graph, D):
    p = {"pu": rng.randrange(1, D),
         "t": [[rng.randrange(1, D) for _ in range(6)] for _ in range(2)]}
    if graph in ("a", "b"):
        p["fa"] = [rng.randrange(1, D), rng.randrange(1, D)]      # f(a=0|u)
    else:
        p["fa"] = rng.randrange(1, D)                             # f(a=0)
    if graph in ("a", "c"):
        p["fb"] = {(u, a): composition3(rng, D)                   # f(b|u,a)
                   for u in (0, 1) for a in (0, 1)}
    else:
        p["fb"] = [composition3(rng, D), composition3(rng, D)]    # f(b|a)
    return p


def masks_for(p, graph, D):
    pu, t = p["pu"], p["t"]

    def fu(u):
        return pu if u == 0 else D - pu

    def fa(u, av):
        n = p["fa"][u] if graph in ("a", "b") else p["fa"]
        return n if av == 0 else D - n

    def fb(u, av, bv):
        row = p["fb"][(u, av)] if graph in ("a", "c") else p["fb"][av]
        return row[bv]

    masks = {}
    for reg, (keep_a, keep_b) in REGIME_KEEP.items():
        keys = []
        for xi, (av, bv) in enumerate(XS):
            m0, m1 = fu(0), fu(1)
            if keep_a:
                m0 *= fa(0, av)
                m1 *= fa(1, av)
            if keep_b:
                m0 *= fb(0, av, bv)
                m1 *= fb(1, av, bv)
            keys.append(F(m0 * t[0][xi] + m1 * t[1][xi], m0 + m1))
        mask = 0
        for idx, (i, j) in enumerate(PAIRS):
            if keys[i] == keys[j]:
                mask |= 1 << idx
        masks[reg] = mask
    return masks


def closure(mask):
    parent = list(range(6))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for idx, (i, j) in enumerate(PAIRS):
        if mask >> idx & 1:
            parent[find(i)] = find(j)
    out = 0
    for idx, (i, j) in enumerate(PAIRS):
        if find(i) == find(j):
            out |= 1 << idx
    return out


def run_chunk(args):
    graph, D, n, seed = args
    rng = random.Random(seed)
    agg = {"n": n, "edge": [0, 0, 0, 0], "any": 0, "uviol": 0,
           "gap": 0, "full_nt": 0}
    for _ in range(n):
        masks = masks_for(draw(rng, graph, D), graph, D)
        viol = [bool(masks[lo] & ~masks[hi]) for lo, hi in EDGES]
        union = closure(masks["doA"] | masks["doB"])
        uviol = bool(union & ~masks["full"])
        for k in range(4):
            agg["edge"][k] += viol[k]
        agg["any"] += any(viol) or uviol
        agg["uviol"] += uviol
        agg["gap"] += (not uviol) and union != masks["full"]
        agg["full_nt"] += masks["full"] != 0
    return graph, D, agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--workers", type=int, default=min(60, cpu_count()))
    args = ap.parse_args()
    N = 10_000 if args.quick else 200_000

    t0 = time.time()
    out = {"meta": {"N": N, "D_list": list(D_LIST), "workers": args.workers,
                    "space": "2x3 (X^A binary, X^B ternary)"},
           "conv": {g: [] for g in GRAPH_KEYS}}
    with Pool(args.workers) as pool:
        for g in GRAPH_KEYS:
            for D in D_LIST:
                chunk = max(2000, N // (args.workers * 3))
                tasks, left, seed = [], N, (hash((g, D, 6)) & 0xFFFF) << 16
                while left > 0:
                    m = min(chunk, left)
                    tasks.append((g, D, m, seed + len(tasks)))
                    left -= m
                tot = {"n": 0, "edge": [0, 0, 0, 0], "any": 0, "uviol": 0,
                       "gap": 0, "full_nt": 0}
                for _, _, agg in pool.imap_unordered(run_chunk, tasks):
                    for k in ("n", "any", "uviol", "gap", "full_nt"):
                        tot[k] += agg[k]
                    for k in range(4):
                        tot["edge"][k] += agg["edge"][k]
                out["conv"][g].append(dict(tot, D=D))
                print(f"graph ({g}) D={D:3d}: viol {tot['any']:6d} "
                      f"({tot['any']/tot['n']:.2e})  gap {tot['gap']:6d} "
                      f"({tot['gap']/tot['n']:.2e})  edges {tot['edge']}",
                      flush=True)
    out["meta"]["seconds"] = round(time.time() - t0, 1)
    fname = "coarsening6_quick.json" if args.quick else "coarsening6_curves.json"
    with open(fname, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"Done in {out['meta']['seconds']}s -> {fname}")


if __name__ == "__main__":
    main()
