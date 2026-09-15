#!/usr/bin/env python3
"""Large-scale stress test of the partial coarsening hierarchy.

For each graph type (a, b, c) we sample CPTs uniformly from a rational grid
on the parameter simplices (every parameter = k/D, k in 1..D-1) and compute,
with EXACT arithmetic (integer weights + Fractions, no float tolerance):

  metric 1  violations of the partial coarsening property on every edge of
            the lattice:  obs <= do(A), obs <= do(B), do(A) <= do(x),
            do(B) <= do(x)   (P <= Q means Q merges whatever P merges);
  metric 2  the union partition  join(do(A), do(B)):  whether union <= full
            ever fails, and the fraction of samples where the union is
            STRICTLY finer than the full partition (the union gap).

Why a grid and not continuous uniform sampling: under a continuous law the
probability of ANY exact density coincidence is zero -- every partition is
all-singletons and every check passes vacuously.  The coincidence sets are
measure-zero; a rational grid gives them positive probability, and refining
the grid (D up) shows the violation fraction shrinking toward zero -- that
is "almost always", made quantitative.

Also produced, for the dashboard:
  - red/green scatter samples in a 2-D projection of the parameter space
    (all violating samples kept, non-violating subsampled);
  - a dense 2-D slice sweep around the paper's Mechanism II: all parameters
    fixed, f(x^B|u, x^A=1) swept over a SxS grid, each point classified
    green (no violation, union = full) / amber (no violation, union gap) /
    red (coarsening violation);
  - a grid-resolution convergence table (violation & gap fraction vs D).

Run:  python3 experiment_coarsening.py            (full run, all cores)
      python3 experiment_coarsening.py --quick    (small benchmark)
"""

import argparse
import json
import random
import time
from fractions import Fraction as F
from itertools import combinations
from multiprocessing import Pool, cpu_count

XS = ((0, 0), (0, 1), (1, 0), (1, 1))            # x = (x^A, x^B)
PAIRS = tuple(combinations(range(4), 2))          # 6 unordered pairs
EDGES = (("obs", "doA"), ("obs", "doB"), ("doA", "full"), ("doB", "full"))
REGIME_KEEP = {"obs": (True, True), "doA": (False, True),
               "doB": (True, False), "full": (False, False)}
GRAPH_KEYS = ("a", "b", "c")

# ----------------------------------------------------------- per-sample core


def draw_params(rng, graph, D):
    """All CPT parameters as integer numerators over denominator D."""
    p = {"pu": rng.randrange(1, D),
         "t": [[rng.randrange(1, D) for _ in range(4)] for _ in range(2)]}
    if graph in ("a", "b"):
        p["fa"] = [rng.randrange(1, D), rng.randrange(1, D)]   # f(a=0|u)
    else:
        p["fa"] = rng.randrange(1, D)                          # f(a=0)
    if graph in ("a", "c"):
        p["fb"] = {(u, a): rng.randrange(1, D)                 # f(b=0|u,a)
                   for u in (0, 1) for a in (0, 1)}
    else:
        p["fb"] = [rng.randrange(1, D), rng.randrange(1, D)]   # f(b=0|a)
    return p


def masks_for(params, graph, D):
    """Merge-mask (6 bits, one per pair of x-points) for each regime."""
    pu, t = params["pu"], params["t"]

    def fu(u):
        return pu if u == 0 else D - pu

    def fa(u, av):
        n = params["fa"][u] if graph in ("a", "b") else params["fa"]
        return n if av == 0 else D - n

    def fb(u, av, bv):
        n = params["fb"][(u, av)] if graph in ("a", "c") else params["fb"][av]
        return n if bv == 0 else D - n

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
            # P(Y=1 | regime, x) up to a constant factor 1/D: exact key
            keys.append(F(m0 * t[0][xi] + m1 * t[1][xi], m0 + m1))
        mask = 0
        for idx, (i, j) in enumerate(PAIRS):
            if keys[i] == keys[j]:
                mask |= 1 << idx
        masks[reg] = mask
    return masks


def closure(mask):
    """Transitive closure of a pair-mask over the 4-point universe."""
    parent = list(range(4))

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


def classify(masks):
    """Edge violations, union<=full violation, union gap flag."""
    viol = [bool(masks[lo] & ~masks[hi]) for lo, hi in EDGES]
    union = closure(masks["doA"] | masks["doB"])
    uviol = bool(union & ~masks["full"])
    gap = (not uviol) and union != masks["full"]
    return viol, uviol, gap, union


# ------------------------------------------------------------- worker chunks


def run_chunk(args):
    graph, D, n, seed, keep_green_p, red_cap = args
    rng = random.Random(seed)
    agg = {"n": n, "edge": [0, 0, 0, 0], "any": 0, "uviol": 0, "gap": 0,
           "obs_nt": 0, "full_nt": 0, "reds": [], "greens": []}
    for _ in range(n):
        params = draw_params(rng, graph, D)
        masks = masks_for(params, graph, D)
        viol, uviol, gap, union = classify(masks)
        anyv = any(viol) or uviol
        for k in range(4):
            agg["edge"][k] += viol[k]
        agg["any"] += anyv
        agg["uviol"] += uviol
        agg["gap"] += gap
        agg["obs_nt"] += masks["obs"] != 0
        agg["full_nt"] += masks["full"] != 0
        # scatter projection: (f(u=0), f(y=1 | u=0, x=(0,0)))
        px, py = params["pu"] / D, params["t"][0][0] / D
        if anyv:
            if len(agg["reds"]) < red_cap:
                agg["reds"].append({
                    "x": px, "y": py, "params": jsonable(params),
                    "edges": [f"{lo}->{hi}" for (lo, hi), v in zip(EDGES, viol) if v]
                             + (["union->full"] if uviol else []),
                    "masks": {r: mask_blocks(m) for r, m in masks.items()},
                })
        elif rng.random() < keep_green_p:
            agg["greens"].append([round(px, 4), round(py, 4), 1 if gap else 0])
    return graph, D, agg


def jsonable(params):
    return {k: ({f"{u}{a}": v for (u, a), v in val.items()}
                if isinstance(val, dict) else val)
            for k, val in params.items()}


def mask_blocks(mask):
    """Human-readable partition from a (closed) pair mask."""
    m = closure(mask)
    parent = list(range(4))
    for idx, (i, j) in enumerate(PAIRS):
        if m >> idx & 1:
            parent[j] = min(parent[j], parent[i])
    groups = {}
    for i in range(4):
        groups.setdefault(parent[i], []).append("".join(map(str, XS[i])))
    return sorted(groups.values())


# ------------------------------------------------------------ slice sweep


SWEEP_S = 200            # grid: i/S for i in 1..S-1

def sweep_row(task):
    """Two 2-D slices of graph (a) around the paper's Mechanism II point.

    kind 'fb': sweep the leaf row f(x^B=0|u, x^A=1) = (i/S, j/S),
               outcome CPT fixed (carries the built-in prior coincidence);
    kind 'ty': sweep the outcome column f(y=1|u, (1,1)) = (i/S, j/S),
               X-mechanism fixed at Mechanism II;
    kind 'fa': sweep the ROOT mechanism f(x^A=0|u) = (i/S, j/S) on the
               ALIGNED (Mechanism I) base, where do(X^A) always merges
               {00,11}; union = full then happens exactly on the lines
               a0 = a1 (X^A independent of U) and a0 + a1 = 1 (the
               symmetric-confounding line of the paper examples).
    """
    kind, i = task
    S = SWEEP_S
    pu = F(1, 2)
    fa = {0: F(3, 4), 1: F(1, 4)}                        # f(a=0|u)
    fb0 = {0: F(3, 5), 1: F(2, 5)}                       # f(b=0|u, a=0)
    t = {(0, 0): F(1, 5), (1, 0): F(4, 5), (0, 1): F(1, 10), (1, 1): F(3, 10),
         (0, 2): F(3, 10), (1, 2): F(1, 10), (0, 3): F(3, 5), (1, 3): F(2, 5)}
    row = []
    for j in range(1, S):
        if kind == "fb":
            fb1 = {0: F(i, S), 1: F(j, S)}               # f(b=0|u, a=1)
        elif kind == "ty":
            fb1 = {0: F(3, 10), 1: F(7, 10)}             # Mechanism II row
            t = dict(t)
            t[(0, 3)], t[(1, 3)] = F(i, S), F(j, S)      # f(y=1|u, (1,1))
        else:                                            # kind 'fa'
            fb1 = {0: F(4, 5), 1: F(1, 5)}               # Mechanism I row
            fa = {0: F(i, S), 1: F(j, S)}                # f(a=0|u)

        def fb(u, av, bv):
            n = fb0[u] if av == 0 else fb1[u]
            return n if bv == 0 else 1 - n

        masks = {}
        for reg, (keep_a, keep_b) in REGIME_KEEP.items():
            keys = []
            for xi, (av, bv) in enumerate(XS):
                m0, m1 = pu, 1 - pu
                if keep_a:
                    m0 *= fa[0] if av == 0 else 1 - fa[0]
                    m1 *= fa[1] if av == 0 else 1 - fa[1]
                if keep_b:
                    m0 *= fb(0, av, bv)
                    m1 *= fb(1, av, bv)
                keys.append((m0 * t[(0, xi)] + m1 * t[(1, xi)]) / (m0 + m1))
            mask = 0
            for idx, (a_, b_) in enumerate(PAIRS):
                if keys[a_] == keys[b_]:
                    mask |= 1 << idx
            masks[reg] = mask
        viol, uviol, gap, _ = classify(masks)
        row.append("2" if (any(viol) or uviol) else ("1" if gap else "0"))
    return (kind, i), "".join(row)


# --------------------------------------------------------------------- main


def run_experiment(pool, workers, graph, D, n_total, keep_green, red_cap):
    chunk = max(2000, n_total // (workers * 4))
    tasks, left, seed = [], n_total, (hash((graph, D)) & 0xFFFF) << 16
    while left > 0:
        m = min(chunk, left)
        tasks.append((graph, D, m, seed + len(tasks), keep_green / n_total, red_cap))
        left -= m
    tot = {"n": 0, "edge": [0, 0, 0, 0], "any": 0, "uviol": 0, "gap": 0,
           "obs_nt": 0, "full_nt": 0, "reds": [], "greens": []}
    for _, _, agg in pool.imap_unordered(run_chunk, tasks):
        for k in ("n", "any", "uviol", "gap", "obs_nt", "full_nt"):
            tot[k] += agg[k]
        for k in range(4):
            tot["edge"][k] += agg["edge"][k]
        tot["reds"].extend(agg["reds"])
        tot["greens"].extend(agg["greens"])
    tot["reds"] = tot["reds"][:red_cap]
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--workers", type=int, default=min(60, cpu_count()))
    args = ap.parse_args()

    D_MAIN = 20
    N_MAIN = 20_000 if args.quick else 500_000
    D_CONV = (5, 10, 20, 40, 80)
    N_CONV = 10_000 if args.quick else 200_000

    t0 = time.time()
    out = {"meta": {"D_main": D_MAIN, "N_main": N_MAIN,
                    "D_conv": list(D_CONV), "N_conv": N_CONV,
                    "workers": args.workers,
                    "edges": [f"{lo}->{hi}" for lo, hi in EDGES]},
           "main": {}, "conv": {g: [] for g in GRAPH_KEYS}, "sweep": None}

    with Pool(args.workers) as pool:
        for g in GRAPH_KEYS:
            r = run_experiment(pool, args.workers, g, D_MAIN, N_MAIN,
                               keep_green=4000, red_cap=400)
            out["main"][g] = r
            print(f"[main D={D_MAIN}] graph ({g}): n={r['n']:,}  "
                  f"violations={r['any']} ({r['any']/r['n']:.2e})  "
                  f"per-edge={r['edge']}  union-viol={r['uviol']}  "
                  f"gap={r['gap']} ({r['gap']/r['n']:.2e})  "
                  f"full-nontrivial={r['full_nt']} "
                  f"[gap|full-nt = {r['gap']/max(r['full_nt'],1):.3f}]", flush=True)

        for g in GRAPH_KEYS:
            for D in D_CONV:
                r = run_experiment(pool, args.workers, g, D, N_CONV,
                                   keep_green=0, red_cap=5)
                out["conv"][g].append(
                    {"D": D, "n": r["n"], "any": r["any"], "gap": r["gap"],
                     "uviol": r["uviol"], "edge": r["edge"],
                     "full_nt": r["full_nt"]})
                print(f"[conv] graph ({g}) D={D:3d}: viol {r['any']:6d} "
                      f"({r['any']/r['n']:.2e})  gap {r['gap']:6d} "
                      f"({r['gap']/r['n']:.2e})  union-viol {r['uviol']}", flush=True)

        tasks = [(k, i) for k in ("fb", "ty", "fa") for i in range(1, SWEEP_S)]
        rows = dict(pool.imap_unordered(sweep_row, tasks))
    out["sweep_fb"] = {"S": SWEEP_S,
                       "rows": [rows[("fb", i)] for i in range(1, SWEEP_S)],
                       "base": "graph (a), Mech II; x = f(x^B=0|u=0,x^A=1), "
                               "y = f(x^B=0|u=1,x^A=1)"}
    out["sweep_ty"] = {"S": SWEEP_S,
                       "rows": [rows[("ty", i)] for i in range(1, SWEEP_S)],
                       "base": "graph (a), Mech II; x = f(y=1|u=0,(1,1)), "
                               "y = f(y=1|u=1,(1,1))"}
    out["sweep_fa"] = {"S": SWEEP_S,
                       "rows": [rows[("fa", i)] for i in range(1, SWEEP_S)],
                       "base": "graph (a), Mech I; x = f(x^A=0|u=0), "
                               "y = f(x^A=0|u=1)"}

    out["meta"]["seconds"] = round(time.time() - t0, 1)
    fname = "coarsening_experiment.json" if not args.quick else "coarsening_quick.json"
    with open(fname, "w") as fh:
        json.dump(out, fh)
    print(f"\nDone in {out['meta']['seconds']}s -> {fname}", flush=True)


if __name__ == "__main__":
    main()
