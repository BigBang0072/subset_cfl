#!/usr/bin/env python3
"""Paper figures for the subset-CFL examples.

  figures/partial_cct_sweep_2x3.{pdf,png}
      Two-panel curve figure from coarsening6_curves.json:
      (i)  fraction of sampled distributions violating the partial
           coarsening property, per grid resolution D, graphs (a),(b),(c);
      (ii) fraction where the union partition is strictly coarser than its
           parts but strictly finer than the full one (the union gap).

  figures/union_ladder_2x3.{pdf,png}
      Diagram of the strict ladder example on the 2x3 X-space
      (values verified exactly by union_ladder6.py).
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 8,
    "axes.linewidth": 0.6,
    "xtick.direction": "out", "ytick.direction": "out",
})

INK = "#1A2233"
GREY = "#8A93A8"
COLORS = {"a": "#3D6CD4", "b": "#B26F0E", "c": "#175E50"}
MARKERS = {"a": "o", "b": "s", "c": "^"}

os.makedirs("figures", exist_ok=True)

# ------------------------------------------------------------ curve figure

data = json.load(open("coarsening6_curves.json"))
DS = data["meta"]["D_list"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.6, 2.45))
for ax in (ax1, ax2):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(DS)
    ax.set_xticklabels([str(d) for d in DS])
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.grid(axis="y", which="major", lw=0.4, color="#DDE2EC")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_xlabel("grid resolution $D$")

for g in ("a", "b", "c"):
    rows = data["conv"][g]
    fr = [r["any"] / r["n"] for r in rows]
    ax1.plot(DS, fr, marker=MARKERS[g], ms=3.5, lw=1.1, color=COLORS[g],
             label=f"graph ({g})", clip_on=False)
ax1.set_ylabel("fraction of distributions")
ax1.set_title("(i) partial-coarsening violations", fontsize=8, pad=6)
ax1.legend(frameon=False, fontsize=6.5, handlelength=1.6, borderaxespad=0.2)

rows = data["conv"]["a"]
gap = [r["gap"] / r["n"] for r in rows]
ax2.plot(DS, gap, marker="o", ms=3.5, lw=1.1, color=COLORS["a"],
         label="graph (a)", clip_on=False)
ax2.set_title(r"(ii) union gap:  $\Pi_\cup \subsetneq \Pi_{\mathrm{do}(x)}$",
              fontsize=8, pad=6)
ax2.legend(frameon=False, fontsize=6.5, handlelength=1.6, borderaxespad=0.2)
ax2.text(0.03, 0.06, "graphs (b), (c): identically 0",
         transform=ax2.transAxes, fontsize=6.5, color=GREY)

fig.tight_layout(w_pad=2.2)
fig.savefig("figures/partial_cct_sweep_2x3.pdf")
fig.savefig("figures/partial_cct_sweep_2x3.png", dpi=220)
plt.close(fig)
print("wrote figures/partial_cct_sweep_2x3.{pdf,png}")

# ------------------------------------------------------------ ladder figure
# exact values from union_ladder6.py

PALE, PALE_BD = "#E9EDF4", "#C2CAD9"
BLK = {"A": ("#3D6CD4", "w"), "B": ("#9E630B", "w"), "T": ("#175E50", "w"),
       "U": ("#A987E0", INK), "F": ("#5B2D9E", "w"), "p": (PALE, INK)}

STAGES = {
    "obs":   dict(name="observational", math=r"$f(y\,|\,x)$", k="6 classes",
                  cells=[["1/5", "p"], ["23/50", "p"], ["33/170", "p"],
                         ["1/2", "p"], ["11/35", "p"], ["33/130", "p"]]),
    "doA":   dict(name="intervene on $X^A$", math=r"$f(y\,|\,x^B\!,\,\mathrm{do}(x^A))$", k="5 classes",
                  cells=[["17/50", "A"], ["17/30", "p"], ["27/110", "p"],
                         ["1/2", "p"], ["17/50", "A"], ["31/110", "p"]]),
    "doB":   dict(name="intervene on $X^B$", math=r"$f(y\,|\,x^A\!,\,\mathrm{do}(x^B))$", k="4 classes",
                  cells=[["3/10", "p"], ["2/5", "B"], ["3/20", "T"],
                         ["1/2", "p"], ["2/5", "B"], ["3/20", "T"]]),
    "union": dict(name="union (derived)", math=r"$\Pi_{\mathrm{do}(A)} \vee \Pi_{\mathrm{do}(B)}$", k="3 classes",
                  cells=[["0,0", "U"], ["0,1", "U"], ["0,2", "T"],
                         ["1,0", "p"], ["1,1", "U"], ["1,2", "T"]]),
    "full":  dict(name="full intervention", math=r"$f(y\,|\,\mathrm{do}(x))$", k="2 classes",
                  cells=[["1/2", "F"], ["1/2", "F"], ["1/5", "T"],
                         ["1/2", "F"], ["1/2", "F"], ["1/5", "T"]]),
}
CW, CH, GAP = 0.66, 0.52, 0.055                    # cell size (data units)
GW, GH = 3 * CW + 2 * GAP, 2 * CH + GAP            # grid size
POS = {"obs": (0.55, 2.75), "doA": (3.30, 4.65), "doB": (3.30, 0.85),
       "union": (6.20, 2.75), "full": (8.85, 2.75)}

fig, ax = plt.subplots(figsize=(7.2, 3.85))
ax.set_xlim(0, 11.15)
ax.set_ylim(-1.20, 7.05)
ax.axis("off")

for key, (gx, gy) in POS.items():
    st = STAGES[key]
    dashed = key == "union"
    for i, (txt, bk) in enumerate(st["cells"]):
        r, c = divmod(i, 3)
        x = gx + c * (CW + GAP)
        y = gy + (1 - r) * (CH + GAP)
        face, tc = BLK[bk]
        ax.add_patch(Rectangle((x, y), CW, CH, facecolor=face,
                               edgecolor=PALE_BD if bk == "p" else "none",
                               linewidth=0.5))
        ax.text(x + CW / 2, y + CH / 2, txt, ha="center", va="center",
                fontsize=5.6, color="white" if tc == "w" else tc,
                family="DejaVu Sans Mono" if key != "union" else None,
                style="italic" if key == "union" else "normal")
    ax.add_patch(Rectangle((gx - 0.10, gy - 0.10), GW + 0.20, GH + 0.20,
                           fill=False, edgecolor=PALE_BD, linewidth=0.7,
                           linestyle=(0, (2.5, 2)) if dashed else "solid"))
    ax.text(gx + GW / 2, gy + GH + 0.62, st["name"], ha="center",
            fontsize=6.4, color=GREY)
    ax.text(gx + GW / 2, gy + GH + 0.28, st["math"], ha="center", fontsize=7.2,
            color=INK)
    ax.text(gx + GW / 2, gy - 0.66, st["k"], ha="center", fontsize=6.2,
            color=INK, weight="bold")

# axis labels on the observational grid only
ox, oy = POS["obs"]
for c, lb in enumerate(("b=0", "b=1", "b=2")):
    ax.text(ox + c * (CW + GAP) + CW / 2, oy - 0.30, lb, ha="center",
            fontsize=5.0, color=GREY, family="DejaVu Sans Mono")
for r, lb in enumerate(("a=0", "a=1")):
    ax.text(ox - 0.22, oy + (1 - r) * (CH + GAP) + CH / 2, lb, ha="right",
            va="center", fontsize=5.2, color=GREY, family="DejaVu Sans Mono")

def arrow(p1, p2, label, ly=0.16, lx=0.0):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=7,
                                 lw=0.8, color=GREY, shrinkA=2, shrinkB=2))
    mx, my = (p1[0] + p2[0]) / 2 + lx, (p1[1] + p2[1]) / 2 + ly
    ax.text(mx, my, label, fontsize=5.4, color=INK, ha="center",
            style="italic")

gy_mid = lambda k: POS[k][1] + GH / 2
arrow((POS["obs"][0] + GW + 0.14, gy_mid("obs") + 0.35),
      (POS["doA"][0] - 0.16, POS["doA"][1] - 0.02), "{00,11}", ly=0.34, lx=-0.12)
arrow((POS["obs"][0] + GW + 0.14, gy_mid("obs") - 0.35),
      (POS["doB"][0] - 0.16, POS["doB"][1] + GH + 0.06), "{01,11}, {02,12}",
      ly=-0.72, lx=-0.90)
arrow((POS["doA"][0] + GW + 0.14, POS["doA"][1] - 0.02),
      (POS["union"][0] - 0.16, gy_mid("union") + 0.35),
      r"$\Rightarrow 00 \sim 01$", ly=0.36, lx=0.10)
arrow((POS["doB"][0] + GW + 0.14, POS["doB"][1] + GH + 0.06),
      (POS["union"][0] - 0.16, gy_mid("union") - 0.35), "join", ly=-0.40, lx=0.10)
arrow((POS["union"][0] + GW + 0.20, gy_mid("union")),
      (POS["full"][0] - 0.16, gy_mid("full")),
      r"$+\{10\} \to \kappa_1$", ly=-0.85, lx=0.10)

ax.text(5.55, 6.85, r"$\Pi_{\mathrm{do}(A)},\; \Pi_{\mathrm{do}(B)} \;\subsetneq\;"
                    r" \Pi_\cup \;\subsetneq\; \Pi_{\mathrm{do}(x)}$"
                    "        6 $\\to$ 5, 4 $\\to$ 3 $\\to$ 2 classes",
        ha="center", fontsize=7.6, color=INK)

legend = [[("p", "singleton"), ("A", r"$\{00,11\}$ via do($X^A$)"),
           ("B", r"$\{01,11\}$ via do($X^B$)")],
          [("T", r"$\kappa_2=\{02,12\}$ via do($X^B$)"),
           ("U", r"union chain $\{00,01,11\}$"), ("F", r"$\kappa_1$ complete")]]
for row, items in enumerate(legend):
    y = -0.62 - 0.42 * row
    for (bk, lab), lx in zip(items, (2.2, 5.0, 8.0)):
        face, _ = BLK[bk]
        ax.add_patch(Rectangle((lx, y - 0.10), 0.26, 0.20, facecolor=face,
                               edgecolor=PALE_BD if bk == "p" else "none", lw=0.5))
        ax.text(lx + 0.36, y, lab, fontsize=5.8, va="center", color=INK)

fig.savefig("figures/union_ladder_2x3.pdf", bbox_inches="tight")
fig.savefig("figures/union_ladder_2x3.png", dpi=220, bbox_inches="tight")
plt.close(fig)
print("wrote figures/union_ladder_2x3.{pdf,png}")
