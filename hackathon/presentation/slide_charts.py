"""Slide versions of the main charts, in one consistent style (presentation/style.py).

Run from hackathon/:  uv run python presentation/slide_charts.py
Writes to results/figures/slides/:

  1_speed.png        one disorder study at 16x16: 7 days -> 2 hours (RESULTS.md R12)
  2_speedup_split.png  where the speed-up comes from: smarter maths x GPU (R12)
  3_disorder_switch.png  disorder switches topology on: fraction topological vs disorder (R11)
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import (
    BLUE,
    GRID,
    INK,
    INK_2,
    MUTED,
    ORANGE,
    SURFACE,
    TINT,
    new_slide,
    save,
    titled,
)

HERE = Path(__file__).resolve().parent.parent
COST = HERE / "results" / "kaggle" / "20261121-213825" / "out" / "sweep_cost.csv"  # R12
PILOT = HERE / "results" / "sweep" / "pilot_8x8_c_up.csv"  # R11
FSS12 = HERE / "results" / "sweep" / "fss_12x12_c_up.csv"  # R13
FSS16 = HERE / "results" / "sweep" / "fss_16x16_c_up.csv"  # R13
PHASEMAP = HERE / "results" / "sweep" / "phasemap_12x12_c_up.csv"  # R15


def rows(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def nice_time(hours: float) -> str:
    if hours >= 48:
        return f"{hours / 24:.0f} days"
    if hours >= 1.5:
        return f"{hours:.1f} h" if hours < 10 else f"{hours:.0f} h"
    return f"{hours * 60:.0f} min"


def rounded_bar(ax, y: float, width: float, height: float, colour: str) -> None:
    ax.add_patch(FancyBboxPatch((0, y - height / 2), width, height, mutation_aspect=0.02,
                                boxstyle=f"round,pad=0,rounding_size={min(height / 2, width / 2)}",
                                linewidth=0, facecolor=colour))


# ---------------------------------------------------------------- 1. speed


def slide_speed(size: int = 16) -> Path:
    r = next(r for r in rows(COST) if int(r["supercell"]) == size)
    bars = [
        ("PythTB today", "4 CPU cores", float(r["sweep_h_pythtb_cpu"]), "#b9b8b1"),
        ("Our method", "same 4 CPU cores", float(r["sweep_h_c_up_cpu"]), ORANGE),
        ("Our method", "one GPU", float(r["sweep_h_c_up_gpu"]), BLUE),
    ]
    fig, ax = new_slide()
    longest = bars[0][2]
    for y, (name, where, h, colour) in zip([2, 1, 0], bars, strict=True):
        ax.barh(y, longest, height=0.56, color=SURFACE, edgecolor=GRID, linewidth=1.2)  # track
        ax.barh(y, max(h, longest * 0.006), height=0.56, color=colour)
        ax.text(-longest * 0.02, y + 0.09, name, ha="right", va="center", fontsize=19,
                fontweight="bold", color=INK)
        ax.text(-longest * 0.02, y - 0.17, where, ha="right", va="center", fontsize=14, color=INK_2)
        inside = h > longest * 0.5
        ax.text(h - longest * 0.015 if inside else h + longest * 0.015, y, nice_time(h),
                ha="right" if inside else "left", va="center", fontsize=26, fontweight="bold",
                color=INK)
    ax.set_xlim(0, longest)
    ax.set_ylim(-0.55, 2.55)
    ax.axis("off")
    titled(fig, f"{float(r['x_total']):.0f}× faster: from a week to two hours",
           f"Time for one disorder study: 16,400 imperfect samples of a {size}×{size} supercell",
           "Estimated from per-sample times measured on one Kaggle machine (4 CPU cores + 1 Tesla T4). "
           "RESULTS.md R12.")
    return save(fig, "1_speed.png", left=0.27, right=0.96, bottom=0.12, top=0.76)


# ---------------------------------------------------------------- 2. split


def slide_split(size: int = 16) -> Path:
    r = next(r for r in rows(COST) if int(r["supercell"]) == size)
    algo, gpu, total = float(r["x_algorithm"]), float(r["x_gpu_on_top"]), float(r["x_total"])
    fig, ax = new_slide()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")

    def card(x: float, colour: str, big: str, line1: str, line2: str, fill: bool) -> None:
        ax.add_patch(FancyBboxPatch((x, 0.75), 2.5, 2.5, boxstyle="round,pad=0,rounding_size=0.28",
                                    facecolor=colour if fill else SURFACE, edgecolor=colour,
                                    linewidth=0 if fill else 3))
        text = "white" if fill else INK
        ax.text(x + 1.25, 2.35, big, ha="center", va="center", fontsize=54, fontweight="bold",
                color=text)
        ax.text(x + 1.25, 1.45, line1, ha="center", va="center", fontsize=17, fontweight="bold",
                color=text)
        ax.text(x + 1.25, 1.05, line2, ha="center", va="center", fontsize=13,
                color="white" if fill else INK_2)

    card(0.3, ORANGE, f"{algo:.1f}×", "smarter maths", "a better invariant, any computer", True)
    ax.text(3.15, 2.0, "×", ha="center", va="center", fontsize=48, color=MUTED)
    card(3.75, BLUE, f"{gpu:.1f}×", "the GPU", "batching everything at once", True)
    ax.text(6.6, 2.0, "=", ha="center", va="center", fontsize=48, color=MUTED)
    card(7.2, INK, f"{total:.0f}×", "faster overall", f"vs PythTB, {size}×{size} supercells", False)
    titled(fig, "Where the speed-up comes from",
           "Two separate gains that multiply, and the GPU's share grows with system size",
           f"Same machine: PythTB on all 4 CPU cores vs our method on CPU and on one T4. "
           f"At 12×12: {float(next(x for x in rows(COST) if x['supercell'] == '12')['x_total']):.0f}× "
           "overall. RESULTS.md R12.")
    return save(fig, "2_speedup_split.png", left=0.04, right=0.96, bottom=0.08, top=0.8)


# ---------------------------------------------------------------- 3. sweep


def slide_disorder_switch() -> Path:
    data = rows(FSS12)
    fig, ax = new_slide()
    ax.axvspan(1.25, 2.75, color=TINT, zorder=0)
    for lam, colour, label in ((1.0, BLUE, "topological when perfect"),
                               (1.65, ORANGE, "ordinary when perfect")):
        pts = sorted((float(r["W"]), float(r["frac_topological"]), float(r["stderr"]))
                     for r in data if abs(float(r["lambda_nu"]) - lam) < 1e-9)
        W, f, e = (np.array(v) for v in zip(*pts, strict=True))
        ax.fill_between(W, np.clip(f - e, 0, 1), np.clip(f + e, 0, 1), color=colour, alpha=0.18,
                        linewidth=0)
        ax.plot(W, f, color=colour, lw=3.5)
        ax.text(W[-1] + 0.15, f[-1] + (0.06 if lam == 1.0 else -0.02), label, color=INK_2,
                fontsize=14, va="center")
    ax.add_patch(FancyArrowPatch((0.55, 0.62), (1.75, 0.5), arrowstyle="-|>", mutation_scale=22,
                                 color=INK, lw=1.8, connectionstyle="arc3,rad=-0.25"))
    ax.text(0.05, 0.7, "disorder switches\ntopology ON", fontsize=17, fontweight="bold", color=INK,
            va="bottom")
    ax.text(8.3, 0.74, "too much disorder\ndestroys it", fontsize=16, color=INK_2, va="bottom")
    ax.add_patch(FancyArrowPatch((8.9, 0.72), (7.35, 0.42), arrowstyle="-|>", mutation_scale=20,
                                 color=INK_2, lw=1.5, connectionstyle="arc3,rad=0.25"))
    ax.set_xlim(-0.2, 12.2)
    ax.set_ylim(-0.03, 1.08)
    ax.set_yticks([0, 0.5, 1], ["0%", "50%", "100%"])
    ax.set_xticks([0, 2, 4, 6, 8, 10])
    ax.yaxis.grid(True)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("disorder strength (how imperfect the material is)")
    ax.set_ylabel("samples that are topological")
    titled(fig, "Imperfections can make a material topological",
           "Fraction of 100 random imperfect samples per point that are topological, 12×12 supercells",
           r"Kane-Mele model, $\lambda_{SO}$ = 0.3, $\lambda_R$ = 0; $\lambda_\nu$ = 1.0 (blue) and 1.65 "
           "(orange). Bands: ± 1 standard error. 4,200 samples in 10 min on one T4. RESULTS.md R13.")
    return save(fig, "3_disorder_switch.png", left=0.09, right=0.86, bottom=0.17, top=0.76)


# ---------------------------------------------------------------- 4. sizes


def slide_sizes() -> Path:
    """Finite-size scaling of the switch-on: lambda_nu = 1.65, three supercell sizes."""
    from style import BLUE_DARK

    curves = [(PILOT, 8, "#86b6ef"), (FSS12, 12, "#3987e5"), (FSS16, 16, BLUE_DARK)]
    fig, ax = new_slide()
    ax.axvspan(1.85, 2.0, color=TINT, zorder=0)
    for path, n, colour in curves:
        pts = sorted((float(r["W"]), float(r["frac_topological"]), float(r["stderr"]))
                     for r in rows(path) if abs(float(r["lambda_nu"]) - 1.65) < 1e-9
                     and float(r["W"]) <= 4.0 + 1e-9)
        W, f, e = (np.array(v) for v in zip(*pts, strict=True))
        ax.fill_between(W, np.clip(f - e, 0, 1), np.clip(f + e, 0, 1), color=colour, alpha=0.2,
                        linewidth=0)
        ax.plot(W, f, color=colour, lw=3.5, marker="o", ms=6, markeredgecolor=SURFACE,
                markeredgewidth=1.5, label=f"{n}×{n} supercell")
    ax.text(1.925, 1.06, "curves cross\nW ≈ 1.9–2.0", ha="center", va="bottom", fontsize=14,
            color=INK_2)
    ax.text(2.75, 0.32, "bigger systems:\na sharper switch", fontsize=17, fontweight="bold",
            color=INK, va="center")
    ax.set_xlim(0.9, 4.0)
    ax.set_ylim(-0.03, 1.2)
    ax.set_yticks([0, 0.5, 1], ["0%", "50%", "100%"])
    ax.yaxis.grid(True)
    ax.spines["left"].set_visible(False)
    ax.legend(loc="lower right", fontsize=15)
    ax.set_xlabel("disorder strength (how imperfect the material is)")
    ax.set_ylabel("samples that are topological")
    titled(fig, "The switch gets sharper as the system grows",
           "The hallmark of a real transition: three system sizes, 100 imperfect samples per point",
           r"Kane-Mele, $\lambda_\nu$ = 1.65 (ordinary when perfect), $\lambda_{SO}$ = 0.3, "
           r"$\lambda_R$ = 0. Bands: ± 1 standard error. 16×16: 2,100 samples in 10 min on one T4. "
           "RESULTS.md R13.")
    return save(fig, "4_sizes.png", left=0.09, right=0.95, bottom=0.17, top=0.76)


# ---------------------------------------------------------------- 5. phase map


def slide_phase_map(path: Path = PHASEMAP, out: str = "5_phase_map.png") -> Path:
    """lambda_nu x W map of the fraction topological, with the 50% boundary."""
    from matplotlib.colors import LinearSegmentedColormap

    data = rows(path)
    lams = sorted({float(r["lambda_nu"]) for r in data})
    Ws = sorted({float(r["W"]) for r in data})
    grid = np.full((len(lams), len(Ws)), np.nan)
    for r in data:
        grid[lams.index(float(r["lambda_nu"])), Ws.index(float(r["W"]))] = float(r["frac_topological"])

    # sequential, one hue: near-surface = ordinary, dark blue = topological (validated ramp steps)
    cmap = LinearSegmentedColormap.from_list(
        "topo", ["#f4f3ef", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
    fig, ax = new_slide()
    dW = Ws[1] - Ws[0] if len(Ws) > 1 else 1
    dl = lams[1] - lams[0] if len(lams) > 1 else 1
    im = ax.imshow(grid, origin="lower", aspect="auto", cmap=cmap, vmin=0, vmax=1,
                   extent=(Ws[0] - dW / 2, Ws[-1] + dW / 2, lams[0] - dl / 2, lams[-1] + dl / 2),
                   interpolation="nearest")
    clean = 3 * np.sqrt(3) * 0.3  # clean-limit phase boundary, lambda_nu = 3 sqrt(3) lambda_SO
    above = np.array(lams) > clean  # the 50% line only where disorder does the switching
    if above.sum() > 1 and len(Ws) > 1:
        ax.contour(Ws, np.array(lams)[above], grid[above], levels=[0.5], colors=[INK],
                   linewidths=2.5)
    ax.axhline(clean, color="white", lw=1.5, ls=(0, (5, 4)), alpha=0.8)
    ax.text(Ws[-1] + dW / 2 - 0.08, clean - dl * 0.15, "perfect material changes phase here",
            ha="right", va="top", fontsize=12, color="white")
    ax.text(0.35, lams[0] + 0.4 * (clean - lams[0]), "topological", fontsize=18, fontweight="bold",
            color="white", va="center")
    ax.text(0.35, lams[-1] - 0.25 * (lams[-1] - clean), "ordinary", fontsize=18, fontweight="bold",
            color=INK, va="center")
    ax.text(3.3, clean + 0.22 * (lams[-1] - clean), "disorder made\nit topological", fontsize=17,
            fontweight="bold", color="white", va="center", ha="center")
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, ticks=[0, 0.5, 1])
    cbar.ax.set_yticklabels(["0%", "50%", "100%"])
    cbar.ax.tick_params(labelsize=12, colors=MUTED)
    cbar.outline.set_visible(False)
    cbar.set_label("samples that are topological", fontsize=13, color=INK_2)
    ax.set_xlabel("disorder strength (how imperfect the material is)")
    ax.set_ylabel(r"staggered potential $\lambda_\nu$ (the material's starting point)")
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)
    n_cells = sum(1 for r in data)
    n_samples = sum(int(r["samples"]) for r in data)
    t = sum(float(r["t_gpu"]) + float(r["t_refine"]) for r in data) / 60
    size = data[0]["supercell"]
    titled(fig, "Where imperfections create topology",
           f"Fraction topological over {n_cells} combinations of material and disorder, "
           f"{size}×{size} supercells",
           rf"Kane-Mele, $\lambda_{{SO}}$ = 0.3, $\lambda_R$ = 0. Line: 50% boundary. "
           f"{n_samples:,} samples in {t:.0f} min on one T4. RESULTS.md R15.")
    return save(fig, out, left=0.1, right=0.93, bottom=0.17, top=0.76)


if __name__ == "__main__":
    makers = [slide_speed, slide_split, slide_disorder_switch, slide_sizes]
    if PHASEMAP.exists():
        makers.append(slide_phase_map)
    for make in makers:
        print(f"wrote {make()}")
