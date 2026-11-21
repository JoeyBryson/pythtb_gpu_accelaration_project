"""Clean charts for the slides, from saved results (no new computation).

Run from hackathon/:  uv run python presentation/charts.py
Writes to results/figures/:

  speed_sweep_cost.png   time for a full 16,400-sample disorder sweep, per supercell size:
                         PythTB on all CPU cores vs our algorithm on CPU vs our algorithm on a T4
                         (RESULTS.md R12; estimated from measured per-sample rates)
  speedup_split.png      the total speed-up split into "better algorithm" x "GPU", per size
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent.parent
COST_CSV = HERE / "results" / "kaggle" / "20261121-213825" / "out" / "sweep_cost.csv"  # R12
FIGURES = HERE / "results" / "figures"

# validated default palette (dataviz skill): first three categorical slots pass all-pairs checks
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": AXIS,
    "axes.labelcolor": INK_2, "axes.titlecolor": INK, "axes.titlelocation": "left",
    "axes.titleweight": "bold", "axes.titlesize": 14, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "xtick.color": MUTED, "ytick.color": MUTED, "legend.frameon": False,
    "legend.labelcolor": INK_2, "font.family": "sans-serif",
})


def load_cost() -> list[dict]:
    with COST_CSV.open() as f:
        return sorted(csv.DictReader(f), key=lambda r: int(r["supercell"]))


def nice_time(hours: float) -> str:
    if hours >= 48:
        return f"{hours / 24:.0f} days"
    if hours >= 1.5:
        return f"{hours:.0f} h" if hours >= 10 else f"{hours:.1f} h"
    return f"{hours * 60:.0f} min"


def chart_sweep_cost() -> Path:
    rows = load_cost()
    methods = [
        ("sweep_h_pythtb_cpu", "PythTB 2.0, all 4 CPU cores", MUTED),
        ("sweep_h_c_up_cpu", "our algorithm, all 4 CPU cores", ORANGE),
        ("sweep_h_c_up_gpu", "our algorithm, one T4 GPU", BLUE),
    ]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ys = list(range(len(rows)))[::-1]  # biggest system at the bottom
    for y, r in zip(ys, rows, strict=True):
        hrs = [float(r[key]) for key, _, _ in methods]
        ax.plot([min(hrs), max(hrs)], [y, y], color=GRID, lw=6, solid_capstyle="round", zorder=1)
        for (key, _, colour), h in zip(methods, hrs, strict=True):
            ax.scatter(h, y, s=140, color=colour, zorder=3, edgecolor=SURFACE, linewidth=2)
            below = key == "sweep_h_c_up_gpu"  # GPU label under its dot: neighbours can be close
            ax.annotate(nice_time(h), (h, y), xytext=(0, -20 if below else 13),
                        textcoords="offset points", ha="center", fontsize=9.5,
                        color=INK if below else INK_2, fontweight="bold" if below else "normal")
        ax.annotate(f"{float(r['x_total']):.0f}× faster", (max(hrs), y), xytext=(14, -4),
                    textcoords="offset points", ha="left", fontsize=11, fontweight="bold", color=INK)

    ax.set_xscale("log")
    ax.set_yticks(ys, [f"{r['supercell']}×{r['supercell']} supercell" for r in rows])
    ax.tick_params(axis="y", length=0, labelsize=11, labelcolor=INK_2)
    ax.set_xlim(0.2, 900)
    ticks = [0.25, 1, 4, 24, 24 * 7]
    ax.set_xticks(ticks, ["15 min", "1 h", "4 h", "1 day", "1 week"])
    ax.minorticks_off()
    ax.grid(axis="y", visible=False)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_title("Time for a full disorder sweep (16,400 samples): a week → two hours")
    for _, label, colour in methods:
        ax.scatter([], [], s=90, color=colour, label=label)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=10)
    fig.text(0.01, 0.01, "Estimated from per-sample times measured on the same Kaggle machine (4 CPU cores "
             "+ 1 Tesla T4), at the transition W = 2, λ_ν = 1.65. RESULTS.md R12.",
             color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return save(fig, "speed_sweep_cost.png")


def chart_speedup_split() -> Path:
    rows = load_cost()
    sizes = [f"{r['supercell']}×{r['supercell']}" for r in rows]
    algo = [float(r["x_algorithm"]) for r in rows]
    gpu = [float(r["x_gpu_on_top"]) for r in rows]
    total = [float(r["x_total"]) for r in rows]

    fig, ax = plt.subplots(figsize=(8, 4.8))
    x = list(range(len(rows)))
    ax.plot(x, algo, color=ORANGE, lw=2.5, marker="o", ms=8, markeredgecolor=SURFACE, markeredgewidth=2)
    ax.plot(x, gpu, color=BLUE, lw=2.5, marker="o", ms=8, markeredgecolor=SURFACE, markeredgewidth=2)
    for xi, a, g in zip(x, algo, gpu, strict=True):
        # each label on the side away from the other line (the lines cross at 16x16)
        up, down = (10, -18)
        ax.annotate(f"{a:.1f}×", (xi, a), xytext=(0, up if a >= g else down),
                    textcoords="offset points", ha="center", fontsize=9.5, color=INK_2)
        ax.annotate(f"{g:.1f}×", (xi, g), xytext=(0, up if g > a else down),
                    textcoords="offset points", ha="center", fontsize=9.5, color=INK_2)
    # direct labels at the left end, where the two lines are far apart
    ax.annotate("better algorithm\n(any hardware)", (0, algo[0]), xytext=(-14, 0),
                textcoords="offset points", ha="right", va="center", fontsize=10, color=INK_2)
    ax.annotate("GPU on top", (0, gpu[0]), xytext=(-14, 0), textcoords="offset points",
                ha="right", va="center", fontsize=10, color=INK_2)
    ymax = max(algo + gpu) * 1.45
    for xi, t in zip(x, total, strict=True):
        ax.text(xi, ymax * 0.93, f"total {t:.0f}×", ha="center", fontsize=12, fontweight="bold", color=INK)
    ax.set_ylim(0, ymax)
    ax.set_xticks(x, sizes)
    ax.set_xlim(-1.1, len(rows) - 0.6)
    ax.set_ylabel("speed-up factor")
    ax.set_xlabel("supercell size (bigger = more realistic disorder)")
    ax.set_title("Two separate gains: the GPU's share grows with system size")
    fig.text(0.01, 0.01, "total = algorithm × GPU, vs PythTB on all 4 CPU cores of the same machine "
             "(Kaggle, 1 Tesla T4). RESULTS.md R12.", color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return save(fig, "speedup_split.png")


def chart_speed_simple(size: int = 16) -> Path:
    """One message, big text: how long one full sweep takes, three ways, at one size."""
    r = next(r for r in load_cost() if int(r["supercell"]) == size)
    bars = [
        ("PythTB today\n(4 CPU cores)", float(r["sweep_h_pythtb_cpu"]), MUTED),
        ("Our method\n(same 4 CPU cores)", float(r["sweep_h_c_up_cpu"]), ORANGE),
        ("Our method\n(one GPU)", float(r["sweep_h_c_up_gpu"]), BLUE),
    ]
    fig, ax = plt.subplots(figsize=(11, 5))
    ys = [2, 1, 0]
    for y, (label, h, colour) in zip(ys, bars, strict=True):
        ax.barh(y, h, height=0.62, color=colour)
        ax.text(h + 2, y, nice_time(h), va="center", ha="left", fontsize=22, fontweight="bold",
                color=INK)
    ax.set_yticks(ys, [b[0] for b in bars])
    ax.tick_params(axis="y", length=0, labelsize=16, labelcolor=INK)
    ax.set_xticks([])
    ax.grid(False)
    for side in ("bottom", "left"):
        ax.spines[side].set_visible(False)
    ax.set_xlim(0, float(r["sweep_h_pythtb_cpu"]) * 1.25)
    ax.set_title(f"One disorder study, {size}×{size} supercells: "
                 f"{float(r['x_total']):.0f}× faster", fontsize=22, pad=16)
    fig.text(0.01, 0.01, "16,400 simulated samples. Times estimated from per-sample timings measured "
             "on one Kaggle machine (4 CPU cores + 1 Tesla T4). RESULTS.md R12.",
             color=MUTED, fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return save(fig, f"speed_simple_{size}x{size}.png")


def save(fig, name: str) -> Path:
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / name
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


if __name__ == "__main__":
    for make in (chart_sweep_cost, chart_speedup_split, chart_speed_simple):
        print(f"wrote {make()}")
