"""Charts from the CPU baseline results (results/cpu_*.csv) plus one physics picture.

Run from hackathon/:  uv run python benchmarks/plot_cpu.py
Writes PNGs to results/figures/. Each chart is one small function, so they are easy to
tweak for the slides.

  cpu_scaling.png       how CPU time grows with supercell size (fixed vs scaled mesh)
  cpu_stage_split.png   where the time goes: diagonalisation vs Wilson loop
  cpu_serial_vs_parallel.png  the naive loop vs all CPU cores (the baseline to beat)
  wannier_flow.png      what "topological" looks like: winding vs non-winding Wannier centres
"""

from __future__ import annotations

import csv
from pathlib import Path

import common  # noqa: F401  (macOS warning filter)
import matplotlib

matplotlib.use("Agg")  # write files, no window
import matplotlib.pyplot as plt

from gpu_engine.reference import clean_kane_mele, half_bz_flow, solve, wannier_centres

RESULTS = Path(__file__).resolve().parent.parent / "results"
FIGURES = RESULTS / "figures"

# Colours: validated default palette (dataviz skill), checked colour-blind safe as a pair.
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK_2, MUTED, GRID, AXIS, SURFACE = (
    "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
)

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,  # gridlines behind the bars, not over them
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "font.family": "sans-serif",
        "legend.frameon": False,
        "legend.labelcolor": INK_2,
    }
)


# ---------------------------------------------------------------- reading results


def read_latest(name: str, key: str) -> list[dict]:
    """Rows of results/<name>.csv, keeping only the most recent row for each `key` value."""
    with (RESULTS / f"{name}.csv").open() as f:
        rows = list(csv.DictReader(f))
    latest = {}
    for row in rows:  # file is in time order, so later rows overwrite earlier ones
        latest[row[key]] = row
    return sorted(latest.values(), key=lambda r: int(r[key]))


def hardware_note(row: dict) -> str:
    return f"CPU: {row['cpu']}, {row['n_cpus']} cores, BLAS {row['blas']}"


def footnote(fig, text: str) -> None:
    fig.text(0.01, 0.01, text, color=MUTED, fontsize=8, ha="left", va="bottom")


# ---------------------------------------------------------------- charts


def chart_scaling() -> Path:
    """Total time for one clean supercell vs matrix size, for both mesh choices."""
    fixed = read_latest("cpu_clean", "supercell")
    scaled = read_latest("cpu_scaled", "supercell")

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for rows, colour, label in (
        (fixed, BLUE, "31×31 k-mesh (fixed)"),
        (scaled, ORANGE, "k-mesh ≈ 48/n (scaled)"),
    ):
        x = [int(r["matrix_dim"]) for r in rows]
        y = [float(r["t_total"]) for r in rows]
        ax.plot(x, y, color=colour, lw=2, marker="o", ms=7, label=label,
                markeredgecolor=SURFACE, markeredgewidth=2)
        for r, xi, yi in zip(rows, x, y, strict=True):  # label each point with the supercell
            ax.annotate(f"{r['supercell']}×{r['supercell']}", (xi, yi), textcoords="offset points",
                        xytext=(6, -12), fontsize=8, color=INK_2)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Hamiltonian matrix size (4n² for an n×n supercell)")
    ax.set_ylabel("seconds per configuration")
    ax.set_title("CPU cost of one configuration grows steeply with supercell size")
    ax.legend(loc="upper left")
    footnote(fig, f"PythTB 2.0, clean Kane-Mele, best of 3. {hardware_note(fixed[0])}")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return save(fig, "cpu_scaling.png")


def chart_stage_split() -> Path:
    """Share of time in diagonalisation vs Wilson loop, per supercell (100% stacked bars)."""
    rows = read_latest("cpu_clean", "supercell")
    labels = [f"{r['supercell']}×{r['supercell']}" for r in rows]
    solve_t = [float(r["t_solve"]) for r in rows]
    wilson_t = [float(r["t_wilson"]) for r in rows]
    share_solve = [100 * s / (s + w) for s, w in zip(solve_t, wilson_t, strict=True)]
    share_wilson = [100 - s for s in share_solve]

    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    gap = 0.6  # thin surface gap between the two segments
    ax.barh(labels, share_solve, color=BLUE, height=0.6, label="diagonalise H(k)")
    ax.barh(labels, [w - gap for w in share_wilson], left=[s + gap for s in share_solve],
            color=ORANGE, height=0.6, label="Wilson loop")
    for i, (s, w, st, wt) in enumerate(zip(share_solve, share_wilson, solve_t, wilson_t, strict=True)):
        ax.text(s / 2, i, f"{s:.0f}%  ({st:.2f} s)", ha="center", va="center", color="white", fontsize=9)
        ax.text(s + w / 2, i, f"{w:.0f}%", ha="center", va="center", color="white", fontsize=9)

    ax.set_xlim(0, 100)
    ax.set_xlabel("share of time (%)")
    ax.grid(axis="y", visible=False)
    ax.invert_yaxis()
    ax.set_title("Where the CPU time goes: ~3/4 diagonalisation, ~1/4 Wilson loop")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
    footnote(fig, f"31×31 k-mesh, clean model. {hardware_note(rows[0])}")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return save(fig, "cpu_stage_split.png")


def chart_serial_vs_parallel() -> Path:
    """Seconds per configuration: plain loop vs one process per CPU core."""
    serial = read_latest("cpu_serial", "supercell")
    parallel = read_latest("cpu_parallel", "supercell")
    sizes = [r["supercell"] for r in serial if r["supercell"] in {p["supercell"] for p in parallel}]
    s_per = {r["supercell"]: float(r["wall_s"]) / int(r["configs"]) for r in serial}
    p_per = {r["supercell"]: float(r["wall_s"]) / int(r["configs"]) for r in parallel}
    workers = parallel[0]["workers"]

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    width, x = 0.36, range(len(sizes))
    bars_s = ax.bar([i - width / 2 - 0.01 for i in x], [s_per[n] for n in sizes], width,
                    color=BLUE, label="serial loop (1 process)")
    bars_p = ax.bar([i + width / 2 + 0.01 for i in x], [p_per[n] for n in sizes], width,
                    color=ORANGE, label=f"parallel ({workers} processes)")
    ax.bar_label(bars_s, fmt="%.2f s", padding=3, color=INK_2, fontsize=9)
    ax.bar_label(bars_p, fmt="%.2f s", padding=3, color=INK_2, fontsize=9)
    for i, n in enumerate(sizes):
        ax.text(i, max(s_per[n], p_per[n]) * 1.18, f"{s_per[n] / p_per[n]:.1f}× faster",
                ha="center", color=INK, fontsize=9, fontweight="bold")

    nk = {r["supercell"]: r["nk"] for r in serial}
    ax.set_xticks(list(x), [f"{n}×{n} supercell\n({nk[n]}×{nk[n]} mesh)" for n in sizes])
    ax.set_ylabel("seconds per configuration")
    ax.set_ylim(0, max(s_per.values()) * 1.35)
    ax.grid(axis="x", visible=False)
    ax.set_title("The CPU baseline to beat: every core on its own configuration")
    ax.legend(loc="upper left")
    W, configs = serial[0]["W"], serial[0]["configs"]
    footnote(fig, f"{configs} disorder configs at W = {W}, full pipeline incl. Z2. {hardware_note(serial[0])}")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return save(fig, "cpu_serial_vs_parallel.png")


def chart_wannier_flow(nk: int = 48) -> Path:
    """The physics picture: Wannier centres wind (topological) or don't (trivial)."""
    cases = [("Topological (Z2 = 1)", 1.0), ("Normal insulator (Z2 = 0)", 2.5)]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    for ax, (title, delta) in zip(axes, cases, strict=True):
        model = clean_kane_mele(delta=delta)
        wfa = solve(model, nk)
        flow = half_bz_flow(wfa, wannier_centres(wfa, model.norb))  # (k_x steps, bands)
        kx = [i / (len(flow) - 1) for i in range(len(flow))]  # 0 ... 1 meaning 0 ... pi
        for band in range(flow.shape[1]):
            ax.plot(kx, flow[:, band], ls="none", marker="o", ms=3.5, color=INK)
        ax.set_title(title, fontsize=12)
        ax.set_xticks([0, 0.5, 1], ["0", "π/2", "π"])
        ax.set_xlabel("$k_x$")
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("Wannier centre along y (fraction of cell)")
    fig.suptitle("Topological: the centres wind across the cell.  Normal: they come back.",
                 x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK)
    footnote(fig, f"Clean Kane-Mele, PythTB tutorial parameters (on-site 1.0 vs 2.5), {nk}×{nk} k-mesh.")
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    return save(fig, "wannier_flow.png")


def save(fig, name: str) -> Path:
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / name
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


if __name__ == "__main__":
    for make in (chart_scaling, chart_stage_split, chart_serial_vs_parallel, chart_wannier_flow):
        print(f"wrote {make()}")
