"""Plot a disorder sweep: fraction of samples that are topological vs disorder strength W.

Run from hackathon/:  uv run python analysis/plot_sweep.py results/sweep/pilot_8x8.csv [more.csv ...]
Writes <csv name>.png next to the first CSV (or --out).

Top: fraction topological (among reliable answers) +- binomial standard error, one curve per
lambda_nu (and per supercell size, if several files are given).
Bottom: how many samples per point were unreliable or had the band gap closed, so the
reader can see where the answers are weaker. Never hidden.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# validated default palette (dataviz skill): categorical slots in fixed order
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": AXIS,
    "axes.labelcolor": INK_2, "axes.titlecolor": INK, "axes.titlelocation": "left",
    "axes.titleweight": "bold", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "xtick.color": MUTED, "ytick.color": MUTED, "legend.frameon": False,
    "legend.labelcolor": INK_2, "font.family": "sans-serif",
})

LABELS = {1.0: "λ_ν = 1.0: topological when clean", 1.65: "λ_ν = 1.65: normal when clean"}


def load(paths: list[Path]) -> dict[tuple, list[dict]]:
    """{(lambda_nu, supercell): rows sorted by W}"""
    curves: dict[tuple, list[dict]] = {}
    for path in paths:
        with path.open() as f:
            for r in csv.DictReader(f):
                curves.setdefault((float(r["lambda_nu"]), int(r["supercell"])), []).append(r)
    return {k: sorted(v, key=lambda r: float(r["W"])) for k, v in sorted(curves.items())}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("csv", type=Path, nargs="+")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--lambda-nu", type=float, default=None,
                   help="show only this lambda_nu; sizes then share one colour, darker = bigger")
    p.add_argument("--W-max", type=float, default=None, help="cut the x axis here")
    p.add_argument("--title", default=None)
    args = p.parse_args()
    curves = load(args.csv)
    if args.lambda_nu is not None:
        curves = {k: v for k, v in curves.items() if abs(k[0] - args.lambda_nu) < 1e-9}
    if args.W_max is not None:
        curves = {k: [r for r in v if float(r["W"]) <= args.W_max + 1e-9] for k, v in curves.items()}
    sizes = sorted({n for _, n in curves})
    # one lambda_nu, several sizes: an ordinal blue ramp (validated steps), light -> dark with size
    ramp = {n: c for n, c in zip(sizes, ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"][-len(sizes):],
                                 strict=False)}

    fig, (top, bottom) = plt.subplots(2, 1, figsize=(9, 6.4), sharex=True,
                                      gridspec_kw={"height_ratios": [3, 1]})
    for i, ((lam, n), rows) in enumerate(curves.items()):
        colour = ramp[n] if args.lambda_nu is not None and len(sizes) > 1 else SERIES[i % len(SERIES)]
        W = [float(r["W"]) for r in rows]
        frac = [float(r["frac_topological"]) for r in rows]
        err = [float(r["stderr"]) for r in rows]
        if args.lambda_nu is not None and len(sizes) > 1:
            label = f"{n}×{n} supercell"
        else:
            label = LABELS.get(lam, f"λ_ν = {lam:g}") + (f", {n}×{n}" if len(sizes) > 1 else "")
        top.errorbar(W, frac, yerr=err, color=colour, lw=2, marker="o", ms=5, capsize=2,
                     elinewidth=1, label=label, markeredgecolor=SURFACE, markeredgewidth=1)
        weak = [int(r["n_unreliable"]) + int(r["n_gap_closed"]) for r in rows]
        bottom.plot(W, weak, color=colour, lw=1.5, marker="o", ms=3.5)

    first = next(iter(curves.values()))[0]
    samples = sorted({r["samples"] for v in curves.values() for r in v})
    meshes = sorted({f"{r['nk']}×{r['nk']}" for v in curves.values() for r in v})
    top.set_ylim(-0.03, 1.03)
    top.set_ylabel("fraction topological")
    title = args.title or (f"Does disorder destroy topology, or create it? "
                           f"({', '.join(f'{n}×{n}' for n in sizes)} supercells)")
    top.set_title(title, fontsize=13)
    top.legend(loc="upper right")  # top right stays clear: curves end low at strong W
    bottom.set_ylabel("uncertain or\ngap-closed")
    bottom.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))  # counts
    bottom.set_xlabel("disorder strength W (random on-site energies in [−W/2, W/2])")
    fig.text(0.01, 0.01,
             f"Kane-Mele, λ_SO = 0.3, λ_R = 0{f', λ_ν = {args.lambda_nu:g}' if args.lambda_nu else ''}. "
             f"{'/'.join(samples)} disorder samples per point, k-mesh {', '.join(meshes)} "
             f"(uncertain ones re-run at {first['refine_nk']}).\n"
             "Error bars: binomial standard error over reliable samples.",
             color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    out = args.out or args.csv[0].with_suffix(".png")
    fig.savefig(out, dpi=160)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
