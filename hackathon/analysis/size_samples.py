"""Seven supercell sizes, 100 vs 500 disorder samples: what the extra samples buy.

Written at the hackathon (22 Nov 2026). Input: ensemble CSVs from scripts/ensemble_sweep.py
(C_up, lambda_nu = 1.65, lambda_R = 0, shared seeds), any number of files, e.g. the parts of
one sweep run on several GPUs. One seed = one disorder pattern, kept together across all W.

  - "100 samples": the first 100 seeds; the 5 disjoint groups of 100 are 5 independent
    repeats of a 100-sample experiment (grey dots in c, d).
  - "500 samples": all seeds.
  - Fractions are over RESOLVED samples (C_up passes the reliability check at nk = 6 or 10).
  - Window opens / closes = where the fraction crosses 1/2 going up / down (linear
    interpolation between adjacent grid points). Error bars: 95% bootstrap over seeds.

Run from hackathon/:
  uv run python analysis/size_samples.py results/ensemble_sizes_*.csv
Output: results/size_samples.png and results/size_samples.csv (the edge table).
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.ensemble import load_rows
from analysis.ensemble_plots import INK, MUTED, _trajectories, wilson_interval

SMALL, BIG = 100, 500
N_BOOT = 1000
plt.rcParams.update({"font.size": 11, "axes.titlesize": 12.5, "axes.labelsize": 11,
                     "legend.fontsize": 9.5, "xtick.labelsize": 10, "ytick.labelsize": 10})


def size_colours(Ls) -> dict[int, tuple]:
    """Ordered sizes -> a light-to-dark blue ramp (bigger supercell = darker)."""
    cmap = plt.get_cmap("Blues")
    return {L: cmap(0.4 + 0.6 * i / max(len(Ls) - 1, 1)) for i, L in enumerate(Ls)}


def tables(d: dict, L: int):
    """(Ws, T, R): (seeds, W) tables of 'topological' (|C_up| odd) and 'resolved'."""
    m = d["L"] == L
    _, Ws, Z = _trajectories(d, m, "z2")
    _, _, R = _trajectories(d, m, "reliable", fill=0)
    return Ws, Z == 1, R.astype(bool)


def frac(T, R, idx):
    """Fraction topological among resolved, and (k, n), per W, over seed rows idx."""
    k, n = np.sum(T[idx] & R[idx], 0), np.sum(R[idx], 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return k / n, k, n


def edges(Ws, p) -> tuple[float, float]:
    """(opens, closes): first upward and last downward crossing of 1/2. nan if absent."""
    up, down = np.nan, np.nan
    for i in range(len(p) - 1):
        a, b = p[i], p[i + 1]
        if np.isnan(a) or np.isnan(b) or (a < 0.5) == (b < 0.5):
            continue
        w = Ws[i] + (0.5 - a) * (Ws[i + 1] - Ws[i]) / (b - a)
        if b > a and np.isnan(up):
            up = w
        elif b < a:
            down = w
    return up, down


def boot_edges(Ws, T, R, idx, rng, n_boot=N_BOOT) -> np.ndarray:
    """(n_boot, 2) bootstrap of (opens, closes), resampling whole seed rows of idx."""
    return np.array([edges(Ws, frac(T, R, rng.choice(idx, len(idx)))[0]) for _ in range(n_boot)])


def summarise(d: dict, Ls, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    out = []
    for L in Ls:
        Ws, T, R = tables(d, L)
        n_seeds = T.shape[0]
        groups = [np.arange(g * SMALL, (g + 1) * SMALL) for g in range(n_seeds // SMALL)]
        row = {"L": L, "sites": 2 * L * L, "seeds": n_seeds,
               "resolved_min": float(R.mean(0).min())}
        for label, idx in (("n100", groups[0]), ("n500", np.arange(min(BIG, n_seeds)))):
            est = edges(Ws, frac(T, R, idx)[0])
            b = boot_edges(Ws, T, R, idx, rng)
            for j, name in enumerate(("opens", "closes")):
                row[f"{name}_{label}"] = est[j]
                row[f"{name}_{label}_lo"], row[f"{name}_{label}_hi"] = np.nanpercentile(b[:, j], [2.5, 97.5])
                row[f"{name}_{label}_sd"] = float(np.nanstd(b[:, j]))
        reps = np.array([edges(Ws, frac(T, R, g)[0]) for g in groups])
        row["opens_repeats"], row["closes_repeats"] = reps[:, 0], reps[:, 1]
        sec = [d["batch_seconds"][(d["L"] == L) & (d["W"] == w)][0] / n_seeds for w in Ws]
        row["s_per_sample"] = float(np.mean(sec))
        out.append(row)
    return out


def panel_curves(ax, d, Ls, colours, n, title):
    for L in Ls:
        Ws, T, R = tables(d, L)
        p, k, m = frac(T, R, np.arange(n))
        lo, hi = wilson_interval(k, m)
        ax.fill_between(Ws, lo, hi, color=colours[L], alpha=0.25, lw=0)
        ax.plot(Ws, p, "-", color=colours[L], lw=2, label=f"{L}×{L}")
    ax.axhline(0.5, color=MUTED, lw=0.8, ls=":")
    ax.set(xlabel="Disorder strength W (t)", ylabel="Fraction topological", ylim=(-0.02, 1.02),
           xlim=(0, 8))
    ax.set_title(title, loc="left", fontweight="bold")
    ax.grid(alpha=0.3)


def panel_edge(ax, rows, colours, name, title, ylabel):
    x = np.arange(len(rows))
    for i, r in enumerate(rows):
        reps = r[f"{name}_repeats"]
        ax.scatter(np.full(len(reps), i - 0.15), reps, s=18, color=MUTED, alpha=0.8, lw=0,
                   label="5 separate 100-sample runs" if i == 0 else None, zorder=2)
        for label, dx, col, mk in (("n100", -0.15, MUTED, "o"), ("n500", 0.15, colours[r["L"]], "D")):
            v, lo, hi = r[f"{name}_{label}"], r[f"{name}_{label}_lo"], r[f"{name}_{label}_hi"]
            if label == "n100":
                ax.errorbar(i + dx, v, yerr=[[v - lo], [hi - v]], fmt="none", ecolor=INK, lw=1.2,
                            capsize=3, zorder=3, label="100 samples: 95% range" if i == 0 else None)
            else:
                ax.errorbar(i + dx, v, yerr=[[v - lo], [hi - v]], fmt=mk, color=col, mec=INK,
                            ecolor=INK, ms=7, lw=1.6, capsize=3, zorder=4,
                            label="500 samples: estimate and 95% range" if i == 0 else None)
    ax.set_xticks(x, [f"{r['L']}×{r['L']}\n{r['sites']}" for r in rows])
    ax.set(xlabel="Supercell (top) and number of sites (bottom)", ylabel=ylabel)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.grid(alpha=0.3, axis="y")


def figure(d: dict, rows: list[dict], out: Path, hardware: str) -> None:
    Ls = [r["L"] for r in rows]
    colours = size_colours(Ls)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10.5), layout="constrained")
    panel_curves(axes[0, 0], d, Ls, colours, SMALL, f"a  {SMALL} samples per point")
    panel_curves(axes[0, 1], d, Ls, colours, BIG, f"b  {BIG} samples per point")
    axes[0, 1].legend(title="Supercell", loc="center right", ncol=1, frameon=True)
    panel_edge(axes[1, 0], rows, colours, "opens", "c  Where the topological window opens",
               "W where the fraction first rises past 1/2")
    panel_edge(axes[1, 1], rows, colours, "closes", "d  Where it closes again",
               "W where the fraction falls back below 1/2")
    axes[1, 0].legend(loc="upper right", frameon=True)
    total = sum(r["s_per_sample"] * r["seeds"] for r in rows) * len(np.unique(d["W"]))
    fig.suptitle(f"Disordered Kane-Mele (λ_ν = 1.65, λ_R = 0), {len(Ls)} sizes × {len(np.unique(d['W']))} "
                 f"disorder strengths × {BIG} samples: {total / 60:.0f} GPU-minutes on {hardware}",
                 x=0.01, ha="left", fontsize=14, fontweight="bold")
    fig.text(0.01, -0.01,
             "Shaded bands in a, b: 95% Wilson intervals. Fractions are over samples whose "
             "spin-up Chern number passed the reliability check. Error bars in c, d: 95% "
             "bootstrap over disorder patterns. The 5 grey dots per size are disjoint groups "
             "of 100 samples, i.e. five independent 100-sample experiments.",
             fontsize=9, color=MUTED, ha="left", va="top", wrap=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("csv", type=Path, nargs="+")
    ap.add_argument("--out", type=Path, default=Path("results/size_samples.png"))
    ap.add_argument("--hardware", default="Modal H100s")
    args = ap.parse_args()

    d = load_rows(*args.csv)
    assert len(np.unique(d["delta"])) == 1 and np.all(d["rashba"] == 0), "one lambda_nu, rashba 0"
    Ls = sorted(int(L) for L in np.unique(d["L"]))
    rows = summarise(d, Ls)
    figure(d, rows, args.out, args.hardware)

    keys = [k for k in rows[0] if not k.endswith("repeats")]
    with open(args.out.with_suffix(".csv"), "w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows({k: r[k] for k in keys} for r in rows)
    print(f"{'size':>6} {'opens 100':>16} {'opens 500':>16} {'closes 100':>16} {'closes 500':>16}"
          f" {'resolved':>8} {'s/sample':>8}")
    for r in rows:
        def cell(n, s, r=r):
            return f"{r[f'{n}_{s}']:6.2f} ± {r[f'{n}_{s}_sd']:.2f}"

        print(f"{r['L']:>3}x{r['L']:<2} {cell('opens', 'n100'):>16} {cell('opens', 'n500'):>16} "
              f"{cell('closes', 'n100'):>16} {cell('closes', 'n500'):>16} "
              f"{r['resolved_min']:8.0%} {r['s_per_sample']:8.3f}")
    print(f"Saved {args.out} and {args.out.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
