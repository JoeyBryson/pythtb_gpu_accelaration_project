"""Our disorder ensemble vs the published TAI window (lambda_nu = 1.65, lambda_R = 0).

Written at the hackathon (21 Nov 2026). Overlays, all in OUR disorder convention
(eps in [-W/2, W/2]; Orth's W is doubled):
  - ours: fraction of resolved samples with Z2 = 1 (95% Wilson interval) and the range if the
    unresolved ones were all trivial / all topological, from an ensemble CSV
  - Favata & Marrazzo 2023, Fig. 7 left (Delta/lambda_SO = 5.5, lambda_R = 0): mean +- SD of the
    single-point spin Chern number over M = 50-100 configurations, L = 15 and 42
    (analysis/reference_data/favata2023_fig7_left.csv, extracted from the vector figure)
  - Orth et al. 2016, Fig. 5: onset W_c at lambda_nu = 1.65 from tight-binding conductance, and
    the lowest-order Born curve (analysis/reference_data/orth2016_fig5.csv)

Different observables (to keep in mind when reading the plot): Favata's single-point value is
not an exact integer in a finite cell, so its mean is not the same thing as our fraction of
topological samples; Orth measures edge conductance. We compare where the window opens and
closes (W at which each curve crosses 1/2), not the curves' shapes.

Run from hackathon/:
  uv run python analysis/compare_papers.py results/ensemble_L16_n500_h100.csv
Output: results/validation_tai_window.png, and the edge table printed.
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
from analysis.ensemble_plots import (
    INK,
    L_COLOURS,
    MUTED,
    one_slice,
    outcome_counts,
    wilson_interval,
)

REF = Path(__file__).resolve().parent / "reference_data"
LAMBDA_NU = 1.65


def crossings(W: np.ndarray, y: np.ndarray) -> list[tuple[float, str]]:
    """W where y crosses 1/2 (linear interpolation), labelled 'opens' or 'closes'."""
    out = []
    for i in range(len(y) - 1):
        if np.isnan(y[i]) or np.isnan(y[i + 1]):
            continue
        if (y[i] < 0.5) != (y[i + 1] < 0.5):
            w = W[i] + (0.5 - y[i]) * (W[i + 1] - W[i]) / (y[i + 1] - y[i])
            out.append((float(w), "opens" if y[i + 1] > y[i] else "closes"))
    return out


def favata() -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    with open(REF / "favata2023_fig7_left.csv") as f:
        rows = list(csv.DictReader(f))
    out = {}
    for L in sorted({int(r["L"]) for r in rows}):
        sel = [r for r in rows if int(r["L"]) == L]
        out[L] = tuple(np.array([float(r[k]) for r in sel]) for k in ("W", "C_s_mean", "C_s_sd"))
    return out


def orth_onset() -> tuple[float, float, float]:
    """(tight-binding W_c, its error, Born W_c) at lambda_nu = 1.65, in OUR convention."""
    with open(REF / "orth2016_fig5.csv") as f:
        rows = list(csv.DictReader(f))
    tb = [r for r in rows if r["source"] == "tight-binding" and float(r["lambda_nu"]) == LAMBDA_NU]
    born = [r for r in rows if r["source"] == "born"]
    lam = np.array([float(r["lambda_nu"]) for r in born])
    wc = np.array([float(r["Wc_orth"]) for r in born])
    return 2 * float(tb[0]["Wc_orth"]), 2 * float(tb[0]["err_orth"]), 2 * float(np.interp(LAMBDA_NU, lam, wc))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("csv", type=Path, help="our ensemble CSV at lambda_nu = 1.65, rashba = 0")
    ap.add_argument("--out", type=Path, default=Path("results/validation_tai_window.png"))
    args = ap.parse_args()

    d = one_slice(load_rows(args.csv), None)
    assert np.isclose(d["delta"][0], LAMBDA_NU) and d["rashba"][0] == 0, "needs lambda_nu 1.65, rashba 0"
    L = int(np.max(d["L"]))
    m = d["L"] == L
    Ws, n_all, n_t, n_f, n_u = outcome_counts(d, m)
    with np.errstate(invalid="ignore", divide="ignore"):
        p = n_t / (n_t + n_f)
    lo, hi = wilson_interval(n_t, n_t + n_f)
    S = int(n_all[0])
    tb, tb_err, born = orth_onset()
    fav = favata()

    fig, ax = plt.subplots(figsize=(12, 6.8), layout="constrained")
    ax.fill_between(Ws, n_t / n_all, (n_t + n_u) / n_all, color=L_COLOURS[0], alpha=0.12, lw=0,
                    label="ours: if unresolved were trivial … topological")
    ax.fill_between(Ws, lo, hi, color=L_COLOURS[0], alpha=0.3, lw=0)
    ax.plot(Ws, p, "o-", color=L_COLOURS[0], lw=2.6, ms=5,
            label=f"ours: {L}×{L} ({2 * L * L} sites), fraction topological, {S} samples per W")
    for (Lf, (W, C, sd)), col in zip(fav.items(), (L_COLOURS[1], L_COLOURS[2])):
        ax.errorbar(W, C, yerr=sd, fmt="s-", color=col, ms=3.5, lw=1.3, capsize=2, elinewidth=0.8,
                    label=f"Favata & Marrazzo 2023, L = {Lf} ({2 * Lf * Lf} sites): "
                          "mean spin Chern number ± SD")
    ax.axvspan(tb - tb_err, tb + tb_err, color=INK, alpha=0.12, lw=0)
    ax.axvline(tb, color=INK, lw=1.2, label=f"Orth et al. 2016: onset from edge conductance, W = {tb:.2f}")
    ax.axvline(born, color=MUTED, lw=1.2, ls="--", label=f"Born approximation (Orth's Fig. 5): W = {born:.2f}")
    ax.axhline(0.5, color=MUTED, lw=0.8, ls=":")
    ax.set(xlabel="Disorder strength W (t), on-site energies uniform in [−W/2, W/2]",
           ylabel="Fraction topological  /  mean invariant", ylim=(-0.3, 1.25), xlim=(0, 12.2))
    ax.set_title(f"The topological window at λ_ν = {LAMBDA_NU}, λ_R = 0: ours vs published",
                 loc="left", fontsize=14, fontweight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=2, fontsize=9, frameon=False)
    ax.grid(alpha=0.3)
    fig.text(0.01, -0.13,
             "Different observables: Favata's single-point spin Chern number is not an exact integer in a "
             "finite cell (hence values below 0 or above 1); Orth measures edge conductance (W doubled to "
             "our convention; Orth's plots end at W = 6). Compare where the curves cross 1/2.",
             fontsize=9, color=MUTED, ha="left", va="top", wrap=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")

    print(f"{'curve':<42} {'opens':>6} {'closes':>7}")
    rows = [(f"ours {L}x{L} ({2 * L * L} sites)", crossings(Ws, p))]
    rows += [(f"Favata L={Lf} ({2 * Lf * Lf} sites)", crossings(W, C)) for Lf, (W, C, _) in fav.items()]
    for name, xs in rows:
        op = [w for w, k in xs if k == "opens"]
        cl = [w for w, k in xs if k == "closes"]
        print(f"{name:<42} {op[0] if op else float('nan'):6.2f} {cl[-1] if cl else float('nan'):7.2f}")
    print(f"{'Orth tight-binding onset':<42} {tb:6.2f} ± {tb_err:.2f}")
    print(f"{'Born (Orth normalisation)':<42} {born:6.2f}")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
