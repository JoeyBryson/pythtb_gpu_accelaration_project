"""Slide figures for the finite-size-scaling result (analysis/fss.py), in the slide style of
presentation/style.py (Oscar). Re-run after fss.py whenever new data arrives.

  1  the switch sharpens with system size (data + fits, critical band)
  2  all sizes cross at one point (crossings and midpoints vs 1/L)
  3  how fast it sharpens: the exponent nu (width vs L, log-log, with nu = 1 and nu = 2.5 guides)
  4  where everyone puts the critical disorder (our estimate vs the papers, one axis, our units)

Run from hackathon/:  uv run python analysis/fss_figures.py [--tag fss_estimate]
Writes results/figures/slides/fss_<n>_<name>_<tag>.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "presentation"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fss import load  # noqa: E402
from style import AQUA, BLUE, BLUE_DARK, BLUE_LIGHT, INK, INK_2, MUTED, ORANGE, TINT, new_slide, save, titled  # noqa: E402

PARAMS = "Kane-Mele, λν = 1.65t, λSO = 0.3t, λR = 0, half filling"
CONV = "Disorder W: random on-site energies in [−W/2, W/2] (Orth et al. use [−W, W]: their W ≈ t is our W ≈ 2)."


def shades(n: int) -> list:
    """Light → dark blue for small → large sizes."""
    import matplotlib.colors as mc

    cmap = mc.LinearSegmentedColormap.from_list("sizes", [BLUE_LIGHT, BLUE, BLUE_DARK])
    return [cmap(x) for x in np.linspace(0, 1, n)]


def logistic(W, mid, width):
    return 1 / (1 + np.exp(-(W - mid) / width))


def fig_switch(data, est, tag):
    Ls = est["sizes"]
    fig, ax = new_slide()
    cols = shades(len(Ls))
    lo, hi = est["W_range"]
    Wf = np.linspace(lo, hi, 400)
    if "Wc_weighted_mean" in est:
        wc, e = est["Wc_weighted_mean"]["Wc"], est["Wc_weighted_mean"]["err"]
        ax.axvspan(wc - e, wc + e, color=ORANGE, alpha=0.15, lw=0)
        ax.axvline(wc, color=ORANGE, lw=2)
        ax.text(wc + e + 0.02, 0.06, f"critical disorder\nW_c = {wc:.2f} ± {e:.2f}", color=ORANGE,
                fontsize=15, fontweight="bold", va="bottom")
    for L, c in zip(Ls, cols):
        W, k, n = data[L]
        p = k / np.maximum(n, 1)
        ax.errorbar(W, p, yerr=np.sqrt(p * (1 - p) / np.maximum(n, 1)), fmt="o", ms=6, color=c, elinewidth=1.2,
                    capsize=0, zorder=3)
        ps = est["per_size"][str(L)] if str(L) in est["per_size"] else est["per_size"][L]
        ax.plot(Wf, logistic(Wf, ps["W_mid"], ps["width"]), color=c, lw=2.5, label=f"{L}×{L}  ({int(n.mean())} samples per point)")
    ax.set(xlim=(lo, hi), ylim=(-0.03, 1.03), xlabel="disorder strength W", ylabel="fraction of samples that are topological")
    ax.yaxis.grid(True)
    ax.legend(loc="upper left", title="supercell", title_fontsize=13)
    titled(fig, "The switch gets sharper as the material gets bigger",
           "Fraction of random disorder samples that are topological, per system size, with S-curve fits",
           f"{PARAMS}. Topology per sample: spin-up Chern number C↑ (Z₂ = C↑ mod 2), GPU. Errors: binomial. {CONV}")
    return save(fig, f"fss_1_switch_{tag}.png")


def fig_crossings(est, tag):
    Ls = est["sizes"]
    fig, ax = new_slide()
    if "Wc_weighted_mean" in est:
        wc, e = est["Wc_weighted_mean"]["Wc"], est["Wc_weighted_mean"]["err"]
        ax.axhspan(wc - e, wc + e, color=ORANGE, alpha=0.15, lw=0)
        ax.axhline(wc, color=ORANGE, lw=2)
        ax.text(0.003, wc + e + 0.006, f"weighted mean of crossings  W_c = {wc:.3f} ± {e:.3f}", color=ORANGE, fontsize=14)
    mids = [est["per_size"][str(L)] for L in Ls]
    ax.errorbar([1 / L for L in Ls], [m["W_mid"] for m in mids], yerr=[m["W_mid_err"] for m in mids], fmt="o", ms=9,
                color=BLUE, elinewidth=1.5, label="midpoint of each size (50% topological)")
    for L, m in zip(Ls, mids):
        ax.annotate(f"{L}×{L}", (1 / L, m["W_mid"]), textcoords="offset points", xytext=(10, -4), color=BLUE, fontsize=12)
    cr = est["crossings"]
    keys = list(cr)
    x = [2 / sum(map(int, k.split("/"))) for k in keys]
    ax.errorbar(x, [cr[k]["W"] for k in keys], yerr=[cr[k]["err"] for k in keys], fmt="s", ms=9, color=INK,
                elinewidth=1.5, label="where consecutive sizes cross")
    for xi, k in zip(x, keys):
        ax.annotate(k.replace("/", " & "), (xi, cr[k]["W"]), textcoords="offset points", xytext=(10, 6), color=INK_2, fontsize=11)
    if "pseudo_critical" in est:
        pc = est["pseudo_critical"]
        ax.errorbar([0], [pc["Wc"]], yerr=[pc["Wc_err"]], fmt="D", ms=9, color=AQUA, elinewidth=1.5,
                    label=f"midpoints extrapolated to infinite size: {pc['Wc']:.3f} ± {pc['Wc_err']:.3f}")
    ax.set(xlim=(-0.005, 1 / min(Ls) * 1.15), xlabel="1 / system size   (infinite system at 0)", ylabel="disorder strength W")
    ax.yaxis.grid(True)
    ax.legend(loc="upper left", fontsize=13)
    drift = est.get("drift", {})
    note = (f"Drift of the crossings with size: {'significant' if drift.get('significant') else 'not significant'} at 2σ."
            if drift else "")
    titled(fig, "All sizes cross at one point: the critical disorder",
           "Finite-size scaling, as for the Binder cumulant of the Ising model: crossings of consecutive sizes",
           f"{PARAMS}. Logistic fits per size; errors from 2,000 parametric bootstrap resamples. {note} {CONV}")
    return save(fig, f"fss_2_crossings_{tag}.png")


def fig_exponent(est, tag):
    Ls = np.array(est["sizes"], float)
    wid = np.array([est["per_size"][str(int(L))]["width"] for L in Ls])
    err = np.array([est["per_size"][str(int(L))]["width_err"] for L in Ls])
    fig, ax = new_slide()
    Lf = np.geomspace(Ls.min() * 0.85, Ls.max() * 1.6, 100)
    for nu, ls in ((1.0, ":"), (2.5, "--")):  # guides through the smallest size
        ax.plot(Lf, wid[0] * (Lf / Ls[0]) ** (-1 / nu), ls, color=MUTED, lw=1.5)
        ax.text(Lf[-1], wid[0] * (Lf[-1] / Ls[0]) ** (-1 / nu), f"  ν = {nu:g}", color=MUTED, fontsize=12, va="center")
    if "width_scaling" in est:
        ws = est["width_scaling"]
        B = np.exp(np.mean(np.log(wid) + ws["inv_nu"] * np.log(Ls)))
        ax.plot(Lf, B * Lf ** (-ws["inv_nu"]), color=ORANGE, lw=2.5,
                label=f"fit: width ∝ L^(−1/ν),  ν = {ws['nu']:.2f} ± {ws['nu_err']:.2f}")
    ax.errorbar(Ls, wid, yerr=err, fmt="o", ms=10, color=BLUE, elinewidth=1.5, zorder=3, label="width of the switch, per size")
    ax.set(xscale="log", yscale="log", xlabel="system size L (supercell L × L)", ylabel="width of the switch (in W)")
    ax.set_xticks(Ls)
    ax.set_xticklabels([f"{int(L)}" for L in Ls])
    ax.minorticks_off()
    ax.legend(loc="lower left")
    titled(fig, "How fast the switch sharpens: the critical exponent ν",
           "The transition width shrinks as a power of the system size; the power gives ν",
           f"{PARAMS}. Guides: ν = 1 (clean, mass-driven gap closing) and ν ≈ 2.5 (quantum-Hall-type transitions, e.g. "
           "disordered Haldane model, Mildner et al.). Small sizes can sit in a crossover between them: to check with larger L.")
    return save(fig, f"fss_3_exponent_{tag}.png")


def fig_comparison(est, tag):
    """One axis, our units. Rows: theory, published methods, ours."""
    rows = [  # label, value, err (None = no error bar), colour, note
        ("Lowest-order Born approximation (theory)", 1.74, 0.06, MUTED, "1.68–1.80 depending on normalisation"),
        ("Orth et al. 2016: edge conductance, 93 × 150 ribbon", 2.14, 0.07, INK_2, "their Fig. 5 point; definition of W_c not stated"),
        ("Huang & Liu 2018: spin Bott index, 10 samples", 2.0, 0.2, INK_2, "'around W = 1.0t' (their units), read from text"),
        ("Favata & Marrazzo 2023: spin Chern number, L = 15", 1.99, None, INK_2, "midpoint of their curve (our interpolation)"),
        ("Favata & Marrazzo 2023: spin Chern number, L = 42", 1.92, None, INK_2, "midpoint of their curve (our interpolation)"),
    ]
    big = max(est["sizes"])
    pb = est["per_size"][str(big)]
    rows.append((f"Ours: midpoint at {big}×{big}", pb["W_mid"], pb["W_mid_err"], BLUE, "same estimator as the Favata rows"))
    if "Wc_weighted_mean" in est:
        w = est["Wc_weighted_mean"]
        rows.append((f"Ours: finite-size scaling ({len(est['sizes'])} sizes)", w["Wc"], w["err"], ORANGE,
                     "crossings of consecutive sizes, weighted mean"))
    fig, ax = new_slide()
    y = np.arange(len(rows))[::-1]
    for yi, (lab, v, e, c, note) in zip(y, rows):
        if e is not None:
            ax.errorbar([v], [yi], xerr=[e], fmt="o", ms=11, color=c, elinewidth=3, capsize=0)
        else:
            ax.plot([v], [yi], "o", ms=11, color=c, mfc="white", mew=2.5)
        ax.text(1.43, yi, lab, ha="left", va="center", fontsize=14, color=INK if c in (BLUE, ORANGE) else INK_2,
                fontweight="bold" if c in (BLUE, ORANGE) else "normal")
        ax.text(2.42, yi, note, ha="left", va="center", fontsize=10.5, color=MUTED)
    ax.axvspan(1.8, 2.3, color=TINT, zorder=0)
    ax.set(xlim=(1.42, 3.25), ylim=(-0.8, len(rows) - 0.2), yticks=[], xlabel="critical disorder W_c (our units)")
    ax.set_xticks([1.6, 1.8, 2.0, 2.2, 2.4])
    ax.spines["left"].set_visible(False)
    titled(fig, "Where everyone puts the critical disorder",
           "Same model and parameters, four independent methods; open circles: no error bar given",
           f"{PARAMS}. All values converted to our disorder convention. Shaded: our target band [1.8, 2.3]. "
           "The observables differ (edge conductance vs bulk invariants), so agreement is on location, not a precision test.")
    return save(fig, f"fss_4_comparison_{tag}.png", left=0.04)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="fss_estimate")
    args = ap.parse_args()
    est = json.loads((HERE / "results" / "sweep" / f"{args.tag}.json").read_text())
    est["per_size"] = {str(k): v for k, v in est["per_size"].items()}
    data = load([Path(f) for f in est["files"]], est["lambda_nu"], *est["W_range"])
    for f in (fig_switch(data, est, args.tag), fig_crossings(est, args.tag), fig_exponent(est, args.tag),
              fig_comparison(est, args.tag)):
        print("saved", f)


if __name__ == "__main__":
    main()
