"""Plots of a disorder ensemble: what the *population* of samples tells us beyond the average.

Written at the hackathon (21 Nov 2026). Input: CSVs from scripts/ensemble_sweep.py.

Run from hackathon/:
  uv run python analysis/ensemble_plots.py results/ensemble_tai_demo.csv \\
      --pinned results/ensemble_tai_demo_pinned.csv --out results/ensemble_tai_demo

Figures (PNG):
  <out>_pitch.png       resolved fractions by size, Wilson intervals and unresolved bounds;
                       Born estimates in our W convention, not measured boundaries
  <out>_overview.png   backup diagnostics:
    a  all three outcomes among all draws of the largest supercell
    b  control: uniform vs sublattice-pinned disorder (only if --pinned is given)
    c  band-gap distribution per W (largest supercell)
    d  gap vs W per sample, coloured by outcome
    e  P(Z2 = 1) vs the sample's own sublattice imbalance (eps_A - eps_B)/2
    f  each sample's own critical disorder W* (needs seeds_mode = shared)
    g  running estimate of P(Z2 = 1) vs number of samples (how many do we need?)
    h  reliability of the Z2 call vs W
    i  gap of individual samples followed across W (same seed, scaled disorder)
  <out>_mosaic.png      one tile per sample, coloured by Z2, for several W (pitch visual)
  <out>_wannier.png     pooled centre counts (backup only; needs _flows.npz)

Optional --onset-bracket LOW HIGH prints exploratory paired size crossings to the console;
imbalance fit summaries are also printed rather than drawn over the observations.

Everything marked "to check" in titles/notes is a hypothesis for the physics side.
"""

from __future__ import annotations

import argparse
import sys
from itertools import pairwise
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.ensemble import CLEAN_BOUNDARY_FACTOR, load_rows

# Colours (validated categorical palette: first slots are colour-blind safe as a set)
L_COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#008300", "#e34948"]
TOPO, TRIVIAL, UNSURE = "#6250d6", "#eda100", "#b8b7b0"
INK, MUTED = "#0b0b0b", "#52514e"

plt.rcParams.update({
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "font.size": 9, "legend.frameon": False, "lines.linewidth": 1.6,
})


# ----------------------------------------------------------------------------- statistics


def wilson_interval(k: np.ndarray, n: np.ndarray, z: float = 1.96):
    """95% Wilson score interval for a binomial fraction k/n. Works when k = 0 or k = n."""
    k, n = np.asarray(k, float), np.asarray(n, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        p = k / n
        centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
        half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return np.clip(centre - half, 0, 1), np.clip(centre + half, 0, 1)


def p_topo_vs_W(d: dict, mask: np.ndarray, reliable_only: bool = True):
    """(W values, P(Z2=1), lower, upper, n) over the rows in `mask`."""
    m = mask & d["reliable"] if reliable_only else mask
    Ws = np.unique(d["W"][mask])
    k = np.array([d["z2"][m & (d["W"] == w)].sum() for w in Ws])
    n = np.array([np.sum(m & (d["W"] == w)) for w in Ws])
    with np.errstate(invalid="ignore", divide="ignore"):
        p = k / n
    lo, hi = wilson_interval(k, n)
    return Ws, p, lo, hi, n


def outcome_counts(d: dict, mask: np.ndarray):
    """Per W: (Ws, n_all, n_topo, n_trivial, n_unresolved). 'Unresolved' = failed a
    reliability check (gap, mesh, tracking); counted out of ALL draws, not dropped, because
    dropping them biases P exactly where gaps shrink."""
    Ws = np.unique(d["W"][mask])
    out = []
    for w in Ws:
        m = mask & (d["W"] == w)
        r = d["reliable"][m]
        out.append((m.sum(), np.sum(r & (d["z2"][m] == 1)), np.sum(r & (d["z2"][m] == 0)),
                    np.sum(~r)))
    return (Ws, *np.array(out).T)


def _trajectories(d: dict, mask: np.ndarray, field: str, fill=-1):
    """(seeds, W) table of `field`, one row per seed (= one disorder pattern when seeds are
    shared across W). Raises if a (seed, W) cell appears twice, e.g. CSVs with several
    delta values pooled together."""
    seeds, Ws = np.unique(d["seed"][mask]), np.unique(d["W"][mask])
    T = np.full((len(seeds), len(Ws)), fill, dtype=float)
    for j, w in enumerate(Ws):
        m = mask & (d["W"] == w)
        rows = np.searchsorted(seeds, d["seed"][m])
        if len(np.unique(rows)) != len(rows):
            raise ValueError("duplicate (seed, W) rows: select one parameter slice first")
        T[rows, j] = d[field][m]
    return seeds, Ws, T


def _crossings(Ws, p1, p2) -> np.ndarray:
    """W values where p1 - p2 changes sign or touches zero (linear interpolation)."""
    diff = p1 - p2
    ok = ~np.isnan(diff)
    Ws, diff = Ws[ok], diff[ok]
    i = np.nonzero((np.sign(diff[:-1]) * np.sign(diff[1:]) < 0) | (diff[:-1] == 0))[0]
    with np.errstate(invalid="ignore", divide="ignore"):
        x = Ws[i] - diff[i] * (Ws[i + 1] - Ws[i]) / (diff[i + 1] - diff[i])
    return np.where(diff[i] == 0, Ws[i], x)


def size_crossing(d: dict, L1: int, L2: int, bracket: tuple[float, float], n_boot: int = 1000,
                  seed: int = 0, min_resolved: int = 20):
    """Does P(Z2=1 | resolved) of size L1 cross that of L2 inside a W bracket chosen in
    advance (e.g. the onset)? Paired bootstrap over seed IDs: the same seeds are resampled
    for both sizes, since the ensembles reuse seeds across sizes and W.

    Returns (W_cross of the full data or nan, 2.5%, 97.5% of bootstrap crossings in the
    bracket, fraction of replicates with a crossing in the bracket). Uses W values with
    >= min_resolved resolved samples at both sizes. A finite-size estimate only.
    """
    rng = np.random.default_rng(seed)
    tabs = [[_trajectories(d, d["L"] == L, f, fill=0)[2] for f in ("z2", "reliable")]
            for L in (L1, L2)]
    s1, Ws = _trajectories(d, d["L"] == L1, "z2")[:2]
    s2 = _trajectories(d, d["L"] == L2, "z2")[0]
    if not np.array_equal(s1, s2):
        raise ValueError("paired bootstrap needs the same seeds at both sizes")
    tabs = [(Z, R.astype(bool)) for Z, R in tabs]
    enough = np.all([R.sum(0) >= min_resolved for _, R in tabs], axis=0)
    Ws = Ws[enough]
    tabs = [(Z[:, enough], R[:, enough]) for Z, R in tabs]

    def first_in_bracket(idx):
        curves = []
        for Z, R in tabs:
            z, r = Z[idx], R[idx]
            with np.errstate(invalid="ignore", divide="ignore"):
                curves.append(np.sum((z == 1) & r, 0) / np.sum(r, 0))
        x = _crossings(Ws, *curves)
        x = x[(x >= bracket[0]) & (x <= bracket[1])]
        return x[0] if len(x) else np.nan

    x0 = first_in_bracket(np.arange(len(s1)))
    xb = np.array([first_in_bracket(rng.integers(0, len(s1), len(s1))) for _ in range(n_boot)])
    found = ~np.isnan(xb)
    lo, hi = np.percentile(xb[found], [2.5, 97.5]) if found.any() else (np.nan, np.nan)
    return x0, lo, hi, found.mean()


def per_sample_critical_W(d: dict, mask: np.ndarray):
    """Each sample's own transition, from its RESOLVED calls only (shared seeds).

    Along each seed's W trajectory keep only reliable calls; the first change of Z2 between
    two consecutive reliable calls brackets the sample's transition. Returns
    (W_star = bracket midpoint or nan, bracket width or nan, z2 at the first reliable W).
    nan means no resolved flip (never flips, or censored by unresolved calls).
    """
    seeds, Ws, Z = _trajectories(d, mask, "z2")
    _, _, R = _trajectories(d, mask, "reliable", fill=0)
    W_star = np.full(len(seeds), np.nan)
    width = np.full(len(seeds), np.nan)
    z0 = np.full(len(seeds), -1)
    for i in range(len(seeds)):
        j_ok = np.nonzero(R[i] == 1)[0]
        if len(j_ok) == 0:
            continue
        z0[i] = Z[i, j_ok[0]]
        for a, b in pairwise(j_ok):
            if Z[i, a] != Z[i, b]:
                W_star[i], width[i] = 0.5 * (Ws[a] + Ws[b]), Ws[b] - Ws[a]
                break
    return W_star, width, z0


def is_separated(x: np.ndarray, y: np.ndarray) -> bool:
    """True if a threshold on x perfectly splits y = 0 from y = 1 (logistic fit diverges)."""
    x0, x1 = x[y == 0], x[y == 1]
    if len(x0) == 0 or len(x1) == 0:
        return False
    return x1.max() < x0.min() or x0.max() < x1.min()


def logistic_slope(x: np.ndarray, y: np.ndarray, iters: int = 30) -> tuple[float, float]:
    """Fit P(y=1) = 1 / (1 + exp(-(a + b x))) by Newton's method. Returns (b, std error of b)."""
    X = np.column_stack([np.ones_like(x), x])
    beta = np.zeros(2)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ beta))
        H = X.T @ (X * (p * (1 - p))[:, None]) + 1e-9 * np.eye(2)
        beta += np.linalg.solve(H, X.T @ (y - p))
    p = 1 / (1 + np.exp(-X @ beta))
    cov = np.linalg.inv(X.T @ (X * (p * (1 - p))[:, None]) + 1e-9 * np.eye(2))
    return float(beta[1]), float(np.sqrt(cov[1, 1]))


def born_lines(delta: float, soc: float, rashba: float) -> list[tuple[float, str]]:
    """Born-approximation W_c (our convention) for the TAI side, from analysis/born.py."""
    if rashba != 0 or delta <= CLEAN_BOUNDARY_FACTOR * soc:
        return []
    from analysis.born import born_wc

    return [(born_wc(delta, soc), "Born (BZ average)"),
            (born_wc(delta, soc, orth_normalisation=True), "Born (Orth normalisation)")]


# ----------------------------------------------------------------------------- panels


W_LABEL = "Disorder strength W (t)"
CLASSES = ((TOPO, "Topological"), (TRIVIAL, "Trivial"), (UNSURE, "Unresolved"))


def _key(ax, **kwargs):
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.24), borderaxespad=0,
              fontsize=9, **kwargs)


def pitch(d, out):
    sizes = np.unique(d["L"])
    refs = born_lines(d["delta"][0], d["soc"][0], d["rashba"][0])
    paper = (np.isclose(d["delta"][0], 1.65) and np.isclose(d["soc"][0], 0.3)
             and d["rashba"][0] == 0 and d["disorder"][0] == "uniform")
    fig, axes = plt.subplots(1, len(sizes), figsize=(max(11, 4 * len(sizes)), 6.6),
                             sharex=True, sharey=True, squeeze=False)
    fig.subplots_adjust(left=0.085, right=0.98, top=0.75, bottom=0.32, wspace=0.13)
    for index, (ax, n) in enumerate(zip(axes.flat, sizes)):
        c = L_COLOURS[index % len(L_COLOURS)]
        m = d["L"] == n
        Ws, p, lo, hi, _ = p_topo_vs_W(d, m)
        _, n_all, n_t, _, n_u = outcome_counts(d, m)
        ax.fill_between(Ws, n_t / n_all, (n_t + n_u) / n_all, color=c, alpha=0.25, lw=0)
        yerr = np.clip([p - lo, hi - p], 0, None)  # clip float rounding at p = 0 or 1
        ax.errorbar(Ws, p, yerr=yerr, fmt="o-", color=c, ms=4, capsize=3, lw=2)
        ax.set_title(f"{n} × {n} supercell", color=c, fontsize=15, pad=10)
        ax.set_ylim(-0.04, 1.045)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1], ["0%", "25%", "50%", "75%", "100%"])
        ax.tick_params(labelsize=12)
        ax.grid(False, axis="x")
        for wc, _ in refs:
            ax.plot(wc, 1.02, marker="|", color=INK, ms=10)
        if paper:
            ax.plot(2, 1.02, marker="v", color=MUTED, ms=5)
    axes.flat[0].set_ylabel("Topological samples (%)", fontsize=13)
    headline = "How often does disorder produce a topological sample?"
    if len(p) >= 3 and Ws[0] == 0 and p[0] == 0 and np.any(p[1:-1] > 0.5) and p[-1] < 0.5:
        headline = f"At {n}×{n}, most resolved samples are topological only at intermediate disorder"
    fig.text(0.04, 0.93, headline, fontsize=19, weight="bold")
    fig.text(0.04, 0.87, "Disordered Kane–Mele model   /   Each panel tests a different simulated material size.", fontsize=13)
    fig.text(0.5, 0.23, "Disorder width W (hopping units t)   ·   site energies uniform from −W/2 to +W/2", ha="center", fontsize=13)
    fig.text(0.04, 0.15, "Dots: fraction among resolved samples (bars: 95% Wilson intervals).", fontsize=12)
    fig.text(0.04, 0.10, "Bands: fraction among all samples if every unresolved draw were trivial (bottom) or topological (top).", fontsize=12)
    if refs:
        note = f"Top ticks: Born onset at {refs[1][0]:.2f} (Orth norm.) and {refs[0][0]:.2f} (BZ average)."
        if paper:
            note += "   ▼  Paper ≈ 2 (to check); our W = 2 × Orth’s W."
        fig.text(0.04, 0.04, note, fontsize=11, color=MUTED)
    fig.savefig(out, dpi=150, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


def panel_control(ax, d, dp):
    if dp is None:
        ax.text(0.5, 0.5, "no --pinned data", ha="center", va="center", color=MUTED,
                transform=ax.transAxes)
        ax.set_title("b  Control: sublattice-pinned disorder")
        return
    for c, n in zip(L_COLOURS, np.unique(d["L"])):
        for data, ls, lab in ((d, "-", "uniform"), (dp, "--", "pinned")):
            m = data["L"] == n
            if not m.any():
                continue
            Ws, p, *_ = p_topo_vs_W(data, m, reliable_only=False)
            rel = np.array([data["reliable"][m & (data["W"] == w)].mean() for w in Ws])
            ax.plot(Ws, p, ls, color=c, label=f"{n}×{n} {lab}")
            ax.scatter(Ws[rel >= 0.5], p[rel >= 0.5], color=c, s=10, zorder=3)
            ax.scatter(Ws[rel < 0.5], p[rel < 0.5], facecolors="white", edgecolors=c, s=14,
                       zorder=3)
    ax.axhline(0.5, color=MUTED, lw=0.6)
    ax.set(xlabel=W_LABEL, ylabel="Raw fraction called topological", ylim=(-0.03, 1.03),
           title="b   What changes if sublattice means are pinned?")
    _key(ax, ncol=2, title="Includes failed calls; hollow = fewer than half pass")


def panel_gap_violin(ax, d, n):
    m = d["L"] == n
    Ws = np.unique(d["W"][m])
    data = [d["band_gap"][m & (d["W"] == w)] for w in Ws]
    width = 0.8 * np.min(np.diff(Ws)) if len(Ws) > 1 else 0.5
    parts = ax.violinplot(data, positions=Ws, widths=width, showextrema=False)
    for b in parts["bodies"]:
        b.set_facecolor(MUTED)
        b.set_alpha(0.35)
    ax.plot(Ws, [np.median(x) for x in data], "o-", color=INK, ms=3, label="median")
    ax.plot(Ws, [np.mean(x) for x in data], "s--", color=MUTED, ms=3, lw=1, label="mean")
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set(xlabel=W_LABEL, ylabel="Indirect band gap (t)",
           title=f"c   How widely do band gaps vary?  ·  {n}×{n}")
    _key(ax, ncol=2)


def panel_gap_scatter(ax, d, n, rng):
    m = d["L"] == n
    jitter = rng.uniform(-0.12, 0.12, m.sum())
    W, g, z, r = d["W"][m] + jitter, d["band_gap"][m], d["z2"][m], d["reliable"][m]
    ax.scatter(W[~r], g[~r], s=10, facecolors="none", edgecolors=UNSURE, lw=0.7,
               label="Unresolved")
    for val, col, lab in ((1, TOPO, "Topological"), (0, TRIVIAL, "Trivial")):
        k = r & (z == val)
        ax.scatter(W[k], g[k], s=8, color=col, lw=0, alpha=0.75, label=lab)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set(xlabel=W_LABEL + " · jittered", ylabel="Indirect band gap (t)",
           title=f"d   Each sample’s gap and outcome  ·  {n}×{n}")
    _key(ax, ncol=3, markerscale=1.5)


def panel_imbalance(ax, d, n):
    """P(Z2=1) in quantile bins of (eps_A - eps_B)/2, for W values inside the transition."""
    m = (d["L"] == n) & d["reliable"]
    Ws, p, *_ = p_topo_vs_W(d, d["L"] == n)
    inside = Ws[(p > 0.1) & (p < 0.9)]
    cmap = plt.get_cmap("Blues")
    notes = []
    for i, w in enumerate(inside):
        k = m & (d["W"] == w)
        x, y = d["eps_A_minus_B"][k] / 2, d["z2"][k]
        edges = np.quantile(x, np.linspace(0, 1, 6))
        idx = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, 4)
        centres = [x[idx == b].mean() for b in np.unique(idx)]
        frac = [y[idx == b].mean() for b in np.unique(idx)]
        col = cmap(0.4 + 0.6 * i / max(1, len(inside) - 1))
        ax.plot(centres, frac, "o-", color=col, ms=3, label=f"W = {w:g}")
        if is_separated(x, y):
            notes.append(f"W={w:g}: fully separated by imbalance (n={k.sum()})")
        else:
            b, se = logistic_slope(x, y.astype(float))
            notes.append(f"W={w:g}: slope {b:+.1f}±{se:.1f} (n={k.sum()})")
    ax.set(xlabel="Sample imbalance (mean εA − mean εB)/2 (t)", ylabel="Fraction topological",
           ylim=(-0.03, 1.03), title=f"e   Topology vs sublattice imbalance  ·  {n}×{n}")
    if notes:
        _key(ax, ncol=4, title="Resolved only · points are imbalance quintiles")
        print("Imbalance fits (logistic slope ± Wald SE; to check):", *notes, sep="\n  ")
    else:
        ax.text(0.5, 0.5, "no W with 0.1 < P < 0.9", ha="center", color=MUTED,
                transform=ax.transAxes)


def panel_critical_W(ax, d):
    ax.set_title("f   When does a sample first change class?")
    if d["seeds_mode"][0] != "shared":
        ax.text(0.5, 0.5, "needs seeds_mode = shared", ha="center", color=MUTED,
                transform=ax.transAxes)
        return
    bins = np.unique(d["W"])  # W* is a midpoint between grid points, so grid points are edges
    for c, n in zip(L_COLOURS, np.unique(d["L"])):
        W_star, width, _ = per_sample_critical_W(d, d["L"] == n)
        ok = ~np.isnan(W_star)
        if ok.any():
            ax.hist(W_star[ok], bins=bins, density=True, histtype="step", color=c, lw=1.6,
                    label=f"{n}×{n}: {ok.sum()} flips · median bracket {np.median(width[ok]):.1f} t")
    ax.set(xlabel="Midpoint between differing resolved calls, W* (t)", ylabel="Density (1/t)")
    if ax.get_legend_handles_labels()[0]:
        _key(ax, title="Only samples with an observed flip")


def panel_convergence(ax, d, n):
    m = (d["L"] == n) & d["reliable"]
    Ws, p, *_ = p_topo_vs_W(d, d["L"] == n)
    if not np.isfinite(p).any():
        ax.set_title(f"g   No resolved samples  ·  {n}×{n}")
        return
    w = Ws[np.nanargmin(np.abs(p - 0.5))]
    z = d["z2"][m & (d["W"] == w)][np.argsort(d["sample"][m & (d["W"] == w)])]
    N = np.arange(1, len(z) + 1)
    k = np.cumsum(z)
    lo, hi = wilson_interval(k, N)
    ax.fill_between(N, lo, hi, color=MUTED, alpha=0.2, lw=0, label="95% Wilson interval")
    ax.plot(N, k / N, color=INK)
    for target in (10, 100, 1000):
        if target <= N[-1]:
            ax.axvline(target, color=MUTED, lw=0.6, ls=":")
    ax.set(xscale="log", xlabel="Resolved samples counted (log scale)", ylabel="Running fraction topological",
           ylim=(-0.03, 1.03), title=f"g   Estimate as draws accumulate  ·  {n}×{n}, W = {w:g}")
    _key(ax)


def panel_trajectories(ax, d, n, k: int = 8):
    """Gap vs W for k individual samples (same seed = same disorder pattern, scaled by W),
    chosen across the spread of their own W*. Markers: Z2 at each W."""
    m = d["L"] == n
    if d["seeds_mode"][0] != "shared":
        ax.set_title("i   Following band gaps requires shared seeds")
        return
    _, Ws, G = _trajectories(d, m, "band_gap", fill=np.nan)
    _, _, Z = _trajectories(d, m, "z2")
    _, _, R = _trajectories(d, m, "reliable", fill=0)
    W_star, *_ = per_sample_critical_W(d, m)
    order = np.argsort(np.nan_to_num(W_star, nan=np.inf))
    pick = order[np.unique(np.linspace(0, len(order) - 1, k).round().astype(int))]
    for i in pick:
        ax.plot(Ws, G[i], color=MUTED, lw=0.8, alpha=0.7, zorder=1)
        col = np.where(R[i] == 0, UNSURE, np.where(Z[i] == 1, TOPO, TRIVIAL))
        ax.scatter(Ws, G[i], c=col, s=14, zorder=2, edgecolors="white", linewidths=0.4)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set(xlabel=W_LABEL, ylabel="Indirect band gap (t)",
           title=f"i   Following {len(pick)} disorder patterns  ·  {n}×{n}")
    _key(ax, ncol=3, handles=[matplotlib.patches.Patch(color=c, label=s) for c, s in CLASSES])


def panel_reliability(ax, d):
    for c, n in zip(L_COLOURS, np.unique(d["L"])):
        m = d["L"] == n
        Ws = np.unique(d["W"][m])
        ax.plot(Ws, [d["reliable"][m & (d["W"] == w)].mean() for w in Ws], "o-", color=c,
                ms=3, label=f"{n}×{n} reliable")
        ax.plot(Ws, [np.mean(d["band_gap"][m & (d["W"] == w)] > 0) for w in Ws], ":",
                color=c, lw=1.2, label=f"{n}×{n} gapped")
    ax.set(xlabel=W_LABEL, ylabel="Fraction of all draws", ylim=(-0.03, 1.03),
           title="h   How many samples pass the reliability checks?")
    _key(ax, ncol=2, title="Gapped = positive indirect gap on the mesh")


# ----------------------------------------------------------------------------- figures


def overview(d, dp, out: Path, onset_bracket=None):
    rng = np.random.default_rng(0)
    n_big = int(np.max(d["L"]))
    fig, ax = plt.subplots(3, 3, figsize=(16, 13.5), constrained_layout=True)
    ax = ax.ravel()
    Ws, total, topo, trivial, unsure = outcome_counts(d, d["L"] == n_big)
    ax[0].stackplot(Ws, topo / total, trivial / total, unsure / total,
                    colors=[c for c, _ in CLASSES], labels=[s for _, s in CLASSES])
    ax[0].set(xlabel=W_LABEL, ylabel="Fraction of all draws", ylim=(0, 1),
              title=f"a   What happened to every draw?  ·  {n_big}×{n_big}")
    _key(ax[0], ncol=3)
    if onset_bracket is not None:
        for small, large in pairwise(np.unique(d["L"])):
            x, lo, hi, frac = size_crossing(d, small, large, onset_bracket)
            print(f"Size crossing {small}/{large} in {onset_bracket} (to check): W={x:.2f}; "
                  f"bootstrap [{lo:.2f}, {hi:.2f}] among {frac:.0%} with a crossing")
    panel_control(ax[1], d, dp)
    panel_gap_violin(ax[2], d, n_big)
    panel_gap_scatter(ax[3], d, n_big, rng)
    panel_imbalance(ax[4], d, n_big)
    panel_critical_W(ax[5], d)
    panel_convergence(ax[6], d, n_big)
    panel_reliability(ax[7], d)
    panel_trajectories(ax[8], d, n_big)
    fig.suptitle(
        f"Ensemble diagnostics   /   Kane–Mele model\nδ={d['delta'][0]:g}, "
        f"soc={d['soc'][0]:g}, rashba={d['rashba'][0]:g}, "
        f"mesh nk = {', '.join(map(str, np.unique(d['nk'])))}, ε ~ U[−W/2, W/2]; energies in hopping units t",
        fontweight="bold", x=0.01, ha="left")
    fig.savefig(out, dpi=140)
    plt.close(fig)


def mosaic(d, out: Path, n_panels: int = 3):
    """One tile per sample; the pitch picture of 'a population of imperfect materials'."""
    n = int(np.max(d["L"]))
    m = d["L"] == n
    Ws = np.unique(d["W"][m])
    show = np.unique([Ws[np.argmin(abs(Ws - target))] for target in np.linspace(Ws[0], Ws[-1], n_panels)])
    samples = np.unique(d["sample"][m])
    side = int(np.ceil(np.sqrt(len(samples))))
    rgb = {1: matplotlib.colors.to_rgb(TOPO), 0: matplotlib.colors.to_rgb(TRIVIAL)}
    fig, ax = plt.subplots(1, len(show), figsize=(max(10, 3.6 * len(show)), 5.8))
    fig.subplots_adjust(left=0.04, right=0.96, top=0.76, bottom=0.23, wspace=0.18)
    for a, w in zip(np.atleast_1d(ax), show):
        k = m & (d["W"] == w)
        img = np.ones((side * side, 4))  # RGBA; unused tiles stay transparent
        img[:, 3] = 0
        for i, z, r in zip(np.searchsorted(samples, d["sample"][k]), d["z2"][k], d["reliable"][k]):
            img[i, :3] = rgb[int(z)] if r else matplotlib.colors.to_rgb(UNSURE)
            img[i, 3] = 1
        a.imshow(img.reshape(side, side, 4), interpolation="nearest")
        for sp in a.spines.values():
            sp.set_visible(False)
        a.set_xticks([])
        a.set_yticks([])
        a.set_xticks(np.arange(-0.5, side, 1), minor=True)
        a.set_yticks(np.arange(-0.5, side, 1), minor=True)
        a.grid(which="minor", color="white", lw=0.8)
        a.tick_params(which="both", length=0, labelbottom=False, labelleft=False)
        a.grid(which="major", visible=False)
        r = d["reliable"][k]
        n_t, n_u = np.sum(r & (d["z2"][k] == 1)), np.sum(~r)
        a.set_title(f"W = {w:g} t", fontsize=16)
        a.set_xlabel(f"{n_t} of {k.sum()} topological ({n_t / k.sum():.0%})\n"
                     f"{n_u / k.sum():.0%} unresolved", fontsize=13, labelpad=8)
    same = "Tile positions track the same pattern across W." if d["seeds_mode"][0] == "shared" else "Each W uses independent samples."
    fig.text(0.04, 0.93, "One tile, one simulated material", weight="bold", fontsize=22)
    fig.text(0.04, 0.86, f"{n} × {n} supercells   /   {same}   W is the full disorder width.", fontsize=13)
    handles = [matplotlib.patches.Patch(color=c, label=s) for c, s in CLASSES]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=14)
    fig.savefig(out, dpi=140)
    plt.close(fig)


def wannier_density(flows_path: Path, d, out: Path, n_panels: int = 4):
    """2D histogram of Wannier centres vs k_x over all stored samples, for several W."""
    flows = np.load(flows_path)
    n = int(np.max(d["L"]))
    keys = sorted((k for k in flows.files if k.startswith(f"L{n}_")),
                  key=lambda k: float(k.split("_W")[1]))
    keys = [keys[i] for i in np.unique(np.linspace(0, len(keys) - 1, n_panels).round().astype(int))]
    fig, ax = plt.subplots(1, len(keys), figsize=(3.6 * len(keys), 3.4), constrained_layout=True,
                           sharey=True)
    images = []
    for a, key in zip(np.atleast_1d(ax), keys):
        f = flows[key]  # (samples, nk/2+1, n_occ)
        S, nkx, n_occ = f.shape
        kx = np.repeat(np.linspace(0, 0.5, nkx)[None, :, None], S, 0).repeat(n_occ, 2)
        H, xe, _ = np.histogram2d(kx.ravel(), f.ravel(), bins=[nkx, 60],
                                   range=[[-0.25 / (nkx - 1), 0.5 + 0.25 / (nkx - 1)], [0, 1]])
        images.append(a.imshow(H.T, origin="lower", aspect="auto", cmap="Blues",
                               extent=[xe[0], xe[-1], 0, 1]))
        a.grid(False)
        a.set(xlabel="Momentum kₓ (reduced units)", title=f"W = {key.split('_W')[1]} t  ·  {S} samples")
    norm = matplotlib.colors.PowerNorm(0.5, vmin=0, vmax=max(im.get_array().max() for im in images))
    for im in images:
        im.set_norm(norm)
    fig.colorbar(images[0], ax=np.atleast_1d(ax).tolist(), label="Centre counts per bin")
    np.atleast_1d(ax)[0].set_ylabel("Wannier centre along y (cell units)")
    fig.suptitle(f"Where Wannier centres accumulate  ·  {n}×{n}\n"
                 "Pooling samples hides partner switching; this view cannot establish topology.",
                 x=0.01, ha="left", fontsize=12)
    fig.savefig(out, dpi=140)
    plt.close(fig)


def one_slice(d: dict, delta: float | None) -> dict:
    """Restrict to one parameter set; the panels pool by supercell size only."""
    if delta is not None:
        keep = np.isclose(d["delta"], delta)
        d = {k: v[keep] for k, v in d.items()}
    for key in ("delta", "soc", "rashba", "disorder", "seeds_mode", "precision"):
        if len(np.unique(d[key])) > 1:
            raise ValueError(f"CSV mixes several {key} values; pick one (e.g. --delta)")
    return d


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("csv", type=Path, help="ensemble CSV (uniform disorder)")
    ap.add_argument("--pinned", type=Path, default=None, help="matching CSV, --disorder pinned")
    ap.add_argument("--out", type=Path, default=None, help="output prefix (default: csv stem)")
    ap.add_argument("--delta", type=float, default=None, help="pick one delta if the CSV has several")
    ap.add_argument("--onset-bracket", type=float, nargs=2, default=None,
                    help="W range in which to test for a size crossing (choose in advance)")
    args = ap.parse_args()
    out = args.out or args.csv.with_suffix("")
    d = one_slice(load_rows(args.csv), args.delta)
    dp = one_slice(load_rows(args.pinned), args.delta) if args.pinned else None

    pitch(d, out.with_name(out.name + "_pitch.png"))
    overview(d, dp, out.with_name(out.name + "_overview.png"), args.onset_bracket)
    mosaic(d, out.with_name(out.name + "_mosaic.png"))
    flows = args.csv.with_name(args.csv.stem + "_flows.npz")
    if flows.exists():
        wannier_density(flows, d, out.with_name(out.name + "_wannier.png"))
    print(f"Saved {out}_pitch.png, _overview.png, _mosaic.png" + (", _wannier.png" if flows.exists() else ""))


if __name__ == "__main__":
    main()
