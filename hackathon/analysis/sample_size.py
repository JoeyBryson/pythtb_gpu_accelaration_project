"""Why many samples matter: the same analysis with few vs many disorder realisations.

Written at the hackathon (21 Nov 2026). Input: one ensemble CSV from scripts/ensemble_sweep.py
with many samples per W (e.g. results/ensemble_n1000_L3.csv, 1000 shared-seed samples), and
optionally a multi-size CSV (e.g. results/ensemble_tai165_v3.csv) for the backup figure.

Every "experiment of N samples" is a subset of the big ensemble (one seed = one disorder
pattern, kept together across all W):
  - panels a-c: DISJOINT groups of seeds, i.e. truly independent repeats of the experiment;
  - panel d and the backup figure: bootstrap resamples of whole seed trajectories.
    These estimate repeated sampling from the empirical pool; patterns that are absent from
    the pool can't show up, so the spread can come out slightly too small.
  - panel e: exact (hypergeometric) probabilities for subsets drawn without replacement.
Fractions are over RESOLVED draws (passing the reliability checks); unresolved draws are
dropped inside each subset. Panel c also shows the bounds if unresolved were all trivial /
all topological.

Run from hackathon/:
  uv run python analysis/sample_size.py results/ensemble_n1000_L3.csv \\
      --sizes-csv results/ensemble_tai165_v3.csv
  # optional: --gpu-speedup X   adds GPU-time labels from a MEASURED end-to-end speed-up

Output: <csv stem>_sample_size.png (a-f) and <csv stem>_sample_size_backup.png (how many
samples it takes to see the imbalance effect; telling two supercell sizes apart).
"""

from __future__ import annotations

import argparse
import sys
from math import lgamma
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.ensemble import load_rows
from analysis.ensemble_plots import (
    INK,
    L_COLOURS,
    MUTED,
    TOPO,
    TRIVIAL,
    _trajectories,
    one_slice,
    wilson_interval,
)

ACCENT = "#2a78d6"
plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
                     "legend.fontsize": 10, "xtick.labelsize": 10, "ytick.labelsize": 10})
Y_RESOLVED = "Topological fraction\namong resolved draws"


# ----------------------------------------------------------------------------- statistics


def tables(d: dict, L: int):
    """(Ws, Z, R, X): (seeds, W) tables of Z2, reliable flag, and imbalance (eps_A-eps_B)/2."""
    m = d["L"] == L
    _, Ws, Z = _trajectories(d, m, "z2")
    _, _, R = _trajectories(d, m, "reliable", fill=0)
    _, _, X = _trajectories(d, m, "eps_A_minus_B", fill=np.nan)
    return Ws, Z.astype(int), R.astype(bool), X / 2


def p_curve(Z, R, idx):
    """P(Z2=1 | resolved) per W over the seeds in idx, plus (k, n)."""
    z, r = Z[idx], R[idx]
    k, n = np.sum((z == 1) & r, 0), np.sum(r, 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return k / n, k, n


def onset(Ws, p) -> float:
    """First W where P crosses 1/2 from below, by linear interpolation between ADJACENT grid
    points. Points exactly at 1/2 (common at small N) are interpolated across rather than
    counted as a crossing. nan if there is no crossing, or if the crossing would span a grid
    point with no resolved draws."""
    below = None  # index of the last grid point with p < 1/2
    for i, v in enumerate(p):
        if np.isnan(v):
            below = None  # don't interpolate across a missing point
        elif v < 0.5:
            below = i
        elif v > 0.5 and below is not None:
            return float(Ws[below] + (0.5 - p[below]) * (Ws[i] - Ws[below]) / (v - p[below]))
    return np.nan


def rank_z(x_topo: np.ndarray, x_triv: np.ndarray) -> float:
    """Mann-Whitney U as a z-score (normal approximation, average ranks for ties).
    Negative = topological samples tend to have SMALLER x than trivial ones."""
    x = np.concatenate([x_topo, x_triv])
    ranks = _average_ranks(x)
    n1, n2 = len(x_topo), len(x_triv)
    U = ranks[:n1].sum() - n1 * (n1 + 1) / 2
    return float((U - n1 * n2 / 2) / np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12))


def _average_ranks(x: np.ndarray) -> np.ndarray:
    """Ranks 1..n, tied values get the average of their ranks."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x))
    ranks[order] = np.arange(1, len(x) + 1)
    _, group, counts = np.unique(x, return_inverse=True, return_counts=True)
    return (np.bincount(group, weights=ranks) / counts)[group]


def rank_permutation_p(x_topo, x_triv, rng, n_perm: int = 999) -> float:
    """One-sided permutation p-value that topological samples have smaller x (rank sum,
    ties averaged). Exact enough at small counts, unlike the normal approximation:
    3 vs 3 perfectly separated gives p ~ 1/20."""
    x = np.concatenate([x_topo, x_triv])
    ranks = _average_ranks(x)
    n1 = len(x_topo)
    obs = ranks[:n1].sum()
    perm = np.argsort(rng.random((n_perm, len(x))), axis=1)[:, :n1]
    null = ranks[perm].sum(axis=1)
    return float((1 + np.sum(null <= obs + 1e-9)) / (n_perm + 1))


def imbalance_detected(x, z, rng=None, alpha: float = 0.025) -> bool:
    """Is 'larger (eps_A - eps_B)/2 -> less often topological' visible in this subset?
    Needs >= 3 of each class and a one-sided rank permutation test at level alpha."""
    if min(np.sum(z == 1), np.sum(z == 0)) < 3:
        return False
    rng = rng or np.random.default_rng(0)
    return rank_permutation_p(x[z == 1], x[z == 0], rng) < alpha


def prob_miss_all(pool: int, k: int, N: int) -> float:
    """Chance that N draws without replacement from `pool` items contain none of k special
    ones: C(pool - k, N) / C(pool, N)."""
    if N > pool - k:
        return 0.0
    lc = lambda a, b: lgamma(a + 1) - lgamma(b + 1) - lgamma(a - b + 1)
    return float(np.exp(lc(pool - k, N) - lc(pool, N)))


# ----------------------------------------------------------------------------- helpers


def seconds_per_sample_curve(d: dict, L: int) -> float:
    """Measured wall-clock seconds per sample for one full W sweep (sum over W points)."""
    m = d["L"] == L
    return float(sum(d["batch_seconds"][m & (d["W"] == w)][0] / np.sum(m & (d["W"] == w))
                     for w in np.unique(d["W"][m])))


def _fmt_time(s: float) -> str:
    return f"{s:.0f} s" if s < 90 else (f"{s / 60:.0f} min" if s < 5400 else f"{s / 3600:.1f} h")


def _note(ax, text, loc=(0.03, 0.95)):
    ax.text(*loc, text, transform=ax.transAxes, va="top", fontsize=10.5, color=INK,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 2})


# ----------------------------------------------------------------------------- main figure


def figure(d: dict, L: int, out: Path, gpu_speedup: float | None = None, seed: int = 0,
           n_boot: int = 300):
    rng = np.random.default_rng(seed)
    Ws, Z, R, _ = tables(d, L)
    S = len(Z)
    perm = rng.permutation(S)
    p_all, k_all, n_all = p_curve(Z, R, np.arange(S))
    t_sample = seconds_per_sample_curve(d, L)
    t_total = t_sample * S

    fig = plt.figure(figsize=(18, 12.5), layout="constrained")
    fig.get_layout_engine().set(rect=(0, 0, 1, 0.93))
    gs = fig.add_gridspec(2, 12, height_ratios=[1, 1.05])
    fig.text(0.01, 0.985, "More samples turn a noisy guess into a measurement",
             fontsize=19, fontweight="bold", ha="left", va="top")
    fig.text(0.01, 0.952,
             f"{L}×{L} Kane–Mele supercells (δ = {d['delta'][0]:g}, λ_SO = {d['soc'][0]:g}, "
             f"λ_R = {d['rashba'][0]:g}), disorder ε ~ U[−W/2, W/2]. {S} disorder patterns per W "
             f"in total ({_fmt_time(t_total)} on a laptop CPU); every 'experiment' below is a "
             "subset of them.", fontsize=11.5, color=MUTED, ha="left", va="top")

    # --- a-c: the phase curve from independent experiments of size N -------------------------
    sizes_n = [n for n in (10, 100) if S // n >= 2] + [S]
    axes = []
    for col, N in enumerate(sizes_n):
        ax = fig.add_subplot(gs[0, 4 * col : 4 * col + 4], sharey=axes[0] if axes else None)
        axes.append(ax)
        n_exp = S // N
        shown = min(n_exp, 20)
        if N < S:
            ons = []
            for e in range(n_exp):
                p, *_ = p_curve(Z, R, perm[e * N : (e + 1) * N])
                ons.append(onset(Ws, p))
                if e < shown:
                    ax.plot(Ws, p, color=ACCENT, lw=1.0, alpha=0.35)
            ons = np.array(ons)
            q10, q90 = np.nanpercentile(ons, [10, 90])
            title = f"{chr(97 + col)}  {N} samples per W: {shown} of {n_exp} experiments shown"
            note = (f"onset W across all {n_exp} experiments:\n"
                    f"middle 80%: {q10:.1f}–{q90:.1f}   "
                    f"extremes: {np.nanmin(ons):.1f}–{np.nanmax(ons):.1f}")
            if np.isnan(ons).any():
                note += f"\nno clear onset in {np.isnan(ons).sum()}"
        else:
            # bounds over ALL draws if every unresolved draw were trivial / topological
            unres = np.sum(~R, 0)
            lo_b, hi_b = k_all / S, (k_all + unres) / S
            ax.fill_between(Ws, lo_b, hi_b, color=MUTED, alpha=0.15, lw=0,
                            label="if unresolved were all\ntrivial … all topological")
            lo, hi = wilson_interval(k_all, n_all)
            ax.fill_between(Ws, lo, hi, color=ACCENT, alpha=0.35, lw=0, label="95% interval")
            ax.plot(Ws, p_all, color=ACCENT, lw=2.4, marker="o", ms=4)
            ax.legend(loc="upper right", fontsize=9.5)
            title = f"{chr(97 + col)}  All {S} samples per W"
            note = (f"onset W = {onset(Ws, p_all):.2f}\n"
                    f"(unresolved bounds: {onset(Ws, hi_b):.2f}–{onset(Ws, lo_b):.2f})")
        ax.axhline(0.5, color=MUTED, lw=0.8, ls=":")
        ax.set(xlabel="Disorder strength W (t)", ylim=(-0.03, 1.03), title=title)
        ax.set_ylabel(Y_RESOLVED if col == 0 else "")
        _note(ax, note)

    # --- d: precision of the onset vs N -----------------------------------------------------
    Ns = np.array([n for n in (5, 10, 20, 50, 100, 200, 500, 1000, 2000) if n <= S])
    sd, missing = [], []
    for N in Ns:
        ons = np.array([onset(Ws, p_curve(Z, R, rng.integers(0, S, N))[0])
                        for _ in range(n_boot)])
        sd.append(np.nanstd(ons))
        missing.append(np.mean(np.isnan(ons)))
    sd, missing = np.array(sd), np.array(missing)
    ax = fig.add_subplot(gs[1, 0:4])
    ax.plot(Ns, sd, "o-", color=ACCENT, lw=2.2, ms=6, label="spread between experiments")
    ref = sd[np.argmin(np.abs(Ns - 100))] * np.sqrt(100 / Ns)
    ax.plot(Ns, ref, ls="--", color=MUTED, lw=1.2, label="∝ 1/√N (expected)")
    target = 0.1
    ax.axhline(target, color=TOPO, lw=1.2, ls=":")
    hit = Ns[sd < target]
    ax.text(Ns[0], target * 1.08, f"target: onset to ±{target:g} t"
            + (f"  → first reached at N = {hit[0]} (of the N tested)" if len(hit) else ""),
            fontsize=10, color=TOPO)
    for n, m in zip(Ns, missing):
        if m >= 0.01:
            ax.annotate(f"{m:.0%} no\nonset", (n, sd[list(Ns).index(n)]), fontsize=8,
                        color=MUTED, xytext=(4, 4), textcoords="offset points")
    ax.set(xscale="log", yscale="log", title="d  How precisely do we locate the onset?",
           ylabel="SD of onset W between experiments (t)")
    yt = [v for v in (0.02, 0.05, 0.1, 0.2, 0.5, 1, 2) if sd.min() / 2 <= v <= sd.max() * 2]
    ax.set_yticks(yt, [f"{v:g}" for v in yt])
    labels = [f"{n}\n{_fmt_time(n * t_sample)}" for n in Ns]
    xl = (f"samples per W (N)  ·  CPU time per curve, scaled from the measured "
          f"{S}-sample sweep")
    if gpu_speedup:
        labels = [f"{lab}\n{_fmt_time(n * t_sample / gpu_speedup)}" for lab, n in zip(labels, Ns)]
        xl += f"  ·  GPU (×{gpu_speedup:g}, measured)"
    ax.set_xticks(Ns, labels, fontsize=9)
    ax.minorticks_off()
    ax.set_xlabel(xl, fontsize=9.5)
    ax.legend(loc="upper right")

    # --- e: rare events -----------------------------------------------------------------------
    ax = fig.add_subplot(gs[1, 4:8])
    j_peak = int(np.nanargmax(p_all))
    k_topo_all = k_all  # resolved topological draws, counted out of ALL S draws
    rare = [j for j in range(j_peak + 1, len(Ws)) if 0 < k_topo_all[j] <= 0.1 * S]
    Nr = np.unique(np.round(np.logspace(0, np.log10(S), 40)).astype(int))
    for j, colr in zip(rare[-2:], (TOPO, ACCENT)):
        k = int(k_topo_all[j])
        miss = [prob_miss_all(S, k, int(n)) for n in Nr]
        ax.plot(Nr, miss, color=colr, lw=2.2,
                label=f"W = {Ws[j]:g}: {k} of {S} samples topological ({k / S:.1%})")
        for n in (10, 100):
            if n <= S:
                pm = prob_miss_all(S, k, n)
                ax.plot([n], [pm], "o", color=colr, ms=6)
                ax.annotate(f"{pm:.0%}", (n, pm), xytext=(6, 4), textcoords="offset points",
                            fontsize=10, color=colr)
    if not rare:
        _note(ax, "no W in this sweep where 0–10% of samples\nare topological past the peak",
              loc=(0.03, 0.5))
    ax.set(xscale="log", ylim=(-0.03, 1.03), xlabel="samples per W (N)",
           ylabel="chance the experiment sees\nNO topological sample at all",
           title="e  Strong disorder: rare topological samples get missed")
    if rare:
        ax.legend(loc="upper right", fontsize=9.5)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))

    # --- f: one distribution, few vs many samples ------------------------------------------
    j_on = onset_index(Ws, p_all)
    _, _, _, X = tables(d, L)
    imbalance_histograms(fig, gs[1, 8:12], Ws, Z, R, X, j_on, perm,
                         f"f  Which samples turn topological? (W = {Ws[j_on]:g}, near onset)")

    fig.text(0.01, -0.005,
             "Fractions use resolved draws only (unresolved = failed the numerical reliability "
             "checks). a–c: disjoint groups of disorder patterns. d: 300 bootstrap experiments per "
             "N, resampling whole patterns across W. e: exact probability for subsets drawn "
             "without replacement. f: resolved samples of one random 20-sample subset vs all; "
             "why the imbalance matters is to check with the physics side. CPU times are scaled "
             "from one measured sweep.",
             fontsize=9.5, color=MUTED, ha="left", va="top", wrap=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


def panel_size_difference(ax, d: dict, rng, n_boot: int):
    """95% bootstrap interval of P(L_big) - P(L_small) at the W where the full data differ
    most, vs N (paired: the same seed IDs at both sizes)."""
    Ls = np.unique(d["L"])
    Ls = (int(Ls[0]), int(Ls[-1]))
    tabs = {L: tables(d, L) for L in Ls}
    Ws = tabs[Ls[0]][0]
    S = len(tabs[Ls[0]][1])
    diff_full = p_curve(*tabs[Ls[1]][1:3], np.arange(S))[0] - \
        p_curve(*tabs[Ls[0]][1:3], np.arange(S))[0]
    j = int(np.nanargmax(np.abs(diff_full)))
    Ns = np.array([n for n in (10, 20, 50, 100, 200, 500, 1000) if n <= S])
    lo, hi = [], []
    for N in Ns:
        ds = []
        for _ in range(n_boot):
            idx = rng.integers(0, S, N)
            ps = [p_curve(tabs[L][1][:, [j]], tabs[L][2][:, [j]], idx)[0][0] for L in Ls]
            ds.append(ps[1] - ps[0])
        lo_, hi_ = np.nanpercentile(ds, [2.5, 97.5])
        lo.append(lo_)
        hi.append(hi_)
    lo, hi = np.array(lo), np.array(hi)
    ax.fill_between(Ns, lo, hi, color=L_COLOURS[2], alpha=0.3, lw=0, label="95% range")
    ax.axhline(diff_full[j], color=L_COLOURS[2], lw=2.2,
               label=f"all {S} samples: {diff_full[j]:+.0%}")
    ax.axhline(0, color=MUTED, lw=1)
    sep = (lo > 0) if diff_full[j] > 0 else (hi < 0)
    # first N from which the interval excludes 0 at every larger N tested
    clear = [n for i, n in enumerate(Ns) if sep[i:].all()]
    _note(ax, f"W = {Ws[j]:g}: is the {Ls[1]}×{Ls[1]} fraction\nreally different from "
              f"{Ls[0]}×{Ls[0]}?" + (f"\nclearly yes from N = {clear[0]}" if len(clear) else ""))
    ax.set(xscale="log", xlabel="samples per W (N), at each size",
           ylabel=f"P({Ls[1]}×{Ls[1]}) − P({Ls[0]}×{Ls[0]})",
           title="Telling two supercell sizes apart")
    ax.set_xticks(Ns, [str(n) for n in Ns])
    ax.minorticks_off()
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.legend(loc="lower right", fontsize=9.5)


def onset_index(Ws, p_all) -> int:
    """Grid index closest to P = 1/2 on the RISING side of the window (near the onset)."""
    rising = np.arange(len(Ws)) <= np.nanargmax(p_all)
    return int(np.nanargmin(np.where(rising, np.abs(p_all - 0.5), np.inf)))


def imbalance_histograms(fig, slot, Ws, Z, R, X, j: int, perm, title: str, small: int = 20):
    """Mirrored histograms (topological up, trivial down) of the sample's sublattice
    imbalance at grid index j: one subset of `small` samples above, all samples below."""
    S = len(Z)
    sub = slot.subgridspec(2, 1, hspace=0.08)
    bins = np.linspace(np.nanmin(X[:, j]), np.nanmax(X[:, j]), 26)
    for row, (idx, lab) in enumerate(((perm[:small], f"{small} samples"),
                                      (np.arange(S), f"all {S} samples"))):
        ax = fig.add_subplot(sub[row])
        r = R[idx, j]
        top = 1
        for val, colr, name, sign in ((1, TOPO, "topological", 1), (0, TRIVIAL, "trivial", -1)):
            x = X[idx, j][r & (Z[idx, j] == val)]
            h, _ = np.histogram(x, bins=bins)
            ax.bar(bins[:-1], sign * h, width=np.diff(bins), align="edge", color=colr,
                   edgecolor="white", linewidth=0.5,
                   label=f"{name} ({len(x)}) {'↑' if sign > 0 else '↓'}")
            top = max(top, h.max(initial=1))
        ax.set_ylim(-1.15 * top, 1.15 * top)
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{abs(v):g}"))
        ax.set_ylabel(f"{lab}\n(count)")
        ax.legend(loc="upper right", fontsize=9)
        if row == 0:
            ax.set_title(title)
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel("Sample's sublattice imbalance (ε̄_A − ε̄_B)/2 (t)")


# ----------------------------------------------------------------------------- backup figure


def backup_figure(d: dict, L: int, out: Path, sizes: dict | None = None, seed: int = 0,
                  n_boot: int = 200):
    """Backup: (left) how many samples an experiment needs to see the imbalance effect, near
    onset / at the peak / on the falling side; (right) how many to tell two supercell sizes
    apart (needs a multi-size CSV)."""
    rng = np.random.default_rng(seed)
    Ws, Z, R, X = tables(d, L)
    S = len(Z)
    p_all = p_curve(Z, R, np.arange(S))[0]
    j_peak = int(np.nanargmax(p_all))
    j_on = onset_index(Ws, p_all)

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(15, 6), layout="constrained")
    fig.suptitle(f"Backup: what else needs many samples ({L}×{L} unless stated)",
                 x=0.01, ha="left", fontweight="bold")
    falling = [j for j in range(j_peak + 1, len(Ws)) if 0.25 <= p_all[j] <= 0.75]
    cols = [(j_on, "near onset"), (j_peak, "peak")] + (
        [(falling[0], "falling side")] if falling else [])
    Ns = np.array([n for n in (5, 10, 20, 50, 100, 200, 500, 1000) if n <= S])
    for (j, name), colr in zip(cols, (TOPO, ACCENT, MUTED)):
        power = []
        for N in Ns:
            hits = 0
            for _ in range(n_boot):
                idx = rng.integers(0, S, N)
                r = R[idx, j]
                hits += imbalance_detected(X[idx, j][r], Z[idx, j][r], rng)
            power.append(hits / n_boot)
        ax.plot(Ns, power, "o-", color=colr, lw=2.2, ms=6,
                label=f"W = {Ws[j]:g} ({name}, {p_all[j]:.0%} topological)")
    ax.axhline(0.8, color=MUTED, lw=0.8, ls=":")
    ax.set(xscale="log", ylim=(-0.03, 1.03), xlabel="samples per W (N)",
           ylabel="fraction of experiments that see the effect",
           title="Seeing the imbalance effect (rank permutation test, one-sided 2.5%)")
    ax.set_xticks(Ns, [str(n) for n in Ns])
    ax.minorticks_off()
    ax.legend(loc="lower right", fontsize=9.5)
    if sizes is not None:
        panel_size_difference(ax2, sizes, rng, n_boot)
    else:
        ax2.set_visible(False)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("csv", type=Path)
    ap.add_argument("--L", type=int, default=None, help="supercell size (default: largest)")
    ap.add_argument("--sizes-csv", type=Path, default=None,
                    help="CSV with >= 2 supercell sizes and shared seeds (backup figure)")
    ap.add_argument("--gpu-speedup", type=float, default=None,
                    help="measured end-to-end GPU speed-up, for the time labels in panel d")
    args = ap.parse_args()
    d = one_slice(load_rows(args.csv), None)
    L = args.L or int(np.max(d["L"]))
    sizes = one_slice(load_rows(args.sizes_csv), None) if args.sizes_csv else None
    out = args.csv.with_name(args.csv.stem + "_sample_size.png")
    figure(d, L, out, args.gpu_speedup)
    backup_figure(d, L, out.with_name(out.stem + "_backup.png"), sizes)
    print(f"Saved {out} and {out.stem}_backup.png")


if __name__ == "__main__":
    main()
