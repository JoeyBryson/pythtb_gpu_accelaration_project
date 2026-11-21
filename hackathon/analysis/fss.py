"""Finite-size scaling of the TAI switch-on: best estimate of W_c and the exponent nu.

Method adapted from Joey's Ising lab report (hackathon/Ising_Model.pdf), with the fraction
topological f(W, L) in the role of the Binder cumulant U_N (dimensionless, size-independent at
the critical point), disorder W for temperature and supercell size L for N:

  1. per size: binomial logistic fit f_L(W) = 1/(1 + exp(-(W - W_mid)/width))
     (analysis/crossing.py, Oscar; an analytic form instead of splines, as the report suggests)
  2. crossings of CONSECUTIVE sizes: zero of the difference f_L1 - f_L2 of the fitted curves
  3. W_c = inverse-variance weighted mean of those crossings (errors from the bootstrap)
  4. drift check: crossings vs 1/L_mean (the report's Fig. 5); linear extrapolation to 1/L -> 0
     is reported as a systematic, not used as the estimate unless the trend is significant
  5. pseudo-critical fit: W_mid(L) = W_c + A L^(-1/nu)       (needs >= 4 sizes; 3 parameters)
  6. slope scaling: width(L) = B L^(-1/nu), i.e. log width vs log L is a line of slope -1/nu
     (2 parameters: the report's most robust kind of fit)
  7. parametric bootstrap of every count (Binomial(n, k/n)), 2,000 resamples, through steps 1-6

Unlike Monte Carlo there is no equilibration or autocorrelation: samples are independent random
materials, so the binomial errors are exact.

Input: sweep CSVs (analysis/sweep.py format), one or more per size; files for the same size are
merged (counts added per W). Default: results/sweep/fssfine_*.csv, else the R11/R13 files.

Run from hackathon/:
  uv run python analysis/fss.py                                  # default files
  uv run python analysis/fss.py results/sweep/fssfine_*.csv --W-min 1.5 --W-max 2.5
Writes results/sweep/fss_estimate.json and results/figures/fss_estimate.png.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from crossing import crossing, fit_logistic  # noqa: E402  (Oscar's fit, reused unchanged)

HERE = Path(__file__).resolve().parent.parent
SWEEP = HERE / "results" / "sweep"
FALLBACK = [SWEEP / "pilot_8x8_c_up.csv", SWEEP / "fss_12x12_c_up.csv", SWEEP / "fss_16x16_c_up.csv",
            SWEEP / "dense_16x16_c_up.csv"]


def load(paths: list[Path], lambda_nu: float, w_lo: float, w_hi: float) -> dict[int, tuple]:
    """{L: (W, k, n)} with counts merged over files of the same size."""
    acc: dict[int, dict[float, list[int]]] = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for p in paths:
        with Path(p).open() as f:
            for r in csv.DictReader(f):
                if r.get("invariant", "c_up") not in ("c_up", ""):
                    continue
                w = round(float(r["W"]), 4)
                if abs(float(r["lambda_nu"]) - lambda_nu) > 1e-9 or not (w_lo - 1e-9 <= w <= w_hi + 1e-9):
                    continue
                c = acc[int(r["supercell"])][w]
                c[0] += int(r["n_topological"])
                c[1] += int(r["n_reliable"])
    out = {}
    for L, pts in acc.items():
        Ws = sorted(pts)
        out[L] = (np.array(Ws), np.array([pts[w][0] for w in Ws], float), np.array([pts[w][1] for w in Ws], float))
    return out


def power_fit(L: np.ndarray, y: np.ndarray, sig: np.ndarray | None = None):
    """log y = log B - (1/nu) log L, weighted least squares. Returns (inv_nu, B)."""
    x, z = np.log(L), np.log(y)
    w = np.ones_like(x) if sig is None else (y / sig) ** 2  # var(log y) ~ (sig/y)^2
    A = np.stack([np.ones_like(x), x], axis=1)
    coef = np.linalg.solve(A.T @ (w[:, None] * A), A.T @ (w * z))
    return -coef[1], np.exp(coef[0])


def pseudo_critical_fit(L: np.ndarray, mid: np.ndarray, sig: np.ndarray):
    """mid = Wc + A L^(-1/nu). Returns (Wc, A, inv_nu) or None if the fit fails."""
    from scipy.optimize import curve_fit

    f = lambda L, Wc, A, inv_nu: Wc + A * L ** (-inv_nu)  # noqa: E731
    try:
        p, _ = curve_fit(f, L, mid, p0=[mid[-1], (mid[0] - mid[-1]) * L[0], 1.0], sigma=sig,
                         absolute_sigma=True, maxfev=20000, bounds=([0, -1e3, 0.05], [10, 1e3, 5]))
        return tuple(p)
    except Exception:  # noqa: BLE001  (too few sizes / no curvature: report as unavailable)
        return None


def analyse(data: dict[int, tuple], fit_pairs: list[tuple[int, int]]) -> dict:
    """Steps 1-6 on one data set (the real counts, or one bootstrap resample)."""
    Ls = sorted(data)
    fits = {L: fit_logistic(*data[L]) for L in Ls}
    res = {"mid": {L: fits[L][0] for L in Ls}, "width": {L: fits[L][1] for L in Ls},
           "cross": {f"{a}/{b}": crossing(*fits[a], *fits[b]) for a, b in fit_pairs}}
    widths = np.array([fits[L][1] for L in Ls])
    if len(Ls) >= 2 and np.all(widths > 0):
        res["inv_nu_width"], _ = power_fit(np.array(Ls, float), widths)
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("files", nargs="*", type=Path)
    ap.add_argument("--lambda-nu", type=float, default=1.65)
    ap.add_argument("--W-min", type=float, default=1.0)
    ap.add_argument("--W-max", type=float, default=3.0)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--tag", default="fss_estimate", help="output file stem")
    args = ap.parse_args()

    files = args.files or [Path(p) for p in sorted(glob.glob(str(SWEEP / "fssfine_*.csv")))] \
        or [p for p in FALLBACK if p.exists()]
    data = load(files, args.lambda_nu, args.W_min, args.W_max)
    Ls = sorted(data)
    if len(Ls) < 2:
        raise SystemExit(f"need >= 2 sizes, got {Ls} from {[str(f) for f in files]}")
    pairs = list(zip(Ls[:-1], Ls[1:]))  # consecutive sizes (the report's method)
    print(f"files: {[f.name for f in files]}")
    print(f"lambda_nu = {args.lambda_nu}, W in [{args.W_min}, {args.W_max}], sizes {Ls}, "
          f"{args.boot} bootstrap resamples")
    for L in Ls:
        W, k, n = data[L]
        print(f"  L = {L:2d}: {len(W)} W points, {int(n.sum())} reliable samples")

    # ---- real data + parametric bootstrap
    best = analyse(data, pairs)
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(args.boot):
        bd = {L: (W, rng.binomial(n.astype(int), np.clip(k / np.maximum(n, 1), 0, 1)).astype(float), n)
              for L, (W, k, n) in data.items()}
        boots.append(analyse(bd, pairs))

    def spread(get):
        v = np.array([get(b) for b in boots], float)
        v = v[np.isfinite(v)]
        return float(v.std()) if len(v) > 1 else float("nan"), v

    out = {"sizes": Ls, "files": [str(f) for f in files], "lambda_nu": args.lambda_nu,
           "W_range": [args.W_min, args.W_max], "per_size": {}, "crossings": {}}
    print("\nper size (logistic fit):")
    for L in Ls:
        sm, _ = spread(lambda b, L=L: b["mid"][L])
        sw, _ = spread(lambda b, L=L: b["width"][L])
        out["per_size"][L] = {"W_mid": best["mid"][L], "W_mid_err": sm, "width": best["width"][L], "width_err": sw}
        print(f"  L = {L:2d}: W_mid = {best['mid'][L]:.4f} +- {sm:.4f}   width = {best['width'][L]:.4f} +- {sw:.4f}")

    # ---- steps 2-3: consecutive crossings, weighted mean
    print("\nconsecutive crossings:")
    xs, es, lbar = [], [], []
    for a, b in pairs:
        key = f"{a}/{b}"
        x = best["cross"][key]
        e, v = spread(lambda bb, key=key: bb["cross"][key] if args.W_min < bb["cross"][key] < args.W_max else np.nan)
        out["crossings"][key] = {"W": x, "err": e, "n_boot_inside": int(len(v))}
        print(f"  {key:>6}: W = {x:.4f} +- {e:.4f}   ({len(v)}/{args.boot} resamples cross inside the range)")
        if np.isfinite(x) and np.isfinite(e) and e > 0 and args.W_min < x < args.W_max:
            xs.append(x), es.append(e), lbar.append(0.5 * (a + b))
    xs, es, lbar = map(np.array, (xs, es, lbar))
    if len(xs):
        w = 1 / es**2
        wc, wc_err = float((w * xs).sum() / w.sum()), float(w.sum() ** -0.5)
        chi2 = float((w * (xs - wc) ** 2).sum()) / max(1, len(xs) - 1)
        out["Wc_weighted_mean"] = {"Wc": wc, "err": wc_err, "chi2_per_dof": chi2, "n_crossings": int(len(xs))}
        print(f"\n  W_c (weighted mean of {len(xs)} crossings) = {wc:.4f} +- {wc_err:.4f}"
              f"   (chi2/dof = {chi2:.2f}; > 1 means the crossings scatter more than their errors: drift)")

        # ---- step 4: drift vs 1/L_mean
        if len(xs) >= 2:
            A = np.stack([np.ones_like(lbar), 1 / lbar], axis=1)
            cov = np.linalg.inv(A.T @ (w[:, None] * A))
            c = cov @ (A.T @ (w * xs))
            slope, slope_err = float(c[1]), float(np.sqrt(cov[1, 1]))
            out["drift"] = {"intercept_1overL_to_0": float(c[0]), "intercept_err": float(np.sqrt(cov[0, 0])),
                            "slope": slope, "slope_err": slope_err, "significant": bool(abs(slope) > 2 * slope_err)}
            print(f"  drift: crossings vs 1/L_mean, slope {slope:.3f} +- {slope_err:.3f} "
                  f"({'SIGNIFICANT' if abs(slope) > 2 * slope_err else 'not significant'} at 2 sigma); "
                  f"extrapolated to 1/L -> 0: {c[0]:.4f} +- {np.sqrt(cov[0, 0]):.4f}")

    # ---- step 5: pseudo-critical fit (>= 4 sizes)
    L_arr = np.array(Ls, float)
    mids = np.array([best["mid"][L] for L in Ls])
    mid_err = np.array([out["per_size"][L]["W_mid_err"] for L in Ls])
    if len(Ls) >= 4:
        p = pseudo_critical_fit(L_arr, mids, mid_err)
        if p:
            bp = []
            for b in boots:
                q = pseudo_critical_fit(L_arr, np.array([b["mid"][L] for L in Ls]), mid_err)
                if q:
                    bp.append(q)
            bp = np.array(bp)
            out["pseudo_critical"] = {"Wc": p[0], "Wc_err": float(bp[:, 0].std()), "inv_nu": p[2],
                                      "inv_nu_err": float(bp[:, 2].std()), "n_boot_ok": int(len(bp))}
            print(f"\n  pseudo-critical fit W_mid(L) = W_c + A L^(-1/nu): W_c = {p[0]:.4f} +- {bp[:, 0].std():.4f}, "
                  f"1/nu = {p[2]:.3f} +- {bp[:, 2].std():.3f}  ({len(bp)}/{args.boot} resamples converged)")
    else:
        print(f"\n  pseudo-critical fit skipped: needs >= 4 sizes (have {len(Ls)})")

    # ---- step 6: slope scaling width ~ L^(-1/nu)
    if "inv_nu_width" in best:
        e, _ = spread(lambda b: b.get("inv_nu_width", np.nan))
        inv = best["inv_nu_width"]
        out["width_scaling"] = {"inv_nu": inv, "inv_nu_err": e, "nu": 1 / inv,
                                "nu_err": e / inv**2}
        print(f"  width scaling width(L) ~ L^(-1/nu): 1/nu = {inv:.3f} +- {e:.3f}  ->  nu = {1 / inv:.2f} +- {e / inv**2:.2f}"
              f"\n    (compare: ~2.4-2.6 quantum-Hall class (Mildner, Haldane); 1 = clean mass-driven transition;"
              f" small sizes can sit in a crossover)")

    js = SWEEP / f"{args.tag}.json"
    js.write_text(json.dumps(out, indent=2, default=float))
    print(f"\nsaved {js}")
    plot(data, best, out, Ls, pairs, HERE / "results" / "figures" / f"{args.tag}.png")


def plot(data, best, out, Ls, pairs, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 2, figsize=(11, 8.5))
    cmap = plt.get_cmap("viridis")
    lo, hi = min(d[0].min() for d in data.values()), max(d[0].max() for d in data.values())
    Wf = np.linspace(lo, hi, 400)
    for i, L in enumerate(Ls):
        W, k, n = data[L]
        p = k / np.maximum(n, 1)
        col = cmap(i / max(1, len(Ls) - 1))
        ax[0, 0].errorbar(W, p, yerr=np.sqrt(p * (1 - p) / np.maximum(n, 1)), fmt="o", ms=3, color=col,
                          label=f"L = {L} ({int(n.mean())} per point)")
        m, wd = best["mid"][L], best["width"][L]
        ax[0, 0].plot(Wf, 1 / (1 + np.exp(-(Wf - m) / wd)), color=col, lw=1)
    if "Wc_weighted_mean" in out:
        wc, e = out["Wc_weighted_mean"]["Wc"], out["Wc_weighted_mean"]["err"]
        ax[0, 0].axvspan(wc - e, wc + e, color="red", alpha=0.2, label=f"W_c = {wc:.3f} ± {e:.3f}")
    ax[0, 0].set(xlabel="disorder W (energies in [−W/2, W/2])", ylabel="fraction topological",
                 title="(a) switch-on for each size, logistic fits")
    ax[0, 0].legend(fontsize=7)

    keys = [f"{a}/{b}" for a, b in pairs]
    xs = [out["crossings"][k]["W"] for k in keys]
    es = [out["crossings"][k]["err"] for k in keys]
    il = [2 / (a + b) for a, b in pairs]
    ax[0, 1].errorbar(il, xs, yerr=es, fmt="s", color="k")
    for x, y, k in zip(il, xs, keys):
        ax[0, 1].annotate(k, (x, y), textcoords="offset points", xytext=(5, 5), fontsize=7)
    if "Wc_weighted_mean" in out:
        ax[0, 1].axhspan(wc - e, wc + e, color="red", alpha=0.2, label="weighted mean")
    ax[0, 1].set(xlabel="1 / L_mean of the pair", ylabel="crossing W", title="(b) consecutive crossings vs 1/L")
    ax[0, 1].legend(fontsize=7)

    L_arr = np.array(Ls, float)
    mids = [out["per_size"][L]["W_mid"] for L in Ls]
    merr = [out["per_size"][L]["W_mid_err"] for L in Ls]
    ax[1, 0].errorbar(L_arr, mids, yerr=merr, fmt="o", color="tab:blue")
    if "pseudo_critical" in out:
        pc = out["pseudo_critical"]
        Lf = np.linspace(L_arr.min(), L_arr.max() * 2, 200)
        A = (mids[-1] - pc["Wc"]) * L_arr[-1] ** pc["inv_nu"]
        ax[1, 0].plot(Lf, pc["Wc"] + A * Lf ** (-pc["inv_nu"]), "--", color="tab:blue",
                      label=f"W_c = {pc['Wc']:.3f} ± {pc['Wc_err']:.3f}, 1/ν = {pc['inv_nu']:.2f}")
        ax[1, 0].legend(fontsize=7)
    ax[1, 0].set(xlabel="L", ylabel="midpoint W_mid(L)", title="(c) pseudo-critical points")

    wid = [out["per_size"][L]["width"] for L in Ls]
    werr = [out["per_size"][L]["width_err"] for L in Ls]
    ax[1, 1].errorbar(L_arr, wid, yerr=werr, fmt="o", color="tab:green")
    if "width_scaling" in out:
        ws = out["width_scaling"]
        B = np.exp(np.mean(np.log(wid) + ws["inv_nu"] * np.log(L_arr)))
        ax[1, 1].plot(L_arr, B * L_arr ** (-ws["inv_nu"]), "--", color="tab:green",
                      label=f"1/ν = {ws['inv_nu']:.2f} ± {ws['inv_nu_err']:.2f}  (ν = {ws['nu']:.2f} ± {ws['nu_err']:.2f})")
        ax[1, 1].legend(fontsize=7)
    ax[1, 1].set(xscale="log", yscale="log", xlabel="L", ylabel="width of the switch",
                 title="(d) width ~ L^(−1/ν)")
    fig.suptitle("Kane-Mele TAI switch-on (λν = 1.65t, λSO = 0.3t, λR = 0): finite-size scaling", fontsize=11)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
