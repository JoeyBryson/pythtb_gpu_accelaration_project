"""Where does disorder switch topology on? Estimate the critical W from several system sizes.

For each supercell size L, fit a logistic curve to the fraction of topological samples vs W
around the switch-on (lambda_nu = 1.65):

    P(topological | W) = 1 / (1 + exp(-(W - W_mid) / width))

by binomial maximum likelihood on the actual counts (k topological out of n reliable per point;
a logistic regression, fitted by iteratively reweighted least squares). Then:

  W_mid(L)   where that size's curve passes 50%
  width(L)   how sharp the switch is: it should SHRINK with L if the transition is real
  W_x(L1,L2) where the fitted curves of two sizes cross: the finite-size estimate of W_c

Error bars: parametric bootstrap. Resample every point's count from Binomial(n, k/n), refit,
repeat 2,000 times; report mean +- standard deviation.

Caveats (say them with the number): logistic is a convenient shape, not a theory; crossings
can drift with L (small sizes); our 8x8 has only 0.5-spaced points. A full analysis would use
more sizes and a scaling collapse.

Run from hackathon/:  uv run python analysis/crossing.py
"""

from __future__ import annotations

import csv
from itertools import combinations
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
SWEEP = HERE / "results" / "sweep"
FILES = {8: SWEEP / "pilot_8x8_c_up.csv", 12: SWEEP / "fss_12x12_c_up.csv",
         16: SWEEP / "fss_16x16_c_up.csv"}
DENSE = SWEEP / "dense_16x16_c_up.csv"  # 1,000 samples per point, if available: preferred
LAMBDA_NU, W_LO, W_HI = 1.65, 1.0, 3.0
N_BOOT = 2000


def load(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """W, k (topological), n (reliable) for lambda_nu = 1.65 within [W_LO, W_HI]."""
    W, k, n = [], [], []
    with path.open() as f:
        for r in csv.DictReader(f):
            w = float(r["W"])
            if abs(float(r["lambda_nu"]) - LAMBDA_NU) < 1e-9 and W_LO - 1e-9 <= w <= W_HI + 1e-9:
                W.append(w)
                k.append(int(r["n_topological"]))
                n.append(int(r["n_reliable"]))
    return np.array(W), np.array(k, float), np.array(n, float)


def fit_logistic(W: np.ndarray, k: np.ndarray, n: np.ndarray) -> tuple[float, float]:
    """Binomial logistic regression logit(p) = a + b W by IRLS. Returns (W_mid, width)."""
    X = np.stack([np.ones_like(W), W], axis=1)
    beta = np.array([-2.0 * W.mean(), 2.0])  # start: a switch near the middle of the range
    for _ in range(100):
        eta = np.clip(X @ beta, -30, 30)
        p = 1 / (1 + np.exp(-eta))
        w = np.maximum(n * p * (1 - p), 1e-9)
        z = eta + (k - n * p) / w
        new = np.linalg.solve(X.T @ (w[:, None] * X) + 1e-9 * np.eye(2), X.T @ (w * z))
        if np.abs(new - beta).max() < 1e-10:
            beta = new
            break
        beta = new
    a, b = beta
    return -a / b, 1 / b


def crossing(m1: float, w1: float, m2: float, w2: float) -> float:
    """W where two logistic curves are equal: (W - m1)/w1 = (W - m2)/w2."""
    return (m1 * w2 - m2 * w1) / (w2 - w1) if abs(w2 - w1) > 1e-12 else np.nan


def main() -> None:
    rng = np.random.default_rng(0)
    files = dict(FILES)
    if DENSE.exists():
        files[16] = DENSE
    data = {L: load(p) for L, p in files.items() if p.exists()}

    fits, boots = {}, {}
    print(f"lambda_nu = {LAMBDA_NU}, W in [{W_LO}, {W_HI}], {N_BOOT} bootstrap resamples\n")
    for L, (W, k, n) in sorted(data.items()):
        fits[L] = fit_logistic(W, k, n)
        boot = []
        for _ in range(N_BOOT):
            kb = rng.binomial(n.astype(int), np.clip(k / n, 0, 1))
            boot.append(fit_logistic(W, kb, n))
        boots[L] = np.array(boot)
        mid, width = fits[L]
        print(f"{L:2d}x{L:<2d} ({len(W)} points, {int(n.sum())} samples): "
              f"W_mid = {mid:.3f} +- {boots[L][:, 0].std():.3f}   "
              f"width = {width:.3f} +- {boots[L][:, 1].std():.3f}")

    rows = []
    print()
    for L1, L2 in combinations(sorted(data), 2):
        x = crossing(*fits[L1], *fits[L2])
        xb = np.array([crossing(*b1, *b2) for b1, b2 in zip(boots[L1], boots[L2], strict=True)])
        xb = xb[np.isfinite(xb) & (xb > W_LO) & (xb < W_HI)]
        print(f"crossing {L1}x{L1} / {L2}x{L2}: W = {x:.3f} +- {xb.std():.3f}  "
              f"({len(xb)}/{N_BOOT} resamples cross inside [{W_LO}, {W_HI}])")
        rows.append({"pair": f"{L1}/{L2}", "W_cross": f"{x:.4f}", "err": f"{xb.std():.4f}"})
    for L, (mid, width) in sorted(fits.items()):
        rows.append({"pair": f"{L} only", "W_mid": f"{mid:.4f}", "W_mid_err": f"{boots[L][:, 0].std():.4f}",
                     "width": f"{width:.4f}", "width_err": f"{boots[L][:, 1].std():.4f}",
                     "source": files[L].name})

    out = SWEEP / "crossing_estimate.csv"
    fields = ["pair", "W_cross", "err", "W_mid", "W_mid_err", "width", "width_err", "source"]
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(rows)
    print(f"\nsaved to {out}")


if __name__ == "__main__":
    main()
