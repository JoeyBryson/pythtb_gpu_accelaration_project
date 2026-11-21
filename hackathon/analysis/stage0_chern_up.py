"""Stage 0b: spin-up Chern number C_up on the stage-0 configurations, at nk = 6, 10, 14.

At lambda_R = 0, S_z is conserved: H is block diagonal in spin, and the spin-up block is a
Chern problem. C_up = winding of the SUM of the spin-up Wannier centres along k_x, i.e. of
arg det(Wilson loop) = sum over links of arg det(M). No gap tracking. Z2 = C_up mod 2.
Reliable if every k_x step of the sum is well below 1/2 (column max_step).

Found 21 Nov: C_up identical on nk = 6/10/14 for 39/40 configs (W = 2, 3, L = 8); the one
change had max_step = 0.48 at nk = 6. Gap-tracking Z2 disagreed with it in 6/20 (nk = 6)
and 4/20 (nk = 10) configs at W = 3, including some flagged reliable.

Run from hackathon/:  uv run python analysis/stage0_chern_up.py   (about 5 min on 8 cores)
Writes results/stage0_chern_up.csv.
"""
import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from gpu_engine.reference import clean_kane_mele, disordered_kane_mele

OUT = Path(__file__).resolve().parent.parent / "results" / "stage0_chern_up.csv"
P = {"delta": 1.65, "t": 1.0, "soc": 0.3, "rashba": 0.0}

def c_up(model, nk):
    k = np.arange(nk) / nk
    kk = np.stack(np.meshgrid(k, k, indexing="ij"), -1).reshape(-1, 2)
    H = model.hamiltonian(kk, flatten_spin_axis=True)
    assert np.abs(H[:, 0::2, 1::2]).max() < 1e-12          # no spin mixing at lambda_R = 0
    Hu = H[:, 0::2, 0::2]
    e, v = np.linalg.eigh(Hu)
    n = Hu.shape[-1] // 2                                     # half filling of the up block
    u = np.swapaxes(v[..., :n], -1, -2).reshape(nk, nk, n, -1)  # (kx, ky, n, norb) rows = states
    gap = float((e[:, n] - e[:, n - 1]).min())
    D = np.exp(-2j * np.pi * np.asarray(model.orb_vecs) @ np.array([0.0, 1.0]))
    nxt = np.roll(u, -1, axis=1); nxt[:, -1] = nxt[:, -1] * D
    M = u.conj() @ np.swapaxes(nxt, -1, -2)                   # (kx, ky, n, n)
    sign, _ = np.linalg.slogdet(M)
    phase = np.angle(sign).sum(axis=1)                        # arg det W(kx), one number per line
    p = -phase / (2 * np.pi)                                  # sum of up-spin Wannier centres, mod 1
    d = np.diff(np.append(p, p[0]))
    d = (d + 0.5) % 1.0 - 0.5                                 # wrap each step into (-1/2, 1/2]
    return round(float(d.sum())), float(np.abs(d).max()), gap

def task(a):
    L, W, s, nk = a
    m = clean_kane_mele(L, **P) if W == 0 else disordered_kane_mele(L, W, np.random.default_rng(s), **P)
    C, step, gap = c_up(m, nk)
    return L, W, s, nk, C, step, gap

if __name__ == "__main__":
    # sanity: clean unit cell, topological (1.45) and trivial (1.65)
    for d in (1.45, 1.65):
        print("clean 1x1 delta", d, "C_up =", c_up(clean_kane_mele(1, **{**P, "delta": d}), 24)[0])
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=8)
    ap.add_argument("--W", type=float, nargs="+", default=[2.0, 3.0])
    ap.add_argument("--nk", type=int, nargs="+", default=[6, 10, 14])
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--out", default=None, help="file name in results/ (default stage0_chern_up.csv)")
    a = ap.parse_args()
    out = OUT if a.out is None else OUT.with_name(a.out)
    tasks = [(a.L, W, s, nk) for W in a.W for s in range(a.seeds) for nk in a.nk]
    with ProcessPoolExecutor(8) as ex:
        res = list(ex.map(task, tasks))
    with out.open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["L", "W", "seed", "nk", "C_up", "max_step", "gap_up"]); w.writerows(res)
    print(f"saved {len(res)} rows to {out}")
