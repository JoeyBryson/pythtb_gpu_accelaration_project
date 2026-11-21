"""Sample simulation with the *existing* PythTB pipeline on CPU (no GPU code).

Written at the hackathon (21 Nov 2026) to see how the stages fit together and where the time goes
before we replace them. Steps, for each model:

  1. build    Kane-Mele supercell + Anderson disorder (random on-site energy per site)
  2. H(k)     PythTB builds the Bloch Hamiltonian on a k-mesh             (timed separately)
  3. solve    WFArray.solve_model: H(k) + numpy eigh at every k-point     (~70% in our profiles)
  4. wilson   WFArray.berry_phase: Wilson loop along k2 for every k1-line  (~30%)
  5. Z2       winding of the hybrid Wannier centres over half the BZ      (provisional method)

Run:  uv run python scripts/sample_cpu_sim.py            (~1 min on a laptop)
      uv run python scripts/sample_cpu_sim.py --quick    (smaller sweep)

Outputs: results/sample_cpu_sweep.csv, results/sample_cpu_sim.png

TO CHECK (physics side):
  - The Z2 rule below is the "largest gap" method of Soluyanov & Vanderbilt, PRB 83, 235401
    (2011). The team has not agreed the Z2 algorithm yet; this is a stand-in.
  - Disorder convention follows Orth et al. Eq. (1): eps_i uniform in [-W, W], same for both
    spins. How PythTB's kane_mele(delta, t, soc, rashba) maps onto their lambda_nu, lambda_SO,
    lambda_R is not confirmed, so W values here are NOT directly comparable with the paper.
"""

from __future__ import annotations

import argparse
import copy
import csv
import platform
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pythtb import Mesh, TBModel, WFArray

from gpu_engine.reference import clean_kane_mele

RESULTS = Path(__file__).resolve().parents[1] / "results"


# ----------------------------------------------------------------------------- model


def disordered_kane_mele(n: int, W: float, rng: np.random.Generator, **params) -> TBModel:
    """n x n Kane-Mele supercell with Anderson disorder eps_i ~ U[-W, W] on every site.

    Each site gets one random number, applied to both spin components (scalar spinful onsite).
    """
    model = copy.deepcopy(clean_kane_mele(supercell=n, **params))
    if W > 0:
        eps = rng.uniform(-W, W, size=model.norb)  # (norb,) one value per site
        model.set_onsite(list(eps), mode="add")
    return model


# ----------------------------------------------------------------------------- pipeline


def run_pipeline(model: TBModel, nk: int, use_tf: bool = False) -> dict:
    """Run PythTB's solve + Wilson loop on an nk x nk mesh (k in [0, 1)), timing each stage.

    use_tf: pass use_tensorflow=True to solve_model (PythTB's built-in TF eigh; needs tensorflow).

    Returns hybrid Wannier centres for k1 in [0, 0.5] (enough for Z2, by time reversal):
      wcc: (nk//2 + 1, n_occ) float in [0, 1)
    """
    assert nk % 2 == 0, "nk must be even so the mesh contains k1 = 0 and k1 = 0.5"
    n_occ = model.norb  # half filling: 2*norb spinful states, norb occupied

    mesh = Mesh(["k", "k"])
    mesh.build_grid(shape=(nk, nk))  # k_i = 0, 1/nk, ..., (nk-1)/nk

    out = {"M": 2 * model.norb, "n_occ": n_occ, "nk": nk}

    # Stage 2 on its own (solve_model rebuilds it internally; timed here only to see its share)
    k1d = np.arange(nk) / nk
    k_flat = np.stack(np.meshgrid(k1d, k1d, indexing="ij"), -1).reshape(-1, 2)  # (nk*nk, 2)
    t0 = time.perf_counter()
    model.hamiltonian(k_flat, flatten_spin_axis=True)  # (nk*nk, M, M) complex
    out["t_H"] = time.perf_counter() - t0

    # Stage 3: H(k) + eigh at every k-point
    t0 = time.perf_counter()
    wfa = WFArray(model.lattice, mesh, spinful=True)
    wfa.solve_model(model=model, use_tensorflow=use_tf)
    out["t_solve"] = time.perf_counter() - t0

    # Stage 4: Wilson loop along k2 for every k1 (PythTB loops over k1-lines in Python)
    t0 = time.perf_counter()
    phases = wfa.berry_phase(
        axis_idx=1, state_idx=list(range(n_occ)), contin=False, berry_evals=True
    )  # (nk, n_occ)
    out["t_wilson"] = time.perf_counter() - t0

    wcc = np.mod(np.asarray(phases) / (2 * np.pi), 1.0)
    out["wcc"] = np.sort(wcc[: nk // 2 + 1], axis=-1)
    out["k1"] = np.arange(nk // 2 + 1) / nk

    # Energy gap at half filling over the whole mesh (negative = no gap on this mesh)
    E = np.real(wfa.energies)  # (nk, nk, M); the TF path returns complex dtype
    out["gap"] = float(E[..., n_occ].min() - E[..., n_occ - 1].max())
    return out


def z2_largest_gap(wcc: np.ndarray) -> tuple[int, np.ndarray]:
    """Z2 from Wannier-centre flow, largest-gap method (Soluyanov & Vanderbilt 2011). TO CHECK.

    wcc: (n_k1, n_occ) sorted centres in [0, 1) for k1 from 0 to 0.5.
    At each k1, find the midpoint z of the largest gap between neighbouring centres. Moving from
    k1_i to k1_{i+1}, count centres at k1_{i+1} that lie between z_i and z_{i+1}. Z2 = total mod 2.
    Using the straight interval rather than the shorter arc is fine: the two counts differ by
    n_occ, which is even (Kramers pairs).

    Returns (z2, z) with z: (n_k1,) gap midpoints for plotting.
    """
    gaps = np.diff(np.concatenate([wcc, wcc[:, :1] + 1.0], axis=1), axis=1)  # (n_k1, n_occ)
    j = gaps.argmax(axis=1)
    z = np.mod(wcc[np.arange(len(wcc)), j] + 0.5 * gaps[np.arange(len(wcc)), j], 1.0)
    count = 0
    for i in range(len(wcc) - 1):
        lo, hi = sorted((z[i], z[i + 1]))
        count += int(np.sum((wcc[i + 1] > lo) & (wcc[i + 1] < hi)))
    return count % 2, z


# ----------------------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--supercell", type=int, default=3, help="n for the n x n disorder supercell")
    ap.add_argument("--nk", type=int, default=16, help="k-mesh per direction for the sweep (even)")
    ap.add_argument("--configs", type=int, default=6, help="disorder configurations per W")
    ap.add_argument("--W", type=float, nargs="+", default=[0, 0.5, 1, 2, 3, 4, 6, 8, 12])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true", help="fewer W values and configs")
    ap.add_argument("--tensorflow", action="store_true",
                    help="use PythTB's use_tensorflow=True eigh (install tensorflow separately)")
    ap.add_argument("--tag", default="", help="suffix for output file names")
    args = ap.parse_args()
    if args.quick:
        args.W, args.configs = [0, 1, 4, 12], 2

    RESULTS.mkdir(exist_ok=True)
    rng = np.random.default_rng(args.seed)
    print(f"Hardware: {platform.machine()} {platform.processor() or ''} | {platform.platform()}")
    print(f"NumPy {np.__version__} | plain PythTB | eigh via "
          f"{'TensorFlow (use_tensorflow=True)' if args.tensorflow else 'NumPy'}\n")
    csv_path = RESULTS / f"sample_cpu_sweep{args.tag}.csv"
    png_path = RESULTS / f"sample_cpu_sim{args.tag}.png"

    # --- 1. sanity checks on clean primitive cells -------------------------------------------
    print("1. Clean primitive cell, nk=32")
    clean = run_pipeline(clean_kane_mele(), nk=32, use_tf=args.tensorflow)
    z2_clean, z_clean = z2_largest_gap(clean["wcc"])
    print(f"   tutorial params (delta=1, soc=0.3, rashba=0.25): Z2={z2_clean}  (expect 1)")
    triv = run_pipeline(clean_kane_mele(delta=4.0), nk=32, use_tf=args.tensorflow)
    z2_triv, _ = z2_largest_gap(triv["wcc"])
    print(f"   large staggered potential (delta=4):            Z2={z2_triv}  (expect 0)\n")

    # --- 2. where the time goes vs supercell size (clean, fixed mesh) -------------------------
    print("2. Stage timings vs supercell size (clean, nk=24, seconds)")
    print(f"   {'n':>2} {'M':>4} {'H(k)':>8} {'solve':>8} {'wilson':>8}  wilson share")
    timing_rows = []
    for n in (1, 2, 3, 4):
        r = run_pipeline(clean_kane_mele(supercell=n), nk=24, use_tf=args.tensorflow)
        share = r["t_wilson"] / (r["t_solve"] + r["t_wilson"])
        print(f"   {n:>2} {r['M']:>4} {r['t_H']:8.3f} {r['t_solve']:8.3f} {r['t_wilson']:8.3f}"
              f"  {share:5.0%}")
        timing_rows.append(r)
    print()

    # --- 3. disorder sweep ----------------------------------------------------------------------
    n = args.supercell
    print(f"3. Disorder sweep: {n}x{n} supercell (M={4 * n * n}), nk={args.nk}, "
          f"{args.configs} configs per W")
    print(f"   {'W':>5} {'Z2=1':>6} {'gapped':>7} {'mean gap':>9} {'s/config':>9}")
    rows, wcc_examples = [], {}
    t_sweep = time.perf_counter()
    for W in args.W:
        z2s, gaps, times = [], [], []
        for c in range(args.configs):
            r = run_pipeline(disordered_kane_mele(n, W, rng), nk=args.nk, use_tf=args.tensorflow)
            z2, z = z2_largest_gap(r["wcc"])
            z2s.append(z2)
            gaps.append(r["gap"])
            times.append(r["t_solve"] + r["t_wilson"])
            rows.append({"W": W, "config": c, "z2": z2, "gap": r["gap"],
                         "t_solve": r["t_solve"], "t_wilson": r["t_wilson"]})
            if c == 0:
                wcc_examples[W] = (r["k1"], r["wcc"], z)
        z2s, gaps = np.array(z2s), np.array(gaps)
        print(f"   {W:5.1f} {z2s.mean():6.0%} {np.mean(gaps > 0):7.0%} {gaps.mean():9.3f}"
              f" {np.mean(times):9.3f}")
    t_sweep = time.perf_counter() - t_sweep
    print(f"   total sweep time: {t_sweep:.1f} s for {len(rows)} disordered systems\n")
    print("   Note: Z2 is only meaningful for gapped configs (gap > 0); see the CSV.")

    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # --- figure ----------------------------------------------------------------------------------
    fig, ax = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)

    def wcc_panel(a, k1, wcc, z, title):
        a.plot(k1, wcc, "k.", ms=3)
        a.plot(k1, z, "r-", lw=1, label="largest-gap midpoint")
        a.set(xlim=(0, 0.5), ylim=(0, 1), xlabel="k1 (reduced)", ylabel="Wannier centre (x2)",
              title=title)

    wcc_panel(ax[0, 0], clean["k1"], clean["wcc"], z_clean, f"clean primitive, Z2={z2_clean}")
    ax[0, 0].legend(loc="upper right", fontsize=8)
    W_show = sorted(wcc_examples)
    for a, W in zip((ax[0, 1], ax[0, 2]), (W_show[1] if len(W_show) > 1 else W_show[0], W_show[-1])):
        k1, wcc, z = wcc_examples[W]
        wcc_panel(a, k1, wcc, z, f"{n}x{n} supercell, W={W}, Z2={z2_largest_gap(wcc)[0]}")

    Ws = np.array([r["W"] for r in rows])
    Z2 = np.array([r["z2"] for r in rows])
    G = np.array([r["gap"] for r in rows])
    uW = np.unique(Ws)
    ax[1, 0].plot(uW, [Z2[Ws == w].mean() for w in uW], "o-", label="fraction Z2 = 1")
    ax[1, 0].plot(uW, [np.mean(G[Ws == w] > 0) for w in uW], "s--", label="fraction gapped")
    ax[1, 0].set(xlabel="disorder W (units of t)", ylabel="fraction of configs", ylim=(-0.05, 1.05),
                 title=f"{n}x{n}, {args.configs} configs per W")
    ax[1, 0].legend(fontsize=8)

    ax[1, 1].plot(Ws, G, "k.", alpha=0.6)
    ax[1, 1].axhline(0, color="grey", lw=0.8)
    ax[1, 1].set(xlabel="disorder W (units of t)", ylabel="half-filling gap on mesh (t)",
                 title="energy gap per config")

    M = [r["M"] for r in timing_rows]
    ax[1, 2].plot(M, [r["t_H"] for r in timing_rows], "o-", label="H(k) build")
    ax[1, 2].plot(M, [r["t_solve"] for r in timing_rows], "o-", label="solve (H + eigh)")
    ax[1, 2].plot(M, [r["t_wilson"] for r in timing_rows], "o-", label="Wilson loop")
    ax[1, 2].set(xscale="log", yscale="log", xlabel="matrix size M", ylabel="seconds",
                 title="CPU stage timings, nk=24x24")
    ax[1, 2].set_xticks(M, [str(m) for m in M], minor=False)
    ax[1, 2].minorticks_off()
    ax[1, 2].legend(fontsize=8)

    fig.suptitle(f"Plain PythTB on CPU ({platform.machine()}): Kane-Mele + Anderson disorder")
    fig.savefig(png_path, dpi=130)
    print(f"Saved {csv_path} and {png_path}")


if __name__ == "__main__":
    main()
