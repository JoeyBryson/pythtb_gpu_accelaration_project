"""PythTB's built-in TensorFlow path (use_tensorflow=True) with Apple's Metal plugin vs plain NumPy.

Written at the hackathon (21 Nov 2026). Mac only. TensorFlow is NOT a project dependency; run with
a pinned throwaway environment (tensorflow-metal 1.2.0 fails to load with newer TF):

  uv run --with "tensorflow==2.18.1" --with tensorflow-metal python benchmarks/tf_metal_vs_numpy.py

Four parts:
  1. which linear-algebra ops the Metal plugin can actually place on the GPU (strict placement)
  2. WFArray.solve_model timings, NumPy vs use_tensorflow=True, plus a correctness check
  3. the one op that does run on Metal (float32 matmul) vs NumPy, for reference
  4. where the TF speed-up comes from: solve_model (6x6) with 1, 4 and default TF CPU threads,
     each in a fresh process (TF fixes its thread pool at start-up)

Without the Metal plugin (shows that part 2's speed-up does not come from Metal):
  uv run --with "tensorflow==2.18.1" python benchmarks/tf_metal_vs_numpy.py --intra-op-threads 0
"""

from __future__ import annotations

import argparse
import importlib.metadata as md
import os
import platform
import subprocess
import sys
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import tensorflow as tf
from pythtb import Mesh, WFArray

from gpu_engine.reference import clean_kane_mele

tf.get_logger().setLevel("ERROR")


def best_of(f, repeats: int = 3) -> float:
    """Minimum wall time over `repeats` calls, after one warm-up call (CPU-side timing)."""
    f()
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        f()
        ts.append(time.perf_counter() - t0)
    return min(ts)


def gpu_placement_table() -> None:
    """Try each op on /GPU:0 with soft placement off; report whether Metal has a kernel."""
    tf.config.set_soft_device_placement(False)
    ops = {
        "matmul": lambda x: tf.linalg.matmul(x, x),
        "eigh": lambda x: tf.linalg.eigh(x)[0],
        "svd": lambda x: tf.linalg.svd(x)[0],
        "eig": lambda x: tf.linalg.eig(x)[0],
    }
    dtypes = [np.complex128, np.complex64, np.float64, np.float32]
    print(f"   {'':8s}" + "".join(f"{np.dtype(d).name:>12s}" for d in dtypes))
    for name, op in ops.items():
        row = []
        for dt in dtypes:
            a = np.eye(16, dtype=dt)[None].repeat(4, 0)
            try:
                with tf.device("/GPU:0"):
                    op(tf.constant(a))
                row.append("GPU")
            except Exception:  # noqa: BLE001  any TF error here means no Metal kernel
                row.append("no")
        print(f"   {name:8s}" + "".join(f"{r:>12s}" for r in row))
    tf.config.set_soft_device_placement(True)


def solve(model, nk: int, use_tf: bool) -> WFArray:
    mesh = Mesh(["k", "k"])
    mesh.build_grid(shape=(nk, nk))
    wfa = WFArray(model.lattice, mesh, spinful=True)
    wfa.solve_model(model=model, use_tensorflow=use_tf)
    return wfa


def metal_version() -> str:
    try:
        return md.version("tensorflow-metal")
    except md.PackageNotFoundError:
        return "not installed"


def time_threads(threads: int) -> None:
    """Part 4 worker: solve_model on 6x6 with a fixed TF intra-op thread count (0 = default)."""
    if threads:
        tf.config.threading.set_intra_op_parallelism_threads(threads)
    model = clean_kane_mele(supercell=6)
    t = best_of(lambda: solve(model, 24, True))
    print(f"   tf, intra-op threads {threads or 'default':>7}, metal plugin {metal_version():>13}: "
          f"{t:.3f} s")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--intra-op-threads", type=int, default=None,
                   help="only run the part-4 timing with this many TF threads (0 = default)")
    args = p.parse_args()
    if args.intra_op_threads is not None:
        time_threads(args.intra_op_threads)
        return
    print(f"Hardware: {platform.machine()} | {platform.platform()}")
    print(f"TensorFlow {tf.__version__}, tensorflow-metal {metal_version()}, "
          f"NumPy {np.__version__}")
    print(f"Devices: {[d.name for d in tf.config.list_physical_devices()]}\n")

    print("1. Can the Metal plugin run this op on the GPU? (soft placement off)")
    gpu_placement_table()
    print("   -> PythTB's TF path uses complex128 eigh, so on a Mac it runs on the CPU.\n")

    nk = 24
    print(f"2. WFArray.solve_model, clean Kane-Mele supercell, {nk}x{nk} mesh, complex128 (s)")
    print(f"   {'n':>2} {'M':>4} {'numpy':>8} {'tf':>8} {'tf/numpy':>9} {'max|dE|':>9}")
    for n in (1, 2, 3, 4, 6, 8):
        model = clean_kane_mele(supercell=n)
        reps = 1 if n >= 8 else 3
        t_np = best_of(lambda m=model: solve(m, nk, False), reps)
        t_tf = best_of(lambda m=model: solve(m, nk, True), reps)
        dE = np.abs(solve(model, nk, False).energies - solve(model, nk, True).energies).max()
        print(f"   {n:>2} {4 * n * n:>4} {t_np:8.3f} {t_tf:8.3f} {t_tf / t_np:8.2f}x {dE:9.1e}")
    print("   (tf/numpy > 1 means TensorFlow is slower)\n")

    print("3. Batched float32 matmul (the one op Metal runs), 576 x M x M (ms)")
    print(f"   {'M':>4} {'numpy':>8} {'metal':>8}")
    rng = np.random.default_rng(0)
    for M in (64, 256, 512):
        a = rng.standard_normal((576, M, M)).astype(np.float32)
        with tf.device("/GPU:0"):
            ta = tf.constant(a)
        t_np = best_of(lambda a=a: a @ a)
        t_mt = best_of(lambda ta=ta: tf.linalg.matmul(ta, ta).numpy())  # .numpy() forces completion
        print(f"   {M:>4} {1e3 * t_np:8.1f} {1e3 * t_mt:8.1f}")
    print("   Metal timing includes copying the result back to NumPy.\n")

    print("4. solve_model 6x6, 24x24 mesh, vs TF CPU thread count (fresh process each)")
    model = clean_kane_mele(supercell=6)
    print(f"   numpy: {best_of(lambda: solve(model, 24, False)):.3f} s")
    for threads in (1, 4, 0):
        sys.stdout.flush()  # keep the child's line after ours when output is piped
        subprocess.run([sys.executable, __file__, "--intra-op-threads", str(threads)], check=True)


if __name__ == "__main__":
    main()
