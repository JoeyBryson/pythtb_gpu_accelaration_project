"""Runs ON Kaggle: unpack our code, install it, run one PRESET of jobs on the GPU, save results.

Pick a preset with `PRESET=<name> bash kaggle/gpu_run.sh` from hackathon/ (default "speed");
edit or add presets below. Everything in hackathon/results/*.csv at the end comes back to
your laptop.
"""

import glob
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

# Which preset to run. gpu_run.sh rewrites this line from its PRESET environment variable.
PRESET = "speed"

_TESTS = ["-m", "pytest", "-q", "-p", "no:cacheprovider"]
# "speed" = RESULTS.md R10: Rashba = 0 (team choice) on BOTH sides, ours with the spin-block
# split and the double-precision eigenvalue step; GPU single precision vs PythTB on all CPU
# cores of the same machine (timed without diagnostics).
_SPEED = ["--W", "2.0", "--devices", "cuda", "--precisions", "single", "--skip-serial",
          "--no-reliability", "--rashba", "0", "--spin-blocks"]
PRESETS = {
    "speed": [
        _TESTS,
        ["benchmarks/gpu_pipeline.py", "--sizes", "8", "12", "--configs", "40", *_SPEED],
        # 16x16 with fewer configs: PythTB is slow here even on all cores
        ["benchmarks/gpu_pipeline.py", "--sizes", "16", "--configs", "16", *_SPEED],
    ],
    # "micro": GPU-only design checks, no PythTB baseline (~5-10 min): spin-block crossover
    # by size, and the speed/accuracy of ways to get the Wilson-matrix eigenvalues.
    "micro": [
        _TESTS,
        ["benchmarks/gpu_micro.py"],
    ],
    # Disorder sweeps (analysis/sweep.py): lambda_nu = 1.0 and 1.65, Rashba 0. They write
    # straight into /kaggle/working/out after every W point, so a cancelled run keeps its data.
    # "sweep_pilot": 8x8, W 0..10 step 0.5, 100 samples per point (~10-15 min).
    "sweep_pilot": [
        _TESTS,
        ["analysis/sweep.py", "--size", "8", "--samples", "100", "--W-step", "0.5",
         "--out", "/kaggle/working/out/sweep_pilot_8x8.csv"],
    ],
    # "fss": finite-size scaling with the C_up sweep (~30 min). 12x12 for both lambda_nu over the
    # full W range, 16x16 for lambda_nu = 1.65 around the switch-on transition. With the 8x8 pilot
    # (results/sweep/pilot_8x8_c_up.csv) this gives 8/12/16 curves on one plot.
    "fss": [
        ["analysis/sweep.py", "--size", "12", "--samples", "100", "--W-step", "0.5",
         "--out", "/kaggle/working/out/fss_12x12.csv"],
        ["analysis/sweep.py", "--size", "16", "--deltas", "1.65", "--samples", "100",
         "--W-max", "5", "--W-step", "0.25", "--out", "/kaggle/working/out/fss_16x16.csv"],
    ],
    # "dense": 1,000 samples per point (20x Favata & Marrazzo's Fig. 7, which used 50) around the
    # switch-on, 16x16, lambda_nu = 1.65, W = 1.0..3.0 step 0.25 (~45 min). Tight error bars.
    "dense": [
        ["analysis/sweep.py", "--size", "16", "--deltas", "1.65", "--samples", "1000",
         "--W-min", "1.0", "--W-max", "3.0", "--W-step", "0.25",
         "--out", "/kaggle/working/out/dense_16x16.csv"],
    ],
    # "phasemap": 2D map, lambda_nu 1.45..1.85 (step 0.05) x W 0..4 (step 0.25), 12x12, 50 samples
    # per cell (~20-25 min). Where disorder creates topology; also gives W_c(lambda_nu) for
    # comparison with Orth Fig. 5.
    "phasemap": [
        ["analysis/sweep.py", "--size", "12", "--samples", "50",
         "--deltas", "1.45", "1.5", "1.55", "1.6", "1.65", "1.7", "1.75", "1.8", "1.85",
         "--W-max", "4", "--W-step", "0.25", "--out", "/kaggle/working/out/phasemap_12x12.csv"],
    ],
    # "cost": what a full sweep costs, measured per sample on this machine and extrapolated:
    # PythTB as-is (all cores) vs C_up on all cores vs C_up on the GPU, at 8/12/16 (~10-15 min).
    "cost": [
        ["benchmarks/sweep_cost.py", "--sizes", "8", "12", "16"],
    ],
    # Spin-up Chern number C_up (gpu_engine.spin_chern; decision A in PART2_PLAN.md).
    # "chern_test": C_up GPU vs the validated CPU version + timing at 8/12/16 (~5 min).
    "chern_test": [
        [*_TESTS, "tests/test_spin_chern.py"],
        ["benchmarks/gpu_spin_chern.py", "--sizes", "8", "12", "16", "--configs", "20", "--W", "2.0"],
    ],
    # "chern_pilot": lambda_nu = 1.65 only, W 0..9 step 0.5, 100 samples, 8x8 and 12x12, with both
    # CPU baselines (PythTB as-is, and the same C_up algorithm in NumPy on all cores) (~15-25 min).
    "chern_pilot": [
        [*_TESTS, "tests/test_spin_chern.py"],
        ["analysis/sweep.py", "--size", "8", "--deltas", "1.65", "--samples", "100", "--W-step", "0.5",
         "--W-max", "9", "--cpu-baseline", "--out", "/kaggle/working/out/chern_pilot_8x8.csv"],
        ["analysis/sweep.py", "--size", "12", "--deltas", "1.65", "--samples", "100", "--W-step", "0.5",
         "--W-max", "9", "--cpu-baseline", "--out", "/kaggle/working/out/chern_pilot_12x12.csv"],
    ],
    # "sweep_12": 12x12, W 0..10 step 0.25, 200 samples per point, plus the PythTB CPU estimate
    # (~2-3 h: overnight).
    "sweep_12": [
        ["analysis/sweep.py", "--size", "12", "--samples", "200", "--W-step", "0.25",
         "--cpu-baseline", "--out", "/kaggle/working/out/sweep_12x12.csv"],
    ],
}
JOBS = PRESETS[PRESET]

WORK = Path("/kaggle/working")
PROJ = WORK / "proj"
OUT = WORK / "out"


def run(cmd: list[str], cwd: Path | None = None) -> int:
    start = time.time()
    print(f"\n[{time.strftime('%H:%M:%S')}] $ {' '.join(cmd)}", flush=True)
    # PYTHONUNBUFFERED: child prints appear in the Kaggle log live, not in one lump at the end
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    # check=False: a failing job is reported but doesn't stop the later jobs
    code = subprocess.run(cmd, check=False, cwd=cwd, text=True, env=env).returncode
    print(f"[{time.strftime('%H:%M:%S')}] exit {code} after {time.time() - start:.0f}s", flush=True)
    return code


def main() -> None:
    # 1. get the code from the private dataset. Kaggle usually unpacks code.tar.gz itself
    #    when it builds the dataset, so look for the unpacked tree first.
    unpacked = glob.glob("/kaggle/input/**/pythtb/__init__.py", recursive=True)
    if unpacked:
        shutil.copytree(Path(unpacked[0]).parent.parent, PROJ)  # read-only input -> writable copy
    else:
        tarball = glob.glob("/kaggle/input/**/code.tar.gz", recursive=True)[0]
        PROJ.mkdir(exist_ok=True)
        with tarfile.open(tarball) as t:
            t.extractall(PROJ, filter="data")
    print("code in", PROJ, sorted(p.name for p in PROJ.iterdir()), flush=True)

    # 2. install: the PythTB fork, then our engine (torch/numpy are already on Kaggle)
    pip = [sys.executable, "-m", "pip", "install", "-q"]
    run([*pip, "-e", str(PROJ)])
    run([*pip, "--no-deps", "-e", str(PROJ / "hackathon")])
    run([*pip, "threadpoolctl", "pytest"])

    # 3. run the jobs
    hackathon = PROJ / "hackathon"
    for job in JOBS:
        run([sys.executable, *job], cwd=hackathon)

    # 4. keep only the results (the log is saved by Kaggle automatically)
    OUT.mkdir(exist_ok=True)
    for f in (hackathon / "results").glob("*.csv"):
        shutil.copy(f, OUT / f.name)
    print("\nsaved:", sorted(p.name for p in OUT.iterdir()))
    shutil.rmtree(PROJ)


if __name__ == "__main__":
    main()
