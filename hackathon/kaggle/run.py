"""Runs ON Kaggle: unpack our code, install it, run the JOBS below on the GPU, save results.

Edit JOBS to change what a GPU run does, then `bash kaggle/gpu_run.sh` from hackathon/.
Everything in hackathon/results/*.csv at the end comes back to your laptop.
"""

import glob
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

# Commands to run, from inside hackathon/. Edit freely.
JOBS = [
    ["scripts/check_env.py"],
    ["-m", "pytest", "-q", "-p", "no:cacheprovider"],
    ["benchmarks/gpu_step1.py", "--sizes", "4", "8", "12", "--configs", "20"],
]

WORK = Path("/kaggle/working")
PROJ = WORK / "proj"
OUT = WORK / "out"


def run(cmd: list[str], cwd: Path | None = None) -> int:
    print(f"\n$ {' '.join(cmd)}", flush=True)
    # check=False: a failing job is reported but doesn't stop the later jobs
    code = subprocess.run(cmd, check=False, cwd=cwd, text=True).returncode
    print(f"[exit {code}]", flush=True)
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
