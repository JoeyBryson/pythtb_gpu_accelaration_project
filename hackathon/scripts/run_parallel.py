"""Run several job lists at once, one per GPU (e.g. Kaggle's "GPU T4 x2").

Each positional argument is one GPU's job list: python commands (relative to hackathon/),
separated by ';'. Lists run in parallel (list i gets CUDA_VISIBLE_DEVICES=i); the commands
inside a list run one after another. Output lines are prefixed with [gpu i].

  python scripts/run_parallel.py "analysis/sweep.py --size 16 ... ; analysis/sweep.py --size 8 ..." \
                                 "analysis/sweep.py --size 14 ... ; analysis/sweep.py --size 12 ..."

If fewer GPUs are visible than lists, lists share GPUs round-robin (still correct, just slower).
Exit code: the largest exit code of any command.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
import threading
import time


HACKATHON = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_list(i: int, gpu: int, commands: list[str], codes: list[int]) -> None:
    try:
        _run_list(i, gpu, commands, codes)
    except Exception as e:  # noqa: BLE001  (report it, and make the overall exit code non-zero)
        print(f"[gpu {gpu}] job list failed: {e!r}", flush=True)
        codes[i] = max(codes[i], 1)


def _run_list(i: int, gpu: int, commands: list[str], codes: list[int]) -> None:
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONUNBUFFERED": "1"}
    for cmd in commands:
        t0 = time.time()
        print(f"[gpu {gpu}] $ python {cmd}", flush=True)
        p = subprocess.Popen([sys.executable, *shlex.split(cmd)], cwd=HACKATHON, env=env,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in p.stdout:
            print(f"[gpu {gpu}] {line}", end="", flush=True)
        p.wait()
        codes[i] = max(codes[i], p.returncode)
        print(f"[gpu {gpu}] exit {p.returncode} after {time.time() - t0:.0f}s", flush=True)


def main() -> None:
    lists = [[c.strip() for c in arg.split(";") if c.strip()] for arg in sys.argv[1:]]
    if not lists:
        raise SystemExit(__doc__)
    try:
        import torch

        n_gpu = max(1, torch.cuda.device_count())
    except ImportError:  # no torch here (e.g. a laptop test): one "GPU" slot
        n_gpu = 1
    print(f"{len(lists)} job lists on {n_gpu} visible GPU(s)", flush=True)
    codes = [0] * len(lists)
    threads = [threading.Thread(target=run_list, args=(i, i % n_gpu, cmds, codes)) for i, cmds in enumerate(lists)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    sys.exit(max(codes))


if __name__ == "__main__":
    main()
