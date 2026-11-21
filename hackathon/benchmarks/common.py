"""Shared helpers for benchmarks: hardware info, k-mesh rule, saving results.

Kept separate from the scenarios so that CPU and GPU benchmarks record results the
same way and can be compared row for row.
"""

from __future__ import annotations

import csv
import os
import platform
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_info

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


# ---------------------------------------------------------------- environment

if platform.system() == "Darwin":
    # numpy on Apple Accelerate raises spurious divide/overflow/invalid flags inside
    # det/slogdet/matmul. Checked 21 Nov at 4x4, nk=31: Wilson matrices unitary to 1e-13,
    # |det| = 1, all Wannier centres finite. Filter only these, only on macOS.
    for _msg in ("divide by zero", "overflow", "invalid value"):
        warnings.filterwarnings("ignore", message=f"{_msg} encountered in (det|slogdet|matmul)")


def blas_info() -> str:
    """Which linear-algebra library numpy uses, and with how many threads."""
    pools = [
        f"{i['internal_api']}:{i['num_threads']}t"
        for i in threadpool_info()
        if i.get("user_api") == "blas"
    ]
    if pools:
        return " ".join(pools)
    # Apple Accelerate isn't visible to threadpoolctl; fall back to numpy's build info
    return np.show_config(mode="dicts")["Build Dependencies"]["blas"]["name"]


def hardware_info() -> dict[str, str]:
    """What the numbers were measured on. A benchmark without this is meaningless."""
    return {
        "host": platform.node(),
        "cpu": platform.processor() or platform.machine(),
        "os": platform.platform(),
        "n_cpus": str(os.cpu_count()),
        "blas": blas_info(),
        "numpy": np.__version__,
    }


# ---------------------------------------------------------------- k-mesh rule


def scaled_nk(supercell: int, nk_unit_cell: int = 48, minimum: int = 6) -> int:
    """k-points per direction for an n x n supercell.

    A supercell n times bigger has a Brillouin zone n times smaller, so it needs about
    n times fewer k-points for the same resolution: nk ~ nk_unit_cell / n.
    Rounded to an even number, because the Z2 check needs k = pi on the mesh.

    But the Z2 check pulls the other way: bigger supercells have more Wannier centres
    crowded on the ring, so they need *enough* k_x steps. A clean 12x12 on a 4x4 mesh
    gave the wrong Z2 (21 Nov), hence minimum = 6. Always check ConfigResult.z2_reliable.

    To confirm: whether 6 holds near the transition and for supercells above 16x16.
    """
    nk = 2 * round(nk_unit_cell / supercell / 2)
    return max(minimum, nk)


# ---------------------------------------------------------------- saving results


def save_rows(scenario: str, rows: list[dict]) -> Path:
    """Append rows to results/<scenario>.csv (one file per scenario, header written once)."""
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"{scenario}.csv"
    hw = hardware_info()
    stamp = datetime.now().isoformat(timespec="seconds")
    rows = [{"timestamp": stamp, "scenario": scenario, **row, **hw} for row in rows]

    fields = list(rows[0])
    old_rows: list[dict] = []
    if path.exists():
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            old_fields = list(reader.fieldnames or [])
            if old_fields != fields:  # columns changed: rewrite with the union of columns,
                old_rows = list(reader)  # so no value ends up under the wrong heading
                fields = old_fields + [k for k in fields if k not in old_fields]

    if old_rows or not path.exists():
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields, restval="")
            writer.writeheader()
            writer.writerows(old_rows + rows)
    else:  # same columns: just append
        with path.open("a", newline="") as f:
            csv.DictWriter(f, fieldnames=fields, restval="").writerows(rows)
    return path


_START = time.time()


def log(msg: str) -> None:
    """Progress line with wall-clock time and time since the script started, flushed at once
    (so it shows up live in the Kaggle log instead of in one lump at the end)."""
    elapsed = time.time() - _START
    print(f"[{time.strftime('%H:%M:%S')} +{elapsed:6.0f}s] {msg}", flush=True)


class Progress:
    """Prints 'done/total, elapsed, ETA' every `every` items (and at the end)."""

    def __init__(self, label: str, total: int, every: int | None = None):
        self.label, self.total, self.done = label, total, 0
        self.every = every or max(1, total // 10)  # ~10 updates per loop
        self.t0 = time.time()

    def step(self, n: int = 1) -> None:
        self.done += n
        if self.done % self.every == 0 or self.done == self.total:
            dt = time.time() - self.t0
            eta = dt / self.done * (self.total - self.done)
            log(f"{self.label}: {self.done}/{self.total} configs, {dt:.0f}s elapsed, ~{eta:.0f}s left")


def print_header(scenario: str, description: str) -> None:
    hw = hardware_info()
    print(f"=== {scenario}: {description}")
    print(f"    on {hw['cpu']}, {hw['n_cpus']} cpus, BLAS {hw['blas']}, numpy {hw['numpy']}\n",
          flush=True)
