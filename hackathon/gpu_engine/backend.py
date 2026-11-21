"""Device and precision selection, plus timing that is correct for async GPU calls.

Usage:
    be = get_backend(precision="double")      # picks CUDA if available, else CPU
    H = torch.as_tensor(H_np, dtype=be.cdtype, device=be.device)
    with timed("eigh", be) as t:
        evals, evecs = torch.linalg.eigh(H)
    print(t.seconds)

We deliberately skip Apple MPS: it has no complex128 and no complex eigh, so laptops
run the CPU path (fine for correctness tests, meaningless for benchmarks).
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass

import torch

_PRECISIONS = {
    "double": (torch.complex128, torch.float64),
    "single": (torch.complex64, torch.float32),
}

# Tolerances for comparing final physical outputs (Wannier centres) with the CPU
# reference, as agreed in CLAUDE.md. Revisit once the physics side confirms.
TOLERANCE = {"double": 1e-8, "single": 1e-4}


@dataclass(frozen=True)
class Backend:
    device: torch.device
    precision: str  # "double" or "single"

    @property
    def cdtype(self) -> torch.dtype:
        return _PRECISIONS[self.precision][0]

    @property
    def rdtype(self) -> torch.dtype:
        return _PRECISIONS[self.precision][1]

    @property
    def tol(self) -> float:
        return TOLERANCE[self.precision]

    @property
    def is_gpu(self) -> bool:
        return self.device.type == "cuda"

    def describe(self) -> str:
        if self.is_gpu:
            name = torch.cuda.get_device_name(self.device)
            mem = torch.cuda.get_device_properties(self.device).total_memory / 1e9
            return f"{name} ({mem:.0f} GB), {self.precision} precision"
        return f"CPU, {self.precision} precision"


def get_backend(precision: str = "double", device: str | None = None) -> Backend:
    """Return a Backend. device=None picks 'cuda' when available, else 'cpu'."""
    if precision not in _PRECISIONS:
        raise ValueError(f"precision must be one of {list(_PRECISIONS)}, got {precision!r}")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return Backend(device=torch.device(device), precision=precision)


def sync(backend: Backend | None = None) -> None:
    """Block until queued GPU work has finished. No-op on CPU."""
    if torch.cuda.is_available() and (backend is None or backend.is_gpu):
        torch.cuda.synchronize()


class _Timer:
    seconds: float = float("nan")


@contextmanager
def timed(label: str | None = None, backend: Backend | None = None, verbose: bool = True):
    """Wall-clock timer that synchronises the GPU before starting and stopping."""
    t = _Timer()
    sync(backend)
    start = time.perf_counter()
    try:
        yield t
    finally:
        sync(backend)
        t.seconds = time.perf_counter() - start
        if verbose and label:
            print(f"{label}: {t.seconds:.4f} s")
