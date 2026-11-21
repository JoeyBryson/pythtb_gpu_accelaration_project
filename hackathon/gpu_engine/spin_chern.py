"""Spin-up Chern number C_up for many disorder configurations at once (lambda_R = 0 only).

Why: at lambda_R = 0 spin S_z is conserved (our Anderson disorder is spin-independent, so it
stays conserved). H is then block diagonal in spin, and the spin-up block alone is a Chern
problem with an integer C_up; Z2 = C_up mod 2. On 21 Nov this was far more robust than
gap-tracking Z2 near the TAI transition (analysis/stage0_chern_up.py: same answer on
nk = 6 and 10 for 137/140 configs; gap tracking disagreed in up to 6/20).

How (no gap tracking, no Newton-Schulz, no Wilson-matrix eigenvalues):
  1. spin-up block of H(k) = H_clean(k) + diag(disorder), half the size of the full H
  2. one batched eigh; occupied = lower half of the up-block spectrum (half filling)
  3. overlaps M = <u(k)|u(k + dk_y)> along k_y, loop closed with PythTB's boundary phase
  4. sum of the up-spin Wannier centres on each k_x line = -sum_j arg det M_j / 2 pi
     (one number per line: the phase of det of the Wilson loop), via batched slogdet
  5. C_up = winding of that number as k_x goes once round the zone; each step is wrapped
     into (-1/2, 1/2], so a step near 1/2 is ambiguous -> max_step is the reliability check

Mirrors analysis/stage0_chern_up.py::c_up (the validated CPU version) line by line.

Shapes: C configs, nk x nk mesh (k = i/nk, k_x first), n_orb = 2 L^2 orbitals (spin-up),
        n = n_orb / 2 occupied
    H        (C, nk*nk, n_orb, n_orb)  complex
    u        (C, nk, nk, n, n_orb)     occupied states as rows
    M        (C, nk, nk, n, n)         one per link along k_y
    p        (C, nk)                   sum of up-spin Wannier centres per k_x line, mod 1
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch
from pythtb import TBModel

from .backend import Backend, sync
from .batched import configs_per_chunk
from .reference import clean_kane_mele

# Reliability: wrapped k_x steps of the Wannier-centre sum must stay below this.
# From 140 configs (L = 8, 12; W = 1-8; 21 Nov): every config whose C_up changed between
# nk = 6 and 10 had max_step >= 0.46 at nk = 6; 0.4 flags all of them (and ~5% stable ones).
STEP_THRESHOLD = 0.4


def k_grid(nk: int) -> np.ndarray:
    """k = (i/nk, j/nk), k_x-major, the same grid as analysis/stage0_chern_up.py. (nk*nk, 2)."""
    k = np.arange(nk) / nk
    return np.stack(np.meshgrid(k, k, indexing="ij"), -1).reshape(-1, 2)


def clean_up_hamiltonians(model: TBModel, nk: int) -> np.ndarray:
    """Spin-up block of the clean H(k) on the grid. Raises if spin is not conserved.

    PythTB's flattened order is (orbital, spin): orb0 up, orb0 down, orb1 up, ... so the
    up block is every second row and column. Returns (nk*nk, n_orb, n_orb) complex.
    """
    H = model.hamiltonian(k_grid(nk), flatten_spin_axis=True)
    if np.abs(H[:, 0::2, 1::2]).max() > 1e-12:
        raise ValueError("spin-up block needs S_z conservation: build the model with rashba=0")
    return H[:, 0::2, 0::2]


def up_disorder(norb: int, W: float, seeds: list[int]) -> np.ndarray:
    """Random on-site energies, the SAME numbers as reference.disordered_kane_mele for the
    same seed (uniform in [-W/2, W/2], one per orbital). Returns (C, norb)."""
    return np.stack([np.random.default_rng(s).uniform(-W / 2, W / 2, size=norb) for s in seeds])


def boundary_phase_up(model: TBModel) -> np.ndarray:
    """u(k + G_y) = D u(k) in PythTB's convention, D = exp(-2 pi i tau_y). Shape (n_orb,)."""
    return np.exp(-2j * np.pi * np.asarray(model.orb_vecs) @ np.array([0.0, 1.0]))


def _det_phase(M: torch.Tensor) -> torch.Tensor:
    """sign = det M / |det M| for a batch of matrices (complex, on M's device).

    On CUDA, slogdet uses a batched LU factorisation. With PyTorch's default MAGMA backend that
    crashed on a Kaggle T4 (torch 2.11 + CUDA 13, 21 Nov: "illegal memory access in
    magma_cgetrf_batched", complex64, 8x8 supercell). cuSOLVER is used for this call instead;
    the previous backend is restored afterwards. A crash like that corrupts the CUDA context,
    so it can't be caught and retried in the same process.
    """
    if not M.is_cuda:
        return torch.linalg.slogdet(M)[0]
    previous = torch.backends.cuda.preferred_linalg_library()
    torch.backends.cuda.preferred_linalg_library("cusolver")
    try:
        return torch.linalg.slogdet(M)[0]
    finally:
        torch.backends.cuda.preferred_linalg_library(previous)


def chern_from_states(u: torch.Tensor, phase: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """C_up and max wrapped step per configuration.

    u: (C, nk, nk, n, n_orb) occupied states as rows; phase: (n_orb,).
    Returns C_up (C,) int64 and max_step (C,) real.
    """
    nxt = torch.roll(u, shifts=-1, dims=2)  # u(k_y + dk); the last one wraps round ...
    nxt[:, :, -1] = nxt[:, :, -1] * phase  # ... and is really u(k_y = 0) + G
    M = u.conj() @ nxt.transpose(-1, -2)  # (C, nk, nk, n, n)
    sign = _det_phase(M)  # det M = sign * exp(logabsdet); sign is complex
    p = -torch.angle(sign).sum(dim=2) / (2 * torch.pi)  # (C, nk): sum of centres, mod 1
    d = torch.roll(p, shifts=-1, dims=1) - p  # step k_x -> k_x + dk, closing the loop
    d = torch.remainder(d + 0.5, 1.0) - 0.5  # wrap each step into (-1/2, 1/2]
    return torch.round(d.sum(dim=1)).to(torch.int64), d.abs().amax(dim=1)


def solve_chern(
    H0: torch.Tensor, shifts: torch.Tensor, nk: int, phase: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One batch: H0 (K, n_orb, n_orb), shifts (C, n_orb) -> C_up, max_step, gap (each (C,))."""
    H = H0.unsqueeze(0) + torch.diag_embed(shifts.to(H0.dtype)).unsqueeze(1)  # (C, K, n_orb, n_orb)
    e, v = torch.linalg.eigh(H)
    n = H.shape[-1] // 2  # half filling of the up block
    gap = (e[..., n] - e[..., n - 1]).amin(dim=1)  # direct gap, smallest over k
    C, n_orb = H.shape[0], H.shape[-1]
    u = v[..., :n].transpose(-1, -2).reshape(C, nk, nk, n, n_orb)
    c_up, max_step = chern_from_states(u, phase)
    return c_up, max_step, gap


@dataclass
class SpinChernResult:
    """Per-configuration answers (one entry per seed) plus timings for the whole batch."""

    seeds: list[int]
    c_up: np.ndarray  # (C,) integer spin-up Chern number
    max_step: np.ndarray  # (C,) largest wrapped k_x step (reliability check)
    gap_up: np.ndarray  # (C,) smallest direct gap of the up block over the mesh
    nk_used: np.ndarray  # (C,) mesh that gave the final answer (nk, or refine_nk if refined)
    timings: dict = field(default_factory=dict)

    @property
    def z2(self) -> np.ndarray:
        return np.abs(self.c_up) % 2

    @property
    def reliable(self) -> np.ndarray:
        return (self.max_step <= STEP_THRESHOLD) & (self.gap_up > 0)


def _run(H0, norb, W, seeds, nk, phase, backend, chunk):
    out = {"c": [], "s": [], "g": []}
    for start in range(0, len(seeds), chunk):
        part = seeds[start : start + chunk]
        d = torch.as_tensor(up_disorder(norb, W, part), dtype=backend.rdtype, device=backend.device)
        c, s, g = solve_chern(H0, d, nk, phase)
        out["c"] += c.cpu().tolist()
        out["s"] += s.double().cpu().tolist()
        out["g"] += g.double().cpu().tolist()
    return np.array(out["c"]), np.array(out["s"]), np.array(out["g"])


def run_spin_chern(
    supercell: int, W: float, nk: int, seeds: list[int], backend: Backend,
    delta: float, soc: float = 0.3, t: float = 1.0, refine_nk: int | None = 10,
    budget_gb: float = 4.0,
) -> SpinChernResult:
    """C_up for every seed at Kane-Mele (delta, soc, t, rashba = 0), L = supercell.

    Configs failing the reliability check at `nk` are re-run at `refine_nk` (adaptive
    refinement; None to skip). Refinement time counts as pipeline time.
    """
    model = clean_kane_mele(supercell, delta=delta, t=t, soc=soc, rashba=0.0)
    norb = model.norb

    def device_inputs(n):
        H0 = torch.as_tensor(clean_up_hamiltonians(model, n), dtype=backend.cdtype, device=backend.device)
        ph = torch.as_tensor(boundary_phase_up(model), dtype=backend.cdtype, device=backend.device)
        return H0, ph, configs_per_chunk(n, norb, backend, budget_gb)

    sync(backend)
    t0 = time.perf_counter()
    H0, ph, chunk = device_inputs(nk)
    c, s, g = _run(H0, norb, W, seeds, nk, ph, backend, chunk)
    nk_used = np.full(len(seeds), nk)
    sync(backend)
    t1 = time.perf_counter()

    n_refined = 0
    if refine_nk is not None:
        bad = np.where((s > STEP_THRESHOLD) | (g <= 0))[0]
        n_refined = len(bad)
        if n_refined:
            del H0
            H0, ph, chunk = device_inputs(refine_nk)
            c2, s2, g2 = _run(H0, norb, W, [seeds[i] for i in bad], refine_nk, ph, backend, chunk)
            c[bad], s[bad], g[bad], nk_used[bad] = c2, s2, g2, refine_nk
    sync(backend)
    t2 = time.perf_counter()

    timings = {"first_pass": t1 - t0, "refine": t2 - t1, "total": t2 - t0, "n_refined": n_refined}
    return SpinChernResult(list(seeds), c, s, g, nk_used, timings)
