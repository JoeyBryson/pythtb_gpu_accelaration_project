"""Batched pipeline, step 1: build H(k) for many disorder configurations at once and
diagonalise them all in one call. Runs on CPU or GPU (PyTorch), same code.

The trick that makes disorder cheap to batch
--------------------------------------------
Anderson disorder only adds a random number to each orbital's on-site energy. On-site
terms don't depend on k (in PythTB's convention their phase factor is exactly 1), so for
configuration c:

    H[c, k] = H_clean[k] + diag(d_c)

We build H_clean(k) once with PythTB, then every configuration is the same stack plus a
different diagonal: one broadcast addition on the device, no Python loop over
configurations and no rebuilding of models.

Array shapes (C = configurations, K = k-points, N = 4 n^2 states, spin flattened):
    H_clean  (K, N, N)      complex
    d        (C, N)         real, random on-site shift per state
    H        (C, K, N, N)   complex
"""

from __future__ import annotations

import numpy as np
import torch
from pythtb import Mesh, TBModel

from .backend import Backend


def k_mesh(nk: int) -> np.ndarray:
    """The same gamma-centred nk x nk mesh the CPU reference uses. Returns (nk, nk, 2)."""
    mesh = Mesh(["k", "k"])
    mesh.build_grid(shape=(nk, nk), gamma_centered=True)
    return np.asarray(mesh.points)


def clean_hamiltonians(model: TBModel, nk: int) -> np.ndarray:
    """H_clean(k) at every mesh point, from PythTB (built once, on the CPU).

    Returns (nk*nk, N, N) complex, k-points in the mesh's row-major order.
    """
    k = k_mesh(nk).reshape(-1, 2)
    return model.hamiltonian(k, flatten_spin_axis=True)


def disorder_shifts(norb: int, W: float, seeds: list[int]) -> np.ndarray:
    """Random on-site energies for each configuration, identical to the CPU reference.

    Same random numbers as `reference.disordered_kane_mele` for the same seed: one value
    per orbital, uniform in [-W/2, W/2], the same for both spins. Returns (C, 2*norb), in
    PythTB's flattened (orbital, spin) order: orb0 up, orb0 down, orb1 up, ...
    """
    per_orbital = np.stack(
        [np.random.default_rng(s).uniform(-W / 2, W / 2, size=norb) for s in seeds]
    )
    return np.repeat(per_orbital, 2, axis=1)  # same shift for spin up and down


def batched_hamiltonians(H_clean: torch.Tensor, shifts: torch.Tensor) -> torch.Tensor:
    """H[c, k] = H_clean[k] + diag(shifts[c]).  (K,N,N) + (C,N) -> (C,K,N,N)."""
    return H_clean.unsqueeze(0) + torch.diag_embed(shifts.to(H_clean.dtype)).unsqueeze(1)


def solve(H: torch.Tensor, n_occ: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Diagonalise every matrix in the batch in one call.

    H: (..., N, N). Returns
      energies: (..., N) real, ascending
      u_occ:    (..., n_occ, N) complex, occupied states as rows (PythTB's layout)
    """
    energies, vecs = torch.linalg.eigh(H)  # vecs: eigenvectors are columns
    u_occ = vecs[..., :n_occ].transpose(-1, -2)
    return energies, u_occ


def configs_per_chunk(nk: int, N: int, backend: Backend, budget_gb: float = 4.0) -> int:
    """How many configurations fit in one batch without running out of GPU memory.

    One configuration's Hamiltonians take nk^2 * N^2 * (bytes per complex number). eigh needs
    extra workspace for eigenvectors and scratch, so we allow ~4x that per configuration.
    budget_gb is deliberately well under a T4's 16 GB.
    """
    bytes_per_complex = 16 if backend.precision == "double" else 8
    per_config = 4 * nk * nk * N * N * bytes_per_complex
    return max(1, int(budget_gb * 1e9 // per_config))


def solve_configs(
    model: TBModel, nk: int, W: float, seeds: list[int], backend: Backend,
    chunk: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Whole step 1 for C configurations of one clean model, `chunk` configurations per batch
    (default: as many as fit in memory, see configs_per_chunk).

    Returns energies (C, nk, nk, N) and occupied states (C, nk, nk, n_occ, N) on the device.
    """
    n_occ = model.norb  # half filling
    H0 = torch.as_tensor(clean_hamiltonians(model, nk), dtype=backend.cdtype, device=backend.device)
    N = H0.shape[-1]
    chunk = chunk or configs_per_chunk(nk, N, backend)

    all_e, all_u = [], []
    for start in range(0, len(seeds), chunk):  # usually just one or a few big batches
        part = seeds[start : start + chunk]
        d = torch.as_tensor(
            disorder_shifts(model.norb, W, part), dtype=backend.rdtype, device=backend.device
        )
        e, u = solve(batched_hamiltonians(H0, d), n_occ)
        all_e.append(e)
        all_u.append(u)
    C = len(seeds)
    energies = torch.cat(all_e).reshape(C, nk, nk, N)
    u_occ = torch.cat(all_u).reshape(C, nk, nk, n_occ, N)
    return energies, u_occ
