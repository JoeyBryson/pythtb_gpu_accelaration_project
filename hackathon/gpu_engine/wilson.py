"""Batched pipeline, step 2: Wilson loops for every line of every configuration at once.

What PythTB does (wfarray.py, `berry_phase` -> `_berry_loop` -> `_wilson_loop`), one line
at a time in a Python loop:
  1. overlaps between neighbouring k-points along the line   M = <u(k)|u(k+dk)>
  2. unitary part of each overlap with an SVD                U = V @ Wh
  3. multiply them in order around the closed loop          W = U_0 @ U_1 @ ... @ U_{n-1}
  4. Wannier centres = -angle(eigenvalues of W) / 2pi

What we do differently (all lines and configurations in one batch):
  2. Newton-Schulz instead of SVD: only matrix multiplications, which GPUs are best at.
     Converges in a few steps because the overlaps are already close to unitary.
     Checked, and SVD used as a fallback for any matrix where it hasn't converged.
  3. Tree product instead of a sequential loop: multiply neighbouring pairs, then pairs of
     pairs, ... (log2(n) rounds). Matrix products are associative, so the order is kept.

Closing the loop at the zone boundary: the neighbour of the last k-point is the first one
plus a reciprocal lattice vector G, and in PythTB's convention u(k + G) = D u(k) with
D = diag(exp(-2 pi i G.tau)) (tau = orbital positions). Checked numerically to 1e-15 (21 Nov).

Shapes: C configs, nk_x x nk_y mesh, n = n_occ occupied states, N = all states (spin flattened)
    u      (C, nk_x, nk_y, n, N)   occupied states as rows
    M, U   (C, nk_x, nk_y, n, n)   one per link along k_y
    W      (C, nk_x, n, n)         one Wilson matrix per line
    wcc    (C, nk_x, n)            Wannier centres in [0, 1)
"""

from __future__ import annotations

import numpy as np
import torch
from pythtb import TBModel

# Newton-Schulz settings. Overlap singular values are ~0.75-1 on the meshes we use (see
# topology.RELIABLE_OVERLAP), where it converges quadratically; 12 steps is generous.
NS_MAX_STEPS = 12
NS_TOL = {torch.complex128: 1e-12, torch.complex64: 1e-5}


def boundary_phase(model: TBModel, G: tuple[float, float] = (0.0, 1.0)) -> np.ndarray:
    """D = exp(-2 pi i G.tau) for each state (spin flattened): u(k + G) = D * u(k). Shape (N,)."""
    tau = np.repeat(np.asarray(model.orb_vecs), 2, axis=0)  # same position for both spins
    return np.exp(-2j * np.pi * tau @ np.asarray(G))


def overlaps_along_ky(u: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
    """M[..., j] = <u(k_j)|u(k_{j+1})> along the k_y axis (dim -3), loop closed with `phase`.

    u: (C, nk_x, nk_y, n, N). Returns (C, nk_x, nk_y, n, n).
    """
    nxt = torch.roll(u, shifts=-1, dims=-3)  # u(k_{j+1}); last one wraps round to k_0 ...
    nxt[..., -1, :, :] = nxt[..., -1, :, :] * phase  # ... which is really k_0 + G
    return u.conj() @ nxt.transpose(-1, -2)


def unitary_part(M: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Closest unitary matrix to each M (the 'U' in M = U H), by Newton-Schulz:

        X <- X (3I - X^H X) / 2

    Stops early once every matrix has converged (checked every 2 steps from step 4: each
    check costs one extra product and, on a GPU, a sync). Returns (U, number of matrices
    that needed the SVD fallback).
    """
    eye = torch.eye(M.shape[-1], dtype=M.dtype, device=M.device)
    tol = NS_TOL[M.dtype] * M.shape[-1]
    X = M
    for step in range(1, NS_MAX_STEPS + 1):
        XhX = X.mH @ X
        if step >= 4 and step % 2 == 0:
            residual = torch.linalg.matrix_norm(XhX - eye, ord="fro")  # how far from unitary
            if bool((residual <= tol).all()):
                break
        X = 0.5 * X @ (3 * eye - XhX)
    # final convergence check: how far is each X from unitary?
    residual = torch.linalg.matrix_norm(X.mH @ X - eye, ord="fro")
    bad = residual > tol
    n_bad = int(bad.sum())
    if n_bad:  # rare: overlaps far from unitary (coarse mesh near a gap closing)
        V, _, Wh = torch.linalg.svd(M[bad], full_matrices=False)
        X[bad] = V @ Wh
    return X, n_bad


def unitary_part_svd(M: torch.Tensor) -> torch.Tensor:
    """PythTB's way: U = V @ Wh from the SVD. Used for comparison and as the fallback."""
    V, _, Wh = torch.linalg.svd(M, full_matrices=False)
    return V @ Wh


def ordered_product(U: torch.Tensor) -> torch.Tensor:
    """U_0 @ U_1 @ ... @ U_{m-1} along dim -3, by a tree of pairwise products.

    (..., m, n, n) -> (..., n, n). Each round multiplies neighbouring pairs in order, so
    the result equals the left-to-right product exactly (up to rounding).
    """
    while U.shape[-3] > 1:
        m = U.shape[-3]
        pairs = U[..., 0 : m - m % 2 : 2, :, :] @ U[..., 1 : m - m % 2 : 2, :, :]
        if m % 2:  # odd count: carry the last matrix to the next round, still at the end
            pairs = torch.cat([pairs, U[..., m - 1 : m, :, :]], dim=-3)
        U = pairs
    return U[..., 0, :, :]


def wannier_centres(
    u: torch.Tensor, model: TBModel, method: str = "newton-schulz", eig_method: str = "hermitian"
) -> tuple[torch.Tensor, dict]:
    """Wannier centres for every line of every configuration.

    u: (C, nk_x, nk_y, n_occ, N) occupied states from batched.solve_configs.
    Returns wcc (C, nk_x, n_occ) in [0, 1) on the CPU, sorted per line, and info
    (fallback count and seconds per sub-step).
    """
    on_gpu = u.is_cuda
    clock = _Clock(on_gpu)

    phase = torch.as_tensor(boundary_phase(model), dtype=u.dtype, device=u.device)
    M = overlaps_along_ky(u, phase)
    clock.lap("overlaps")
    if method == "newton-schulz":
        U, n_fallback = unitary_part(M)
    else:
        U, n_fallback = unitary_part_svd(M), 0
    clock.lap("unitary")
    W = ordered_product(U)  # (C, nk_x, n, n)
    clock.lap("product")
    if eig_method == "hermitian":
        evals, n_eig_fallback = unitary_eigenvalues(W)
        evals = evals.cpu()
    else:  # "general": the plain non-Hermitian eigensolver, on the CPU (slow on GPUs)
        evals, n_eig_fallback = torch.linalg.eigvals(W.cpu()), 0
    wcc = torch.remainder(-torch.angle(evals) / (2 * torch.pi), 1.0)
    wcc, _ = torch.sort(wcc, dim=-1)
    clock.lap("eigvals")
    info = {
        "ns_fallbacks": n_fallback, "eig_fallbacks": n_eig_fallback,
        "n_links": M.shape[:-2].numel(), **clock.laps,
    }
    return wcc, info


# Mixing weight for unitary_eigenvalues. Any fixed number not 0 works; an irrational-looking
# one makes accidental ties (two different phases giving the same mixed value) unlikely.
_MIX = 0.6180339887
EIG_TOL = {torch.complex128: 1e-10, torch.complex64: 1e-4}


def unitary_eigenvalues(W: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Eigenvalues of unitary matrices with a Hermitian eigensolver (fast and batched on GPUs).

    A unitary W has eigenvalues e^{i theta} and an orthonormal set of eigenvectors. The
    Hermitian matrix  H = Re-part + MIX * Im-part = (W + W^H)/2 + MIX (W - W^H)/(2i)
    has the SAME eigenvectors (eigenvalues cos theta + MIX sin theta), so we diagonalise H
    with `eigh` and read each eigenvalue of W back as v^H W v.

    The one way this can go wrong: two different phases giving the same value of
    cos + MIX sin, so eigh mixes their eigenvectors. We check every matrix
    (|W V - V diag(lambda)| small?) and recompute any that fail with the general solver.

    W: (..., n, n). Returns (eigenvalues (..., n) complex, number of matrices recomputed).
    """
    H = 0.5 * (W + W.mH) + _MIX * (W - W.mH) / 2j
    _, V = torch.linalg.eigh(H)
    lam = (V.conj() * (W @ V)).sum(dim=-2)  # diagonal of V^H W V, one per eigenvector
    residual = torch.linalg.matrix_norm(W @ V - V * lam.unsqueeze(-2), ord="fro")
    bad = residual > EIG_TOL[W.dtype] * W.shape[-1]
    n_bad = int(bad.sum())
    if n_bad:
        lam[bad] = torch.linalg.eigvals(W[bad].cpu()).to(lam.device)
    return lam, n_bad


class _Clock:
    """Lap timer for sub-steps; synchronises the GPU so the laps are real."""

    def __init__(self, on_gpu: bool):
        import time

        self._now, self.on_gpu, self.laps = time.perf_counter, on_gpu, {}
        self._last = self._sync_now()

    def _sync_now(self) -> float:
        if self.on_gpu:
            torch.cuda.synchronize()
        return self._now()

    def lap(self, name: str) -> None:
        now = self._sync_now()
        self.laps[f"t_{name}"] = now - self._last
        self._last = now
