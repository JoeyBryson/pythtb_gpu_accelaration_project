"""Batched Wilson loop (step 2) vs PythTB's berry_phase, configuration by configuration.

Wannier centres live on a circle [0, 1), and their order within a line is arbitrary, so
we compare each line's sorted set with the periodic distance.
Tolerances: 1e-8 for complex128, 1e-4 for complex64 (CLAUDE.md testing rules).
"""

import numpy as np
import pytest
import torch

from gpu_engine import get_backend
from gpu_engine.batched import solve_configs
from gpu_engine.reference import (
    clean_kane_mele,
    disordered_kane_mele,
    half_bz_flow,
    periodic_distance,
    solve,
)
from gpu_engine.reference import wannier_centres as pythtb_wannier_centres
from gpu_engine.topology import z2_from_wannier_flow
from gpu_engine.wilson import ordered_product, unitary_part, unitary_part_svd, wannier_centres

TOL = {"double": 1e-8, "single": 1e-4}


def assert_same_centres(ours: np.ndarray, ref: np.ndarray, tol: float) -> None:
    """Both (nk_x, n): compare each line as a set of points on the circle."""
    ours = np.sort(ours % 1.0, axis=-1)
    ref = np.sort(ref % 1.0, axis=-1)
    # sorting on a circle can split a pair across 0/1; align by trying the cyclic shifts
    for line_ours, line_ref in zip(ours, ref, strict=True):
        best = min(
            periodic_distance(np.roll(line_ours, s), line_ref).max() for s in range(len(line_ref))
        )
        assert best < tol, f"Wannier centres differ by {best:.2e} (tol {tol:.0e})"


def test_tree_product_equals_sequential_product():
    U = torch.randn(3, 7, 4, 4, dtype=torch.complex128)  # 7 = odd, exercises the carry
    seq = U[:, 0]
    for i in range(1, 7):
        seq = seq @ U[:, i]
    torch.testing.assert_close(ordered_product(U), seq)


def test_newton_schulz_matches_svd_on_near_unitary_matrices():
    Q, _ = torch.linalg.qr(torch.randn(50, 8, 8, dtype=torch.complex128))
    M = 0.9 * Q + 0.02 * torch.randn(50, 8, 8, dtype=torch.complex128)  # like real overlaps
    U_ns, n_fallback = unitary_part(M)
    torch.testing.assert_close(U_ns, unitary_part_svd(M), atol=1e-10, rtol=0)
    assert n_fallback == 0


def test_newton_schulz_falls_back_to_svd_when_it_cannot_converge():
    M = torch.randn(5, 6, 6, dtype=torch.complex128) * 0.05  # far from unitary
    U, n_fallback = unitary_part(M)
    assert n_fallback > 0
    torch.testing.assert_close(U, unitary_part_svd(M), atol=1e-10, rtol=0)


@pytest.mark.parametrize("dtype", [torch.complex128, torch.complex64])
def test_unitary_eigenvalues_match_general_solver(dtype):
    """Includes exactly degenerate pairs, as Kramers pairs give at k = 0 and pi."""
    from gpu_engine.wilson import unitary_eigenvalues

    torch.manual_seed(0)
    Q, _ = torch.linalg.qr(torch.randn(40, 16, 16, dtype=torch.complex128))
    phases = torch.rand(40, 16, dtype=torch.float64) * 2 * torch.pi
    phases[:20, 1::2] = phases[:20, 0::2]  # half the batch: every eigenvalue doubly degenerate
    W = (Q * torch.exp(1j * phases).unsqueeze(-2)) @ Q.mH
    lam, _ = unitary_eigenvalues(W.to(dtype))
    got = torch.sort(torch.remainder(torch.angle(lam.to(torch.complex128)), 2 * torch.pi)).values
    want = torch.sort(torch.remainder(phases, 2 * torch.pi)).values
    tol = 1e-9 if dtype == torch.complex128 else 1e-4
    assert (got - want).abs().max() < tol


def test_large_single_precision_matrix_stays_accurate():
    """Regression (21 Nov): at n = 512 in single precision the eigenvalues came out ~1e-3 off
    while passing a matrix-wide check. Dense spectrum + single-precision input must still give
    phases within the single-precision tolerance (1e-4) of the exact ones."""
    from gpu_engine.wilson import unitary_eigenvalues

    torch.manual_seed(1)
    n = 384
    phases = torch.sort(torch.rand(2, n, dtype=torch.float64) * 2 * torch.pi).values
    Q, _ = torch.linalg.qr(torch.randn(2, n, n, dtype=torch.complex128))
    W = ((Q * torch.exp(1j * phases).unsqueeze(-2)) @ Q.mH).to(torch.complex64)
    exact = torch.linalg.eigvals(W.to(torch.complex128))  # same (rounded) matrices, exactly
    lam, stats = unitary_eigenvalues(W)
    assert stats["eig_cpu"] == 0
    got = torch.sort(torch.remainder(torch.angle(lam), 2 * torch.pi)).values
    want = torch.sort(torch.remainder(torch.angle(exact), 2 * torch.pi)).values
    assert (got - want).abs().max() < 1e-4


def _unitary_with_phases(phases: list[float]) -> tuple[torch.Tensor, torch.Tensor]:
    p = torch.tensor([phases], dtype=torch.float64)
    Q, _ = torch.linalg.qr(torch.randn(1, len(phases), len(phases), dtype=torch.complex128))
    return (Q * torch.exp(1j * p).unsqueeze(-2)) @ Q.mH, p


def _sorted_phases(lam: torch.Tensor) -> torch.Tensor:
    return torch.sort(torch.remainder(torch.angle(lam), 2 * torch.pi)).values


def test_mixing_failure_is_fixed_by_the_gpu_retry():
    """Two phases with equal cos + MIX*sin get mixed by eigh: caught, and fixed by the retry
    (different MIX, double precision) without needing the slow CPU solver."""
    import math

    from gpu_engine.wilson import _MIX, unitary_eigenvalues

    phi = math.atan(_MIX)  # cos(t) + MIX sin(t) is symmetric about t = phi
    W, phases = _unitary_with_phases([phi + 0.7, phi - 0.7, 2.0, 4.0])
    lam, stats = unitary_eigenvalues(W)
    assert stats == {"eig_retried": 1, "eig_cpu": 0}
    assert (_sorted_phases(lam) - _sorted_phases(torch.exp(1j * phases))).abs().max() < 1e-9


def test_cpu_fallback_when_both_mixings_fail():
    """Ties under BOTH mixing values in one matrix: only then the CPU solver is used."""
    import math

    from gpu_engine.wilson import _MIX, _MIX_RETRY, unitary_eigenvalues

    p1, p2 = math.atan(_MIX), math.atan(_MIX_RETRY)
    W, phases = _unitary_with_phases([p1 + 0.5, p1 - 0.5, p2 + 1.9, p2 - 1.9])
    lam, stats = unitary_eigenvalues(W)
    assert stats == {"eig_retried": 1, "eig_cpu": 1}
    assert (_sorted_phases(lam) - _sorted_phases(torch.exp(1j * phases))).abs().max() < 1e-9


@pytest.mark.parametrize("supercell", [1, 2])
def test_clean_model_matches_pythtb(supercell):
    be = get_backend("double")
    model, nk = clean_kane_mele(supercell), 8
    _, u = solve_configs(model, nk, 0.0, [0], be)
    wcc, info = wannier_centres(u, model)
    ref = pythtb_wannier_centres(solve(model, nk), model.norb)
    assert_same_centres(wcc[0].cpu().numpy(), ref, TOL["double"])
    assert info["ns_fallbacks"] == 0


@pytest.mark.parametrize("precision", ["double", "single"])
@pytest.mark.parametrize("method", ["newton-schulz", "svd"])
def test_disordered_configs_match_pythtb_one_by_one(precision, method):
    be = get_backend(precision)
    n, nk, W, seeds = 2, 8, 3.0, [0, 1, 2]
    _, u = solve_configs(clean_kane_mele(n), nk, W, seeds, be)
    wcc, _ = wannier_centres(u, clean_kane_mele(n), method=method)
    for c, s in enumerate(seeds):
        model = disordered_kane_mele(n, W, np.random.default_rng(s))
        ref = pythtb_wannier_centres(solve(model, nk), model.norb)
        assert_same_centres(wcc[c].double().cpu().numpy(), ref, TOL[precision])


@pytest.mark.parametrize("precision", ["double", "single"])
def test_z2_matches_reference_config_by_config(precision):
    """The number we actually care about: same Z2 as the CPU pipeline for each config."""
    be = get_backend(precision)
    n, nk, seeds = 2, 8, [0, 1, 2, 3]
    for W, expected in [(0.5, 1), (20.0, 0)]:
        _, u = solve_configs(clean_kane_mele(n), nk, W, seeds, be)
        wcc, _ = wannier_centres(u, clean_kane_mele(n))
        for c, s in enumerate(seeds):
            model = disordered_kane_mele(n, W, np.random.default_rng(s))
            wfa = solve(model, nk)
            flow = half_bz_flow(wfa, wcc[c].double().cpu().numpy())
            assert z2_from_wannier_flow(flow) == expected
