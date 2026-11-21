"""Smoke tests: the environment works and the CPU reference behaves as expected."""

import numpy as np
import pytest
import torch

from gpu_engine import get_backend
from gpu_engine.reference import clean_kane_mele, periodic_distance, solve, wannier_centres


def test_reference_wannier_centres_shape_and_range():
    model = clean_kane_mele()
    wfa = solve(model, nk=21)
    wc = wannier_centres(wfa, n_occ=2)
    assert wc.shape == (21, 2)
    assert np.all(np.isfinite(wc))


def test_periodic_distance_wraps():
    assert periodic_distance(0.95, 0.05) == pytest.approx(0.1)
    assert periodic_distance(0.2, 0.2) == 0.0


@pytest.mark.parametrize("precision", ["double", "single"])
def test_torch_eigh_matches_numpy(precision):
    """Eigenvalues (unique, unlike eigenvectors) agree with NumPy on this backend."""
    be = get_backend(precision)
    rng = np.random.default_rng(1)
    a = rng.standard_normal((16, 32, 32)) + 1j * rng.standard_normal((16, 32, 32))
    h = a + np.conj(np.swapaxes(a, -1, -2))
    ref = np.linalg.eigvalsh(h)
    got = torch.linalg.eigvalsh(torch.as_tensor(h, dtype=be.cdtype, device=be.device))
    # single precision: relative error on O(10) eigenvalues, so scale tolerance
    tol = 1e-10 if precision == "double" else 1e-3
    np.testing.assert_allclose(got.cpu().numpy(), ref, atol=tol * np.abs(ref).max())


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA GPU")
def test_gpu_is_picked_by_default():
    assert get_backend().is_gpu
