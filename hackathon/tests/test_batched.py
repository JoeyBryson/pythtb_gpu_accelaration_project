"""Batched step 1 (H(k) + diagonalisation) vs the CPU reference, same seeds.

Tolerance: eigenvalues to 1e-10 in double precision (unique quantities; eigenvectors are
only defined up to phase, so we compare projectors instead of raw vectors).
"""

import numpy as np
import pytest
import torch

from gpu_engine import get_backend
from gpu_engine.batched import clean_hamiltonians, disorder_shifts, solve_configs
from gpu_engine.reference import clean_kane_mele, disordered_kane_mele, solve


@pytest.mark.parametrize("supercell", [1, 2])
def test_clean_hamiltonian_matches_pythtb_energies(supercell):
    model = clean_kane_mele(supercell)
    H = clean_hamiltonians(model, nk=6)
    ours = np.linalg.eigvalsh(H).reshape(6, 6, -1)
    ref = np.asarray(solve(model, 6).energies)
    np.testing.assert_allclose(ours, ref, atol=1e-10)


def test_disorder_shifts_are_the_same_random_numbers():
    model = clean_kane_mele(2)
    rng_ref = np.random.default_rng(42).uniform(-1.0, 1.0, size=model.norb)
    ours = disorder_shifts(model.norb, W=2.0, seeds=[42])[0]
    np.testing.assert_array_equal(ours[0::2], rng_ref)  # spin up
    np.testing.assert_array_equal(ours[1::2], rng_ref)  # spin down


@pytest.mark.parametrize("precision", ["double", "single"])
def test_batched_disorder_matches_reference_config_by_config(precision):
    be = get_backend(precision)
    n, nk, W, seeds = 2, 6, 3.0, [0, 1, 2, 3]
    energies, u_occ = solve_configs(clean_kane_mele(n), nk, W, seeds, be)
    tol = 1e-10 if precision == "double" else 1e-4

    for c, s in enumerate(seeds):
        model = disordered_kane_mele(n, W, np.random.default_rng(s))
        wfa = solve(model, nk)
        ref_e = np.asarray(wfa.energies)
        np.testing.assert_allclose(energies[c].cpu().numpy(), ref_e, atol=tol * 10)

        # projector onto occupied states is unique (no phase ambiguity): compare that
        ref_u = np.asarray(wfa[...]).reshape(nk, nk, 4 * n * n, -1)[:, :, : model.norb]
        ref_P = np.swapaxes(ref_u, -1, -2) @ np.conj(ref_u)
        u = u_occ[c].to(torch.complex128).cpu().numpy()
        ours_P = np.swapaxes(u, -1, -2) @ np.conj(u)
        np.testing.assert_allclose(ours_P, ref_P, atol=tol * 10)
