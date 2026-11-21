"""Batched spin-up Chern number (gpu_engine.spin_chern) vs the validated CPU version
(analysis/stage0_chern_up.py::c_up, plain NumPy + PythTB), configuration by configuration.

C_up is an integer, so it must match exactly. max_step is a smooth quantity: compare with
1e-8 (double) and 1e-3 (single; it is a sum of ~nk angles, each good to ~1e-5).
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from gpu_engine import get_backend
from gpu_engine.reference import clean_kane_mele, disordered_kane_mele
from gpu_engine.spin_chern import (
    clean_up_hamiltonians,
    run_spin_chern,
    up_disorder,
)

PARAMS = {"delta": 1.65, "t": 1.0, "soc": 0.3, "rashba": 0.0}
STEP_TOL = {"double": 1e-8, "single": 1e-3}

_spec = importlib.util.spec_from_file_location(
    "stage0_chern_up", Path(__file__).resolve().parent.parent / "analysis" / "stage0_chern_up.py"
)
_cpu = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cpu)
cpu_c_up = _cpu.c_up  # (model, nk) -> (C_up, max_step, gap_up)


@pytest.mark.parametrize(("delta", "expected"), [(1.45, 1), (1.65, 0)])
def test_clean_unit_cell(delta, expected):
    """Clean limit either side of 3*sqrt(3)*lambda_SO = 1.5588: |C_up| = 1 topological, 0 trivial."""
    res = run_spin_chern(1, 0.0, 12, [0], get_backend("double"), delta=delta, refine_nk=None)
    assert abs(int(res.c_up[0])) == expected


def test_rashba_is_rejected():
    with pytest.raises(ValueError):
        clean_up_hamiltonians(clean_kane_mele(2, delta=1.65, rashba=0.25), 4)


def test_disorder_is_the_reference_disorder():
    """Up block of our H equals the up block of PythTB's disordered model, same seed."""
    L, W, s, nk = 2, 3.0, 7, 4
    ours = clean_up_hamiltonians(clean_kane_mele(L, **PARAMS), nk) + np.diag(
        up_disorder(clean_kane_mele(L, **PARAMS).norb, W, [s])[0]
    )
    model = disordered_kane_mele(L, W, np.random.default_rng(s), **PARAMS)
    from gpu_engine.spin_chern import k_grid

    ref = model.hamiltonian(k_grid(nk), flatten_spin_axis=True)[:, 0::2, 0::2]
    np.testing.assert_allclose(ours, ref, atol=1e-12)


@pytest.mark.parametrize("precision", ["double", "single"])
def test_matches_cpu_config_by_config(precision):
    """Same seeds -> same C_up as the CPU version (L = 4, W around the transition)."""
    L, nk, seeds = 4, 6, list(range(6))
    be = get_backend(precision)
    for W in (2.0, 3.0, 8.0):
        res = run_spin_chern(L, W, nk, seeds, be, delta=1.65, refine_nk=None)
        for i, s in enumerate(seeds):
            C, step, gap = cpu_c_up(disordered_kane_mele(L, W, np.random.default_rng(s), **PARAMS), nk)
            assert res.c_up[i] == C, (W, s)
            assert abs(res.max_step[i] - step) < STEP_TOL[precision], (W, s)
            assert abs(res.gap_up[i] - gap) < 1e-4


def test_refinement_reruns_only_flagged_configs():
    res = run_spin_chern(4, 2.0, 6, list(range(6)), get_backend("double"), delta=1.65, refine_nk=10)
    flagged_first = res.timings["n_refined"]
    assert (res.nk_used == 10).sum() == flagged_first
    assert set(res.nk_used) <= {6, 10}


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA GPU")
def test_gpu_matches_cpu_backend():
    seeds = list(range(8))
    a = run_spin_chern(4, 2.5, 6, seeds, get_backend("double", "cpu"), delta=1.65, refine_nk=None)
    b = run_spin_chern(4, 2.5, 6, seeds, get_backend("double", "cuda"), delta=1.65, refine_nk=None)
    np.testing.assert_array_equal(a.c_up, b.c_up)
