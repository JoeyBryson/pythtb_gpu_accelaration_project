"""Ensemble analysis helpers (analysis/ensemble.py, analysis/ensemble_plots.py).

Exact checks (integers / identities), so tolerances are float rounding only (1e-12).
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.ensemble import disorder_statistics, pinned_shifts, sublattice_index
from analysis.ensemble_plots import is_separated, per_sample_critical_W, wilson_interval
from gpu_engine.batched import disorder_shifts


def test_sublattice_index_matches_onsite_signs():
    sub = sublattice_index(3, delta=1.7)
    assert sub.shape == (18,) and sub.sum() == 9


def test_pinned_shifts_have_zero_sublattice_means_and_same_fluctuations():
    norb, seeds = 18, [0, 1, 2]
    pinned = pinned_shifts(norb, 4.0, seeds)
    plain = disorder_shifts(norb, 4.0, seeds)
    assert np.array_equal(pinned[:, 0::2], pinned[:, 1::2])  # still spin independent
    eps, sub = pinned[:, 0::2], np.arange(norb) % 2
    np.testing.assert_allclose(eps[:, sub == 0].mean(1), 0, atol=1e-12)
    np.testing.assert_allclose(eps[:, sub == 1].mean(1), 0, atol=1e-12)
    # only a constant per sublattice was removed
    diff = plain[:, 0::2] - eps
    np.testing.assert_allclose(diff[:, sub == 0].std(1), 0, atol=1e-12)


def test_shared_seeds_scale_one_pattern_with_W():
    a = disorder_shifts(8, 1.0, [5])
    b = disorder_shifts(8, 3.0, [5])
    np.testing.assert_allclose(b, 3.0 * a, atol=1e-12)


def test_disorder_statistics_imbalance():
    eps = np.array([[1.0, -1.0, 1.0, -1.0]])  # A = +1, B = -1
    st = disorder_statistics(eps, np.array([0, 1, 0, 1]))
    assert st["eps_A_minus_B"][0] == 2.0 and st["eps_mean"][0] == 0.0


def test_wilson_interval_contains_p_and_handles_edges():
    lo, hi = wilson_interval(np.array([0, 5, 10]), np.array([10, 10, 10]))
    assert lo[0] == 0 and hi[2] == 1
    assert lo[1] < 0.5 < hi[1]


def test_per_sample_critical_W_uses_resolved_calls_only():
    # seed 0: resolved flip between W=1 and W=3 (the call at W=2 is unresolved);
    # seed 1: never flips
    d = {
        "W": np.array([0, 1, 2, 3, 0, 1, 2, 3], float),
        "seed": np.array([0, 0, 0, 0, 1, 1, 1, 1]),
        "z2": np.array([1, 1, 0, 0, 1, 1, 1, 1]),
        "reliable": np.array([1, 1, 0, 1, 1, 1, 1, 1]),
    }
    W_star, width, z0 = per_sample_critical_W(d, np.ones(8, bool))
    assert W_star[0] == 2.0 and width[0] == 2.0
    assert np.isnan(W_star[1]) and list(z0) == [1, 1]


def test_sublattice_index_accepts_negative_delta():
    assert sublattice_index(2, delta=-1.0).sum() == 4


def test_is_separated():
    assert is_separated(np.array([0.1, 0.2, 0.5, 0.6]), np.array([1, 1, 0, 0]))
    assert not is_separated(np.array([0.1, 0.5, 0.2, 0.6]), np.array([1, 1, 0, 0]))


def test_rank_tests_and_imbalance_detection():
    from analysis.sample_size import _average_ranks, imbalance_detected, rank_z

    # fully separated, 3 vs 3: U = 0, mean 4.5, sd sqrt(9 * 7 / 12) -> z = -1.964
    assert abs(rank_z(np.array([1.0, 2, 3]), np.array([4.0, 5, 6])) + 4.5 / np.sqrt(5.25)) < 1e-12
    np.testing.assert_allclose(_average_ranks(np.array([3.0, 1, 3, 2])), [3.5, 1, 3.5, 2])
    rng = np.random.default_rng(0)
    # 3 vs 3 perfectly separated: exact one-sided p = 1/20 = 0.05, NOT significant at 2.5%
    assert not imbalance_detected(np.arange(6.0), np.array([1, 1, 1, 0, 0, 0]), rng)
    # 6 vs 6 perfectly separated: p = 1/924
    assert imbalance_detected(np.arange(12.0), np.array([1] * 6 + [0] * 6), rng)
    assert not imbalance_detected(np.arange(12.0), np.array([0] * 6 + [1] * 6), rng)


def test_onset_rule():
    from analysis.sample_size import onset

    Ws = np.array([0.0, 1, 2, 3])
    assert onset(Ws, np.array([0.0, 0.25, 0.75, 1])) == 1.5
    assert onset(Ws, np.array([0.0, 0.5, 1.0, 1])) == 1.0  # interpolate across a touch of 1/2
    assert np.isnan(onset(Ws, np.array([0.0, np.nan, 1.0, 1])))  # no bridging a gap
    assert np.isnan(onset(Ws, np.array([0.0, 0.2, 0.5, 0.4])))  # touch only, no crossing


def test_prob_miss_all():
    from analysis.sample_size import prob_miss_all

    assert abs(prob_miss_all(10, 1, 1) - 0.9) < 1e-12
    assert abs(prob_miss_all(1000, 46, 10) - np.prod([(954 - i) / (1000 - i) for i in range(10)])) < 1e-12
