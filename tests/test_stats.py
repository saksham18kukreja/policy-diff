import math

import pytest

from policy_diff import stats


def test_wilson_known_value():
    lo, hi = stats.wilson(7, 10)
    assert lo == pytest.approx(0.3968, abs=1e-3)
    assert hi == pytest.approx(0.8922, abs=1e-3)
    assert stats.wilson(0, 0) == (0.0, 1.0)
    assert stats.wilson(0, 10)[0] == 0.0 and stats.wilson(10, 10)[1] == 1.0


def test_mcnemar_exact():
    assert stats.mcnemar_exact(6, 0) == pytest.approx(2 / 2 ** 6)
    assert stats.mcnemar_exact(0, 0) == 1.0
    assert stats.mcnemar_exact(3, 3) == 1.0


def test_fisher_exact_tea_tasting():
    # Fisher's lady tasting tea: 3 of 4 vs 1 of 4
    assert stats.fisher_exact(3, 4, 1, 4) == pytest.approx(0.4857, abs=1e-4)
    assert stats.fisher_exact(0, 10, 0, 10) == 1.0


def test_stratified_reduces_to_mcnemar_for_pairs():
    pairs = [(1, 1, 0, 1)] * 6 + [(0, 1, 1, 1)] * 1 + [(1, 1, 1, 1)] * 3
    assert stats.stratified_exact(pairs) == pytest.approx(stats.mcnemar_exact(6, 1))


def test_stratified_reduces_to_fisher_for_one_stratum():
    assert stats.stratified_exact([(23, 30, 1, 10)]) == pytest.approx(stats.fisher_exact(23, 30, 1, 10))


def test_stratified_ignores_uninformative_strata():
    informative = [(3, 3, 0, 1), (2, 3, 0, 1)]
    noise = [(3, 3, 1, 1), (0, 3, 0, 1), (5, 5, 0, 0)]
    assert stats.stratified_exact(informative + noise) == pytest.approx(stats.stratified_exact(informative))
    assert stats.stratified_exact([]) == 1.0


def test_stratified_is_symmetric_in_a_and_b():
    s = [(3, 3, 0, 1), (2, 3, 1, 1), (0, 3, 1, 1)]
    flipped = [(kb, nb, ka, na) for ka, na, kb, nb in s]
    assert stats.stratified_exact(s) == pytest.approx(stats.stratified_exact(flipped))


def test_benjamini_hochberg():
    q = stats.benjamini_hochberg([0.01, 0.04, 0.03, 0.20])
    assert q == pytest.approx([0.04, 0.0533, 0.0533, 0.20], abs=1e-3)
    assert stats.benjamini_hochberg([]) == []


def test_labels():
    assert stats.label(0.001, 0.01) == stats.LIKELY
    assert stats.label(0.03, 0.2) == stats.POSSIBLE
    assert stats.label(0.5, 1.0) == stats.NOISE
    assert math.isclose(stats.LIKELY_Q, 0.05)
