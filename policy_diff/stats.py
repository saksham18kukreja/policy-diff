"""Small-sample statistics for success rates. Standard library only.

Robot evaluations have 10-50 episodes per task, so everything here is exact
or built for small n: Wilson intervals, an exact test stratified by initial
state (which is the exact McNemar test for one-to-one pairs and Fisher's
exact test for a single stratum), and Benjamini-Hochberg across tasks.
"""

from __future__ import annotations

import math
from typing import Sequence

LIKELY_Q = 0.05  # FDR-adjusted p at or below this: likely real
POSSIBLE_P = 0.10  # unadjusted p at or below this: possible

LIKELY, POSSIBLE, NOISE = "likely real", "possible", "noise"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for k successes out of n."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value.

    b: pairs where A succeeded and B failed; c: the reverse. Concordant pairs
    carry no information about the difference.
    """
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def fisher_exact(k1: int, n1: int, k2: int, n2: int) -> float:
    """Two-sided Fisher exact p-value for k1/n1 vs k2/n2 (unpaired)."""
    k, n = k1 + k2, n1 + n2
    if n1 == 0 or n2 == 0 or k in (0, n):
        return 1.0
    denom = math.comb(n, k)

    def prob(x):
        return math.comb(n1, x) * math.comb(n2, k - x) / denom

    observed = prob(k1)
    lo, hi = max(0, k - n2), min(k, n1)
    return min(1.0, sum(p for p in map(prob, range(lo, hi + 1)) if p <= observed * (1 + 1e-9)))


def stratified_exact(strata: Sequence[tuple[int, int, int, int]]) -> float:
    """Two-sided exact test of A vs B, conditioning on each stratum.

    strata: (a_success, a_n, b_success, b_n) per initial state. Given each
    stratum's total successes, B's success count is hypergeometric under "no
    difference"; the statistic is B's total over strata. Strata with only one
    run, or where every episode succeeded or every one failed, carry no
    information. One-to-one pairs give the exact McNemar test; one stratum
    gives Fisher's exact test.
    """
    dist = {0: 1.0}  # pmf of B's successes over the informative strata
    observed = 0
    for ka, na, kb, nb in strata:
        k, n = ka + kb, na + nb
        if na == 0 or nb == 0 or k in (0, n):
            continue
        observed += kb - max(0, k - na)
        denom = math.comb(n, k)
        pmf = [math.comb(nb, x) * math.comb(na, k - x) / denom for x in range(max(0, k - na), min(k, nb) + 1)]
        new: dict[int, float] = {}
        for s, ps in dist.items():
            for j, pj in enumerate(pmf):
                new[s + j] = new.get(s + j, 0.0) + ps * pj
        dist = new
    p_obs = dist.get(observed, 0.0)
    return min(1.0, sum(p for p in dist.values() if p <= p_obs * (1 + 1e-9)))


def benjamini_hochberg(pvalues: Sequence[float]) -> list[float]:
    """FDR-adjusted p-values (q-values), in the input order."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    q = [1.0] * m
    running = 1.0
    for rank in range(m, 0, -1):
        i = order[rank - 1]
        running = min(running, pvalues[i] * m / rank)
        q[i] = min(1.0, running)
    return q


def label(p: float, q: float) -> str:
    """likely real: survives FDR correction; possible: significant on its own; else noise."""
    if q <= LIKELY_Q:
        return LIKELY
    if p <= POSSIBLE_P:
        return POSSIBLE
    return NOISE
