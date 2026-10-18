"""Compare two runs task by task: pair episodes, test, correct, label."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional

from . import stats
from .schema import Episode, Run


def pair_episodes(a: list[Episode], b: list[Episode]):
    """Match episodes with the same task and initial state.

    Within a (task, init) group, episodes with equal seeds are matched first,
    then the rest in seed order. Episodes without an init are matched only on
    an equal seed. Returns (pairs, a_unpaired, b_unpaired).
    """
    def groups(eps):
        g: dict[tuple, list[Episode]] = {}
        for e in eps:
            key = (e.task, e.init, None) if e.init is not None else (e.task, None, e.seed)
            g.setdefault(key, []).append(e)
        return g

    ga, gb = groups(a), groups(b)
    pairs, a_left, b_left = [], [], []
    for key in ga.keys() | gb.keys():
        xs, ys = list(ga.get(key, [])), list(gb.get(key, []))
        if key[1] is None and key[2] is None:  # no init, no seed: nothing to pair on
            a_left += xs
            b_left += ys
            continue
        for x in list(xs):
            y = next((y for y in ys if y.seed == x.seed), None)
            if y is not None:
                pairs.append((x, y))
                xs.remove(x)
                ys.remove(y)
        xs.sort(key=lambda e: (e.seed is None, e.seed))
        ys.sort(key=lambda e: (e.seed is None, e.seed))
        n = min(len(xs), len(ys))
        pairs += list(zip(xs[:n], ys[:n]))
        a_left += xs[n:]
        b_left += ys[n:]
    return pairs, a_left, b_left


@dataclass
class TaskDiff:
    task: str
    a_success: int
    a_n: int
    b_success: int
    b_n: int
    pairs: int  # matched episodes
    a_only_wins: int  # pairs where A succeeded and B failed
    b_only_wins: int  # pairs where B succeeded and A failed
    test: str  # exact test stratified by initial state, or unpaired without inits
    p: float
    q: float = 1.0  # FDR-adjusted across the compared tasks
    label: str = stats.NOISE

    @property
    def a_rate(self) -> float:
        return self.a_success / self.a_n if self.a_n else 0.0

    @property
    def b_rate(self) -> float:
        return self.b_success / self.b_n if self.b_n else 0.0

    @property
    def delta(self) -> float:
        return self.b_rate - self.a_rate

    @property
    def direction(self) -> str:
        if self.label == stats.NOISE or self.delta == 0:
            return "unchanged"
        return "improved" if self.delta > 0 else "regressed"

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update(a_rate=self.a_rate, b_rate=self.b_rate, delta=self.delta, direction=self.direction,
                 a_ci=stats.wilson(self.a_success, self.a_n), b_ci=stats.wilson(self.b_success, self.b_n))
        return d


@dataclass
class Comparison:
    a: str
    b: str
    tasks: list[TaskDiff]
    overall: Optional[TaskDiff]
    only_in_a: list[str] = field(default_factory=list)
    only_in_b: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"a": self.a, "b": self.b, "tasks": [t.to_dict() for t in self.tasks],
                "overall": self.overall.to_dict() if self.overall else None,
                "only_in_a": self.only_in_a, "only_in_b": self.only_in_b}


def _strata(a: list[Episode], b: list[Episode]) -> list[tuple[int, int, int, int]]:
    """(a_success, a_n, b_success, b_n) per (task, initial state); no init: one stratum per task."""
    g: dict[tuple, list[int]] = {}
    for side, eps in ((0, a), (1, b)):
        for e in eps:
            c = g.setdefault((e.task, e.init), [0, 0, 0, 0])
            c[2 * side] += e.success
            c[2 * side + 1] += 1
    return [tuple(c) for c in g.values()]


def _diff(task: str, a: list[Episode], b: list[Episode]) -> TaskDiff:
    pairs, _, _ = pair_episodes(a, b)
    a_wins = sum(x.success and not y.success for x, y in pairs)
    b_wins = sum(y.success and not x.success for x, y in pairs)
    ka, kb = sum(e.success for e in a), sum(e.success for e in b)
    stratified = any(e.init is not None for e in a + b)
    test = "exact, by initial state" if stratified else "exact, unpaired"
    return TaskDiff(task, ka, len(a), kb, len(b), len(pairs), a_wins, b_wins, test,
                    stats.stratified_exact(_strata(a, b)))


def compare(a: Run, b: Run) -> Comparison:
    ta, tb = a.by_task(), b.by_task()
    common = [t for t in ta if t in tb]
    diffs = [_diff(t, ta[t], tb[t]) for t in common]
    for d, q in zip(diffs, stats.benjamini_hochberg([d.p for d in diffs])):
        d.q = q
        d.label = stats.label(d.p, q)
    overall = None
    if common:
        ea = [e for t in common for e in ta[t]]
        eb = [e for t in common for e in tb[t]]
        overall = _diff("overall", ea, eb)
        overall.q = overall.p  # a single test
        overall.label = stats.label(overall.p, overall.q)
    return Comparison(a.run_id, b.run_id, diffs, overall,
                      only_in_a=[t for t in ta if t not in tb], only_in_b=[t for t in tb if t not in ta])
