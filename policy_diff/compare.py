"""Compare two runs task by task: pair episodes, test, correct, label."""

from __future__ import annotations

import json
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
    test: str  # exact test stratified by initial state (or seed), or unpaired
    p: float
    a_unpaired: int = 0  # episodes with no counterpart in the other run (still used by the test)
    b_unpaired: int = 0
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
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"a": self.a, "b": self.b, "tasks": [t.to_dict() for t in self.tasks],
                "overall": self.overall.to_dict() if self.overall else None,
                "only_in_a": self.only_in_a, "only_in_b": self.only_in_b, "warnings": self.warnings}


def _strata(a: list[Episode], b: list[Episode]) -> tuple[list[tuple[int, int, int, int]], int]:
    """(a_success, a_n, b_success, b_n) per (task, initial state), and how many strata are matched.

    Without an init id, episodes with the same seed form a stratum (a seeded
    reset gives the same start). Episodes whose stratum has no counterpart in
    the other run are pooled into one unmatched stratum per task, so they still
    count (as an unpaired comparison) instead of being dropped. Strata depend
    only on ids, never on outcomes, so the conditional test stays exact.
    """
    def key(e):
        if e.init is not None:
            return (e.task, "init", e.init)
        if e.seed is not None:
            return (e.task, "seed", e.seed)
        return (e.task, "none", None)

    keys_a, keys_b = {key(e) for e in a}, {key(e) for e in b}
    matched = (keys_a & keys_b) - {k for k in keys_a | keys_b if k[1] == "none"}
    g: dict[tuple, list[int]] = {}
    for side, eps in ((0, a), (1, b)):
        for e in eps:
            k = key(e)
            c = g.setdefault(k if k in matched else (e.task, "unmatched"), [0, 0, 0, 0])
            c[2 * side] += e.success
            c[2 * side + 1] += 1
    return [tuple(c) for c in g.values()], len(matched)


def _diff(task: str, a: list[Episode], b: list[Episode]) -> TaskDiff:
    pairs, a_left, b_left = pair_episodes(a, b)
    a_wins = sum(x.success and not y.success for x, y in pairs)
    b_wins = sum(y.success and not x.success for x, y in pairs)
    ka, kb = sum(e.success for e in a), sum(e.success for e in b)
    strata, matched = _strata(a, b)
    if not matched:
        test = "exact, unpaired"
    elif any(e.init is not None for e in a + b):
        test = "exact, by initial state"
    else:
        test = "exact, by seed"
    return TaskDiff(task, ka, len(a), kb, len(b), len(pairs), a_wins, b_wins, test,
                    stats.stratified_exact(strata), len(a_left), len(b_left))


def _fingerprint(e: Episode) -> tuple:
    """An episode's outcome without its task name: identical fingerprints mean identical logs."""
    stages = tuple((st.name, st.reached, st.first_true, st.final) for st in e.stages)
    extra = tuple(sorted((k, json.dumps(v, sort_keys=True)) for k, v in e.extra.items()))
    return (e.init, e.seed, e.success, e.length, stages, extra)


def integrity_warnings(a: Run, b: Run) -> list[str]:
    """Signs that two runs are not two independent evaluations."""
    out = []
    if a.episodes and sorted(map(_fingerprint, a.episodes), key=repr) == sorted(map(_fingerprint, b.episodes), key=repr):
        out.append(f"{a.run_id} and {b.run_id} have identical per-episode results ({len(a.episodes)} episodes"
                   f"{', ignoring task names' if a.tasks() != b.tasks() else ''}): the same evaluation logged twice?")
    shared = {e.video for e in a.episodes if e.video} & {e.video for e in b.episodes if e.video}
    if shared:
        out.append(f"{len(shared)} video paths appear in both runs, e.g. {sorted(shared)[0]}")
    return out


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
    return Comparison(a.run_id, b.run_id, diffs, overall, only_in_a=[t for t in ta if t not in tb],
                      only_in_b=[t for t in tb if t not in ta], warnings=integrity_warnings(a, b))
