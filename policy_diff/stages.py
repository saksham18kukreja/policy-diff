"""Where episodes fail: stage outcomes, failure classes and failure signatures.

Works with whatever a run logged:

  stages with timing    first_true / final / ever_lost (simulator predicates, detectors)
  stages without timing reached only (milestones such as "grasp", "lift")
  no stages             the failure reason, or the last harness event

Failure classes (failed episodes only):

  none           no stage reached
  partial        some stages reached, none lost
  undone         a stage was reached and then lost (or not held at the end)
  no stage data  the episode logged no stages

A failure signature is a short, deterministic description of how an episode
failed, e.g. "stopped after grasp [timeout]" for ordered milestones, or
"missing on moka_pot_1 [out of steps]" otherwise. Failures that share a
signature are the same failure mode, so counting signatures is Tier-1 failure
clustering: cheap, explainable, and comparable across runs.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field
from typing import Optional

from .schema import Episode, Run

NONE, PARTIAL, UNDONE, NO_DATA = "none", "partial", "undone", "no stage data"
CLASSES = (NONE, PARTIAL, UNDONE, NO_DATA)


def is_undone(stage) -> bool:
    return bool(stage.reached and (stage.ever_lost or stage.final is False))


def failure_class(e: Episode) -> Optional[str]:
    if e.success:
        return None
    if not e.stages:
        return NO_DATA
    if any(is_undone(s) for s in e.stages):
        return UNDONE
    return PARTIAL if any(s.reached for s in e.stages) else NONE


FAILURE_WORDS = re.compile(r"time[sd]?[ -]?out|fail|error|abort|drop|lost|collision|stuck|no retry|exceed", re.I)


def _reason(e: Episode, max_length: Optional[int] = None) -> Optional[str]:
    """Why the episode failed: the logged failure reason; else the last event if it reads like
    a failure (numbers masked, so "step 1 timed out after 160 steps" -> "step # timed out after # steps");
    else "out of steps" if the episode used the task's full length."""
    if e.failure:
        return e.failure
    if e.events and FAILURE_WORDS.search(e.events[-1][1]):
        return re.sub(r"\d+(\.\d+)?", "#", e.events[-1][1])
    if max_length is not None and e.length is not None and e.length >= max_length:
        return "out of steps"
    return None


def signature(e: Episode, max_length: Optional[int] = None) -> Optional[str]:
    """How the episode failed, as a short string; None for successes.

    max_length: the task's step limit, if known (episodes that reach it ran out of steps).
    """
    if e.success:
        return None
    reason = _reason(e, max_length)
    tail = f" [{reason}]" if reason else ""
    if not e.stages:
        return reason or "failed"
    reached = [s.reached for s in e.stages]
    undone = [s.name for s in e.stages if is_undone(s)]
    k = sum(reached)
    if reached == [True] * k + [False] * (len(reached) - k):  # progress through the stages in order
        head = f"stopped after {e.stages[k - 1].name}" if k else "no stage reached"
    else:  # name only what is missing; reached stages (and preconditions) are implied
        head = "missing " + ", ".join(s.name for s in e.stages if not s.reached)
    if undone:
        head += "; undone: " + ", ".join(undone)
    return head + tail


def _max_length(episodes: list[Episode]) -> Optional[int]:
    lengths = [e.length for e in episodes if e.length is not None]
    return max(lengths) if lengths else None


@dataclass
class StageStats:
    name: str
    n: int  # episodes that logged this stage
    reached: int
    final: Optional[int]  # held at the end (None when not logged)
    lost: Optional[int]
    median_first: Optional[float]  # median step at which it was first reached

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class TaskAnalysis:
    task: str
    n: int
    successes: int
    stages: list[StageStats]
    classes: dict[str, int]  # failure class -> count
    signatures: list[tuple[str, int]]  # most common first

    def to_dict(self) -> dict:
        return {"task": self.task, "n": self.n, "successes": self.successes,
                "stages": [s.to_dict() for s in self.stages], "classes": self.classes,
                "signatures": [list(s) for s in self.signatures]}


def stage_stats(episodes: list[Episode]) -> list[StageStats]:
    order: list[str] = []
    for e in episodes:
        for s in e.stages:
            if s.name not in order:
                order.append(s.name)
    out = []
    for name in order:
        st = [s for e in episodes for s in e.stages if s.name == name]
        finals = [s.final for s in st if s.final is not None]
        losts = [s.ever_lost for s in st if s.ever_lost is not None]
        firsts = [s.first_true for s in st if s.first_true is not None]
        out.append(StageStats(name, len(st), sum(s.reached for s in st),
                              sum(finals) if finals else None, sum(losts) if losts else None,
                              statistics.median(firsts) if firsts else None))
    return out


def analyze_task(task: str, episodes: list[Episode], max_length: Optional[int] = None) -> TaskAnalysis:
    """max_length defaults to the longest episode of the task."""
    max_length = max_length if max_length is not None else _max_length(episodes)
    classes = {c: 0 for c in CLASSES}
    sigs: dict[str, int] = {}
    for e in episodes:
        c = failure_class(e)
        if c is None:
            continue
        classes[c] += 1
        s = signature(e, max_length)
        sigs[s] = sigs.get(s, 0) + 1
    return TaskAnalysis(task, len(episodes), sum(e.success for e in episodes), stage_stats(episodes),
                        {c: k for c, k in classes.items() if k},
                        sorted(sigs.items(), key=lambda kv: (-kv[1], kv[0])))


def analyze(run: Run) -> list[TaskAnalysis]:
    return [analyze_task(t, eps) for t, eps in run.by_task().items()]


@dataclass
class Shift:
    """A stage, or a failure signature, whose rate changed between runs."""
    kind: str  # "stage" (reached) or "signature"
    name: str
    a: int
    a_n: int
    b: int
    b_n: int
    p: float
    q: float = 1.0
    label: str = ""

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class TaskBreakdown:
    task: str
    a: TaskAnalysis
    b: TaskAnalysis
    shifts: list[Shift] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"task": self.task, "a": self.a.to_dict(), "b": self.b.to_dict(),
                "shifts": [s.to_dict() for s in self.shifts]}


def breakdown(task: str, a: list[Episode], b: list[Episode]) -> TaskBreakdown:
    """What moved within a task: each stage's reach rate and each failure signature's rate, tested like success."""
    from . import stats
    from .compare import _strata

    def test(kind, name, pred):
        # same stratified exact test as task success, on an episode-level indicator
        flag = lambda eps: [Episode(e.task, bool(pred(e)), init=e.init, seed=e.seed) for e in eps]
        fa, fb = flag(a), flag(b)
        strata, _ = _strata(fa, fb)
        return Shift(kind, name, sum(e.success for e in fa), len(fa), sum(e.success for e in fb), len(fb),
                     stats.stratified_exact(strata))

    max_length = _max_length(a + b)
    ta, tb = analyze_task(task, a, max_length), analyze_task(task, b, max_length)
    shifts = []
    names = [s.name for s in ta.stages] + [s.name for s in tb.stages if s.name not in {x.name for x in ta.stages}]
    for name in names:
        shifts.append(test("stage", name, lambda e, n=name: any(s.name == n and s.reached for s in e.stages)))
    for sig in dict.fromkeys([s for s, _ in ta.signatures] + [s for s, _ in tb.signatures]):
        shifts.append(test("signature", sig, lambda e, g=sig: signature(e, max_length) == g))
    for s, q in zip(shifts, stats.benjamini_hochberg([s.p for s in shifts])):
        s.q, s.label = q, stats.label(s.p, q)
    return TaskBreakdown(task, ta, tb, shifts)


def reason_shifts(a: list[Episode], b: list[Episode]) -> list[Shift]:
    """How often each failure reason occurs across all the given tasks, A vs B.

    A regression in configuration (a timeout that no longer fits) shows up as
    one reason rising across many tasks, even when no single task changes
    detectably. Stratified by task and initial state, FDR-corrected across reasons.
    """
    from . import stats
    from .compare import _strata

    lengths: dict[str, int] = {}
    for e in a + b:
        if e.length is not None:
            lengths[e.task] = max(lengths.get(e.task, 0), e.length)
    reason = lambda e: None if e.success else _reason(e, lengths.get(e.task))
    names = [r for r in dict.fromkeys(map(reason, a + b)) if r]
    shifts = []
    for r in names:
        flag = lambda eps: [Episode(e.task, reason(e) == r, init=e.init, seed=e.seed) for e in eps]
        fa, fb = flag(a), flag(b)
        strata, _ = _strata(fa, fb)
        shifts.append(Shift("reason", r, sum(e.success for e in fa), len(fa), sum(e.success for e in fb),
                            len(fb), stats.stratified_exact(strata)))
    for s, q in zip(shifts, stats.benjamini_hochberg([s.p for s in shifts])):
        s.q, s.label = q, stats.label(s.p, q)
    return sorted(shifts, key=lambda s: s.p)
