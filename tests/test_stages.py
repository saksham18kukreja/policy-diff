import pytest

from policy_diff import Episode, Run, Stage, stats
from policy_diff.cli import main
from policy_diff.schema import save_run
from policy_diff.stages import NO_DATA, NONE, PARTIAL, UNDONE, analyze, breakdown, failure_class, signature

MILESTONES = ["reach", "grasp", "lift"]


def milestones(k, success=False, failure=None, init=None):
    return Episode("pick", success, init=init, failure=failure,
                   stages=[Stage(n, reached=i < k) for i, n in enumerate(MILESTONES)])


def timed(first, final, lost, success=False, events=(), length=520):
    st = [Stage(n, f, fin, l) for n, f, fin, l in zip(["a", "b", "pre"], first, final, lost)]
    return Episode("t", success, stages=st, events=list(events), length=length)


def test_failure_classes():
    assert failure_class(milestones(3, success=True)) is None
    assert failure_class(milestones(0)) == NONE
    assert failure_class(milestones(2)) == PARTIAL
    assert failure_class(timed([5, None, 1], [False, False, True], [True, False, False])) == UNDONE
    assert failure_class(Episode("t", False)) == NO_DATA


def test_signatures_for_ordered_progress():
    assert signature(milestones(2, failure="timeout")) == "stopped after grasp [timeout]"
    assert signature(milestones(0, failure="target_not_detected")) == "no stage reached [target_not_detected]"
    assert signature(milestones(3, success=True)) is None


def test_signatures_name_only_what_is_missing():
    e = timed([None, 40, 1], [False, True, True], [False, False, False])
    assert signature(e) == "missing a"
    e = timed([30, None, 1], [False, False, True], [True, False, False])
    assert signature(e, max_length=520) == "missing b; undone: a [out of steps]"


def test_reason_from_events():
    fail = timed([None, None, 1], [False] * 2 + [True], [False] * 3,
                 events=[(0, "step 0: start state ok"), (160, "step 0 timed out after 160 steps, no retry (limit)")])
    assert signature(fail, 520).endswith("[step # timed out after # steps, no retry (limit)]")
    benign = timed([None, None, 1], [False] * 2 + [True], [False] * 3, events=[(30, "recovered in 29 steps")])
    assert signature(benign, 520).endswith("[out of steps]")
    assert signature(benign, 600) == "missing a, b"  # didn't use the full length, no known reason
    assert signature(Episode("t", False, failure="dropped")) == "dropped"
    assert signature(Episode("t", False)) == "failed"


def test_analyze_stage_stats_and_signatures():
    run = Run("r", [timed([10, 50, 1], [True] * 3, [False] * 3, success=True, length=60),
                    timed([12, None, 1], [True, False, True], [False] * 3),
                    timed([None, None, 1], [False, False, True], [False] * 3)])
    (t,) = analyze(run)
    a, b, pre = t.stages
    assert (a.reached, a.final, a.lost, a.median_first) == (2, 2, 0, 11)
    assert (b.reached, b.median_first, pre.reached) == (1, 50, 3)
    assert t.classes == {PARTIAL: 2}  # classes with no episodes are left out
    assert dict(t.signatures) == {"missing b [out of steps]": 1, "missing a, b [out of steps]": 1}


def test_breakdown_finds_the_stage_that_moved():
    a = [milestones(3, success=True, init=i) for i in range(20)]
    b = [milestones(1 if i < 15 else 3, success=i >= 15, failure="timeout", init=i) for i in range(20)]
    bd = breakdown("pick", a, b)
    by = {(s.kind, s.name): s for s in bd.shifts}
    assert by[("stage", "reach")].label == stats.NOISE
    assert by[("stage", "grasp")].label == stats.LIKELY and by[("stage", "grasp")].b == 5
    sig = by[("signature", "stopped after reach [timeout]")]
    assert (sig.a, sig.b, sig.label) == (0, 15, stats.LIKELY)


def test_cli_failures_and_details(tmp_path, capsys):
    a = Run("a", [milestones(3, success=True, init=i) for i in range(20)])
    b = Run("b", [milestones(1, failure="timeout", init=i) if i < 15 else milestones(3, True, init=i)
                  for i in range(20)])
    save_run(a, tmp_path / "a")
    save_run(b, tmp_path / "b")
    assert main(["summary", str(tmp_path / "b"), "--failures"]) == 0
    out = capsys.readouterr().out
    assert "15 failures (15 partial)" in out and "15  stopped after reach [timeout]" in out
    assert main(["compare", str(tmp_path / "a"), str(tmp_path / "b"), "--json", str(tmp_path / "d.json")]) == 0
    out = capsys.readouterr().out
    assert "Where the changes come from" in out and "likely real: more often" in out
    import json
    assert json.loads((tmp_path / "d.json").read_text())["breakdowns"][0]["task"] == "pick"
    assert main(["compare", str(tmp_path / "a"), str(tmp_path / "b"), "--details", "none"]) == 0
    assert "Where the changes" not in capsys.readouterr().out


def test_reason_shifts_pool_across_tasks():
    from policy_diff.stages import reason_shifts
    # each task alone: 2 of 4 timeouts vs 0 of 4, noise; across 8 tasks: 16 vs 0
    a = [Episode(f"t{t}", False, init=i, failure="timeout" if i < 2 else "missed") for t in range(8) for i in range(4)]
    b = [Episode(f"t{t}", i >= 2, init=i, failure=None if i >= 2 else "missed") for t in range(8) for i in range(4)]
    by = {s.name: s for s in reason_shifts(a, b)}
    assert (by["timeout"].a, by["timeout"].b, by["timeout"].label) == (16, 0, stats.LIKELY)
    assert by["missed"].label == stats.NOISE
