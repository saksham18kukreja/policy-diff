"""The benchmark-agnostic paths: logging API, table importer, LeRobot adapter, filters, seed pairing."""

import json

import pytest

from policy_diff import Episode, Run, RunWriter, Stage, StageTracker, compare, load_run
from policy_diff.adapters import import_lerobot, import_table
from policy_diff.cli import main
from policy_diff.schema import FormatError


def test_stage_tracker_and_run_writer(tmp_path):
    with RunWriter(tmp_path / "ckpt", policy="p@1", settings={"hz": 10}) as run:
        for init in range(3):
            st = StageTracker(["grasped", "lifted"])
            for t, (g, l) in enumerate([(0, 0), (1, 0), (1, 1), (init != 2, 0)]):
                st.update(t, grasped=g, lifted=l)
            run.episode("pick_mug", success=init == 0, init=init, length=4, stages=st.stages(),
                        failure=None if init == 0 else "dropped", robot="r1")
        run.episode("wipe", True, stages={"contact": True, "wiped": True})
    r = load_run(tmp_path / "ckpt")
    assert r.run_id == "ckpt" and r.policy == "p@1" and r.settings == {"hz": 10} and len(r.episodes) == 4
    e = r.episodes[2]
    assert [(s.name, s.first_true, s.final, s.ever_lost) for s in e.stages] == [
        ("grasped", 1, False, True), ("lifted", 2, False, True)]
    assert e.failure == "dropped" and e.extra == {"robot": "r1"}
    assert r.episodes[3].stages[0].reached and r.episodes[3].stages[0].first_true is None
    with pytest.raises(KeyError, match="unknown stage"):
        StageTracker(["a"]).update(0, b=True)


def test_run_writer_validates(tmp_path):
    with RunWriter(tmp_path / "r") as run, pytest.raises(FormatError):
        run.episode("", True)


def test_stage_reached_without_timing():
    assert Stage("x", reached=True).reached and not Stage("x").reached
    assert Stage("x", first_true=0).reached and Stage("x", final=True).reached
    assert Stage.from_dict(Stage("x", reached=True).to_dict()).reached


def test_table_csv_with_stage_columns(tmp_path):
    f = tmp_path / "eps.csv"
    f.write_text("task,episode,ok,grasp_step,lifted,note,why\n"
                 "pick,0,yes,12,true,a,\n"
                 "pick,1,no,,false,,slipped\n"
                 "pick,2,1,0,1,,\n")
    eps = import_table([f], success="ok", init="episode", failure="why",
                       stages={"grasp": "grasp_step"}, stage_values="step")
    assert [(e.init, e.success, e.stages[0].first_true, e.failure) for e in eps] == [
        (0, True, 12, None), (1, False, None, "slipped"), (2, True, 0, None)]
    assert eps[0].extra == {"lifted": "true", "note": "a"}
    eps = import_table([f], success="ok", stages={"lifted": "lifted"})
    assert [e.stages[0].reached for e in eps] == [True, False, True]
    with pytest.raises(FormatError, match="no column"):
        import_table([f], success="success")
    with pytest.raises(FormatError, match="not a step number"):
        import_table([f], success="ok", stages={"n": "note"}, stage_values="step")


def test_table_tsv_furthest_stage_and_const(tmp_path):
    f = tmp_path / "eps.tsv"
    f.write_text("idx\tfirst_stage\tsuccess\treason\n0\thold\t1\tnone\n1\tgrasp\t0\ttimeout\n2\tnone\t0\tmissed\n")
    eps = import_table([f], task_value="pick", init="idx", failure="reason", furthest="first_stage",
                       order=["reach", "grasp", "hold"], const={"checkpoint": "s0"})
    assert [[s.reached for s in e.stages] for e in eps] == [[True] * 3, [True, True, False], [False] * 3]
    assert [e.failure for e in eps] == [None, "timeout", "missed"] and eps[0].extra == {"checkpoint": "s0"}
    with pytest.raises(FormatError, match="eps.tsv:2: furthest stage 'hold'"):
        import_table([f], task_value="pick", furthest="first_stage", order=["reach"])


def test_table_jsonl(tmp_path):
    f = tmp_path / "eps.jsonl"
    f.write_text(json.dumps({"task": "t", "success": True, "seed": 3}) + "\n")
    (e,) = import_table([f])
    assert (e.task, e.success, e.seed, e.init) == ("t", True, 3, None)


def _eval_info(path, successes):
    path.write_text(json.dumps({"per_task": [{"task_group": "pusht", "task_id": 0, "metrics": {
        "successes": successes, "sum_rewards": [1.0] * len(successes),
        "video_paths": [f"/out/videos/ep{i}.mp4" for i in range(len(successes))]}}]}))
    return path


def test_lerobot_adapter(tmp_path):
    f = _eval_info(tmp_path / "eval_info.json", [True, False])
    eps = import_lerobot([f], start_seed=1000, video_base="/out")
    assert [(e.task, e.seed, e.init, e.success, e.video) for e in eps] == [
        ("pusht/0", 1000, None, True, "videos/ep0.mp4"), ("pusht/0", 1001, None, False, "videos/ep1.mp4")]
    assert eps[0].extra["sum_reward"] == 1.0
    assert import_lerobot([f], index_as_init=True)[1].init == 1
    old = tmp_path / "old.json"
    old.write_text(json.dumps({"per_episode": [{"episode_ix": 0, "success": True, "seed": 7}]}))
    assert import_lerobot([old])[0].seed == 7
    (tmp_path / "bad.json").write_text("{}")
    with pytest.raises(FormatError):
        import_lerobot([tmp_path / "bad.json"])


def test_without_init_episodes_pair_on_seed():
    a = Run("a", [Episode("t", True, seed=s) for s in range(8)])
    b = Run("b", [Episode("t", s == 0, seed=s) for s in range(8)])
    (d,) = compare(a, b).tasks
    assert d.test == "exact, by seed" and (d.pairs, d.a_only_wins) == (8, 7)
    assert d.p == pytest.approx(2 / 2 ** 7)
    no_ids = compare(Run("a", [Episode("t", True)] * 8), Run("b", [Episode("t", False)] * 8)).tasks[0]
    assert no_ids.test == "exact, unpaired" and no_ids.pairs == 0 and no_ids.b_unpaired == 8


def test_where_filters_fields_and_extra(tmp_path):
    run = Run("r", [Episode("pick", True, extra={"robot": "r1"}), Episode("pick", False, extra={"robot": "r2"}),
                    Episode("wipe", True, extra={"robot": "r1"})])
    assert len(run.where(robot="r1").episodes) == 2 and run.where(robot="r1").run_id == "r[robot=r1]"
    assert len(run.where(task="pick", success="false").episodes) == 1


def test_cli_table_import_and_filtered_compare(tmp_path, capsys):
    f = tmp_path / "eps.csv"
    f.write_text("idx,success,ckpt\n" + "".join(f"{i},{int(i % 5 != 0)},a\n{i},1,b\n" for i in range(20)))
    out = tmp_path / "run"
    assert main(["import", "table", str(f), "--task-value", "pick", "--init", "idx", "--out", str(out)]) == 0
    capsys.readouterr()
    assert main(["compare", str(out), str(out), "--where-a", "ckpt=a", "--where-b", "ckpt=b"]) == 0
    text = capsys.readouterr().out
    assert "run[ckpt=a]" in text and "16/20" in text and "20/20" in text
    assert main(["compare", str(out), str(out), "--where", "ckpt=zzz"]) == 2


def test_runs_with_disjoint_seeds_fall_back_to_unpaired():
    """Two runs evaluated on different seeds can't pair, but their rates still compare."""
    a = Run("a", [Episode("t", i < 65, seed=1000 + i) for i in range(100)])
    b = Run("b", [Episode("t", i < 40, seed=5000 + i) for i in range(100)])
    (d,) = compare(a, b).tasks
    from policy_diff import stats
    assert d.test == "exact, unpaired" and d.p == pytest.approx(stats.fisher_exact(65, 100, 40, 100))
    assert d.label == stats.LIKELY


def test_partly_matched_runs_keep_unmatched_episodes():
    a = Run("a", [Episode("t", True, init=i) for i in range(10)])
    b = Run("b", [Episode("t", i >= 5, init=i) for i in range(5, 15)])  # inits 5-9 shared
    from policy_diff.compare import _strata
    strata, matched = _strata(a.episodes, b.episodes)
    assert matched == 5 and sum(s[1] + s[3] for s in strata) == 20
