import json
from pathlib import Path

import pytest

from policy_diff import load_run
from policy_diff.adapters.libero import import_libero
from policy_diff.cli import main
from policy_diff.schema import FormatError

PREDS = ["on moka_pot_1 stove", "on moka_pot_2 stove", "turnon stove"]
PLAN = {"8": [{"instruction": "left pot", "check": "on moka_pot_2 stove"},
              {"instruction": "right pot", "check": "on moka_pot_1 stove"}]}


def row(init, seed, success, first_true, autoreset=False, steps=None):
    return {"task_id": 8, "init_state": init, "seed": seed, "autoreset": autoreset, "predicates": PREDS,
            "steps": steps or (300 if success else 520), "success": success,
            "first_true": first_true, "final": [f is not None for f in first_true], "ever_lost": [False] * 3}


@pytest.fixture
def logs(tmp_path):
    rows = [row(0, 1000, True, [200, 120, 1]),
            row(1, 1001, False, [None, 150, 1]),
            row(6, None, False, [None, None, 1], autoreset=True, steps=40),
            row(7, 1002, False, [None, None, 1])]  # episode 2 started from init 7 after an autoreset
    sg = tmp_path / "subgoals.jsonl"
    sg.write_text("".join(json.dumps(r) + "\n" for r in rows))
    ev = tmp_path / "eval_info.json"
    ev.write_text(json.dumps({"per_task": [{"task_id": 8, "metrics": {
        "successes": [True, False, False], "video_paths": [f"/runs/x/videos/ep{i}.mp4" for i in range(3)]}}]}))
    hz = tmp_path / "harness.jsonl"
    hz.write_text("".join(json.dumps({"task_id": 8, "seed": s, "plan": ["left pot", "right pot"],
                                      "reached_step": 1, "events": [[120, "step 0 done"]], "steps": n,
                                      "success": ok}) + "\n"
                          for s, ok, n in [(1000, True, 300), (1001, False, 520), (1002, False, 520)]))
    return tmp_path, sg, ev, hz


def test_import_full_task(logs):
    _, sg, ev, hz = logs
    eps = import_libero([sg], [ev], [hz], video_base="/runs/x", plan=PLAN)
    assert [(e.task, e.init, e.seed, e.success) for e in eps] == [
        ("libero_10/8", 0, 1000, True), ("libero_10/8", 1, 1001, False), ("libero_10/8", 7, 1002, False)]
    assert eps[2].extra["episode_index"] == 2 and eps[2].video == "videos/ep2.mp4"
    assert [s.name for s in eps[0].stages] == ["on moka_pot_2 stove", "on moka_pot_1 stove", "turnon stove"]
    assert eps[0].events == [(120, "step 0 done")] and eps[0].extra["plan"] == ["left pot", "right pot"]


def test_step_success_uses_the_plan_check(logs):
    _, sg, _, _ = logs
    eps = import_libero([sg], step=0, plan=PLAN)
    assert [(e.task, e.success) for e in eps] == [("libero_10/8:step0", True), ("libero_10/8:step0", True),
                                                  ("libero_10/8:step0", False)]
    with pytest.raises(ValueError, match="needs a plan"):
        import_libero([sg], step=0)
    with pytest.raises(FormatError, match="no step 2"):
        import_libero([sg], step=2, plan=PLAN)


def test_mismatched_eval_file_is_an_error(logs):
    tmp, sg, ev, _ = logs
    info = json.loads(ev.read_text())
    info["per_task"][0]["metrics"]["successes"] = [False, False, False]
    ev.write_text(json.dumps(info))
    with pytest.raises(FormatError, match="same evaluation"):
        import_libero([sg], [ev])
    with pytest.raises(ValueError, match="2 eval files"):
        import_libero([sg], [ev, ev])


def test_plan_with_unknown_predicate(logs):
    _, sg, _, _ = logs
    with pytest.raises(FormatError, match="not goal predicates"):
        import_libero([sg], plan={"8": [{"check": "on nothing"}]})


def test_cli_import_and_append(logs, capsys):
    tmp, sg, ev, hz = logs
    out = tmp / "run"
    assert main(["import", "libero", "--subgoals", str(sg), "--eval", str(ev), "--harness", str(hz),
                 "--out", str(out), "--setting", "n_action_steps=50", "--setting", "harness=m3"]) == 0
    (tmp / "plan.json").write_text(json.dumps(PLAN))
    assert main(["import", "libero", "--append", "--subgoals", str(sg), "--plan", str(tmp / "plan.json"),
                 "--step", "1", "--out", str(out)]) == 0
    run = load_run(out)
    assert run.run_id == "run" and run.settings == {"n_action_steps": 50, "harness": "m3"}
    assert run.tasks() == ["libero_10/8", "libero_10/8:step1"] and len(run.sources) == 4
    assert main(["import", "libero", "--subgoals", str(sg), "--step", "0", "--out", str(out)]) == 2
    assert "needs a plan" in capsys.readouterr().err


EXAMPLES = Path(__file__).parent.parent / "examples" / "libero_long" / "runs"


@pytest.mark.skipif(not EXAMPLES.exists(), reason="example runs not converted")
@pytest.mark.parametrize("run_id, n, successes", [
    ("smolvla_libero", 300, 162),  # 54% baseline
    ("libero90_smolvla", 300, 0),  # LIBERO-90 model on full instructions
    ("libero90_smolvla_conductor", 180, 122),  # 67.8% with Conductor
    ("libero90_smolvla_steps", 540, 371),  # 68.7% commanded step
    ("finetune_v1_50k_steps", 50, 28),  # 15/30 new steps + 13/20 covered
    ("finetune_v1_50k_conductor_t160", 100, 20),
    ("finetune_v1_50k_conductor_t220", 100, 28),
    ("finetune_v1_50k_conductor_t220_n10", 100, 41),
])
def test_examples_match_reported_results(run_id, n, successes):
    run = load_run(EXAMPLES / run_id)
    assert (len(run.episodes), sum(e.success for e in run.episodes)) == (n, successes)
