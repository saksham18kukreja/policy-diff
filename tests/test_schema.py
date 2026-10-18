import json

import pytest

from policy_diff import Episode, Run, Stage, load_run, save_run
from policy_diff.schema import FormatError


def _run():
    eps = [Episode("libero_10/2", False, init=7, seed=1007, length=520,
                   stages=[Stage("turnon flat_stove_1", 82, True), Stage("on moka_pot_1 x", None, False)],
                   events=[(82, "step 0 done")], video="v/7.mp4", extra={"episode_index": 7}),
           Episode("libero_10/10", True), Episode("libero_10/9", True, init="a")]
    return Run("base", eps, policy="lerobot/smolvla_libero", benchmark="libero_10", settings={"n_action_steps": 50})


def test_roundtrip(tmp_path):
    run = _run()
    loaded = load_run(save_run(run, tmp_path / "base"))
    assert loaded.run_id == "base" and loaded.settings == {"n_action_steps": 50}
    assert [e.to_dict() for e in loaded.episodes] == [e.to_dict() for e in run.episodes]
    assert loaded.episodes[0].stages[0].reached and not loaded.episodes[0].stages[1].reached


def test_tasks_sort_naturally():
    assert _run().tasks() == ["libero_10/2", "libero_10/9", "libero_10/10"]


def test_bare_episodes_file(tmp_path):
    f = tmp_path / "mine.jsonl"
    f.write_text(json.dumps({"task": "t", "success": True}) + "\n\n")
    run = load_run(f)
    assert run.run_id == "mine" and run.episodes[0].init is None


@pytest.mark.parametrize("row", [{"task": "t"}, {"success": True}, {"task": "", "success": True},
                                 {"task": "t", "success": 1}, {"task": "t", "success": True, "stages": [{}]}])
def test_invalid_episodes(tmp_path, row):
    f = tmp_path / "episodes.jsonl"
    f.write_text(json.dumps({"task": "ok", "success": False}) + "\n" + json.dumps(row) + "\n")
    with pytest.raises(FormatError, match=":2:"):
        load_run(tmp_path)


def test_unsupported_version(tmp_path):
    save_run(_run(), tmp_path)
    (tmp_path / "run.json").write_text(json.dumps({"format_version": "9.0"}))
    with pytest.raises(FormatError, match="not supported"):
        load_run(tmp_path)
