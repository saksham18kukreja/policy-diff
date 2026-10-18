"""W&B import and export, through real offline W&B runs (skipped without wandb)."""

import json

import pytest

from policy_diff.adapters.wandb_tables import import_wandb, table_rows
from policy_diff.cli import main
from policy_diff.schema import FormatError

try:
    import wandb
except ImportError:
    wandb = None

pytestmark = pytest.mark.skipif(wandb is None, reason="needs wandb")


def _offline_run(tmp_path, name, config, rows):
    run = wandb.init(project="pd-test", name=name, config=config, mode="offline", dir=str(tmp_path),
                     reinit="finish_previous")
    run.log({"episodes": wandb.Table(columns=["position", "first_stage", "success", "failure_reason"],
                                     data=rows)})
    path = run.dir
    run.finish()
    from pathlib import Path
    return str(Path(path).parent)  # the wandb/offline-run-* directory


@pytest.fixture
def runs(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_SILENT", "true")
    monkeypatch.chdir(tmp_path)  # exports write their offline run to ./wandb
    v1 = [[i, "hold" if i < 10 else "grasp", i < 10, "none" if i < 10 else "timeout"] for i in range(20)]
    v2 = [[i, "hold", True, "none"] for i in range(20)]
    return (_offline_run(tmp_path, "v1", {"label": "v1", "mode": "pick", "start": "", "ckpt": "s0"}, v1),
            _offline_run(tmp_path, "v2", {"label": "v2", "mode": "pick", "start": "", "ckpt": "s0"}, v2),
            _offline_run(tmp_path, "v2", {"label": "v2", "mode": "pick", "start": "", "ckpt": "s0"}, v2))


def test_import_offline_runs(runs):
    eps, ids, warnings = import_wandb([runs[0]], task_value="pick", const={"ckpt": "s0"}, init="position",
                                      failure="failure_reason", furthest="first_stage",
                                      order=["reach", "grasp", "hold"])
    assert len(ids) == 1 and not warnings and len(eps) == 20
    e = eps[19]
    assert (e.task, e.init, e.success, e.failure) == ("pick", 19, False, "timeout")
    assert [s.reached for s in e.stages] == [True, True, False]
    assert e.extra["ckpt"] == "s0" and e.extra["wandb_run"] == ids[0]
    with pytest.raises(FormatError, match="offline run"):
        import_wandb([runs[0]], task_from_config=["mode"])


def _synced_run(tmp_path, name, config, rows):
    """A run directory as W&B writes it for online / synced runs: config.yaml + summary + table file."""
    files = tmp_path / f"run-20261018_000000-{name}" / "files"
    (files / "media" / "table").mkdir(parents=True)
    (files / "media" / "table" / "episodes_0_abc.table.json").write_text(
        json.dumps({"columns": ["position", "success"], "data": rows}))
    (files / "wandb-summary.json").write_text(json.dumps(
        {"episodes": {"_type": "table-file", "path": "media/table/episodes_0_abc.table.json"}}))
    (files / "config.yaml").write_text("_wandb:\n  value: {}\n" + "".join(
        f"{k}:\n  value: {json.dumps(v)}\n" for k, v in config.items()))
    return str(files.parent)


def test_config_options_on_synced_runs(tmp_path):
    rows = [[i, i < 5] for i in range(10)]
    a = _synced_run(tmp_path, "aaa", {"label": "v2", "mode": "pick", "start": "", "skill": {"obj": "red"}}, rows)
    b = _synced_run(tmp_path, "bbb", {"label": "v2", "mode": "pick", "start": "", "skill": {"obj": "red"}}, rows)
    c = _synced_run(tmp_path, "ccc", {"label": "v1", "mode": "pick", "start": "", "skill": {"obj": "red"}}, rows)
    eps, ids, warnings = import_wandb([a, b, c], filters={"label": "v2"}, init="position",
                                      task_from_config=["mode", "start", "skill.obj"], extra_from_config=["label"])
    assert ids == ["aaa", "bbb"] and {e.task for e in eps} == {"pick/red"}  # empty start left out
    assert eps[0].extra["label"] == "v2"
    assert len(warnings) == 1 and "2 runs have identical configs" in warnings[0]


def test_table_rows_media_cells():
    rows = table_rows({"columns": ["ok", "video"], "data": [[True, {"_type": "video-file", "path": "media/v.mp4"}]]})
    assert rows == [{"ok": True, "video": "media/v.mp4"}]


def test_cli_import_and_export(runs, tmp_path, capsys):
    for i, r in enumerate(runs[:2]):
        assert main(["import", "wandb", r, "--task-value", "pick", "--init", "position",
                     "--failure", "failure_reason", "--out", str(tmp_path / f"r{i}")]) == 0
    assert main(["compare", str(tmp_path / "r0"), str(tmp_path / "r1"), "--wandb", "pd-test",
                 "--wandb-mode", "offline"]) == 0
    out = capsys.readouterr().out
    assert "likely real: improved" in out and "logged to W&B:" in out
    files = out.split("logged to W&B:")[1].strip()
    from pathlib import Path
    tables = {p.name.split("_")[0]: json.loads(p.read_text()) for p in Path(files, "media", "table").glob("*.json")}
    assert set(tables) >= {"tasks", "flips", "reasons"}
    assert table_rows(tables["tasks"])[0]["verdict"] == "likely real: improved"
    flips = table_rows(tables["flips"])
    assert len(flips) == 10 and flips[0]["a_signature"] == "timeout" and flips[0]["b_success"] is True
