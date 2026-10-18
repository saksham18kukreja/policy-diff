"""Import per-episode tables logged to Weights & Biases.

Many teams already log one row per evaluation episode as a `wandb.Table`
(e.g. `run.log({"episodes": wandb.Table(columns=[...], data=rows)})`). This
reads those tables, from the W&B API or from local run directories
(`wandb/run-*`, including offline runs), and maps their columns with the
same options as the table importer.

Runs are selected by path ("entity/project/run_id"), or all runs of a
project whose config matches `filters` (e.g. {"label": "v5.1"}). Values from
each run's config can name the task (task_from_config) and tag episodes
(extra_from_config); every episode also gets `wandb_run` in its extra.

Needs the `wandb` package for the API (and PyYAML for local configs); the
rest of policy-diff does not.
"""

from __future__ import annotations

import json
import tempfile
from urllib.parse import unquote
from pathlib import Path
from typing import Any, Optional, Sequence

from ..schema import Episode, FormatError
from .table import Mapping, episodes_from_rows


def _lookup(config: dict, key: str) -> Any:
    """config value by dotted key ("skill_args.object")."""
    value: Any = config
    for part in key.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def _cell(v):
    """Media cells (videos, images) become their file path."""
    if isinstance(v, dict) and "path" in v:
        return v["path"]
    return v


def table_rows(table_json: dict) -> list[dict]:
    """Rows of a W&B table file ({"columns": [...], "data": [[...], ...]})."""
    cols = table_json.get("columns")
    if cols is None or "data" not in table_json:
        raise FormatError("not a W&B table file (no columns / data)")
    return [{c: _cell(v) for c, v in zip(cols, row)} for row in table_json["data"]]


class _LocalRun:
    """A local run directory (wandb/run-* or wandb/offline-run-*).

    Table files are plain JSON under files/media/table, so they can be read
    from offline runs too. The config is in files/config.yaml for online and
    synced runs; offline runs keep it only in their binary log until
    `wandb sync`, so config-based options need a synced run there.
    """

    def __init__(self, path: Path):
        self.path = path
        self.files = path / "files"
        if not self.files.is_dir():
            raise FileNotFoundError(f"{path}: not a W&B run directory (no files/)")
        self.id = path.name.split("-")[-1]
        self.name = self.id
        summary = self.files / "wandb-summary.json"
        self.summary = json.loads(summary.read_text()) if summary.exists() else {}
        self.config = self._config()

    def _config(self) -> Optional[dict]:
        f = self.files / "config.yaml"
        if not f.exists():
            return None
        import yaml  # installed with wandb
        raw = yaml.safe_load(f.read_text()) or {}
        return {k: v.get("value") if isinstance(v, dict) and "value" in v else v
                for k, v in raw.items() if not k.startswith("_")}

    def table(self, key: str) -> dict:
        ref = self.summary.get(key)
        if isinstance(ref, dict) and "path" in ref:
            return json.loads((self.files / ref["path"]).read_text())
        # no summary (offline run): the latest logged step of this table
        found = sorted(self.files.glob(f"media/table/{key}_*.table.json"),
                       key=lambda p: int(p.name[len(key) + 1:].split("_")[0]))
        if not found:
            raise FormatError(f"{self.path}: no table {key!r}")
        return json.loads(found[-1].read_text())


class _ApiRun:
    def __init__(self, run):
        self.run = run
        self.id = unquote("/".join(run.path))
        self.name = run.name
        self.config = dict(run.config)

    def table(self, key: str) -> dict:
        ref = self.run.summary.get(key)
        if ref is None or "path" not in ref:
            raise FormatError(f"{self.id}: no table {key!r} in the run summary")
        with tempfile.TemporaryDirectory() as tmp:
            f = self.run.file(ref["path"]).download(root=tmp, replace=True)
            f.close()
            return json.loads((Path(tmp) / ref["path"]).read_text())


def _api():
    try:
        import wandb
    except ImportError:
        raise ImportError("reading from the W&B API needs the wandb package: pip install wandb") from None
    return wandb.Api(timeout=60)


def select_runs(runs: Sequence[str] = (), project: Optional[str] = None,
                filters: Optional[dict] = None) -> list:
    """Local run directories or "entity/project/id" paths, plus a project's runs matching `filters`."""
    out, api = [], None
    for r in runs:
        if Path(r).is_dir():
            out.append(_LocalRun(Path(r)))
        else:
            api = api or _api()
            out.append(_ApiRun(api.run(r)))
    if project:
        api = api or _api()
        out += [_ApiRun(r) for r in api.runs(project, per_page=200)]
    if filters:
        out = [r for r in out if r.config is not None]
        out = [r for r in out if all(str(_lookup(r.config, k)).lower() == str(v).lower() for k, v in filters.items())]
    return out


def import_wandb(runs: Sequence[str] = (), *, project: Optional[str] = None, filters: Optional[dict] = None,
                 table: str = "episodes", task_from_config: Sequence[str] = (),
                 extra_from_config: Sequence[str] = (), **mapping) -> tuple[list[Episode], list[str], list[str]]:
    """Episodes from the `table` of each selected run, the run ids they came from, and warnings.

    Keyword arguments are table-importer Mapping fields (success, init, failure, furthest, ...).
    Runs whose whole config is identical but that land in the same task are
    warned about: something that distinguishes them (a setting, a condition)
    was not logged, so their episodes will be pooled as if they were repeats.
    """
    m = Mapping(**mapping)
    selected = select_runs(runs, project, filters)
    if not selected:
        raise ValueError("no W&B runs selected (check the run paths, --project and --filter)")
    uses_config = bool(task_from_config or extra_from_config or filters)
    episodes, ids, seen = [], [], {}
    for r in selected:
        if r.config is None:
            if uses_config:
                raise FormatError(f"{r.path}: offline run, its config is not readable until `wandb sync`; "
                                  "use --task-value / --const instead of config options")
            r.config = {}
        try:
            data = r.table(table)
        except FormatError:
            if project and not runs:  # a project listing can include runs without this table
                continue
            raise
        const = {k: _lookup(r.config, k) for k in extra_from_config}
        const["wandb_run"] = r.id
        if task_from_config:
            parts = [_lookup(r.config, k) for k in task_from_config]  # empty values are left out of the name
            name = "/".join(str(p) for p in parts if p not in (None, ""))
            if not name:
                raise FormatError(f"{r.id}: config has none of {list(task_from_config)}")
            m_run = Mapping(**{**m.__dict__, "task": None, "task_value": name})
        else:
            m_run = m
        episodes += episodes_from_rows(table_rows(data), m_run, f"{r.id}:{table}", 1, const)
        ids.append(r.id)
        key = json.dumps(r.config, sort_keys=True, default=str)
        seen.setdefault(key, []).append(r)
    warnings = []
    for group in seen.values():
        if len(group) > 1:
            names = sorted({r.name for r in group})
            warnings.append(f"{len(group)} runs have identical configs ({', '.join(names)}: "
                            f"{', '.join(r.id for r in group)}); their episodes are pooled as repeats of one "
                            "condition. If they differ in something, log it to the config.")
    return episodes, ids, warnings
