"""Import any per-episode table: CSV, TSV or JSON lines, one row per episode.

Map your columns onto the format; columns you don't map are kept in `extra`.

  task       column with the task name, or a constant (task_value) for one-task files
  success    column with 1/0, true/false, yes/no, success/fail
  init, seed, length, video, failure
             optional columns (used by default when columns with these names exist)
  stages     {stage name: column}; the column holds whether the stage was reached
             (stage_values="bool", the default) or the first step it held
             (stage_values="step"; empty = never)
  furthest + order
             a column naming the furthest stage reached, with the stages in order,
             for logs that record progress as one milestone ("grasp", "lift", ...)
  const      {key: value} added to every row's `extra` (checkpoint, robot id, operator, ...)
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

from ..schema import Episode, FormatError, Stage

TRUE = {"1", "true", "t", "yes", "y", "success", "succeeded", "ok", "pass"}
FALSE = {"0", "false", "f", "no", "n", "failure", "fail", "failed"}
EMPTY = {"", "none", "null", "nan", "-"}
JSON_SUFFIXES = (".jsonl", ".ndjson", ".json")


def read_rows(path) -> list[dict]:
    path = Path(path)
    text = path.read_text()
    if path.suffix in JSON_SUFFIXES:
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    delimiter = "\t" if path.suffix in (".tsv", ".tab") else ","
    return list(csv.DictReader(text.splitlines(), delimiter=delimiter))


def _empty(v) -> bool:
    return v is None or (isinstance(v, str) and v.strip().lower() in EMPTY)


def _bool(v, where) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)) and v in (0, 1):
        return bool(v)
    s = str(v).strip().lower()
    if s in TRUE:
        return True
    if s in FALSE:
        return False
    raise FormatError(f"{where}: can't read {v!r} as success/failure")


def _number(v):
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return int(v) if float(v).is_integer() else v
    try:
        f = float(v)
    except (TypeError, ValueError):
        return v
    return int(f) if f.is_integer() else f


def _stage(name, v, kind, where) -> Stage:
    if _empty(v):
        return Stage(name, reached=False)
    if kind == "step":
        n = _number(v)
        if not isinstance(n, int):
            raise FormatError(f"{where}: stage {name!r}: {v!r} is not a step number")
        return Stage(name, first_true=n)
    return Stage(name, reached=_bool(v, where))


@dataclass
class Mapping:
    """How a table's columns map onto episode fields (see the module docstring)."""
    task: Optional[str] = None
    task_value: Optional[str] = None
    success: str = "success"
    init: Optional[str] = None
    seed: Optional[str] = None
    length: Optional[str] = None
    video: Optional[str] = None
    failure: Optional[str] = None
    stages: dict = field(default_factory=dict)
    stage_values: str = "bool"
    furthest: Optional[str] = None
    order: Optional[Sequence[str]] = None
    const: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.stage_values not in ("bool", "step"):
            raise ValueError(f"stage_values is 'bool' or 'step', not {self.stage_values!r}")
        if self.furthest and not self.order:
            raise ValueError("furthest needs the stage order")
        if self.task is None and self.task_value is None:
            self.task = "task"
        self.stages = dict(self.stages or {})
        self.const = dict(self.const or {})


def episodes_from_rows(rows: Sequence[dict], m: Mapping, source: str = "table", first_row: int = 1,
                       const: Optional[dict] = None) -> list[Episode]:
    """Rows (dicts, one per episode) from any source, mapped onto episodes.

    `source` and `first_row` only label error messages; `const` adds to m.const for these rows.
    """
    if not rows:
        return []
    cols = set(rows[0])
    # optional fields default to same-named columns
    fields = {k: (v if v is not None else (k if k in cols else None))
              for k, v in dict(init=m.init, seed=m.seed, length=m.length, video=m.video, failure=m.failure).items()}
    needed = [c for c in [m.task, m.success, m.furthest, *m.stages.values(), *fields.values()] if c]
    missing = [c for c in needed if c not in cols]
    if missing:
        raise FormatError(f"{source}: no column(s) {missing}; columns are {sorted(cols)}")
    used = set(needed)
    tags = {**m.const, **(const or {})}
    episodes = []
    for n, r in enumerate(rows, first_row):
        where = f"{source}:{n}"
        ok = _bool(r[m.success], where)
        st = [_stage(name, r[col], m.stage_values, where) for name, col in m.stages.items()]
        if m.furthest:
            v = r[m.furthest]
            if _empty(v):
                k = -1
            elif v in m.order:
                k = list(m.order).index(v)
            else:
                raise FormatError(f"{where}: furthest stage {v!r} is not one of {list(m.order)}")
            st += [Stage(name, reached=i <= k) for i, name in enumerate(m.order)]
        get = {k: (None if c is None or _empty(r[c]) else r[c]) for k, c in fields.items()}
        episodes.append(Episode(
            task=str(r[m.task]) if m.task else m.task_value, success=ok,
            init=_number(get["init"]) if get["init"] is not None else None,
            seed=_number(get["seed"]) if get["seed"] is not None else None,
            length=_number(get["length"]) if get["length"] is not None else None,
            stages=st, video=get["video"],
            failure=None if ok or get["failure"] is None else str(get["failure"]),
            extra={**tags, **{k: _number(v) for k, v in r.items() if k not in used and not _empty(v)}}))
    return episodes


def import_table(paths: Sequence[str], **mapping) -> list[Episode]:
    """Episodes from CSV / TSV / JSONL files; keyword arguments are Mapping fields."""
    m = Mapping(**mapping)
    episodes = []
    for path in paths:
        first = 1 if Path(path).suffix in JSON_SUFFIXES else 2  # CSV/TSV line 1 is the header
        episodes += episodes_from_rows(read_rows(path), m, str(path), first)
    return episodes
