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


def import_table(paths: Sequence[str], *, task: Optional[str] = None, task_value: Optional[str] = None,
                 success: str = "success", init: Optional[str] = None, seed: Optional[str] = None,
                 length: Optional[str] = None, video: Optional[str] = None, failure: Optional[str] = None,
                 stages: Optional[dict[str, str]] = None, stage_values: str = "bool",
                 furthest: Optional[str] = None, order: Optional[Sequence[str]] = None,
                 const: Optional[dict] = None) -> list[Episode]:
    if stage_values not in ("bool", "step"):
        raise ValueError(f"stage_values is 'bool' or 'step', not {stage_values!r}")
    if furthest and not order:
        raise ValueError("furthest needs the stage order")
    if task is None and task_value is None:
        task = "task"
    stages = dict(stages or {})
    episodes = []
    for path in paths:
        rows = read_rows(path)
        if not rows:
            continue
        cols = set(rows[0])
        # optional fields default to same-named columns
        fields = {k: (v if v is not None else (k if k in cols else None))
                  for k, v in dict(init=init, seed=seed, length=length, video=video, failure=failure).items()}
        needed = [c for c in [task, success, furthest, *stages.values(), *fields.values()] if c]
        missing = [c for c in needed if c not in cols]
        if missing:
            raise FormatError(f"{path}: no column(s) {missing}; columns are {sorted(cols)}")
        used = set(needed)
        first_line = 1 if Path(path).suffix in JSON_SUFFIXES else 2  # CSV/TSV line 1 is the header
        for n, r in enumerate(rows, first_line):
            where = f"{path}:{n}"  # line number
            ok = _bool(r[success], where)
            st = [_stage(name, r[col], stage_values, where) for name, col in stages.items()]
            if furthest:
                v = r[furthest]
                if _empty(v):
                    k = -1
                elif v in order:
                    k = list(order).index(v)
                else:
                    raise FormatError(f"{where}: furthest stage {v!r} is not one of {list(order)}")
                st += [Stage(name, reached=i <= k) for i, name in enumerate(order)]
            get = {k: (None if c is None or _empty(r[c]) else r[c]) for k, c in fields.items()}
            episodes.append(Episode(
                task=str(r[task]) if task else task_value, success=ok,
                init=_number(get["init"]) if get["init"] is not None else None,
                seed=_number(get["seed"]) if get["seed"] is not None else None,
                length=_number(get["length"]) if get["length"] is not None else None,
                stages=st, video=get["video"],
                failure=None if ok or get["failure"] is None else str(get["failure"]),
                extra={**(const or {}), **{k: _number(v) for k, v in r.items() if k not in used and not _empty(v)}}))
    return episodes
