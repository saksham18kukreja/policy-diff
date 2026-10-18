"""The policy-diff log format.

A run is a directory with two files:

  run.json        header: run_id, policy, benchmark, settings, sources, ...
  episodes.jsonl  one JSON object per episode

Only `task` and `success` are required per episode. Everything else is
optional, so a team can start with what it already logs and add stage
signals, harness events and videos later.

  {"task": "libero_10/2", "init": 7, "seed": 1007, "success": false, "length": 520,
   "stages": [{"name": "turnon flat_stove_1", "first_true": 82, "final": true}],
   "events": [[82, "step 0 done"]],
   "video": "videos/libero_10_2/eval_episode_7.mp4"}

Episodes with the same task and initial state are paired across runs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

FORMAT_VERSION = "0.1"
RUN_FILE = "run.json"
EPISODES_FILE = "episodes.jsonl"


class FormatError(ValueError):
    """A run or episode doesn't follow the log format."""


@dataclass
class Stage:
    name: str
    first_true: Optional[int] = None  # first step the stage condition held
    final: Optional[bool] = None  # held at the end of the episode
    ever_lost: Optional[bool] = None  # held, then stopped holding

    @property
    def reached(self) -> bool:
        return self.first_true is not None

    def to_dict(self) -> dict:
        d = {"name": self.name, "first_true": self.first_true, "final": self.final}
        if self.ever_lost is not None:
            d["ever_lost"] = self.ever_lost
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Stage":
        if not isinstance(d, dict) or not isinstance(d.get("name"), str):
            raise FormatError(f"stage needs a string 'name': {d!r}")
        return cls(d["name"], d.get("first_true"), d.get("final"), d.get("ever_lost"))


@dataclass
class Episode:
    task: str
    success: bool
    init: Union[int, str, None] = None  # initial-state id, the pairing key with task
    seed: Optional[int] = None
    length: Optional[int] = None  # control steps
    stages: list[Stage] = field(default_factory=list)
    events: list[tuple[int, str]] = field(default_factory=list)
    video: Optional[str] = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> tuple:
        """Pairing key: same task, same initial state."""
        return (self.task, self.init)

    def to_dict(self) -> dict:
        d: dict[str, Any] = {"task": self.task, "init": self.init, "seed": self.seed,
                             "success": self.success, "length": self.length}
        if self.stages:
            d["stages"] = [s.to_dict() for s in self.stages]
        if self.events:
            d["events"] = [[t, m] for t, m in self.events]
        if self.video is not None:
            d["video"] = self.video
        if self.extra:
            d["extra"] = self.extra
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Episode":
        if not isinstance(d, dict):
            raise FormatError(f"episode must be an object, got {type(d).__name__}")
        if not isinstance(d.get("task"), str) or not d["task"]:
            raise FormatError(f"episode needs a non-empty string 'task': {d!r}")
        if not isinstance(d.get("success"), bool):
            raise FormatError(f"episode needs a boolean 'success': {d!r}")
        events = [(int(t), str(m)) for t, m in d.get("events") or []]
        return cls(task=d["task"], success=d["success"], init=d.get("init"), seed=d.get("seed"),
                   length=d.get("length"), stages=[Stage.from_dict(s) for s in d.get("stages") or []],
                   events=events, video=d.get("video"), extra=d.get("extra") or {})


@dataclass
class Run:
    run_id: str
    episodes: list[Episode] = field(default_factory=list)
    policy: Optional[str] = None
    benchmark: Optional[str] = None
    settings: dict[str, Any] = field(default_factory=dict)
    sources: list[str] = field(default_factory=list)  # files the run was imported from
    notes: Optional[str] = None
    created: Optional[str] = None

    def tasks(self) -> list[str]:
        return sorted({e.task for e in self.episodes}, key=_natural)

    def by_task(self) -> dict[str, list[Episode]]:
        out: dict[str, list[Episode]] = {}
        for e in self.episodes:
            out.setdefault(e.task, []).append(e)
        return {t: out[t] for t in sorted(out, key=_natural)}

    def header(self) -> dict:
        return {"format_version": FORMAT_VERSION, "run_id": self.run_id, "policy": self.policy,
                "benchmark": self.benchmark, "settings": self.settings, "sources": self.sources,
                "notes": self.notes, "created": self.created}


def _natural(s: str):
    """Sort 'libero_10/10' after 'libero_10/9'."""
    import re
    return [int(p) if p.isdigit() else p for p in re.split(r"(\d+)", s)]


def save_run(run: Run, path: Union[str, Path]) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    (path / RUN_FILE).write_text(json.dumps(run.header(), indent=1) + "\n")
    with open(path / EPISODES_FILE, "w") as f:
        for e in run.episodes:
            f.write(json.dumps(e.to_dict()) + "\n")
    return path


def load_run(path: Union[str, Path]) -> Run:
    """Load a run directory, or a bare episodes.jsonl (run_id = file stem)."""
    path = Path(path)
    if path.is_dir():
        header_file, episodes_file = path / RUN_FILE, path / EPISODES_FILE
    else:
        header_file, episodes_file = path.parent / RUN_FILE, path
    if not episodes_file.exists():
        raise FileNotFoundError(f"no {EPISODES_FILE} in {path}")
    header = json.loads(header_file.read_text()) if header_file.exists() else {}
    version = header.get("format_version", FORMAT_VERSION)
    if str(version).split(".")[0] != FORMAT_VERSION.split(".")[0]:
        raise FormatError(f"{header_file}: format_version {version} is not supported")
    episodes = []
    with open(episodes_file) as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                episodes.append(Episode.from_dict(json.loads(line)))
            except (json.JSONDecodeError, FormatError) as err:
                raise FormatError(f"{episodes_file}:{n}: {err}") from None
    default_id = path.name if path.is_dir() else path.stem
    return Run(run_id=header.get("run_id") or default_id, episodes=episodes,
               policy=header.get("policy"), benchmark=header.get("benchmark"),
               settings=header.get("settings") or {}, sources=header.get("sources") or [],
               notes=header.get("notes"), created=header.get("created"))
