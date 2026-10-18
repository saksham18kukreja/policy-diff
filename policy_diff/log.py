"""Write policy-diff runs from your own evaluation loop.

    from policy_diff.log import RunWriter, StageTracker

    with RunWriter("runs/ckpt_50k", policy="my_policy@50k", settings={"n_action_steps": 10}) as run:
        for task in tasks:
            for init in range(n_inits):
                obs = env.reset(init)
                stages = StageTracker(["grasped", "lifted", "placed"])
                for t in range(max_steps):
                    obs, done = env.step(policy(obs))
                    stages.update(t, grasped=env.is_grasped(), lifted=env.height() > 0.05,
                                  placed=env.on_target())
                    if done:
                        break
                run.episode(task=task, init=init, success=env.success(), length=t + 1,
                            stages=stages.stages(), video=f"videos/{task}_{init}.mp4")

Stage checks are whatever you can compute: simulator state, a learned
success detector, or nothing at all (then only success is compared).
Episodes are appended as they finish, so a crashed evaluation keeps what it
logged.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Optional, Union

from .schema import EPISODES_FILE, RUN_FILE, Episode, Run, Stage


class StageTracker:
    """Records, per named stage condition, the first step it held, whether it held at the end and whether it was lost."""

    def __init__(self, names: Iterable[str]):
        self.names = list(names)
        self._first: dict[str, Optional[int]] = {n: None for n in self.names}
        self._now: dict[str, bool] = {n: False for n in self.names}
        self._lost: dict[str, bool] = {n: False for n in self.names}

    def update(self, t: int, values: Optional[dict] = None, **kw: bool) -> None:
        """Pass the current value of some or all stages at step t."""
        for name, ok in {**(values or {}), **kw}.items():
            if name not in self._now:
                raise KeyError(f"unknown stage {name!r}; stages are {self.names}")
            ok = bool(ok)
            if ok and self._first[name] is None:
                self._first[name] = t
            if self._now[name] and not ok:
                self._lost[name] = True
            self._now[name] = ok

    def stages(self) -> list[Stage]:
        return [Stage(n, self._first[n], self._now[n], self._lost[n]) for n in self.names]


class RunWriter:
    """Appends episodes to a run directory (run.json + episodes.jsonl)."""

    def __init__(self, path: Union[str, Path], run_id: Optional[str] = None, *, policy: Optional[str] = None,
                 benchmark: Optional[str] = None, settings: Optional[dict] = None, notes: Optional[str] = None,
                 append: bool = False):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.run = Run(run_id or self.path.name, policy=policy, benchmark=benchmark, settings=settings or {},
                       notes=notes, created=date.today().isoformat())
        (self.path / RUN_FILE).write_text(json.dumps(self.run.header(), indent=1) + "\n")
        self._f = open(self.path / EPISODES_FILE, "a" if append else "w")
        self.count = 0

    def episode(self, task: str, success: bool, *, init: Any = None, seed: Optional[int] = None,
                length: Optional[int] = None, stages: Optional[list] = None, events: Optional[list] = None,
                failure: Optional[str] = None, video: Optional[str] = None, **extra: Any) -> Episode:
        """Log one finished episode. Stages may be Stage objects or {name: reached} dicts; extra keys are kept."""
        if isinstance(stages, dict):
            stages = [Stage(n, reached=bool(v)) for n, v in stages.items()]
        e = Episode(task=task, success=bool(success), init=init, seed=seed, length=length,
                    stages=list(stages or []), events=[(int(t), str(m)) for t, m in events or []],
                    video=video, failure=failure, extra=extra)
        Episode.from_dict(e.to_dict())  # validate before writing
        self._f.write(json.dumps(e.to_dict()) + "\n")
        self._f.flush()
        self.count += 1
        return e

    def close(self) -> None:
        self._f.close()

    def __enter__(self) -> "RunWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
