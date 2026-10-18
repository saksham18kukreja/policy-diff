"""Import LIBERO evaluations: goal-condition logs, lerobot-eval output, Conductor harness logs.

Inputs, per evaluation (one `lerobot-eval` invocation, one seed):

  subgoals  JSON lines from Conductor's `eval_with_subgoals.py`: one object per
            episode with task_id, init_state, seed, steps, success and, per BDDL
            goal predicate, first_true / final / ever_lost. Required.
  eval      lerobot-eval's eval_info.json: per-task successes and video paths,
            indexed by episode. Optional; successes are cross-checked.
  harness   JSON lines from Conductor's harness rollout: plan, reached_step and
            timestamped events per episode. Optional.

A plan ({task_id: [{"instruction": ..., "check": "<predicate>"}, ...]}) orders
the stages by plan step instead of BDDL order, and is required with `step`:
plan step N checks a named predicate, which is not necessarily predicate N
(LIBERO-Long task 8's goal lists the right moka pot first and also includes
"turnon flat_stove_1", which holds from the first step).

Episodes the vector env started on its own (autoreset at the end of a batch)
are not evaluated episodes and are dropped.

`init` is the LIBERO initial state the episode actually used. It is not always
the episode index: when an episode ends early, LeRobot's LiberoEnv advances
its init-state counter on the autoreset, so later batches can start from
different states in two runs with the same seed. Pairing uses `init`.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional, Sequence

from ..schema import Episode, FormatError, Run, Stage


def _jsonl(path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _eval_episodes(path) -> dict[int, dict]:
    """task_id -> {"successes": [...], "video_paths": [...]} from eval_info.json."""
    info = json.loads(Path(path).read_text())
    out = {}
    for t in info.get("per_task", []):
        m = t["metrics"]
        out[int(t["task_id"])] = {"successes": [bool(s) for s in m["successes"]],
                                  "video_paths": m.get("video_paths") or []}
    return out


def import_libero(subgoals: Sequence[str], evals: Optional[Sequence[str]] = None,
                  harness: Optional[Sequence[str]] = None, *, suite: str = "libero_10",
                  step: Optional[int] = None, plan: Optional[dict] = None,
                  video_base: Optional[str] = None) -> list[Episode]:
    """Convert LIBERO logs into episodes.

    `subgoals`, `evals` and `harness` are aligned lists, one entry per
    evaluation. With `step=N`, success means the predicate checked by plan
    step N was reached (for runs that execute one step of a long task) and
    the task is named "<suite>/<task_id>:step<N>". Video paths are made
    relative to `video_base` when given.
    """
    if step is not None and plan is None:
        raise ValueError("step needs a plan: plan step N does not always check goal predicate N")
    plan = {int(k): [s["check"] for s in v] for k, v in (plan or {}).items()}
    evals, harness = list(evals or []), list(harness or [])
    for name, files in (("eval", evals), ("harness", harness)):
        if files and len(files) != len(subgoals):
            raise ValueError(f"got {len(subgoals)} subgoal logs but {len(files)} {name} files")
    episodes = []
    for i, sg_file in enumerate(subgoals):
        rows = [r for r in _jsonl(sg_file) if not r.get("autoreset")]
        ev = _eval_episodes(evals[i]) if evals else {}
        hz = {(r["task_id"], r["seed"]): r for r in _jsonl(harness[i])} if harness else {}
        # lerobot-eval seeds episode k of a task with base + k
        base = {}
        for r in rows:
            if r["seed"] is not None:
                base[r["task_id"]] = min(base.get(r["task_id"], r["seed"]), r["seed"])
        for r in rows:
            episodes.append(_episode(r, sg_file, ev, hz, base, suite, step, plan, video_base))
        if harness:
            missing = set(hz) - {(r["task_id"], r["seed"]) for r in rows}
            if missing:
                raise FormatError(f"{harness[i]}: harness episodes without a subgoal log: {sorted(missing)}")
    return episodes


def _episode(r, sg_file, ev, hz, base, suite, step, plan, video_base) -> Episode:
    task_id, seed = r["task_id"], r["seed"]
    index = seed - base[task_id] if seed is not None else None
    stages = [Stage(name, ft, fin, lost) for name, ft, fin, lost in
              zip(r["predicates"], r["first_true"], r["final"], r.get("ever_lost") or [None] * len(r["predicates"]))]
    checks = plan.get(task_id)
    if checks:
        names = [s.name for s in stages]
        unknown = [c for c in checks if c not in names]
        if unknown:
            raise FormatError(f"{sg_file}: task {task_id}: plan checks {unknown} are not goal predicates {names}")
        stages = [stages[names.index(c)] for c in checks] + [s for s in stages if s.name not in checks]
    if step is None:
        task, success = f"{suite}/{task_id}", bool(r["success"])
    else:
        if not checks or not 0 <= step < len(checks):
            raise FormatError(f"{sg_file}: the plan has no step {step} for task {task_id}")
        task, success = f"{suite}/{task_id}:step{step}", stages[step].reached
    video = None
    if task_id in ev and index is not None:
        listed = ev[task_id]["successes"]
        if index >= len(listed):
            raise FormatError(f"{sg_file}: task {task_id} episode {index} is not in the eval file")
        if listed[index] != bool(r["success"]):
            raise FormatError(f"{sg_file}: task {task_id} episode {index} (seed {seed}) succeeded="
                              f"{r['success']} but the eval file says {listed[index]}; "
                              "are these logs from the same evaluation?")
        paths = ev[task_id]["video_paths"]
        if index < len(paths):
            video = os.path.relpath(paths[index], video_base) if video_base else paths[index]
    extra = {"episode_index": index}
    events = []
    h = hz.get((task_id, seed))
    if h is not None:
        events = [(int(t), str(m)) for t, m in h.get("events", [])]
        extra.update(plan=h.get("plan"), reached_step=h.get("reached_step"))
        if h.get("success") != r["success"] or h.get("steps") != r["steps"]:
            raise FormatError(f"{sg_file}: task {task_id} seed {seed}: harness log disagrees with the "
                              f"subgoal log ({h.get('success')}/{h.get('steps')} vs {r['success']}/{r['steps']})")
    return Episode(task=task, success=success, init=r["init_state"], seed=seed, length=r["steps"],
                   stages=stages, events=events, video=video, extra=extra)


def import_run(run_id: str, subgoals, evals=None, harness=None, **kwargs) -> Run:
    """A Run from one or more LIBERO evaluations (see import_libero)."""
    run_kw = {k: kwargs.pop(k) for k in ("policy", "settings", "notes", "created") if k in kwargs}
    episodes = import_libero(subgoals, evals, harness, **kwargs)
    sources = [str(p) for p in [*subgoals, *(evals or []), *(harness or [])]]
    return Run(run_id=run_id, episodes=episodes, benchmark=kwargs.get("suite", "libero_10"),
               sources=sources, **run_kw)
