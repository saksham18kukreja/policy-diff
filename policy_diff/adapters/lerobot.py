"""Import `lerobot-eval` output (eval_info.json) for any LeRobot environment.

eval_info.json has per-episode success, rewards and video paths, but not the
initial state. lerobot-eval seeds episode k of each task with seed + k, so
pass the evaluation's `--seed` as `start_seed` to pair runs evaluated with
the same seed. Use `index_as_init` only if your environment's reset depends
on nothing but the episode index (not true for LIBERO's vector envs, whose
init-state counter advances on autoresets; use the LIBERO adapter there).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional, Sequence

from ..schema import Episode, FormatError


def import_lerobot(paths: Sequence[str], *, task: Optional[str] = None, start_seed: Optional[int] = None,
                   index_as_init: bool = False, video_base: Optional[str] = None) -> list[Episode]:
    """`task` names the task (older files don't) or replaces the task group ("<task>/<task_id>")."""
    episodes = []
    for path in paths:
        info = json.loads(Path(path).read_text())
        if "per_task" in info:  # LeRobot >= 0.3: per task, metrics as lists indexed by episode
            for t in info["per_task"]:
                m = t["metrics"]
                name = f"{task or t.get('task_group', 'task')}/{t['task_id']}"
                videos = m.get("video_paths") or []
                for k, ok in enumerate(m["successes"]):
                    extra = {"episode_index": k}
                    for metric in ("sum_rewards", "max_rewards"):
                        if metric in m and k < len(m[metric]):
                            extra[metric[:-1]] = m[metric][k]
                    episodes.append(_episode(name, ok, k, start_seed, index_as_init,
                                             videos[k] if k < len(videos) else None, video_base, extra))
        elif "per_episode" in info:  # older LeRobot: one entry per episode, single task, with seeds
            first = len(episodes)
            for e in info["per_episode"]:
                k = e["episode_ix"]
                extra = {"episode_index": k, "sum_reward": e.get("sum_reward"), "max_reward": e.get("max_reward")}
                seed = e.get("seed")
                ep = _episode(task or "task", e["success"], k, start_seed, index_as_init, None, video_base, extra)
                if seed is not None:
                    ep.seed = seed
                episodes.append(ep)
            videos = info.get("video_paths") or []
            for ep, v in zip(episodes[first:], videos):
                ep.video = os.path.relpath(v, video_base) if video_base else v
        else:
            raise FormatError(f"{path}: not a lerobot-eval eval_info.json (no per_task or per_episode)")
    return episodes


def _episode(task, ok, k, start_seed, index_as_init, video, video_base, extra) -> Episode:
    if video and video_base:
        video = os.path.relpath(video, video_base)
    return Episode(task=task, success=bool(ok), init=k if index_as_init else None,
                   seed=start_seed + k if start_seed is not None else None, video=video, extra=extra)
