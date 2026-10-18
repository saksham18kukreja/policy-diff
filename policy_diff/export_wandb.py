"""Log a comparison to Weights & Biases.

policy-diff does the analysis; W&B is where many teams already look at
results. This logs one W&B run (job_type "policy-diff") with:

  summary   counts of regressed / improved tasks, the overall rates, p and verdict
  tasks     one row per compared task: rates, pairs, flips, p, q, verdict
  shifts    stage and failure-signature changes within changed tasks
  reasons   failure reasons across tasks
  flips     the paired episodes that flipped, with both runs' failure signatures
            and videos (as W&B videos when the files exist)

Needs the `wandb` package.
"""

from __future__ import annotations

import os
from typing import Optional

from . import stages, stats
from .compare import Comparison, pair_episodes
from .schema import Run


def _video(wandb, path: Optional[str], base: Optional[str]):
    if not path:
        return None
    full = os.path.join(base, path) if base and not os.path.isabs(path) else path
    return wandb.Video(full) if os.path.exists(full) else path


def log_comparison(c: Comparison, a: Run, b: Run, project: str, *, breakdowns=(), reasons=(),
                   name: Optional[str] = None, mode: Optional[str] = None, video_base_a: Optional[str] = None,
                   video_base_b: Optional[str] = None, max_flips: int = 50) -> str:
    """Log the comparison as a W&B run; returns the run's URL (or its local directory when offline)."""
    try:
        import wandb
    except ImportError:
        raise ImportError("exporting to W&B needs the wandb package: pip install wandb") from None
    entity, _, proj = project.rpartition("/")
    run = wandb.init(project=proj, entity=entity or None, name=name or f"policy-diff: {c.a} vs {c.b}",
                     job_type="policy-diff", mode=mode, reinit="finish_previous",
                     config={"a": c.a, "b": c.b, "a_policy": a.policy, "b_policy": b.policy,
                             "a_settings": a.settings, "b_settings": b.settings,
                             "a_sources": a.sources, "b_sources": b.sources,
                             "likely_q": stats.LIKELY_Q, "possible_p": stats.POSSIBLE_P})

    def verdict(d):
        return "no change" if d.delta == 0 else ("noise" if d.direction == "unchanged" else f"{d.label}: {d.direction}")

    cols = ["task", "a_success", "a_n", "a_rate", "b_success", "b_n", "b_rate", "delta", "pairs",
            "a_only_wins", "b_only_wins", "a_unpaired", "b_unpaired", "test", "p", "q", "verdict"]
    rows = [[d.task, d.a_success, d.a_n, d.a_rate, d.b_success, d.b_n, d.b_rate, d.delta, d.pairs,
             d.a_only_wins, d.b_only_wins, d.a_unpaired, d.b_unpaired, d.test, d.p, d.q, verdict(d)]
            for d in c.tasks]
    log = {"tasks": wandb.Table(columns=cols, data=rows)}

    shift_cols = ["task", "kind", "name", "a", "a_n", "b", "b_n", "p", "q", "label"]
    shift_rows = [[bd.task, s.kind, s.name, s.a, s.a_n, s.b, s.b_n, s.p, s.q, s.label]
                  for bd in breakdowns for s in bd.shifts]
    if shift_rows:
        log["shifts"] = wandb.Table(columns=shift_cols, data=shift_rows)
    if reasons:
        log["reasons"] = wandb.Table(columns=["reason", "a", "a_n", "b", "b_n", "p", "q", "label"],
                                     data=[[r.name, r.a, r.a_n, r.b, r.b_n, r.p, r.q, r.label] for r in reasons])

    ta, tb = a.by_task(), b.by_task()
    flips = []
    for d in c.tasks:
        if d.direction == "unchanged":
            continue
        pairs, _, _ = pair_episodes(ta[d.task], tb[d.task])
        flipped = [(x, y) for x, y in pairs if x.success != y.success]
        # the direction of the change first
        flipped.sort(key=lambda xy: xy[0].success == (d.direction == "improved"))
        max_len = max([e.length for e in ta[d.task] + tb[d.task] if e.length is not None], default=None)
        for x, y in flipped[:max_flips]:
            flips.append([d.task, x.init, x.seed, y.seed, x.success, y.success,
                          stages.signature(x, max_len), stages.signature(y, max_len),
                          _video(wandb, x.video, video_base_a), _video(wandb, y.video, video_base_b)])
    if flips:
        log["flips"] = wandb.Table(columns=["task", "init", "a_seed", "b_seed", "a_success", "b_success",
                                            "a_signature", "b_signature", "a_video", "b_video"], data=flips)
    run.log(log)
    o = c.overall
    run.summary.update({
        "tasks_compared": len(c.tasks),
        "tasks_regressed": sum(d.direction == "regressed" for d in c.tasks),
        "tasks_improved": sum(d.direction == "improved" for d in c.tasks),
        "likely_regressions": sum(d.direction == "regressed" and d.label == stats.LIKELY for d in c.tasks),
        "overall_a_rate": o.a_rate if o else None, "overall_b_rate": o.b_rate if o else None,
        "overall_p": o.p if o else None, "overall_verdict": verdict(o) if o else None,
        "warnings": len(c.warnings)})
    if c.warnings:
        run.summary["warning_text"] = " | ".join(c.warnings)
    url = run.url or run.dir
    run.finish()
    return url
