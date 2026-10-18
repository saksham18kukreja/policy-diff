"""policy-diff command line.

  policy-diff import table episodes.csv --task task --success success --init init --out runs/base
  policy-diff import lerobot eval_info.json --start-seed 1000 --out runs/base
  policy-diff import wandb --project team/evals --filter label=v5.1 --out runs/v5.1
  policy-diff summary runs/base
  policy-diff compare runs/base runs/candidate [--where robot=r2] [--json diff.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

from . import stages, stats
from .adapters.lerobot import import_lerobot
from .adapters.libero import import_libero
from .adapters.table import import_table
from .compare import Comparison, compare
from .schema import EPISODES_FILE, FormatError, Run, load_run, save_run


def _pct(k: int, n: int) -> str:
    lo, hi = stats.wilson(k, n)
    return f"{k:>3}/{n:<3} {100 * k / n:3.0f}% [{100 * lo:3.0f}-{100 * hi:3.0f}]" if n else "-"


def _settings(pairs):
    out = {}
    for item in pairs or []:
        k, _, v = item.partition("=")
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out


def _write(args, episodes, sources) -> int:
    """Save imported episodes as a run (or add them to one with --append)."""
    run = Run(args.run_id or Path(args.out).name, episodes, policy=args.policy, benchmark=args.benchmark,
              settings=_settings(args.setting), sources=[str(p) for p in sources], notes=args.notes,
              created=args.created or date.today().isoformat())
    if args.append and (Path(args.out) / EPISODES_FILE).exists():
        old = load_run(args.out)
        old.episodes += run.episodes
        old.sources += run.sources
        old.settings.update(run.settings)
        run = old
    save_run(run, args.out)
    print(f"{args.out}: {len(run.episodes)} episodes, {len(run.tasks())} tasks, "
          f"{sum(e.success for e in run.episodes)} successes")
    return 0


def _mapping(args) -> dict:
    """Column-mapping options shared by `import table` and `import wandb`."""
    stages = {}
    for item in args.stage or []:
        name, _, col = item.partition("=")
        stages[name] = col or name
    return dict(task=args.task, task_value=args.task_value, success=args.success, init=args.init,
                seed=args.seed, length=args.length, video=args.video, failure=args.failure, stages=stages,
                stage_values=args.stage_values, furthest=args.furthest,
                order=args.order.split(",") if args.order else None, const=_settings(args.const))


def cmd_import_table(args) -> int:
    return _write(args, import_table(args.files, **_mapping(args)), args.files)


def cmd_import_wandb(args) -> int:
    from .adapters.wandb_tables import import_wandb
    split = lambda v: [k for k in (v or "").split(",") if k]
    eps, ids, warnings = import_wandb(args.runs, project=args.project, filters=_where(args.filter),
                                      table=args.table, task_from_config=split(args.task_from_config),
                                      extra_from_config=split(args.extra_from_config), **_mapping(args))
    print(f"read {len(ids)} W&B runs")
    for w in warnings:
        print(f"WARNING: {w}", file=sys.stderr)
    return _write(args, eps, [f"wandb:{i}" for i in ids])


def cmd_import_lerobot(args) -> int:
    eps = import_lerobot(args.files, task=args.task, start_seed=args.start_seed, index_as_init=args.index_as_init,
                         video_base=args.video_base)
    return _write(args, eps, args.files)


def cmd_import_libero(args) -> int:
    if args.benchmark is None:
        args.benchmark = args.suite
    eps = import_libero(args.subgoals, args.eval, args.harness, suite=args.suite, step=args.step,
                        plan=json.loads(Path(args.plan).read_text()) if args.plan else None,
                        video_base=args.video_base)
    return _write(args, eps, [*args.subgoals, *(args.eval or []), *(args.harness or [])])


def _where(items) -> dict:
    out = {}
    for item in items or []:
        k, sep, v = item.partition("=")
        if not sep:
            raise ValueError(f"--where takes KEY=VALUE, got {item!r}")
        out[k] = v
    return out


def cmd_summary(args) -> int:
    run = load_run(args.run).where(**_where(args.where))
    print(f"{run.run_id}  policy={run.policy}  benchmark={run.benchmark}")
    if run.settings:
        print("settings: " + ", ".join(f"{k}={v}" for k, v in run.settings.items()))
    w = max([len(t) for t in run.tasks()] + [3])
    for task, eps in run.by_task().items():
        print(f"  {task:<{w}}  {_pct(sum(e.success for e in eps), len(eps))}")
    print(f"  {'all':<{w}}  {_pct(sum(e.success for e in run.episodes), len(run.episodes))}")
    if args.failures:
        for t in stages.analyze(run):
            print()
            print(format_analysis(t, top=args.top))
    return 0


def _frac(k, n) -> str:
    return f"{k}/{n}" if n else "-"


def format_analysis(t: "stages.TaskAnalysis", top: int = 5) -> str:
    """Stage outcomes and the most common failure signatures of one task."""
    lines = [f"{t.task}: {t.successes}/{t.n} succeeded"]
    if t.stages:
        w = max(len(s.name) for s in t.stages)
        lines.append(f"  {'stage':<{w}}  {'reached':>9} {'held':>9} {'lost':>6} {'median step':>11}")
        for s in t.stages:
            med = "-" if s.median_first is None else f"{s.median_first:.0f}"
            lines.append(f"  {s.name:<{w}}  {_frac(s.reached, s.n):>9} "
                         f"{'-' if s.final is None else _frac(s.final, s.n):>9} "
                         f"{'-' if s.lost is None else s.lost:>6} {med:>11}")
    failed = t.n - t.successes
    if failed:
        classes = ", ".join(f"{k} {c}" for c, k in t.classes.items())
        lines.append(f"  {failed} failures ({classes}):")
        for sig, k in t.signatures[:top]:
            lines.append(f"    {k:>4}  {sig}")
        if len(t.signatures) > top:
            lines.append(f"    ... {len(t.signatures) - top} more signatures")
    return "\n".join(lines)


def format_breakdown(bd: "stages.TaskBreakdown", d, top: int = 5) -> str:
    """Within one changed task: which stages and failure signatures moved."""
    verdict = "noise" if d.direction == "unchanged" else f"{d.label}: {d.direction}"
    lines = [f"{bd.task}: {d.a_success}/{d.a_n} -> {d.b_success}/{d.b_n} ({verdict})"]
    st = [s for s in bd.shifts if s.kind == "stage"]
    sg = sorted((s for s in bd.shifts if s.kind == "signature"), key=lambda s: s.p)[:top]
    w = max([len(r.name) for r in st + sg] + [len("failure signature")])
    for title, rows in (("stage reached", st), ("failure signature", sg)):
        if not rows:
            continue
        lines.append(f"  {title:<{w}}  {'A':>9} {'B':>9}  {'p':>6}  verdict")
        for r in rows:
            v = "noise" if r.label == stats.NOISE else f"{r.label}: {'more' if r.b * r.a_n > r.a * r.b_n else 'less'} often"
            lines.append(f"  {r.name:<{w}}  {_frac(r.a, r.a_n):>9} {_frac(r.b, r.b_n):>9}  {r.p:6.3f}  {v}")
    return "\n".join(lines)


def format_comparison(c: Comparison) -> str:
    w = max([len(d.task) for d in c.tasks] + [7])
    lines = [f"A = {c.a}", f"B = {c.b}", "",
             f"{'task':<{w}}  {'A':<23} {'B':<23} {'change':>7}  {'pairs':>5} {'A✓B✗':>5} {'B✓A✗':>5} {'unpaired':>8}"
             f" {'p':>6} {'q':>6}  verdict"]

    def row(d):
        if d.direction != "unchanged":
            verdict = f"{d.label}: {d.direction}"
        else:
            verdict = "no change" if d.a_success * d.b_n == d.b_success * d.a_n else "noise"
        test = {"exact, by seed": " (paired by seed)", "exact, unpaired": " (unpaired)"}.get(d.test, "")
        return (f"{d.task:<{w}}  {_pct(d.a_success, d.a_n):<23} {_pct(d.b_success, d.b_n):<23} "
                f"{100 * d.delta:+6.0f}%  {d.pairs:>5} {d.a_only_wins:>5} {d.b_only_wins:>5} "
                f"{_unpaired(d):>8} "
                f"{d.p:6.3f} {d.q:6.3f}  {verdict}{test}")

    order = {"regressed": 0, "improved": 1, "unchanged": 2}
    for d in sorted(c.tasks, key=lambda d: (order[d.direction], d.p)):
        lines.append(row(d))
    if c.overall:
        lines += ["-" * (w + 110), row(c.overall)]
    counts = {k: sum(d.direction == k for d in c.tasks) for k in order}
    lines += ["", f"{len(c.tasks)} tasks compared: {counts['regressed']} regressed, {counts['improved']} improved, "
              f"{counts['unchanged']} no detectable change"]
    for w in c.warnings:
        lines.append(f"WARNING: {w}")
    if c.only_in_a:
        lines.append(f"only in A (not compared): {', '.join(c.only_in_a)}")
    if c.only_in_b:
        lines.append(f"only in B (not compared): {', '.join(c.only_in_b)}")
    lines.append(f"likely real: FDR-adjusted q <= {stats.LIKELY_Q}; possible: p <= {stats.POSSIBLE_P} "
                 "before correction. Tests are exact and stratified by initial state;\n"
                 "pairs = episodes matched on task and initial state, A✓B✗ / B✓A✗ = pairs that flipped;\n"
                 "unpaired = A/B episodes with no counterpart (counted by the test, not in the flips).")
    return "\n".join(lines)


def _unpaired(d) -> str:
    if not (d.a_unpaired or d.b_unpaired):
        return "-"
    return f"{d.a_unpaired}/{d.b_unpaired}"


def cmd_compare(args) -> int:
    a = load_run(args.a).where(**_where((args.where or []) + (args.where_a or [])))
    b = load_run(args.b).where(**_where((args.where or []) + (args.where_b or [])))
    for r in (a, b):
        if not r.episodes:
            raise ValueError(f"{r.run_id}: no episodes (check --where)")
    c = compare(a, b)
    print(format_comparison(c))
    out = c.to_dict()
    common = [d.task for d in c.tasks]
    ea = [e for e in a.episodes if e.task in common]
    eb = [e for e in b.episodes if e.task in common]
    reasons = stages.reason_shifts(ea, eb) if args.details != "none" else []
    if reasons:
        w = max(len(r.name) for r in reasons[:args.top])
        print(f"\nFailure reasons across the {len(common)} compared tasks:")
        print(f"  {'reason':<{w}}  {'A':>9} {'B':>9}  {'p':>6} {'q':>6}  verdict")
        for r in reasons[:args.top]:
            v = "noise" if r.label == stats.NOISE else f"{r.label}: {'more' if r.b * r.a_n > r.a * r.b_n else 'less'} often"
            print(f"  {r.name:<{w}}  {_frac(r.a, r.a_n):>9} {_frac(r.b, r.b_n):>9}  {r.p:6.3f} {r.q:6.3f}  {v}")
        out["reasons"] = [r.to_dict() for r in reasons]
    if args.details != "none":
        ta, tb = a.by_task(), b.by_task()
        chosen = [d for d in c.tasks if args.details == "all" or d.direction != "unchanged"]
        if chosen:
            print("\nWhere the changes come from (stage reach and failure signatures, same tests, FDR within each task):")
        out["breakdowns"] = []
        for d in chosen:
            bd = stages.breakdown(d.task, ta[d.task], tb[d.task])
            print()
            print(format_breakdown(bd, d, top=args.top))
            out["breakdowns"].append(bd.to_dict())
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=1)
    if args.wandb:
        from .export_wandb import log_comparison
        bds = [stages.breakdown(d.task, a.by_task()[d.task], b.by_task()[d.task])
               for d in c.tasks if d.direction != "unchanged"] if args.details != "none" else []
        url = log_comparison(c, a, b, args.wandb, breakdowns=bds, reasons=reasons, mode=args.wandb_mode,
                             video_base_a=args.video_base_a, video_base_b=args.video_base_b)
        print(f"\nlogged to W&B: {url}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="policy-diff", description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    imp = sub.add_parser("import", help="convert logs into the policy-diff format")
    imp_sub = imp.add_subparsers(dest="source", required=True)

    def importer(name, help, func):
        q = imp_sub.add_parser(name, help=help)
        q.set_defaults(func=func)
        return q

    def run_options(q):
        g = q.add_argument_group("run")
        g.add_argument("--out", required=True, help="run directory to write")
        g.add_argument("--run-id", help="default: the --out directory name")
        g.add_argument("--policy", help="policy / checkpoint name")
        g.add_argument("--benchmark")
        g.add_argument("--setting", action="append", metavar="KEY=VALUE", help="run setting, repeatable")
        g.add_argument("--notes")
        g.add_argument("--created", help="date (default: today)")
        g.add_argument("--append", action="store_true", help="add episodes to an existing run")

    def table_options(t):
        t.add_argument("--task", help="task column (default: 'task')")
        t.add_argument("--task-value", help="the task for every row, for files without a task column")
        t.add_argument("--success", default="success", help="success column (1/0, true/false, ...)")
        for col in ("init", "seed", "length", "video", "failure"):
            t.add_argument(f"--{col}", help=f"{col} column (default: a column named '{col}', if any)")
        t.add_argument("--stage", action="append", metavar="NAME[=COLUMN]", help="a stage column, repeatable")
        t.add_argument("--stage-values", choices=["bool", "step"], default="bool",
                       help="stage columns hold reached yes/no (bool) or the first step it held (step)")
        t.add_argument("--furthest", help="column naming the furthest stage reached")
        t.add_argument("--order", help="comma-separated stage order for --furthest")
        t.add_argument("--const", action="append", metavar="KEY=VALUE",
                       help="add to every episode's extra (e.g. checkpoint=s0), repeatable")

    t = importer("table", "any CSV / TSV / JSONL with one row per episode", cmd_import_table)
    t.add_argument("files", nargs="+")
    table_options(t)
    run_options(t)

    wb = importer("wandb", "per-episode tables logged to Weights & Biases", cmd_import_wandb)
    wb.add_argument("runs", nargs="*", help="entity/project/run_id paths or local wandb/run-* directories")
    wb.add_argument("--project", help="entity/project: every run of it (narrow with --filter)")
    wb.add_argument("--filter", action="append", metavar="KEY=VALUE", help="only runs whose config matches")
    wb.add_argument("--table", default="episodes", help="summary key of the per-episode table")
    wb.add_argument("--task-from-config", help="comma-separated config keys whose values name the task")
    wb.add_argument("--extra-from-config", help="comma-separated config keys to copy into each episode's extra")
    table_options(wb)
    run_options(wb)

    lr = importer("lerobot", "lerobot-eval eval_info.json, any LeRobot environment", cmd_import_lerobot)
    lr.add_argument("files", nargs="+")
    lr.add_argument("--task", help="task name (older eval_info.json files have none) or task group")
    lr.add_argument("--start-seed", type=int, help="lerobot-eval --seed; pairs runs evaluated with the same seed")
    lr.add_argument("--index-as-init", action="store_true",
                    help="treat the episode index as the initial-state id (only if resets depend on it alone)")
    lr.add_argument("--video-base", help="store video paths relative to this directory")
    run_options(lr)

    lib = importer("libero", "LIBERO goal-condition logs (+ lerobot-eval, Conductor harness)", cmd_import_libero)
    lib.add_argument("--subgoals", nargs="+", required=True, help="eval_with_subgoals.py logs, one per evaluation")
    lib.add_argument("--eval", nargs="+", help="lerobot-eval eval_info.json files, aligned with --subgoals")
    lib.add_argument("--harness", nargs="+", help="Conductor harness logs, aligned with --subgoals")
    lib.add_argument("--suite", default="libero_10")
    lib.add_argument("--plan", help="JSON plan: {task_id: [{instruction, check}, ...]}; orders stages by step")
    lib.add_argument("--step", type=int, help="success = the predicate of plan step N reached (needs --plan)")
    lib.add_argument("--video-base", help="store video paths relative to this directory")
    run_options(lib)

    s = sub.add_parser("summary", help="per-task success of one run")
    s.add_argument("run")
    s.add_argument("--where", action="append", metavar="KEY=VALUE", help="only matching episodes (field or extra)")
    s.add_argument("--failures", action="store_true", help="per task: stage outcomes and failure signatures")
    s.add_argument("--top", type=int, default=5, help="failure signatures to show per task")
    s.set_defaults(func=cmd_summary)

    c = sub.add_parser("compare", help="what changed from run A to run B")
    c.add_argument("a")
    c.add_argument("b")
    c.add_argument("--where", action="append", metavar="KEY=VALUE",
                   help="compare only matching episodes (a field such as task, or an extra key), repeatable")
    c.add_argument("--where-a", action="append", metavar="KEY=VALUE", help="filter run A only")
    c.add_argument("--where-b", action="append", metavar="KEY=VALUE", help="filter run B only")
    c.add_argument("--details", choices=["changed", "all", "none"], default="changed",
                   help="stage and failure-signature breakdown for changed tasks (default), all tasks or none")
    c.add_argument("--top", type=int, default=5, help="failure signatures to show per task")
    c.add_argument("--json", help="also write the comparison as JSON")
    c.add_argument("--wandb", metavar="ENTITY/PROJECT", help="also log the comparison to a W&B run (needs wandb)")
    c.add_argument("--wandb-mode", choices=["online", "offline"], help="W&B mode (default: W&B's own setting)")
    c.add_argument("--video-base-a", help="directory run A's video paths are relative to (for W&B videos)")
    c.add_argument("--video-base-b", help="directory run B's video paths are relative to")
    c.set_defaults(func=cmd_compare)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (FormatError, FileNotFoundError, ValueError) as err:
        print(f"policy-diff: error: {err}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
