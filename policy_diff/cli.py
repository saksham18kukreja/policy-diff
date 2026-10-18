"""policy-diff command line.

  policy-diff import libero --subgoals s1000.jsonl --eval eval_info.json --out runs/base
  policy-diff summary runs/base
  policy-diff compare runs/base runs/candidate [--json diff.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date

from . import stats
from .adapters.libero import import_run
from .compare import Comparison, compare
from .schema import FormatError, load_run, save_run


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


def cmd_import_libero(args) -> int:
    run = import_run(args.run_id or args.out.rstrip("/").split("/")[-1], args.subgoals, args.eval,
                     args.harness, suite=args.suite, step=args.step, video_base=args.video_base,
                     plan=json.loads(open(args.plan).read()) if args.plan else None,
                     policy=args.policy, settings=_settings(args.setting), notes=args.notes,
                     created=args.created or date.today().isoformat())
    if args.append:
        try:
            old = load_run(args.out)
        except FileNotFoundError:
            old = None
        if old is not None:
            old.episodes += run.episodes
            old.sources += run.sources
            old.settings.update(run.settings)
            run = old
    save_run(run, args.out)
    print(f"{args.out}: {len(run.episodes)} episodes, {len(run.tasks())} tasks, "
          f"{sum(e.success for e in run.episodes)} successes")
    return 0


def cmd_summary(args) -> int:
    run = load_run(args.run)
    print(f"{run.run_id}  policy={run.policy}  benchmark={run.benchmark}")
    if run.settings:
        print("settings: " + ", ".join(f"{k}={v}" for k, v in run.settings.items()))
    for task, eps in run.by_task().items():
        print(f"  {task:<22} {_pct(sum(e.success for e in eps), len(eps))}")
    n = len(run.episodes)
    print(f"  {'all':<22} {_pct(sum(e.success for e in run.episodes), n)}")
    return 0


def format_comparison(c: Comparison) -> str:
    lines = [f"A = {c.a}", f"B = {c.b}", "",
             f"{'task':<22} {'A':<23} {'B':<23} {'change':>7}  {'pairs':>5} {'A✓B✗':>5} {'B✓A✗':>5}"
             f" {'p':>6} {'q':>6}  verdict"]

    def row(d):
        verdict = "noise" if d.direction == "unchanged" else f"{d.label}: {d.direction}"
        test = "" if d.test == "exact, by initial state" else " (unpaired)"
        return (f"{d.task:<22} {_pct(d.a_success, d.a_n):<23} {_pct(d.b_success, d.b_n):<23} "
                f"{100 * d.delta:+6.0f}%  {d.pairs:>5} {d.a_only_wins:>5} {d.b_only_wins:>5} "
                f"{d.p:6.3f} {d.q:6.3f}  {verdict}{test}")

    order = {"regressed": 0, "improved": 1, "unchanged": 2}
    for d in sorted(c.tasks, key=lambda d: (order[d.direction], d.p)):
        lines.append(row(d))
    if c.overall:
        lines += ["-" * 120, row(c.overall)]
    counts = {k: sum(d.direction == k for d in c.tasks) for k in order}
    lines += ["", f"{len(c.tasks)} tasks compared: {counts['regressed']} regressed, {counts['improved']} improved, "
              f"{counts['unchanged']} no detectable change"]
    if c.only_in_a:
        lines.append(f"only in A (not compared): {', '.join(c.only_in_a)}")
    if c.only_in_b:
        lines.append(f"only in B (not compared): {', '.join(c.only_in_b)}")
    lines.append(f"likely real: FDR-adjusted q <= {stats.LIKELY_Q}; possible: p <= {stats.POSSIBLE_P} "
                 "before correction. Tests are exact and stratified by initial state;\n"
                 "pairs = episodes matched on task and initial state, A✓B✗ / B✓A✗ = pairs that flipped.")
    return "\n".join(lines)


def cmd_compare(args) -> int:
    a, b = load_run(args.a), load_run(args.b)
    c = compare(a, b)
    print(format_comparison(c))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(c.to_dict(), f, indent=1)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="policy-diff", description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    imp = sub.add_parser("import", help="convert logs into the policy-diff format")
    imp_sub = imp.add_subparsers(dest="source", required=True)
    lib = imp_sub.add_parser("libero", help="LIBERO goal-condition logs (+ lerobot-eval, Conductor harness)")
    lib.add_argument("--subgoals", nargs="+", required=True, help="eval_with_subgoals.py logs, one per evaluation")
    lib.add_argument("--eval", nargs="+", help="lerobot-eval eval_info.json files, aligned with --subgoals")
    lib.add_argument("--harness", nargs="+", help="Conductor harness logs, aligned with --subgoals")
    lib.add_argument("--out", required=True, help="run directory to write")
    lib.add_argument("--run-id")
    lib.add_argument("--policy")
    lib.add_argument("--suite", default="libero_10")
    lib.add_argument("--plan", help="JSON plan: {task_id: [{instruction, check}, ...]}; orders stages by step")
    lib.add_argument("--step", type=int, help="success = the predicate of plan step N reached (needs --plan)")
    lib.add_argument("--video-base", help="store video paths relative to this directory")
    lib.add_argument("--setting", action="append", metavar="KEY=VALUE", help="run setting, repeatable")
    lib.add_argument("--notes")
    lib.add_argument("--created", help="date (default: today)")
    lib.add_argument("--append", action="store_true", help="add episodes to an existing run")
    lib.set_defaults(func=cmd_import_libero)

    s = sub.add_parser("summary", help="per-task success of one run")
    s.add_argument("run")
    s.set_defaults(func=cmd_summary)

    c = sub.add_parser("compare", help="what changed from run A to run B")
    c.add_argument("a")
    c.add_argument("b")
    c.add_argument("--json", help="also write the comparison as JSON")
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
