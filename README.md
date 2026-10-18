# policy-diff

**Find out what your new robot policy broke, before your robot does.**

[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
![Status: early](https://img.shields.io/badge/status-early%20(v0.0.1)-orange.svg)
[![test](https://github.com/saksham18kukreja/policy-diff/actions/workflows/test.yml/badge.svg)](https://github.com/saksham18kukreja/policy-diff/actions/workflows/test.yml)

policy-diff is an open-source tool for failure analysis and regression testing of
learned robot policies, for any robot, simulator or benchmark. Give it per-episode
results from two evaluations and it reports which tasks improved or regressed, whether
each change is real or noise, and which runs shouldn't be compared at all.

> **Status: early.** The log format, importers (any CSV/TSV/JSONL, LeRobot, LIBERO), a
> logging API and `compare` work today. The stage breakdown, timing and HTML report
> are next. Design: [design document](docs/policy-diff_design.docx),
> [one-pager](docs/policy-diff_one-pager.docx). Feedback and issues are welcome.

## Why

Robot policy evaluation usually reports one success rate per task. That number hides
what changed between checkpoints, and whether the change is real. From real logs:

| What was actually happening | What the number said |
|---|---|
| LeRobot's published VQ-BeT PushT checkpoint was re-trained in a refactor and dropped from 63.8% to 57.0% on the same 500 seeds | a new checkpoint, same model card |
| The published ACT *insertion* evaluation is the *transfer-cube* evaluation, byte for byte | "83% on insertion" |
| A fine-tune learned three steps no model could do (0/30 → 15/30) and lost a full task (7/10 → 1/10) | "slightly worse" |
| A 160-step retry timeout cut off 7 of 8 successes of a slower model | "the model is bad" |
| A step went 5 → 2 → 5 out of 10 across checkpoints | looked like a regression; it was noise |

The first two come from LeRobot's public checkpoints ([`examples/lerobot_hub`](examples/lerobot_hub));
the rest from building [Conductor](https://github.com/saksham18kukreja/conductor-vla), a
step-by-step harness for VLAs, where each took hours to find by hand.

## Try it

No dependencies beyond Python 3.9+.

```bash
git clone https://github.com/saksham18kukreja/policy-diff && cd policy-diff
python -m policy_diff compare examples/lerobot_hub/runs/vqbet_pusht_2024-07 examples/lerobot_hub/runs/vqbet_pusht_2025-02
```

```
task     A                       B                        change  pairs  A✓B✗  B✓A✗ unpaired      p      q  verdict
pusht    319/500  64% [ 59- 68]  285/500  57% [ 53- 61]      -7%    500   130    96        -  0.028  0.028  likely real: regressed (paired by seed)
```

Rates come with 95% Wilson intervals. Each task gets an exact test stratified by initial
state (or seed): episodes that started the same way are compared with each other, which
detects smaller changes than comparing two averages, and episodes without a counterpart
still count. *Likely real* means the change survives a false-discovery-rate correction
across tasks; *possible* means it is significant on its own; otherwise it is *noise*.
`compare` also warns when two runs look like the same evaluation logged twice.

Three example sets, from three different setups:

| Example | What | Imported with |
|---|---|---|
| [`lerobot_hub`](examples/lerobot_hub) | LeRobot's published Diffusion Policy, VQ-BeT and ACT evaluations (PushT, ALOHA sim) across Hub revisions | `import lerobot` |
| [`cube_lift`](examples/cube_lift) | a MuJoCo Franka cube-lift benchmark logging the furthest milestone reached and a failure reason | `import table` |
| [`libero_long`](examples/libero_long) | SmolVLA checkpoints on LIBERO-Long with per-goal-condition logs and harness events | `import libero` |

## Get your results in

Each run is a directory with `run.json` and `episodes.jsonl`: one JSON object per
episode. Only `task` and `success` are required. The optional fields are `init` (an
initial-state ID used for pairing), `seed`, `length`, named `stages` (reached, or the
first step they held), timestamped `events`, a `failure` reason, `video`, and anything
else under `extra`. See [`policy_diff/schema.py`](policy_diff/schema.py).

**From your evaluation loop** (any simulator or robot; stage checks are whatever you can compute):

```python
from policy_diff import RunWriter, StageTracker

with RunWriter("runs/ckpt_50k", policy="my_policy@50k", settings={"n_action_steps": 10}) as run:
    for init in range(50):
        obs = env.reset(init)
        stages = StageTracker(["grasped", "lifted", "placed"])
        for t in range(max_steps):
            obs, done = env.step(policy(obs))
            stages.update(t, grasped=env.grasped(), lifted=env.lifted(), placed=env.placed())
            if done:
                break
        run.episode("pick_place", success=env.placed(), init=init, length=t + 1,
                    stages=stages.stages(), robot="arm-2")
```

**From a table you already have** (CSV, TSV or JSONL, one row per episode):

```bash
python -m policy_diff import table results.csv --task task_name --success ok --init start_id \
    --stage grasped --stage lifted --failure reason --const checkpoint=50k --out runs/ckpt_50k
```

**From LeRobot** (`lerobot-eval` output, any environment) or **LIBERO**:

```bash
python -m policy_diff import lerobot outputs/eval/eval_info.json --start-seed 1000 --out runs/ckpt_50k
python -m policy_diff import libero --subgoals subgoals.jsonl --eval eval_info.json --out runs/ckpt_50k
```

Then compare runs, or subsets of them (any field or `extra` key):

```bash
python -m policy_diff compare runs/ckpt_40k runs/ckpt_50k --json diff.json
python -m policy_diff compare runs/fleet runs/fleet --where-a robot=arm-1 --where-b robot=arm-2
```

## What it will do

```bash
policy-diff compare runs/ckpt_40k runs/ckpt_50k --out report.html
```

One self-contained HTML report:

- **What improved and what regressed**, per task, with intervals and a label:
  *likely real*, *possible* or *noise*. Episodes with the same task and initial state
  are paired across runs, which makes small differences detectable.
- **Where failures happen**: each failed episode classified as *nothing done*,
  *partial*, *undone* or *timed out*, per stage, and grouped into failure signatures
  such as "step 1 done, step 2 never started".
- **Timing shifts** per stage, which catch configuration problems (a timeout that no
  longer fits a model) that look like model regressions.
- **Side-by-side videos** of the episodes that flipped between checkpoints.
- **Run settings diff**, so a changed `n_action_steps` isn't mistaken for a better model.

```bash
policy-diff check --baseline runs/main --candidate runs/pr-123 --fail-on likely-regression
```

**CI mode**: exit non-zero when a new checkpoint regresses beyond noise.

## How it works

- **Logs in, report out.** No access to the policy, robot or simulator is needed.
- **A simple open log format** with only task and success required, so it works the
  same for simulation benchmarks and for real-robot logs. Importers for any
  CSV/TSV/JSONL, LeRobot and LIBERO, plus a Python logging API.
- **Failure clustering in two tiers**: deterministic rules on stage outcomes and
  events by default; optional VLM captioning and clustering of failure videos for
  real-robot logs without stage signals.
- **Runs on a laptop CPU**; video and VLM features are optional extras.

## Roadmap

| Version | Scope |
|---|---|
| **v0.1** | ✅ Log format, logging API, table / LeRobot / LIBERO importers, confidence-labelled diffs, duplicate-run checks · next: stage breakdown, failure signatures, timing, HTML report |
| v0.2 | `check` for CI, GitHub Action, settings diff, JSON summary |
| v0.3 | VLM-based failure clustering for real-robot logs |
| v0.4 | Regression suites: turn logged failures into named suites to re-run on every checkpoint |
| Later | Real-world failures as reproducible simulation cases |

The v0.1 launch demo is a real HTML report generated from the three example sets.

## Related work

policy-diff analyses evaluation results; it builds on tools that run and score them:
[LeRobot](https://github.com/huggingface/lerobot),
[XPolicyLab](https://arxiv.org/abs/2608.09892) (running any policy in any
environment), Rollproof (statistical A/B verdicts from logs), the TRI/Cornell
evaluation harness, and research on discovering failure taxonomies from deployment
logs ([arXiv 2506.06570](https://arxiv.org/abs/2506.06570)).

## License

Apache License 2.0. See [LICENSE](LICENSE).
