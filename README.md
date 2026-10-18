# policy-diff

**Find out what your new robot policy broke, before your robot does.**

[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
![Status: early](https://img.shields.io/badge/status-early%20(v0.0.1)-orange.svg)

policy-diff is an open-source tool for failure analysis and regression testing of
learned robot policies. Point it at rollout logs from simulation or deployment and it
breaks failures down by stage, clusters them into failure modes, compares checkpoints,
and reports which behaviours improved or regressed, with confidence levels and
side-by-side videos.

> **Status: early.** The log format, the LIBERO adapter and `compare` (per-task diffs
> with exact tests and confidence labels) work today; the stage breakdown, timing and the
> HTML report are next. Design: [design document](docs/policy-diff_design.docx),
> [one-pager](docs/policy-diff_one-pager.docx). Feedback and issues are welcome.

## Why

Robot policy evaluation usually reports one success rate per task. That number hides
what changed between checkpoints. These are real cases from building
[Conductor](https://github.com/saksham18kukreja/conductor-vla), a harness that runs
VLAs step by step on LIBERO-Long; each took hours to find by hand:

| What was actually happening | What the average said |
|---|---|
| A fine-tune learned three steps no model could do (0/30 → 15/30) and lost a full task (7/10 → 1/10) | "slightly worse" |
| A 160-step retry timeout cut off 7 of 8 successes of a slower model | "the model is bad" |
| 24% of an end-to-end VLA's episodes finished part of the task, then stalled | "54% success" |
| A step went 5 → 2 → 5 out of 10 across checkpoints | looked like a regression; it was noise |

## Try it

No dependencies beyond Python 3.9+.

```bash
git clone https://github.com/saksham18kukreja/policy-diff && cd policy-diff
python -m policy_diff compare examples/libero_long/runs/smolvla_libero examples/libero_long/runs/finetune_v1_50k
```

```
task                   A                       B                        change  pairs  A✓B✗  B✓A✗      p      q  verdict
libero_10/2             23/30   77% [ 59- 88]    1/10   10% [  2- 40]     -67%     10     6     0  0.003  0.007  likely real: regressed
libero_10/4              7/30   23% [ 12- 41]    3/10   30% [ 11- 60]      +7%      8     1     2  1.000  1.000  noise
```

Rates come with 95% Wilson intervals. Each task gets an exact test stratified by initial
state, which uses every episode and gets the precision of pairing when runs share initial
states; *likely real* means it survives a false-discovery-rate correction across tasks.
[`examples/libero_long`](examples/libero_long) has 14 real runs to compare.

Import your own LIBERO logs (Conductor's goal-condition logger, plus `lerobot-eval`'s
`eval_info.json` and harness logs if you have them):

```bash
python -m policy_diff import libero --subgoals subgoals_s1000.jsonl --eval eval_info.json --out runs/mine
python -m policy_diff summary runs/mine
```

Or write the format directly: a directory with `run.json` and `episodes.jsonl`, one JSON
object per episode, where only `task` and `success` are required
(see [`policy_diff/schema.py`](policy_diff/schema.py)).

## What it will do

```bash
policy-diff compare runs/smolvla_libero runs/finetune_v1 --out report.html
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
- **A simple open log format** (one JSON object per episode: task, initial state,
  success, and optional stage signals, harness events and video). Adapters convert
  LeRobot `lerobot-eval` output, LIBERO goal-condition logs and Conductor harness
  logs; a CSV adapter covers custom logging.
- **Failure clustering in two tiers**: deterministic rules on stage outcomes and
  events by default; optional VLM captioning and clustering of failure videos for
  real-robot logs without stage signals.
- **Runs on a laptop CPU**; video and VLM features are optional extras.

## Roadmap

| Version | Scope |
|---|---|
| **v0.1** | ✅ Log format, LIBERO adapter, confidence-labelled diffs · next: stage breakdown, failure signatures, timing, HTML report, LeRobot-only adapter |
| v0.2 | `check` for CI, GitHub Action, settings diff, JSON summary |
| v0.3 | VLM-based failure clustering for real-robot logs |
| v0.4 | Regression suites: turn logged failures into named suites to re-run on every checkpoint |
| Later | Real-world failures as reproducible simulation cases |

The v0.1 launch demo is a real report on three SmolVLA checkpoints on LIBERO-Long,
generated from the logs Conductor already records.

## Related work

policy-diff analyses evaluation results; it builds on tools that run and score them:
[LeRobot](https://github.com/huggingface/lerobot),
[XPolicyLab](https://arxiv.org/abs/2608.09892) (running any policy in any
environment), Rollproof (statistical A/B verdicts from logs), the TRI/Cornell
evaluation harness, and research on discovering failure taxonomies from deployment
logs ([arXiv 2506.06570](https://arxiv.org/abs/2506.06570)).

## License

Apache License 2.0. See [LICENSE](LICENSE).
