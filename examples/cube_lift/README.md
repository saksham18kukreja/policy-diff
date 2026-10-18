# Example: a custom MuJoCo benchmark (Franka cube lift)

Logs from [Conductor](https://github.com/saksham18kukreja/conductor-vla)'s cube-lift
benchmark: behaviour-cloned pick policies for a simulated Franka, evaluated on the same
200 held-out cube positions, with 3 training seeds per version. The benchmark writes its
own TSV per checkpoint and condition (`idx`, `first_stage`, `success`, `reason`, ...), not
any standard format. That is the point of this example: `convert.sh` imports it with the
generic table importer and no custom code:

```bash
python -m policy_diff import table v5.1__...__skill-pick__color.tsv --task-value pick/color \
  --init idx --failure reason --furthest first_stage --order reach,close,grasp,rise,lift,hold \
  --const checkpoint=s0 --out runs/cube_v5.1 --append
```

| Run | Version | Tasks (perception, condition) | Success |
|---|---|---|---|
| `cube_v4.1` | pick skill with a fixed grasp timer | 2 | 97.0% |
| `cube_v5.0` | timer replaced by a "settled" trigger | 2 | 90.4% |
| `cube_v5.1` | contact trigger | 8 | 97.9% |

Tasks are `<mode>/<perception>[_<condition>]`: `pick` (the skill) or `policy` (the raw
policy); `gt` (ground-truth cube pose) or `color` (camera-based detection); `d1`/`d2`
(distractor cubes), `after_place`/`after_push` (starting from where another skill left
the arm). Each episode records the furthest milestone reached, the failure reason, and
its checkpoint (in `extra`).

```bash
python -m policy_diff compare examples/cube_lift/runs/cube_v5.0 examples/cube_lift/runs/cube_v5.1
python -m policy_diff compare examples/cube_lift/runs/cube_v5.1 examples/cube_lift/runs/cube_v5.1 \
  --where-a checkpoint=s1 --where-b checkpoint=s2
```

What these show:

- **v5.0 → v5.1 on `pick/color`: 497 → 570 of 600, likely real, and no position got
  worse** (0 pairs flipped to failure). v5.0's failures were 73 timeouts stuck at grasp
  or rise; v5.1's 30 remaining failures are all `target_not_detected`, the same 10
  positions for every checkpoint: a perception limit, not the policy. The breakdown
  shows it directly: `stopped after grasp [timeout]` 41 → 0, `stopped after rise
  [timeout]` 30 → 0, `no stage reached [target_not_detected]` 30 → 30.
- **One v5.1 training seed regresses after a place**: s2 drops or times out on 11 of 200
  `after_place` episodes, s1 on none (likely real); the other 7 conditions are identical.
  The pooled version comparison hides this; `--where` on `extra.checkpoint` finds it.

Rebuild with `CONDUCTOR=path/to/conductor-vla ./convert.sh`.
