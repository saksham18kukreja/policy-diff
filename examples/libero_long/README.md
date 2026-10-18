# Example: SmolVLA checkpoints on LIBERO-Long

Real evaluation logs from [Conductor](https://github.com/saksham18kukreja/conductor-vla),
converted to the policy-diff format. Every run is LIBERO-Long (`libero_10`), 520 control
steps, with per-episode goal-condition logs; success counts match the numbers reported
in Conductor.

| Run | Policy | Episodes | Success |
|---|---|---|---|
| `smolvla_libero` | `lerobot/smolvla_libero`, end to end | 300 (3 seeds) | 54% |
| `libero90_smolvla` | `Kesvill/smolvla_libero_90`, end to end | 300 (3 seeds) | 0% |
| `libero90_smolvla_conductor` | the same, inside Conductor (6 tasks) | 180 (3 seeds) | 67.8% |
| `libero90_smolvla_steps` | the same, one plan step at a time | 540 (3 seeds) | 68.7% |
| `finetune_v1_{10k,30k,50k}` | Conductor fine-tune v1, full tasks 2 and 4 | 20 each | 30%, 25%, 20% |
| `finetune_v1_{10k,30k,50k}_steps` | the same, single steps | 50 each | 28%, 48%, 56% |
| `finetune_v1_50k_conductor_{t160,t220,t220_n10}` | fine-tune v1 inside Conductor | 100 each | 20%, 28%, 41% |

Step runs score a step as reached when the goal predicate its plan step checks holds
(`conductor_plans.json`); their tasks are named `libero_10/<task>:step<k>`.

```bash
python -m policy_diff compare examples/libero_long/runs/smolvla_libero examples/libero_long/runs/finetune_v1_50k
python -m policy_diff compare examples/libero_long/runs/libero90_smolvla_steps examples/libero_long/runs/finetune_v1_50k_steps
python -m policy_diff compare examples/libero_long/runs/finetune_v1_10k_steps examples/libero_long/runs/finetune_v1_30k_steps
```

What these show, each of which took hours by hand:

- the fine-tune **lost full task 2** (77% → 10%, likely real) while task 4 didn't change;
- it **learned steps the LIBERO-90 model can't do** (left moka pot 0/30 → 8/10, mug into
  the microwave 0/30 → 5/10, likely real);
- the mug step's **5 → 2 → 5 swing** across checkpoints is noise;
- the 160 → 220-step retry timeout change (20% → 28%) is not detectable from success
  alone; that is what the timing analysis (next) is for.

Rebuild the runs from a conductor-vla checkout with
`CONDUCTOR=path/to/conductor-vla ./convert.sh`.

**Note on pairing.** The logged `init` is the LIBERO initial state each episode actually
used, which is not always the episode index: when an episode ends early, LeRobot's
`LiberoEnv` advances its init-state counter on the vector env's autoreset, so two runs
with the same seed can start later episodes from different states. policy-diff pairs
episodes on (task, initial state), not on seed.
