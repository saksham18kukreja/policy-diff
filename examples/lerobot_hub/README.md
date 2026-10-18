# Example: LeRobot's published checkpoints (PushT, ALOHA)

LeRobot's official checkpoints on the Hugging Face Hub ship the `eval_info.json` their
evaluation produced: 500 episodes each, with per-episode success, reward and seed. The
Hub keeps every revision, so the results published before and after LeRobot's February 2025
refactor (`remove_hydra`, re-trained and re-evaluated checkpoints) can be compared.
`fetch.sh` downloads them at pinned revisions and converts them with the generic
`import lerobot`; nothing here is specific to these environments. Source models:
Apache-2.0, [huggingface.co/lerobot](https://huggingface.co/lerobot).

| Run | Model @ revision | Seeds | Success |
|---|---|---|---|
| `diffusion_pusht_2024-05` | `lerobot/diffusion_pusht` @ `37628ec9` | 100000+ | 63.8% |
| `diffusion_pusht_2025-02` | `lerobot/diffusion_pusht` @ `a25d928d` | 1000+ | 65.4% |
| `vqbet_pusht_2024-07` | `lerobot/vqbet_pusht` @ `8977f0fc` | 100000+ | 63.8% |
| `vqbet_pusht_2025-02` | `lerobot/vqbet_pusht` @ `43c6c4de` | 100000+ | 57.0% |
| `act_transfer_cube_2024-05` | `lerobot/act_aloha_sim_transfer_cube_human` @ `3c3b58da` | 1000+ | 87.6% |
| `act_transfer_cube_2025-02` | `lerobot/act_aloha_sim_transfer_cube_human` @ `56186765` | 1000+ | 83.0% |
| `act_insertion_2025-01` | `lerobot/act_aloha_sim_insertion_human` @ `04be2f0a` | 1000+ | 83.0% |

```bash
python -m policy_diff compare examples/lerobot_hub/runs/vqbet_pusht_2024-07 examples/lerobot_hub/runs/vqbet_pusht_2025-02
python -m policy_diff compare examples/lerobot_hub/runs/act_transfer_cube_2025-02 examples/lerobot_hub/runs/act_insertion_2025-01
```

What policy-diff finds:

- **VQ-BeT on PushT regressed after the refactor**: 63.8% → 57.0%, on the same 500 seeds
  (paired, q = 0.03, likely real). 130 seeds went from success to failure, 96 the other way.
- **ACT on ALOHA transfer-cube regressed too**: 87.6% → 83.0% on the same seeds (q = 0.02).
- **Diffusion Policy on PushT didn't detectably change** (63.8% → 65.4%, noise). Its two
  evaluations used different seeds, so the comparison is unpaired.
- **Diffusion vs VQ-BeT** (current checkpoints): 65.4% vs 57.0%, likely real (unpaired).
- **The ACT insertion checkpoint's eval_info.json is the transfer-cube evaluation.** It is
  byte-identical to `act_aloha_sim_transfer_cube_human`'s current file, its video paths say
  `act_aloha_transfer`, and no revision of the insertion repo has a different one, so the
  83% published for insertion is not an insertion result. `compare` flags it:

```
WARNING: act_transfer_cube_2025-02 and act_insertion_2025-01 have identical per-episode results
(500 episodes, ignoring task names): the same evaluation logged twice?
```

These are statements about the published evaluation files, not new evaluations: a
regression here means the checkpoint published later did worse in its own published
evaluation, which may also reflect evaluation-code changes in the refactor.
