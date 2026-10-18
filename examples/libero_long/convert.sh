#!/usr/bin/env bash
# Convert the Conductor LIBERO-Long logs into policy-diff runs (examples/libero_long/runs).
#   CONDUCTOR=path/to/conductor-vla ./convert.sh
# Video paths are stored relative to ~/experiments/libero_runs, where the evaluations ran;
# the videos themselves are not in either repository.
set -euo pipefail
# Runs from the conductor-vla results directory, so sources are recorded relative to it.
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="$HERE/runs"
PLAN="$HERE/conductor_plans.json"
cd "${CONDUCTOR:?set CONDUCTOR to a conductor-vla checkout}/results/libero"
R=.
pd() { PYTHONPATH="$HERE/../.." python3 -m policy_diff "$@"; }
seeds=(1000 2000 3000)
each() { for s in "${seeds[@]}"; do printf "$1 " "$s"; done; }

# lerobot/smolvla_libero, end to end on the full instruction (the baseline)
pd import libero --plan "$PLAN" --out "$OUT/smolvla_libero" --policy lerobot/smolvla_libero \
  --subgoals $(each "$R/subgoals_s%s.jsonl") --eval $(each "$R/smolvla_libero10_s%s.json") \
  --video-base "$HOME/experiments/libero_runs" --setting n_action_steps=50 --setting max_steps=520

# Community LIBERO-90 SmolVLA, end to end
pd import libero --plan "$PLAN" --out "$OUT/libero90_smolvla" --policy Kesvill/smolvla_libero_90 \
  --subgoals $(each "$R/kesvill_l90/subgoals_s%s.jsonl") --eval $(each "$R/kesvill_l90/full_instruction_s%s.json") \
  --video-base "$HOME/experiments/libero_runs" --setting n_action_steps=50 --setting max_steps=520

# LIBERO-90 SmolVLA inside Conductor (M3: start-state check, recovery, retry)
pd import libero --plan "$PLAN" --out "$OUT/libero90_smolvla_conductor" --policy Kesvill/smolvla_libero_90 \
  --subgoals $(each "$R/kesvill_l90/m3/subgoals_s%s.jsonl") --harness $(each "$R/kesvill_l90/m3/harness_s%s.jsonl") \
  --eval $(each "$R/kesvill_l90/m3/eval_s%s.json") --video-base "$HOME/experiments/libero_runs" \
  --setting harness=conductor-m3 --setting step_timeout=160 --setting n_action_steps=50

# LIBERO-90 SmolVLA, one step at a time from the episode start (step success)
rm -f "$OUT"/libero90_smolvla_steps/episodes.jsonl
for k in 0 1; do
  pd import libero --plan "$PLAN" --append --out "$OUT/libero90_smolvla_steps" --policy Kesvill/smolvla_libero_90 --step $k \
    --subgoals $(each "$R/kesvill_l90/m1/step${k}_libero90_s%s.jsonl") --setting wording=libero90
done

# Fine-tune v1 checkpoints (seed 1000): full tasks 2 and 4, and single steps
for ck in 10k 30k 50k; do
  pd import libero --plan "$PLAN" --out "$OUT/finetune_v1_$ck" --policy "finetune_v1@$ck" \
    --subgoals "$R/finetune_v1/eval_full$ck/full.jsonl" --setting n_action_steps=50
  rm -f "$OUT/finetune_v1_${ck}_steps/episodes.jsonl"
  for k in 0 1; do
    pd import libero --plan "$PLAN" --append --out "$OUT/finetune_v1_${ck}_steps" --policy "finetune_v1@$ck" --step $k \
      --subgoals "$R/finetune_v1/eval_full$ck/step$k.jsonl" --setting wording=conductor
  done
done

# Fine-tune v1 (50k) inside Conductor: 160-step timeout, 220, 220 + re-planning every 10 actions
for v in "m3_50k t160 160 50" "m3_50k_t220 t220 220 50" "m3_50k_t220_n10 t220_n10 220 10"; do
  set -- $v
  pd import libero --plan "$PLAN" --out "$OUT/finetune_v1_50k_conductor_$2" --policy finetune_v1@50k \
    --subgoals "$R/finetune_v1/$1/subgoals_s1000.jsonl" --harness "$R/finetune_v1/$1/harness_s1000.jsonl" \
    --setting harness=conductor-m3 --setting step_timeout=$3 --setting n_action_steps=$4
done
