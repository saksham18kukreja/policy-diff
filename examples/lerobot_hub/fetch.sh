#!/usr/bin/env bash
# Download the eval_info.json that LeRobot's official checkpoints publish on the Hugging Face
# Hub, at pinned revisions, and convert them with the generic LeRobot importer.
#   ./fetch.sh
# Source models are Apache-2.0 (github.com/huggingface/lerobot, huggingface.co/lerobot).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
RAW="$HERE/eval_info" OUT="$HERE/runs"
mkdir -p "$RAW"
pd() { PYTHONPATH="$HERE/../.." python3 -m policy_diff "$@"; }

# run_id  model  revision  task
while read -r run model rev task; do
  f="$RAW/$run.json"
  [ -s "$f" ] || curl -sfL -m 120 -o "$f" "https://huggingface.co/lerobot/$model/resolve/$rev/eval_info.json"
  seed=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['per_episode'][0]['seed'])" "$f")
  pd import lerobot "$f" --task "$task" --policy "lerobot/$model@${rev:0:8}" --benchmark "$task" \
    --setting start_seed="$seed" --notes "eval_info.json from huggingface.co/lerobot/$model at $rev" \
    --out "$OUT/$run"
done <<'RUNS'
diffusion_pusht_2024-05    diffusion_pusht                   37628ec997e131b87e648f6fde44fe73de3fde30  pusht
diffusion_pusht_2025-02    diffusion_pusht                   a25d928d3dec2281a40103fd92e55c00258c4dbc  pusht
vqbet_pusht_2024-07        vqbet_pusht                       8977f0fc4b75d476f0999acf8bf68fe10f27d298  pusht
vqbet_pusht_2025-02        vqbet_pusht                       43c6c4deda2cb3cfba79c866b6bcdb66970b5c9f  pusht
act_transfer_cube_2024-05  act_aloha_sim_transfer_cube_human 3c3b58dac3f2c26a5c0d8d7e149df632efe69972  aloha_transfer_cube
act_transfer_cube_2025-02  act_aloha_sim_transfer_cube_human 561867653d2f5e94686fa01e3c0b4e3fad34baec  aloha_transfer_cube
act_insertion_2025-01      act_aloha_sim_insertion_human     04be2f0a2f889f89c7e8df5587a63f6b4e1711be  aloha_insertion
RUNS
