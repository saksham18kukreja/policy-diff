#!/usr/bin/env bash
# Convert Conductor's cube-lift benchmark episodes (one TSV per checkpoint and condition)
# into policy-diff runs with the generic table importer; nothing here is cube-specific code.
#   CONDUCTOR=path/to/conductor-vla ./convert.sh
#
# TSV columns: idx (cube start position, the same 200 held-out positions in every run),
# first_stage (furthest of reach/close/grasp/rise/lift/hold reached), success, reason, ...
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="$HERE/runs"
cd "${CONDUCTOR:?set CONDUCTOR to a conductor-vla checkout}/results/episodes"
pd() { PYTHONPATH="$HERE/../.." python3 -m policy_diff "$@"; }

# add FILE TASK RUN [settings...]: one TSV as one task of a run
add() {
  local file=$1 task=$2 run=$3; shift 3
  pd import table "$file" --task-value "$task" --init idx --failure reason \
    --furthest first_stage --order reach,close,grasp,rise,lift,hold \
    --benchmark cube_lift --out "$OUT/$run" --append "$@" >/dev/null
}

rm -rf -- "${OUT:?}"
for f in v4.1__* v5.0__* v5.1__*; do
  version=${f%%__*}
  rest=${f#*__*__}                         # skip the checkpoint name
  ckpt=$(echo "$f" | sed -E 's/.*_(s[0-9])__.*/\1/')
  IFS=_ read -r mode perception condition <<<"$(echo "${rest%.tsv}" | sed 's/__/_/g')"
  task="${mode/skill-/}/$perception${condition:+_$condition}"
  # each version pools its training seeds (v4.1 has no s2 after_place run);
  # compare single checkpoints with --where checkpoint=s1; v5.1 also gets one run per checkpoint
  add "$f" "$task" "cube_$version" --policy "bc pick $version" --const checkpoint=$ckpt
done
for r in "$OUT"/*; do pd summary "$r" | head -1; done
