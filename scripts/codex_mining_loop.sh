#!/bin/sh
set -eu

run_id=${1:?usage: scripts/codex_mining_loop.sh RUN_ID [ROUNDS]}
rounds=${2:-5}

docker compose run --rm --build research \
  python -m alpha_lab.cli mining-init --run "$run_id" --rounds "$rounds"

round_number=1
while [ "$round_number" -le "$rounds" ]; do
  round_label=$(printf '%04d' "$round_number")
  decision="experiments/$run_id/round_$round_label/decision.json"
  if [ -f "$decision" ]; then
    round_number=$((round_number + 1))
    continue
  fi

  factor_number=1000
  while :; do
    factor_id=$(printf 'F%04d' "$factor_number")
    if [ ! -e "src/alpha_lab/factors/candidates/$factor_id.py" ]; then
      break
    fi
    factor_number=$((factor_number + 1))
  done

  proposal="experiments/$run_id/proposals/round_$round_label.json"
  codex exec \
    --sandbox workspace-write \
    --output-schema schemas/proposal.schema.json \
    -o "$proposal" \
    "Use \$factor-mine. Read AGENTS.md, experiments/$run_id/research_brief.md, the run manifest, and prior decisions. Produce exactly one proposal for run_id=$run_id, round_number=$round_number, factor_id=$factor_id. Do not edit locked areas or claim metrics."
  make mining-round RUN="$run_id" PROPOSAL="$proposal"
  round_number=$((round_number + 1))
done

make report RUN="$run_id"
