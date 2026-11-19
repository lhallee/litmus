#!/usr/bin/env bash
# The teacher after embedding, one step after another, inside the litmus container (launch/in_container.sh):
#   bash launch/in_container.sh bash launch/teacher_chain.sh
# Split, head search with its runs of record, the bank, the export, then the baselines, the control's bank and calibration.
# Every step writes under $ROOT and is skipped when its output is there, so a stopped chain is run again as it is.
# SEARCH_MINUTES, TRIAL_STEPS, TOP and SEEDS size the search to the time available.
set -euo pipefail
ROOT="${ROOT:-/work/teacher}"
CORPUS="${CORPUS:-/work/litmus_corpus/corpus/data}"
EMBED="${EMBED:-/work/embed}"
EXPERIMENT="${EXPERIMENT:-2026-11-17_first_build}"
SEARCH_MINUTES="${SEARCH_MINUTES:-12}"
TRIAL_STEPS="${TRIAL_STEPS:-4000}"
TOP="${TOP:-2}"
SEEDS="${SEEDS:-0 1 2}"
SPLIT="$ROOT/split/split.parquet"
PILOT="$EMBED/modernbert_base/pilot"
mkdir -p "$ROOT"

step() {
    echo "== $(date -u +%H:%M:%S) $1" >&2
}

step split
[[ -f "$SPLIT" ]] || python -m litmus_teacher split --corpus "$CORPUS" --embeddings "$EMBED/modernbert_base" --out "$ROOT/split"

step search
[[ -f "$ROOT/heads/selection.json" ]] || python -m litmus_teacher search --corpus "$CORPUS" --split "$SPLIT" \
    --embedding-root "$EMBED" --out "$ROOT/heads" --experiment-id "$EXPERIMENT" --max-minutes "$SEARCH_MINUTES" \
    --trial-steps "$TRIAL_STEPS" --top "$TOP" --seeds $SEEDS

step bank
[[ -f "$ROOT/bank/bank_selection.json" ]] || python -m litmus_teacher bank --corpus "$CORPUS" --split "$SPLIT" \
    --embedding-root "$EMBED" --train-dir "$ROOT/heads" --out "$ROOT/bank" --drift "$PILOT/drift.npz" "$PILOT/reuse_drift.npz"

# The export comes before the baselines and the calibration: the student distills from the head's queries and keys
# alone, so it starts as soon as they are written (the control's bank alone took 7 minutes in a trial run).
step export
[[ -f "$ROOT/export/manifest.json" ]] || python -m litmus_teacher export --train-dir "$ROOT/heads" --embedding-root "$EMBED" \
    --split "$SPLIT" --bank-dir "$ROOT/bank" --out "$ROOT/export"

step baselines
[[ -f "$ROOT/baselines/logistic/modernbert_base/done.json" ]] || python -m litmus_teacher train --corpus "$CORPUS" --split "$SPLIT" \
    --embedding-root "$EMBED" --out "$ROOT/baselines" --experiment-id "$EXPERIMENT" --baselines-only --control --logistic \
    --seeds 0 --max-steps "$TRIAL_STEPS"

step "control bank"
variant="$(python -c 'import json, sys; print(json.load(open(sys.argv[1]))["chosen"]["variant"])' "$ROOT/bank/bank_selection.json")"
size="$(python -c 'import json, sys; print(json.load(open(sys.argv[1]))["chosen"]["size"])' "$ROOT/bank/bank_selection.json")"
control="$(basename "$(ls -d "$ROOT"/baselines/heads/random_* | head -1)")"
[[ -f "$ROOT/bank_control/bank_selection.json" ]] || python -m litmus_teacher bank --corpus "$CORPUS" --split "$SPLIT" \
    --embedding-root "$EMBED" --train-dir "$ROOT/baselines" --head "$control" --variant "$variant" --size "$size" \
    --out "$ROOT/bank_control"

step calibrate
[[ -f "$ROOT/calibration/calibration.json" ]] || python -m litmus_teacher calibrate --corpus "$CORPUS" --bank-dir "$ROOT/bank" \
    --control-bank-dir "$ROOT/bank_control" --logistic "$ROOT"/baselines/logistic/*/predictions.npz --out "$ROOT/calibration"

step done
