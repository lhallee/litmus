#!/usr/bin/env bash
# One student size after the teacher's export, one step after another, inside the litmus container:
#   bash launch/in_container.sh bash -c 'export SIZE=s256_l6 MINUTES=150; bash launch/student_chain.sh'
# Train in a time box, export the release from the checkpoint chosen on validation, evaluate it on validation and
# once on test, then score the benchmarks. Every step writes under $ROOT/$SIZE and is skipped when its output is
# there, so a stopped chain is run again as it is. The rows of the token store (`litmus_student.data --split`) are
# written once, before any size starts.
set -euo pipefail
SIZE="${SIZE:?the student config, for example s256_l6}"
MINUTES="${MINUTES:?the training time box in minutes}"
ROOT="${ROOT:-/work/student}"
DATA="${DATA:-/work/student_store}"
CORPUS="${CORPUS:-/work/litmus_corpus/corpus/data}"
TEACHER="${TEACHER-/work/teacher/export}"  # set empty for a labels-only run on a corpus with no teacher export
EXPERIMENT="${EXPERIMENT:-2026-11-17_first_build}"
STEPS="${STEPS:-60000}"  # a cap the time box closes long before
SETTINGS="${SETTINGS:-}"  # the distillation search summary.json, when its settings are to be taken
TEXTS="${TEXTS:-1024}"
MICRO="${MICRO:-262144}"
EVAL_EVERY="${EVAL_EVERY:-500}"
FRACTION="${FRACTION:-1.0}"
LONG="${LONG:-1500}"
SPLICED="${SPLICED:-100}"
LABELS_ONLY="${LABELS_ONLY:-}"  # non-empty: the control with no teacher, in its own folder `<size>_labels_only`
BANK_RECIPE="${BANK_RECIPE:-}"  # mixed_out for a split with no `bank` set (human and AI bank texts from training rows)
BANK_PER_ORIGIN="${BANK_PER_ORIGIN:-}"  # training's and export's defaults when empty; set, training's validation bank is the release's
INTEGER_ATTENTION="${INTEGER_ATTENTION:-}"  # non-empty: train with integer attention (a format 2 release)
INT8_OUTPUT="${INT8_OUTPUT:-}"  # non-empty: train with the head's int8 output (a format 3 release)
SPLITS="${SPLITS:-validation test}"  # `validation` alone for a candidate checkpoint: the chain then stops before test and the benchmarks
RUN="$ROOT/$SIZE${LABELS_ONLY:+_labels_only}"
mkdir -p "$RUN"

step() {
    echo "== $(date -u +%H:%M:%S) $SIZE $1" >&2
}

step train
if [[ ! -f "$RUN/train/done" ]]; then
    python -m litmus_student.train --config "$SIZE" --data "$DATA" ${TEACHER:+--teacher-export "$TEACHER"} --experiment-id "$EXPERIMENT" \
        --out "$RUN/train" --steps "$STEPS" --minutes "$MINUTES" --texts-per-step "$TEXTS" --micro-tokens "$MICRO" \
        --eval-every "$EVAL_EVERY" --memory-fraction "$FRACTION" ${SETTINGS:+--settings "$SETTINGS"} ${LABELS_ONLY:+--labels-only} \
        ${BANK_RECIPE:+--bank-recipe "$BANK_RECIPE"} ${BANK_PER_ORIGIN:+--bank-per-origin "$BANK_PER_ORIGIN"} \
        ${INTEGER_ATTENTION:+--integer-attention} ${INT8_OUTPUT:+--int8-output}
    touch "$RUN/train/done"
fi

step export
[[ -f "$RUN/release/model.safetensors" ]] || python -m litmus_student.export --checkpoint "$RUN/train/best.pt" --data "$DATA" \
    --corpus "$CORPUS" --out "$RUN/release" ${BANK_RECIPE:+--bank-recipe "$BANK_RECIPE"} ${BANK_PER_ORIGIN:+--bank-per-origin "$BANK_PER_ORIGIN"}

for split in $SPLITS; do
    step "evaluate $split"
    [[ -f "$RUN/evaluation_$split/evaluation.json" ]] || python -m litmus_student.evaluate --checkpoint "$RUN/train/best.pt" \
        --release "$RUN/release" --data "$DATA" --experiment-id "$EXPERIMENT" --out "$RUN/evaluation_$split" --split "$split" \
        --long-documents "$LONG" --spliced-per-cell "$SPLICED"
done

if [[ "$SPLITS" != *test* ]]; then  # a candidate checkpoint: test is read once, on the chosen one, and so are the benchmarks
    step done
    exit 0
fi

step benchmarks
LITMUS_CHECKPOINT="$RUN/train/best.pt" LITMUS_RELEASE="$RUN/release" OUT="$RUN/benchmarks" DETECTORS=Litmus \
    bash launch/benchmark_chain.sh

step done
