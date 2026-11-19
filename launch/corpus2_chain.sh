#!/usr/bin/env bash
# The second corpus, additive, on CPU, inside the litmus container:
#   bash launch/in_container.sh bash launch/corpus2_chain.sh
# 1. Build the new sources (benchmark families' train splits with their tests held out, RAID among them, and the
#    sources the first build did not read) without the first corpus, every benchmark read whole for the hold-out.
#    Benchmark reads whose catalog line is unchanged are taken from the earlier build as hard links: those files are
#    whole and never rewritten (a stage writes to a new name and renames it), so the earlier build stays as it is.
# 2. Extend the earlier corpus: its rows keep their ids, then the new rows (`litmus_corpus extend`).
# Every step resumes: the build reruns only unfinished shards, and the extension runs once its report is missing.
set -euo pipefail
EARLIER="${EARLIER:-/work/litmus_corpus}"
OUT="${OUT:-/work/corpus2}"
WORKERS="${WORKERS:-10}"
LIMIT="${LIMIT:-}"  # rows per source at most, for a trial run
BUILD="$OUT/build${LIMIT:+_limit$LIMIT}"
CORPUS="$OUT/corpus${LIMIT:+_limit$LIMIT}"
REUSED="raid mage mage_ood mage_ood_paraphrase semeval2024_task8_a_mono semeval2024_task8_a_multi beemo mixset apt_eval apt_eval_human codet_m4 droid"
# First-corpus sources the first build held out as their own benchmark, now trained on as a family's train split.
TRAINED_FIRST="${TRAINED_FIRST-raid mage_train mage_validation ghostbuster_train coling2025_en semeval2024_a}"
# Sources read by group_column: an earlier row standing for one of their dropped copies joins its group's doc.
GROUP_SOURCES="${GROUP_SOURCES-raid_train raid_extra_code}"
EARLIER_DATA="${EARLIER_DATA:-$EARLIER/corpus/data}"  # a trial run may point at a few of its first shards

step() {
    echo "== $(date -u +%H:%M:%S) $1" >&2
}

mkdir -p "$BUILD/read"
if [[ -z "$LIMIT" ]]; then
    for name in $REUSED; do
        [[ -e "$BUILD/read/$name" ]] || cp -al "$EARLIER/read/$name" "$BUILD/read/$name"
    done
fi
[[ -e "$BUILD/models" ]] || cp -al "$EARLIER/models" "$BUILD/models"

step build
python -m litmus_corpus build --out "$BUILD" --workers "$WORKERS" --priority 3 --new-token-budget 1e9 --no-first-corpus \
    --family-train --first-index "$EARLIER/index" --earlier-build "$EARLIER" ${LIMIT:+--limit-per-source "$LIMIT"}

step extend
trained=()
for name in $TRAINED_FIRST; do
    trained+=(--trained-first-source "$name")
done
for name in $GROUP_SOURCES; do
    trained+=(--group-source "$name")
done
[[ -f "$CORPUS/report.json" ]] || python -m litmus_corpus extend --earlier "$EARLIER_DATA" --build "$BUILD" --out "$CORPUS" \
    --workers "$WORKERS" ${trained[@]+"${trained[@]}"}

step done
