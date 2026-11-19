#!/usr/bin/env bash
# Score the public benchmarks with some detectors, inside the litmus container:
#   bash launch/in_container.sh bash -c 'export OUT=/work/benchmarks DETECTORS=baselines; bash launch/benchmark_chain.sh'
# DETECTORS is comma-separated (names hold spaces), or the word `baselines` for the open detectors rerun beside
# Litmus. One file per benchmark and detector, each from its own process, so a benchmark that cannot be read or a
# detector that cannot load costs its own scores and no other's, and no two detectors hold the GPU's memory at once;
# the chain still fails at the end, naming what failed. Every detector reads the same sample of a benchmark (the
# seed and --max-rows fix it), so the files compare row for row. Litmus reads its checkpoint and release from
# LITMUS_CHECKPOINT and LITMUS_RELEASE.
set -euo pipefail
OUT="${OUT:?the directory of the score files}"
DETECTORS="${DETECTORS:?comma-separated detector names, or baselines}"
BENCHMARK_ROWS="${BENCHMARK_ROWS:-3000}"  # one number for every detector, so their samples are the same rows
BENCHMARKS="${BENCHMARKS:-mage mage_ood mage_ood_paraphrase raid_extra semeval2024_task8_a_mono semeval2024_task8_a_multi beemo mixset apt_eval apt_eval_human codet_m4 droid hc3 editlens autextification_en autextification_es}"
LITMUS_BATCH="${LITMUS_BATCH:-1024}"  # texts Litmus reads at a time; the other detectors keep the scorer's 32
# Fast-DetectGPT (at the scorer's batch of 32) and Binoculars run out of memory on a 24 GB GPU (an A10, 2026-11-18 and
# 2026-11-19); they ran on a GH200.
GPU_BASELINES="Desklib v1.01,ModernBERT RAID+MAGE (Drayson),e5-small-lora,Fast-DetectGPT,Binoculars"
if [[ "$DETECTORS" == baselines ]]; then
    DETECTORS="ai-vs-human-checker,$GPU_BASELINES"
elif [[ "$DETECTORS" == gpu_baselines ]]; then  # the predecessor checker reads on the CPU, so it runs beside these, not among them
    DETECTORS="$GPU_BASELINES"
fi
IFS=, read -ra detectors <<< "$DETECTORS"
mkdir -p "$OUT"
failed=()
for detector in "${detectors[@]}"; do
    slug="$(echo "$detector" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9\n' '_')"
    batch=32
    if [[ "$detector" == Litmus ]]; then  # its scorer pools a batch's chunks in packed forwards and reads them against the whole bank together
        batch="$LITMUS_BATCH"
    fi
    for name in $BENCHMARKS; do
        echo "== $(date -u +%H:%M:%S) $detector on $name" >&2
        [[ -f "$OUT/scores_${name}__$slug.csv" ]] || python -m litmus_benchmarks score --benchmarks "$name" --detectors "$detector" \
            --max-rows "$BENCHMARK_ROWS" --batch-size "$batch" --out "$OUT/scores_${name}__$slug.csv" || failed+=("$detector on $name")
    done
done
if (( ${#failed[@]} )); then
    printf 'failed: %s\n' "${failed[@]}" >&2
    exit 1
fi
