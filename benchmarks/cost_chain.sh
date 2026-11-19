#!/usr/bin/env bash
# Single-core CPU cost of named detectors, inside the litmus container started with GPU=0 THREADS=1:
#   bash launch/in_container.sh bash -c 'export OUT=/work/cost/costs.csv DETECTORS="e5-small-lora,Litmus engine"; bash benchmarks/cost_chain.sh'
# Each detector runs in its own process on the same fixed texts (the first TEXTS of the cost set) and appends one row
# to OUT. EXTRA_PIP names packages to install first (the engine's wheel, the predecessor's beautifulsoup4 and
# requests); Litmus reads LITMUS_CHECKPOINT and LITMUS_RELEASE, the engine LITMUS_ENGINE_RELEASE. The chain still
# fails at the end, naming what failed.
set -euo pipefail
OUT="${OUT:?the costs CSV to append to}"
DETECTORS="${DETECTORS:?comma-separated detector names}"
TEXTS="${TEXTS:-200}"
BENCHMARK="${BENCHMARK:-mage}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false CHECKER_THREADS=1
if [[ -n "${EXTRA_PIP:-}" ]]; then
    read -ra packages <<< "$EXTRA_PIP"
    pip install --quiet --no-input "${packages[@]}"
fi
mkdir -p "$(dirname "$OUT")"
IFS=, read -ra detectors <<< "$DETECTORS"
failed=()
for detector in "${detectors[@]}"; do
    echo "== $(date -u +%H:%M:%S) cost of $detector on $TEXTS texts of $BENCHMARK" >&2
    python -m litmus_benchmarks cost --benchmark "$BENCHMARK" --detector "$detector" --texts "$TEXTS" --out "$OUT" \
        || failed+=("$detector")
done
if (( ${#failed[@]} )); then
    printf 'failed: %s\n' "${failed[@]}" >&2
    exit 1
fi
echo "== $(date -u +%H:%M:%S) done" >&2
