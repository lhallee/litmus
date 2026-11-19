#!/usr/bin/env bash
# Run one command of a Litmus run in the litmus image, from a copy of the source shipped to the GPU machine. Its
# working directory is the shipped project folder, four levels below the run directory (tree/projects/tooling/<name>):
#   bash launch/in_container.sh python -m litmus_student poc --out /work/poc ...
# The run directory mounts read-only at /workspace, so the container imports exactly what was shipped: the shipped
# root for foundry, and each package's src. RUN_DIR (default ~/litmus_run, on the machine's local disk) mounts at /work
# for every step's outputs, and CACHE (default ~/litmus_cache) at /cache for Hugging Face. Credentials set in the
# environment reach the container by name, never on a command line. CPUS and MEMORY cap the container; several
# containers run side by side under different NAMEs (the corpus build on the cores beside a GPU step).
set -euo pipefail
project="$(pwd)"
here="$(cd ../../../.. && pwd)"
relative="${project#"$here"/}"
if [[ ! -f "$project/launch/in_container.sh" || "$relative" != tree/projects/tooling/* ]]; then
    echo "in_container.sh runs from the shipped project folder, tree/projects/tooling/<project>" >&2
    exit 1
fi
run="${RUN_DIR:-$HOME/litmus_run}"
cache="${CACHE:-$HOME/litmus_cache}"
mkdir -p "$run" "$cache"
names=()
for name in HF_TOKEN WANDB_API_KEY WANDB_ENTITY WANDB_PROJECT WS_PILOT WANDB_MODE; do
    if [[ -n "${!name:-}" ]]; then
        names+=("$name")
    fi
done
passed=()
for name in "${names[@]}"; do
    passed+=(-e "$name")
done
preserve=$(IFS=,; echo "${names[*]}")
inside="/workspace/$relative"
# The shipped import roots (foundry among them) with the run directory rewritten to its mount, then each package.
# The last is the predecessor checker's package (`ai_text_checker`), present only when it is shipped beside this
# project for the benchmark comparison.
shipped="${PYTHONPATH:-}"
paths="${shipped//"$here"//workspace}:$inside/corpus/src:$inside/teacher/src:$inside/student/src:$inside/student:$inside/benchmarks/src:$inside/../ai_text_checker/src"
gpus=(--gpus all)
if [[ "${GPU:-1}" == 0 ]]; then
    gpus=()
fi
exec sudo ${preserve:+--preserve-env="$preserve"} docker run --rm "${gpus[@]}" --ipc=host --name "${NAME:-litmus}" \
    --cpus "${CPUS:-24}" --memory "${MEMORY:-160g}" \
    -v "$here:/workspace:ro" -v "$run:/work" -v "$cache:/cache" "${passed[@]}" \
    -e PYTHONPATH="$paths" -e OMP_NUM_THREADS="${THREADS:-8}" \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -w "$inside" "${IMAGE:-litmus}" "$@"
