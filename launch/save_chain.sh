#!/usr/bin/env bash
# Save the run to private Hub repositories, inside the litmus container (GPU=0, it needs none):
#   bash launch/in_container.sh bash -c 'export WHAT=embeddings; bash launch/save_chain.sh'
# WHAT=embeddings saves /work/embed to the dataset repository; WHAT=run saves the teacher's and the students'
# outputs and the reports to the model repository; the default saves both. An upload resumes where it stopped, so
# the run is saved as stages finish and once more at the end.
set -euo pipefail
WHAT="${WHAT:-all}"
EMBED_REPO="${EMBED_REPO:-lhallee/litmus-embeddings}"
RUN_REPO="${RUN_REPO:-lhallee/litmus-first-build}"
if [[ "$WHAT" == embeddings || "$WHAT" == all ]]; then
    python launch/save_run.py --folder /work/embed --repo "$EMBED_REPO" --repo-type dataset
fi
if [[ "$WHAT" == run || "$WHAT" == all ]]; then
    python launch/save_run.py --folder /work --repo "$RUN_REPO" --repo-type model \
        --allow "teacher/**" "student/**" "benchmarks/**" "poc/*.json" "student_store/meta.json" \
        --ignore "**/wandb/**" "**/last.pt" "**/*.partial"
fi
