#!/bin/bash
# Submit v3_06 array extract + merge with dependency chaining.
#
# Usage: bash v3_06_submit_all.sh

set -euo pipefail

echo "=== v3 Step 06: Per-scale FP extraction ==="

EXTRACT_JOB=$(sbatch --parsable final/05_perscale_fp/05b_perscale_fp.sh)
echo "Extract (array): job ${EXTRACT_JOB}"

MERGE_JOB=$(sbatch --parsable --dependency=afterok:${EXTRACT_JOB} final/05_perscale_fp/05b_merge.sh)
echo "Merge:           job ${MERGE_JOB}"

echo ""
echo "Chain: ${EXTRACT_JOB} → ${MERGE_JOB}"
echo "Monitor: squeue -u \$USER -j ${EXTRACT_JOB},${MERGE_JOB}"
