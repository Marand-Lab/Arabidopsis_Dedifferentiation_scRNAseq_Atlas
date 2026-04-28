#!/bin/bash
# Submit the 3-phase v3_05 workflow with SLURM dependency chaining.
#
# Phase 1: leaf replicates (array 0-19)
# Phase 2: proto replicates (array 0-19, after leaf)
# Phase 3: merge leaf+proto per chunk (after proto)
# Phase 4: concatenate all chunks (after merge)
#
# Usage:
#   bash v3_05_submit_all.sh

set -euo pipefail

echo "=== v3 Step 05: FP extraction pipeline ==="

# Phase 1: leaf
LEAF_JOB=$(sbatch --parsable final/05_perscale_fp/05a_extract_fp.sh leaf)
echo "Phase 1 (leaf):    job ${LEAF_JOB}"

# Phase 2: proto (after leaf — avoids concurrent h5ad access)
PROTO_JOB=$(sbatch --parsable --dependency=afterok:${LEAF_JOB} final/05_perscale_fp/05a_extract_fp.sh proto)
echo "Phase 2 (proto):   job ${PROTO_JOB}"

# Phase 3: merge conditions (after proto)
MERGE_COND_JOB=$(sbatch --parsable --dependency=afterok:${PROTO_JOB} final/05_perscale_fp/05a_merge.sh conditions)
echo "Phase 3 (merge):   job ${MERGE_COND_JOB}"

# Phase 4: merge chunks (after condition merge)
MERGE_CHUNK_JOB=$(sbatch --parsable --dependency=afterok:${MERGE_COND_JOB} final/05_perscale_fp/05a_merge.sh chunks)
echo "Phase 4 (concat):  job ${MERGE_CHUNK_JOB}"

echo ""
echo "Chain: ${LEAF_JOB} → ${PROTO_JOB} → ${MERGE_COND_JOB} → ${MERGE_CHUNK_JOB}"
echo "Monitor: squeue -u \$USER -j ${LEAF_JOB},${PROTO_JOB},${MERGE_COND_JOB},${MERGE_CHUNK_JOB}"
