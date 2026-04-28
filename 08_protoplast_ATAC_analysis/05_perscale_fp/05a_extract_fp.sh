#!/bin/bash
#SBATCH --account=YOURNAME1
#SBATCH --partition=standard
#SBATCH --cpus-per-task=2
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --job-name=v3_fp_extract
#SBATCH --output=_logs/%x_%A_%a.log
#SBATCH --array=0-49

# v3 Step 05: Per-replicate FP extraction at motif hit centers.
# Array job: one condition per submission (leaf or proto).
#
# Usage: via v3_05_submit_all.sh (chains leaf → proto → merge-conditions → merge-chunks)

set -euo pipefail

export HDF5_USE_FILE_LOCKING=FALSE

source ~/home_turbo/fabio_home/LocalInstall/miniconda3/etc/profile.d/conda.sh
conda activate scprinter-cpu

cd /nfs/turbo/lsa-YOURNAME/fabio_home/Projects/13_Arbidopsis_protoplast/5_TF_FP

CONDITION="${1:-leaf}"
CHUNK=$(printf "%02d" "${SLURM_ARRAY_TASK_ID}")
TMP_PRINT="/tmp/v3_05_h5ad_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

echo "[INFO] Condition: ${CONDITION}, Chunk: ${CHUNK}"
echo "Job ID: ${SLURM_JOB_ID}, Task: ${SLURM_ARRAY_TASK_ID}"
echo "Node: $(hostname), Start: $(date)"
echo ""

# Copy h5ad files to /tmp to avoid concurrent NFS read failures
mkdir -p "$TMP_PRINT"
for sid in ${CONDITION}_rep1 ${CONDITION}_rep2 ${CONDITION}_rep3; do
  SRC="3_PRINT_per_rep/printer_${sid}_bulk.h5ad"
  if [[ -f "$SRC" ]]; then
    echo "Copying $SRC to $TMP_PRINT ..."
    cp "$SRC" "$TMP_PRINT/"
  fi
done

python -u final/05_perscale_fp/05a_extract_fp.py \
  --extract \
  --chunk-dir "data/v3_chunks/chunk_${CHUNK}" \
  --condition "${CONDITION}" \
  --print-dir "$TMP_PRINT" \
  --band-edges "20,50"

# Cleanup
rm -rf "$TMP_PRINT"

echo ""
echo "End: $(date)"
