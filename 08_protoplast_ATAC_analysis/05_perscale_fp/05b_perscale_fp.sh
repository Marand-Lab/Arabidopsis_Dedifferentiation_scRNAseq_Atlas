#!/bin/bash
#SBATCH --account=YOURNAME1
#SBATCH --partition=standard
#SBATCH --cpus-per-task=2
#SBATCH --mem=60G
#SBATCH --time=04:00:00
#SBATCH --job-name=v3_perscale_fp
#SBATCH --output=_logs/%x_%A_%a.log
#SBATCH --array=0-49

# v3 Step 06: Per-scale FP extraction (array job).
# Copies h5ad to /tmp for NFS safety.

set -euo pipefail

export HDF5_USE_FILE_LOCKING=FALSE

source ~/home_turbo/fabio_home/LocalInstall/miniconda3/etc/profile.d/conda.sh
conda activate scprinter-cpu

cd /nfs/turbo/lsa-YOURNAME/fabio_home/Projects/13_Arbidopsis_protoplast/5_TF_FP

CHUNK=$(printf "%02d" "${SLURM_ARRAY_TASK_ID}")
TMP_PRINT="/tmp/v3_print_${SLURM_JOB_ID}"

echo "[INFO] Chunk: ${CHUNK}"
echo "Job: ${SLURM_JOB_ID}, Task: ${SLURM_ARRAY_TASK_ID}"
echo "Node: $(hostname), Start: $(date)"
echo ""

# Copy h5ad files to /tmp
mkdir -p "$TMP_PRINT"
for sid in leaf_rep1 leaf_rep2 proto_rep1 proto_rep2; do
  SRC="3_PRINT_per_rep/printer_${sid}_bulk.h5ad"
  if [[ -f "$SRC" ]]; then
    echo "Copying $SRC to $TMP_PRINT ..."
    cp "$SRC" "$TMP_PRINT/"
  fi
done

python -u final/05_perscale_fp/05b_perscale_fp.py \
  --chunk-id "${SLURM_ARRAY_TASK_ID}" \
  --print-dir "$TMP_PRINT"

# Cleanup
rm -rf "$TMP_PRINT"

echo ""
echo "End: $(date)"
