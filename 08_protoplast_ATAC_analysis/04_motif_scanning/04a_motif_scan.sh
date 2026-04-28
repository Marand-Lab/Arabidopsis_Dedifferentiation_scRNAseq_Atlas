#!/bin/bash
#SBATCH --account=YOURNAME1
#SBATCH --partition=standard
#SBATCH --cpus-per-task=2
#SBATCH --mem=80G
#SBATCH --time=02:30:00
#SBATCH --job-name=v3_motif_scan
#SBATCH --output=_logs/%x_%A_%a.log
#SBATCH --array=0-49

# v3 Step 04: MOODS motif scanning with Arabidopsis motif signatures.
#
# Array job: splits native ACR BED into 50 chunks, scans each.
# Much faster than v2 (115 signatures vs 762 motifs, no --max-motif-len filter).
#
# Output per chunk: data/v3_chunks/chunk_{NN}/motif_hits.tsv.gz

set -euo pipefail

export HDF5_USE_FILE_LOCKING=FALSE

source ~/home_turbo/fabio_home/LocalInstall/miniconda3/etc/profile.d/conda.sh
conda activate scprinter-cpu

cd /nfs/turbo/lsa-YOURNAME/fabio_home/Projects/13_Arbidopsis_protoplast/5_TF_FP

FASTA="genome/At.TAIR10.dna_sm.Chr.fa"
BED_FULL="1_ACRs/Athaliana_leaf_protoplast.mergedACRs.bed"
SPLIT_DIR="data/v3_chunks/splits"
PREFIX="${SPLIT_DIR}/native_acr_part_"

mkdir -p "$SPLIT_DIR" _logs "data/v3_chunks"

# Create splits ONCE (task 0 does it; others wait)
if [[ "${SLURM_ARRAY_TASK_ID}" == "0" ]]; then
  if [[ ! -f "${PREFIX}00.bed" ]]; then
    echo "[INFO] Creating BED splits..." >&2
    rm -f "${PREFIX}"*.bed
    sort -k1,1 -k2,2n "$BED_FULL" > "${SPLIT_DIR}/native_acr_sorted.bed"
    split -d -a 2 -n l/50 "${SPLIT_DIR}/native_acr_sorted.bed" "$PREFIX"
    for f in "${PREFIX}"*; do
      [[ "$f" == *.bed ]] && continue
      mv "$f" "$f.bed"
    done
    echo "[INFO] BED splits ready." >&2
  else
    echo "[INFO] BED splits already exist; skipping split." >&2
  fi
fi

# Barrier: wait until splits exist
while [[ ! -f "${PREFIX}00.bed" ]]; do
  sleep 2
done

CHUNK=$(printf "%02d" "${SLURM_ARRAY_TASK_ID}")
BED="${PREFIX}${CHUNK}.bed"
OUT="data/v3_chunks/chunk_${CHUNK}"

echo "[INFO] Chunk ${CHUNK}: scanning ${BED}"
echo "Job ID: ${SLURM_JOB_ID}, Task: ${SLURM_ARRAY_TASK_ID}"
echo "Node:   $(hostname)"
echo "Start:  $(date)"
echo ""

python -u final/04_motif_scanning/04a_motif_scan.py \
  --fasta "$FASTA" \
  --bed-in "$BED" \
  --outdir "$OUT" \
  --meme "data/motif_signatures/At_Motif_SignatureDB.meme" \
  --metadata "data/motif_signatures/signature_metadata.tsv" \
  --n-jobs 2 \
  --chunk-size 100 \
  --pvalue 5e-5

echo ""
echo "End: $(date)"
