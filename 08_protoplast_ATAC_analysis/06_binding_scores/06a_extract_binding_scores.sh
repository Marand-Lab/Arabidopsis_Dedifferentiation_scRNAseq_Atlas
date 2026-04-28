#!/bin/bash
#SBATCH --job-name=v4_03a_bs
#SBATCH --output=_logs/v4_03a_bs_%j.log
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --account=YOURNAME1
#SBATCH --partition=standard

# Extract TFBS & NucBS binding scores for one condition.
#
# Usage:
#   sbatch v4/v4_03a_extract_binding_scores.sh leaf
#   sbatch v4/v4_03a_extract_binding_scores.sh proto

set -euo pipefail

COND=${1:?Usage: sbatch v4/v4_03a_extract_binding_scores.sh leaf|proto}

echo "=== v4_03a: Extract Binding Scores — ${COND} ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "SLURM Job: ${SLURM_JOB_ID}"
echo ""

# ── Copy h5ads to /tmp for fast SSD access (~29 MB each) ────────────
# TFBS and NucBS have the same filename — use separate subdirs to avoid overwrite
TMPDIR=/tmp/v4_03a_${SLURM_JOB_ID}
mkdir -p "${TMPDIR}/TFBS" "${TMPDIR}/NucBS"
echo "[COPY] Copying ${COND} TFBS + NucBS h5ads to ${TMPDIR} ..."
cp "v4/3_PRINT/TFBS/${COND}_merged__ALL.h5ad" "${TMPDIR}/TFBS/"
cp "v4/3_PRINT/NucBS/${COND}_merged__ALL.h5ad" "${TMPDIR}/NucBS/"
echo "[COPY] Done: $(date)"
echo ""

# ── Activate conda ──────────────────────────────────────────────────
source ~/home_turbo/fabio_home/LocalInstall/miniconda3/etc/profile.d/conda.sh
conda activate scprinter-cpu

# ── Run extraction ──────────────────────────────────────────────────
python -u final/06_binding_scores/06a_extract_binding_scores.py \
    --condition "${COND}" \
    --tfbs-dir "${TMPDIR}/TFBS" \
    --nucbs-dir "${TMPDIR}/NucBS"

# ── Cleanup ─────────────────────────────────────────────────────────
rm -rf "${TMPDIR}"

echo ""
echo "=== Done: ${COND} ==="
echo "Date: $(date)"
