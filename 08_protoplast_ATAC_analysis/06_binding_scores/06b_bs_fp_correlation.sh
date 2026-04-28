#!/bin/bash
#SBATCH --job-name=v4_03b
#SBATCH --output=_logs/v4_03b_%j.log
#SBATCH --time=02:00:00
#SBATCH --mem=60G
#SBATCH --cpus-per-task=2
#SBATCH --account=YOURNAME1
#SBATCH --partition=standard

# Correlate TFBS/NucBS binding scores with multi-scale FP.
#
# Usage:
#   sbatch v4/v4_03b_bs_fp_correlation.sh leaf
#   sbatch v4/v4_03b_bs_fp_correlation.sh proto
#   sbatch v4/v4_03b_bs_fp_correlation.sh leaf 10 5   # custom: tfbs_pct=10, nucbs_pct=5
#   sbatch v4/v4_03b_bs_fp_correlation.sh leaf 5 2 --native-only

set -euo pipefail

COND=${1:?Usage: sbatch v4/v4_03b_bs_fp_correlation.sh leaf|proto [tfbs_pct] [nucbs_pct] [--native-only]}
TFBS_PCT=${2:-5}
NUCBS_PCT=${3:-2}
NATIVE_FLAG=${4:-}

echo "=== v4_03b: BS–FP Correlation — ${COND} ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "SLURM Job: ${SLURM_JOB_ID}"
echo ""

# ── Copy FP h5ad to /tmp (~33 GB, worth it for random access) ───────
TMPDIR=/tmp/v4_03b_${SLURM_JOB_ID}
mkdir -p "${TMPDIR}"
echo "[COPY] Copying ${COND} FP h5ad to ${TMPDIR} ..."
cp "v4/3_PRINT/FP/${COND}_merged__ALL.h5ad" "${TMPDIR}/"
echo "[COPY] Done: $(date)"
echo ""

# ── Activate conda ──────────────────────────────────────────────────
source ~/home_turbo/fabio_home/LocalInstall/miniconda3/etc/profile.d/conda.sh
conda activate scprinter-cpu

# ── Run correlation analysis ────────────────────────────────────────
python -u final/06_binding_scores/06b_bs_fp_correlation.py \
    --condition "${COND}" \
    --fp-dir "${TMPDIR}" \
    --tfbs-pct "${TFBS_PCT}" \
    --nucbs-pct "${NUCBS_PCT}" \
    ${NATIVE_FLAG}

# ── Cleanup ─────────────────────────────────────────────────────────
rm -rf "${TMPDIR}"

echo ""
echo "=== Done: ${COND} ==="
echo "Date: $(date)"
