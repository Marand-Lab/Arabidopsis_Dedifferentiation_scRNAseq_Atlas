#!/bin/bash
#SBATCH --job-name=v4_03d
#SBATCH --output=_logs/v4_03d_%j.log
#SBATCH --time=02:00:00
#SBATCH --mem=60G
#SBATCH --cpus-per-task=2
#SBATCH --account=YOURNAME1
#SBATCH --partition=standard

# FP delta distributions at bound/occupied tile positions.
#
# Usage:
#   sbatch v4/v4_03d_binding_deltas.sh
#   sbatch v4/v4_03d_binding_deltas.sh 10 5   # custom: tfbs_pct=10, nucbs_pct=5
#   sbatch v4/v4_03d_binding_deltas.sh 5 2 --native-only

set -euo pipefail

TFBS_PCT=${1:-5}
NUCBS_PCT=${2:-2}
NATIVE_FLAG=${3:-}

echo "=== v4_03d: Binding Deltas ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "SLURM Job: ${SLURM_JOB_ID}"
echo ""

# ── Copy both FP h5ads to /tmp (~66 GB total) ────────────────────────
TMPDIR=/tmp/v4_03d_${SLURM_JOB_ID}
mkdir -p "${TMPDIR}"
echo "[COPY] Copying FP h5ads to ${TMPDIR} ..."
cp v4/3_PRINT/FP/leaf_merged__ALL.h5ad "${TMPDIR}/"
cp v4/3_PRINT/FP/proto_merged__ALL.h5ad "${TMPDIR}/"
echo "[COPY] Done: $(date)"
echo ""

# ── Activate conda ───────────────────────────────────────────────────
source ~/home_turbo/fabio_home/LocalInstall/miniconda3/etc/profile.d/conda.sh
conda activate scprinter-cpu

# ── Run ──────────────────────────────────────────────────────────────
python -u final/06_binding_scores/06d_binding_deltas.py \
    --fp-dir "${TMPDIR}" \
    --tfbs-pct "${TFBS_PCT}" \
    --nucbs-pct "${NUCBS_PCT}" \
    ${NATIVE_FLAG}

# ── Cleanup ──────────────────────────────────────────────────────────
rm -rf "${TMPDIR}"

echo ""
echo "=== Done ==="
echo "Date: $(date)"
