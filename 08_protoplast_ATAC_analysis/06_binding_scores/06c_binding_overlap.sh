#!/bin/bash
#SBATCH --job-name=v4_03c
#SBATCH --output=_logs/v4_03c_%j.log
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --account=YOURNAME1
#SBATCH --partition=standard

# Binding overlap Venns + ACR tile table.
# Runs all threshold combos (2/2, 5/5, 10/10) in one job.
#
# Usage:
#   sbatch v4/v4_03c_binding_overlap.sh                # resized (default)
#   sbatch v4/v4_03c_binding_overlap.sh --native-only  # native ACR tiles only

set -euo pipefail

NATIVE_FLAG=${1:-}

echo "=== v4_03c: Binding Overlap + Tile Table ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "Native flag: '${NATIVE_FLAG}'"
echo ""

source ~/home_turbo/fabio_home/LocalInstall/miniconda3/etc/profile.d/conda.sh
conda activate scprinter-cpu

cd /nfs/turbo/lsa-YOURNAME/fabio_home/Projects/13_Arbidopsis_protoplast/5_TF_FP

# ── Overlap Venns (3 threshold combos) ────────────────────────────
for TFBS in 2 5 10; do
    echo ""
    echo "--- tfbs_pct=${TFBS}, nucbs_pct=${TFBS} ${NATIVE_FLAG} ---"
    python -u final/06_binding_scores/06c_binding_overlap.py \
        --tfbs-pct "${TFBS}" --nucbs-pct "${TFBS}" \
        ${NATIVE_FLAG}
done

# ── ACR tile table (same combos) ─────────────────────────────────
echo ""
echo "--- ACR tile table ${NATIVE_FLAG} ---"
python -u final/06_binding_scores/06c_acr_tile_table.py ${NATIVE_FLAG}

echo ""
echo "=== Done ==="
echo "Date: $(date)"
