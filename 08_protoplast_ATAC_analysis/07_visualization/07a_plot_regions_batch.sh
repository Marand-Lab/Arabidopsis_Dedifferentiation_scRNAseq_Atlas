#!/bin/bash
#SBATCH --job-name=v4_plot_batch
#SBATCH --account=YOURNAME1
#SBATCH --partition=standard
#SBATCH --time=2:00:00
#SBATCH --mem=80G
#SBATCH --cpus-per-task=2
#SBATCH --output=_logs/v4_plot_batch_%j.out
#SBATCH --error=_logs/v4_plot_batch_%j.err

# Extract + plot 6 categories of leaf_gain examples (TFBS + NucBS × 3 overlap groups)
# Each category gets its own subdirectory under results/v4_region_viewer/
#
# Usage: sbatch v4/v4_plot_regions_batch.sh

set -euo pipefail
module load Bioinformatics samtools

eval "$(conda shell.bash hook)"
CONDA_BASE="${HOME}/home_turbo/fabio_home/LocalInstall/miniconda3"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate scprinter-cpu

export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

PROJ="/nfs/turbo/lsa-YOURNAME/fabio_home/Projects/13_Arbidopsis_protoplast/5_TF_FP"
cd "${PROJ}"

# Copy h5ads once (shared across all categories)
echo "[COPY] Copying h5ads to /tmp..."
mkdir -p /tmp/v4_fp /tmp/v4_tfbs /tmp/v4_printer
for cond in leaf proto; do
    cp -v "v4/3_PRINT/FP/${cond}_merged__ALL.h5ad" /tmp/v4_fp/
    cp -v "v4/3_PRINT/TFBS/${cond}_merged__ALL.h5ad" /tmp/v4_tfbs/
    cp -v "v4/3_PRINT/printer_${cond}_merged_bulk.h5ad" /tmp/v4_printer/
done
echo "[COPY] Done."

# Run each category
CATEGORIES=(
    "leaf_gain_tfbs_leaf_only"
    "leaf_gain_tfbs_shared"
    "leaf_gain_tfbs_proto_only"
    "leaf_gain_nucbs_leaf_only"
    "leaf_gain_nucbs_shared"
    "leaf_gain_nucbs_proto_only"
)

for cat in "${CATEGORIES[@]}"; do
    echo ""
    echo "============================================"
    echo "  CATEGORY: ${cat}"
    echo "============================================"
    OUTDIR="results/v4_region_viewer/${cat}"
    REGION_FILE="${OUTDIR}/${cat}_regions.txt"

    python -u final/07_visualization/07a_plot_regions.py \
        --fp-dir /tmp/v4_fp \
        --tfbs-dir /tmp/v4_tfbs \
        --printer-dir /tmp/v4_printer \
        --region-file "${REGION_FILE}" \
        --outdir "${OUTDIR}" \
        --title "${cat}" \
        --force-extract
done

# Replot with zoom to native ACR boundaries
echo ""
echo "============================================"
echo "  REPLOTTING WITH --zoom-native"
echo "============================================"
for cat in "${CATEGORIES[@]}"; do
    echo "  [REPLOT] ${cat}"
    OUTDIR="results/v4_region_viewer/${cat}"
    python -u final/07_visualization/07a_plot_regions.py \
        --plot-only --zoom-native \
        --outdir "${OUTDIR}" \
        --title "${cat}"
done

# Cleanup
rm -rf /tmp/v4_fp /tmp/v4_tfbs /tmp/v4_printer
echo ""
echo "[DONE] All 6 categories extracted + plotted (full + zoomed)."
