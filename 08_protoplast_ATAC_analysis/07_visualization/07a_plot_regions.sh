#!/bin/bash
#SBATCH --job-name=v4_plot_regions
#SBATCH --account=YOURNAME1
#SBATCH --partition=standard
#SBATCH --time=2:00:00
#SBATCH --mem=80G
#SBATCH --cpus-per-task=2
#SBATCH --output=_logs/v4_plot_regions_%j.out
#SBATCH --error=_logs/v4_plot_regions_%j.err

# Usage:
#   # From tile table (top N of a specific group)
#   sbatch v4/v4_plot_regions.sh --tile-table results/v4_03c_binding_overlap/acr_tile_table_tf5_nuc5.tsv \
#       --filter-score TFBS --filter-class leaf_gain --filter-group leaf_only \
#       --top-n 10 --title "TFBS_leaf_only_leaf_gain_top10"
#
#   # Specific regions
#   sbatch v4/v4_plot_regions.sh --regions "chr1:21344421-21346421" "chr3:14193857-14195857" \
#       --title "my_examples"
#
#   # From file
#   sbatch v4/v4_plot_regions.sh --region-file my_regions.txt --title "custom_set"

set -euo pipefail
module load Bioinformatics samtools

# Activate scPrinter environment
eval "$(conda shell.bash hook)"
CONDA_BASE="${HOME}/home_turbo/fabio_home/LocalInstall/miniconda3"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate scprinter-cpu

export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

PROJ="/nfs/turbo/lsa-YOURNAME/fabio_home/Projects/13_Arbidopsis_protoplast/5_TF_FP"
cd "${PROJ}"

# Copy FP + printer h5ads to /tmp for speed (~66 GB FP + ~2 GB printer)
echo "[COPY] Copying h5ads to /tmp..."
mkdir -p /tmp/v4_fp /tmp/v4_tfbs /tmp/v4_printer

for cond in leaf proto; do
    cp -v "v4/3_PRINT/FP/${cond}_merged__ALL.h5ad" /tmp/v4_fp/
    cp -v "v4/3_PRINT/TFBS/${cond}_merged__ALL.h5ad" /tmp/v4_tfbs/
    cp -v "v4/3_PRINT/printer_${cond}_merged_bulk.h5ad" /tmp/v4_printer/
done
echo "[COPY] Done."

python -u final/07_visualization/07a_plot_regions.py \
    --fp-dir /tmp/v4_fp \
    --tfbs-dir /tmp/v4_tfbs \
    --printer-dir /tmp/v4_printer \
    "$@"

# Cleanup
rm -rf /tmp/v4_fp /tmp/v4_tfbs /tmp/v4_printer
echo "[DONE]"

## example usage:
# step 1
# sbatch v4/v4_plot_regions.sh \
#     --regions "chr1:26642945-26644945" \
#     --title "single_chr1_26642945"

# step 2:
# python3 -u v4/v4_plot_regions.py \
#     --plot-only --title "single_chr1_26642945" --zoom-native