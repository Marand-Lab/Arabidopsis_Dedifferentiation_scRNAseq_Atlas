#!/bin/bash
#SBATCH --job-name=v4_03f_fam_regions
#SBATCH --account=YOURNAME1
#SBATCH --partition=standard
#SBATCH --time=4:00:00
#SBATCH --mem=120G
#SBATCH --cpus-per-task=2
#SBATCH --output=_logs/v4_03f_fam_regions_%j.out
#SBATCH --error=_logs/v4_03f_fam_regions_%j.err

# Usage:
#   sbatch v4/v4_03f_plot_regions_families.sh \
#       --score-type TFBS --acr-class leaf_gain --overlap-group leaf_only
#
#   # With options:
#   sbatch v4/v4_03f_plot_regions_families.sh \
#       --score-type NucBS --acr-class proto_gain --overlap-group proto_only \
#       --fdr-threshold 0.10 --zoom-native

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
mkdir -p /tmp/v4_fp /tmp/v4_printer

for cond in leaf proto; do
    cp -v "v4/3_PRINT/FP/${cond}_merged__ALL.h5ad" /tmp/v4_fp/
    cp -v "v4/3_PRINT/printer_${cond}_merged_bulk.h5ad" /tmp/v4_printer/
done
echo "[COPY] Done."

python -u final/07_visualization/07b_plot_regions_families.py \
    --fp-dir /tmp/v4_fp \
    --printer-dir /tmp/v4_printer \
    --zoom-native \
    "$@"

# Cleanup
rm -rf /tmp/v4_fp /tmp/v4_printer
echo "[DONE]"
