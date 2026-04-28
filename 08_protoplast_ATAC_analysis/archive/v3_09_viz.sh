#!/bin/bash
#SBATCH --account=YOURNAME1
#SBATCH --partition=standard
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --job-name=v3_09_viz
#SBATCH --output=_logs/%x_%j.log

# v3 Step 09 viz: SHAP interaction visualization (all figures A–F, both passes).

set -euo pipefail

export HDF5_USE_FILE_LOCKING=FALSE

source ~/home_turbo/fabio_home/LocalInstall/miniconda3/etc/profile.d/conda.sh
conda activate scprinter-cpu

cd /nfs/turbo/lsa-YOURNAME/fabio_home/Projects/13_Arbidopsis_protoplast/5_TF_FP

echo "Node: $(hostname), Start: $(date)"

python -u v3_09_viz.py

echo "End: $(date)"
