#!/bin/bash
#SBATCH --account=YOURNAME1
#SBATCH --partition=standard
#SBATCH --cpus-per-task=2
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --job-name=v3_top_sigs
#SBATCH --output=_logs/%x_%j.log

# v3 Step 07: Top-signature introduction (scatter + heatmap figures).

set -euo pipefail

export HDF5_USE_FILE_LOCKING=FALSE

source ~/home_turbo/fabio_home/LocalInstall/miniconda3/etc/profile.d/conda.sh
conda activate scprinter-cpu

cd /nfs/turbo/lsa-YOURNAME/fabio_home/Projects/13_Arbidopsis_protoplast/5_TF_FP

echo "Node: $(hostname), Start: $(date)"

python -u v3_07_top_signatures.py \
  --top-n 20 \
  --target-scale 10

echo "End: $(date)"
