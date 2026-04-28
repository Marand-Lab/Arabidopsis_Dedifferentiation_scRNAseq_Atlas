#!/bin/bash
#SBATCH --account=YOURNAME1
#SBATCH --partition=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH --time=0:10:00
#SBATCH --job-name=v3_08b_wrky_summary
#SBATCH --output=_logs/%x_%j.log

# v3 Step 08: Gradient boosting + SHAP (3 tiers, 2 passes).

set -euo pipefail

source ~/home_turbo/fabio_home/LocalInstall/miniconda3/etc/profile.d/conda.sh
conda activate scprinter-cpu

cd /nfs/turbo/lsa-YOURNAME/fabio_home/Projects/13_Arbidopsis_protoplast/5_TF_FP

echo "Node: $(hostname), Start: $(date)"

python -u v3_08b_wrky_summary.py

echo "End: $(date)"
