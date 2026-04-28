#!/bin/bash
#SBATCH --account=YOURNAME1
#SBATCH --partition=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --job-name=v3_gb_shap
#SBATCH --output=_logs/%x_%j.log

# v3 Step 08: Gradient boosting + SHAP (3 tiers, 2 passes).

set -euo pipefail

source ~/home_turbo/fabio_home/LocalInstall/miniconda3/etc/profile.d/conda.sh
conda activate scprinter-cpu

cd /nfs/turbo/lsa-YOURNAME/fabio_home/Projects/13_Arbidopsis_protoplast/5_TF_FP

echo "Node: $(hostname), Start: $(date)"

python -u v3_08_gradient_boosting.py \
  --n-permutations 200 \
  --top-n-families 30

echo "End: $(date)"
