#!/bin/bash
#SBATCH --account=YOURNAME1
#SBATCH --partition=standard
#SBATCH --cpus-per-task=2
#SBATCH --mem=90G
#SBATCH --time=04:00:00
#SBATCH --job-name=v3_perscale_merge
#SBATCH --output=_logs/%x_%j.log

# v3 Step 06 merge: combine chunk NPZs + aggregate to delta matrices.

set -euo pipefail

source ~/home_turbo/fabio_home/LocalInstall/miniconda3/etc/profile.d/conda.sh
conda activate scprinter-cpu

cd /nfs/turbo/lsa-YOURNAME/fabio_home/Projects/13_Arbidopsis_protoplast/5_TF_FP

echo "Node: $(hostname), Start: $(date)"

python -u final/05_perscale_fp/05b_perscale_fp.py --merge --n-chunks 50

echo "End: $(date)"
