#!/bin/bash
#SBATCH --account=YOURNAME1
#SBATCH --partition=standard
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --job-name=v3_fp_merge
#SBATCH --output=_logs/%x_%j.log

# v3 Step 05 merge: merge leaf+proto intermediates OR concatenate chunks.
# Usage: via v3_05_submit_all.sh
#   Arg 1: "conditions" or "chunks"

set -euo pipefail

source ~/home_turbo/fabio_home/LocalInstall/miniconda3/etc/profile.d/conda.sh
conda activate scprinter-cpu

cd /nfs/turbo/lsa-YOURNAME/fabio_home/Projects/13_Arbidopsis_protoplast/5_TF_FP

MODE="${1:-conditions}"

echo "[INFO] Mode: ${MODE}"
echo "Node: $(hostname), Start: $(date)"
echo ""

if [[ "$MODE" == "conditions" ]]; then
  python -u final/05_perscale_fp/05a_extract_fp.py --merge-conditions
elif [[ "$MODE" == "chunks" ]]; then
  python -u final/05_perscale_fp/05a_extract_fp.py --merge-chunks
else
  echo "Unknown mode: $MODE" >&2
  exit 1
fi

echo ""
echo "End: $(date)"
