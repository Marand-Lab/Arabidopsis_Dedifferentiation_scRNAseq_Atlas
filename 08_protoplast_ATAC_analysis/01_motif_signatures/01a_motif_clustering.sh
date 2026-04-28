#!/bin/bash
#SBATCH --account=YOURNAME1
#SBATCH --partition=standard
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --job-name=v3_motif_cluster
#SBATCH --output=_logs/%x_%j.log

# v3 Step 0: Per-family motif clustering with motifStack.
# Run from: data/motif_signatures/

set -euo pipefail

#module load R/4.3.1
source ~/home_turbo/fabio_home/LocalInstall/miniconda3/etc/profile.d/conda.sh
conda activate r_motif

cd /nfs/turbo/lsa-YOURNAME/fabio_home/Projects/13_Arbidopsis_protoplast/5_TF_FP

echo "[INFO] Starting per-family motif clustering"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo ""

Rscript final/01_motif_signatures/01a_motif_clustering.R

echo ""
echo "End: $(date)"
