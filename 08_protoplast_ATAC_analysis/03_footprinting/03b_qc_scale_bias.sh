#!/bin/bash
#SBATCH --job-name=v4_qc_scale_bias
#SBATCH --output=_logs/v4_qc_scale_bias_%j.out
#SBATCH --error=_logs/v4_qc_scale_bias_%j.err
#SBATCH --time=4:00:00
#SBATCH --mem=60G
#SBATCH --cpus-per-task=2
#SBATCH --account=YOURNAME1

# ── v4_qc_scale_bias.sh ──────────────────────────────────────────────────────
# Scale-resolved condition bias QC using v4 merged-condition h5ads.
#
# What it tests:
#   1. Scale-resolved mean FP depth at ACR centers (leaf vs proto, all 22k ACRs)
#   2. Scale-resolved delta at null loci (no JASPAR motif within ±50bp)
#
# Interpreting the output:
#   - Panel D flat near 0 → no Tn5 bias; cross-scale sign-flips are real biology
#   - Panel D significant at <20bp → Tn5 bias cannot be excluded for small scales
#   - Panel D significant only at >80bp → large-scale global accessibility shift;
#       small-scale TF signals are genuine
#   - Panel B >> Panel D at small scales → motif-specific, not generic bias
#
# Prerequisites:
#   v4/3_PRINT/FP/leaf_merged__ALL.h5ad
#   v4/3_PRINT/FP/proto_merged__ALL.h5ad
#   (produced by v4_02c_run_print.py for leaf and proto)
#
# Outputs:
#   results/v4_qc_scale_bias/scale_bias_qc.pdf   (4-panel figure)
#   results/v4_qc_scale_bias/scale_bias_qc.png
#   results/v4_qc_scale_bias/scale_bias_qc.npz   (raw arrays)
#   results/v4_qc_scale_bias/scale_bias_summary.txt
#
# Tune:
#   --n-null 5000          number of null loci to sample (more = tighter CIs)
#   --motif-excl-radius 50 exclusion zone around motif hits (bp)
#   --seed 0               RNG seed for reproducibility
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

PROJECT="/nfs/turbo/lsa-YOURNAME/fabio_home/Projects/13_Arbidopsis_protoplast/5_TF_FP"
cd "$PROJECT"

mkdir -p logs results/v4_qc_scale_bias

# Verify inputs exist
for f in \
    "v4/3_PRINT/FP/leaf_merged__ALL.h5ad" \
    "v4/3_PRINT/FP/proto_merged__ALL.h5ad" \
    "data/v3_merged_motif_hits.tsv.gz" \
    "data/acr_native_to_resized.tsv" \
    "v4/data/acr_resized_2000bp.bed"; do
    if [[ ! -f "$f" ]]; then
        echo "[ERR] Missing required file: $f" >&2
        exit 1
    fi
done

echo "[INFO] Starting v4_qc_scale_bias at $(date)"
echo "[INFO] Job ID: $SLURM_JOB_ID"
echo "[INFO] Node:   $SLURM_NODELIST"
echo "[INFO] CPUs:   $SLURM_CPUS_PER_TASK"

export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

CONDA_BASE="$HOME/home_turbo/fabio_home/LocalInstall/miniconda3"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate scprinter-cpu

python -u final/03_footprinting/03b_qc_scale_bias.py \
    --n-null 5000 \
    --motif-excl-radius 50 \
    --seed 0

echo "[DONE] v4_qc_scale_bias at $(date)"
