#!/usr/bin/env bash
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --job-name=v4_02b_merge
#SBATCH --account=YOURNAME1
#SBATCH --partition=standard
#SBATCH --output=_logs/%x_%j.log
#SBATCH --mail-user=gomezcan@umich.edu
#SBATCH --mail-type=FAIL,END

# v4_02b — Merge subsampled BAMs per condition (leaf, proto).
# Produces 2 pooled BAMs at equal per-replicate depth.
#
# Dependency: run after v4_02_subsample_and_count.sh completes.
# Run: sbatch --dependency=afterok:JOBID v4/v4_02b_merge_conditions.sh
#   or: sbatch v4/v4_02b_merge_conditions.sh

set -euo pipefail

source ~/home_turbo/fabio_home/LocalInstall/miniconda3/etc/profile.d/conda.sh
conda activate base

SUBDIR="v4/subsampled_bams"
OUTDIR="v4/merged_bams"
REPORT="v4/subsampling_report.txt"

mkdir -p "$OUTDIR" _logs

# ── Verify inputs ────────────────────────────────────────────────────
for f in \
  "${SUBDIR}/leaf_rep1.subsampled.bam" \
  "${SUBDIR}/leaf_rep2.subsampled.bam" \
  "${SUBDIR}/proto_rep1.subsampled.bam" \
  "${SUBDIR}/proto_rep2.subsampled.bam"; do
  if [[ ! -f "$f" ]]; then
    echo "[ERR] Missing: $f — run v4_02 first" >&2; exit 1
  fi
done

# ── Merge leaf replicates ────────────────────────────────────────────
echo "=== Merging leaf_rep1 + leaf_rep2 ==="
samtools merge -f "${OUTDIR}/leaf_merged.bam" \
  "${SUBDIR}/leaf_rep1.subsampled.bam" \
  "${SUBDIR}/leaf_rep2.subsampled.bam"
samtools index "${OUTDIR}/leaf_merged.bam"

# ── Merge proto replicates ───────────────────────────────────────────
echo "=== Merging proto_rep1 + proto_rep2 ==="
samtools merge -f "${OUTDIR}/proto_merged.bam" \
  "${SUBDIR}/proto_rep1.subsampled.bam" \
  "${SUBDIR}/proto_rep2.subsampled.bam"
samtools index "${OUTDIR}/proto_merged.bam"

# ── Verify read counts ──────────────────────────────────────────────
echo ""
echo "=== Verification ==="
LEAF_N=$(samtools view -c "${OUTDIR}/leaf_merged.bam")
PROTO_N=$(samtools view -c "${OUTDIR}/proto_merged.bam")

echo "  leaf_merged:  $LEAF_N reads"
echo "  proto_merged: $PROTO_N reads"

# Append to report
{
  echo ""
  echo "v4_02b Merged BAMs"
  echo "===================="
  echo "Date: $(date)"
  echo "  leaf_merged:  $LEAF_N reads"
  echo "  proto_merged: $PROTO_N reads"
} >> "$REPORT"

echo ""
echo "=== Done ==="
echo "  ${OUTDIR}/leaf_merged.bam  ($LEAF_N reads)"
echo "  ${OUTDIR}/proto_merged.bam ($PROTO_N reads)"
