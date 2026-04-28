#!/bin/bash
#SBATCH --account=YOURNAME1
#SBATCH --partition=standard
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --job-name=v4_01b_frags
#SBATCH --output=_logs/%x_%j.log
#SBATCH --mail-user=gomezcan@umich.edu
#SBATCH --mail-type=FAIL,END

# v4_01b — Convert merged BAMs (from v4_02b) to scPrinter-compatible
# 1-based fragment files.
#
# Input:  v4/merged_bams/{leaf,proto}_merged.bam
# Output: v4/fragments_1based/{leaf,proto}_merged.bulk.1based.tsv.gz
#
# Dependency: run after v4_02b_merge_conditions.sh completes.
# Run: sbatch v4/v4_01b_prep_merged_fragments.sh

set -euo pipefail

source ~/home_turbo/fabio_home/LocalInstall/miniconda3/etc/profile.d/conda.sh
conda activate scprinter-cpu

CPUS="${SLURM_CPUS_PER_TASK:-8}"
BAM2FRAG="final/lib/bam_to_fragment.py"
INDIR="v4/merged_bams"
OUTDIR="v4/fragments_1based"
TMPDIR_SORT="v4/tmp_namesort"

mkdir -p "$OUTDIR" "$TMPDIR_SORT" _logs

# Verify inputs
if [[ ! -f "$BAM2FRAG" ]]; then
  echo "[ERR] Missing script: $BAM2FRAG" >&2; exit 1
fi

CONDITIONS=(leaf proto)

for COND in "${CONDITIONS[@]}"; do
  BAM="${INDIR}/${COND}_merged.bam"
  FRAG_RAW="${TMPDIR_SORT}/${COND}_merged.fragments.tsv.gz"
  NS_BAM="${TMPDIR_SORT}/${COND}_merged.namesort.bam"
  FRAG_OUT="${OUTDIR}/${COND}_merged.bulk.1based.tsv.gz"

  echo ""
  echo "=========================================="
  echo "  Processing: $COND"
  echo "=========================================="

  if [[ ! -f "$BAM" ]]; then
    echo "[ERR] Missing BAM: $BAM" >&2; exit 1
  fi

  # Skip if output already exists
  if [[ -f "$FRAG_OUT" ]]; then
    echo "[INFO] Output exists, skipping: $FRAG_OUT"
    continue
  fi

  # Step 1: Name-sort the merged BAM
  echo "[INFO] Name-sorting $BAM..."
  samtools sort -n -@ "$CPUS" -o "$NS_BAM" "$BAM"
  echo "[INFO] Name-sorted: $NS_BAM"

  # Step 2: Extract fragments (bulk mode, 0-based 6-col output)
  echo "[INFO] Extracting fragments..."
  python "$BAM2FRAG" \
    --bulk \
    --group "merged" \
    --sample "Athaliana_${COND}" \
    --bam "$NS_BAM" \
    --out "$FRAG_RAW" \
    --min-mapq 20 \
    --threads "$CPUS"

  # Step 3: Convert to 1-based + chrom mapping (same as 01b)
  # Input:  chr start0 end group sample count  (6-col, 0-based)
  # Output: chr start1 end bulk               (4-col, 1-based)
  echo "[INFO] Converting to 1-based + chrom mapping..."
  zcat "$FRAG_RAW" | awk -F'\t' 'BEGIN{OFS="\t"} {
    ch = $1
    gsub(/^Chr/, "", ch)
    gsub(/^chr/, "", ch)
    if (ch == "M" || ch == "Mt" || ch == "MT") ch = "M"
    else if (ch == "C" || ch == "Pt" || ch == "PT") ch = "C"
    ch = "chr" ch

    start1 = $2 + 1
    end = $3

    if (start1 < 1 || end <= start1) next

    print ch, start1, end, "bulk"
  }' | gzip > "$FRAG_OUT"

  # Verify
  N_LINES=$(zcat "$FRAG_OUT" | wc -l)
  echo "[INFO] Output: $FRAG_OUT ($N_LINES fragments)"
  echo "[INFO] First 3 lines:"
  zcat "$FRAG_OUT" | head -3

  # Clean up intermediates
  rm -f "$NS_BAM" "$FRAG_RAW"
  echo "[DONE] $COND"
done

# Clean up temp directory
rmdir "$TMPDIR_SORT" 2>/dev/null || true

echo ""
echo "=========================================="
echo "  All conditions processed"
echo "=========================================="
ls -lh "$OUTDIR/"
