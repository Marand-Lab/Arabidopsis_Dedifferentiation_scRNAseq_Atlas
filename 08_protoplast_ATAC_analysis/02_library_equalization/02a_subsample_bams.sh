#!/usr/bin/env bash
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --job-name=v4_02_subsample
#SBATCH --account=YOURNAME1
#SBATCH --partition=standard
#SBATCH --output=_logs/%x_%j.log
#SBATCH --mail-user=gomezcan@umich.edu
#SBATCH --mail-type=FAIL,END

# v4_02 — Subsample 4 BAMs (reps 1+2) to equal depth, then count per ACR.
# Rep3 excluded (label swap confirmed).
#
# Run: sbatch v4/v4_02_subsample_and_count.sh

set -euo pipefail

source ~/home_turbo/fabio_home/LocalInstall/miniconda3/etc/profile.d/conda.sh
conda activate base

SEED=42
BAM_DIR="bam"
BED="1_ACRs/Athaliana_leaf_protoplast.mergedACRs.bed"
OUTDIR="v4/subsampled_bams"
REPORT="v4/subsampling_report.txt"
COUNTS_OUT="v4/subsampled_counts.tsv"

mkdir -p "$OUTDIR" _logs

# ── Define the 4 active BAMs (reps 1+2 only) ────────────────────────
declare -A BAM_FILES
BAM_FILES=(
  [leaf_rep1]="${BAM_DIR}/Athaliana_leaf_ATAC_rep1.mq30.rmdup.bam"
  [leaf_rep2]="${BAM_DIR}/Athaliana_leaf_ATAC_rep2.mq30.rmdup.bam"
  [proto_rep1]="${BAM_DIR}/Athaliana_protoplasts_ATAC_rep1.mq30.rmdup.bam"
  [proto_rep2]="${BAM_DIR}/Athaliana_protoplasts_ATAC_rep2.mq30.rmdup.bam"
)
# Stable order for output columns
SAMPLE_ORDER=(leaf_rep1 leaf_rep2 proto_rep1 proto_rep2)

# Verify inputs
if [[ ! -f "$BED" ]]; then
  echo "[ERR] Missing BED: $BED" >&2; exit 1
fi
for sid in "${SAMPLE_ORDER[@]}"; do
  if [[ ! -f "${BAM_FILES[$sid]}" ]]; then
    echo "[ERR] Missing BAM: ${BAM_FILES[$sid]}" >&2; exit 1
  fi
done

# ── Step 1: Count total reads per BAM ────────────────────────────────
echo "=== Step 1: Counting total reads per BAM ==="
declare -A TOTAL_READS
MIN_READS=999999999999
MIN_SAMPLE=""

for sid in "${SAMPLE_ORDER[@]}"; do
  bam="${BAM_FILES[$sid]}"
  n=$(samtools view -c "$bam")
  TOTAL_READS[$sid]=$n
  echo "  $sid: $n reads"
  if (( n < MIN_READS )); then
    MIN_READS=$n
    MIN_SAMPLE=$sid
  fi
done

echo "  Minimum: $MIN_SAMPLE = $MIN_READS reads"
echo ""

# ── Step 2: Subsample each BAM to MIN_READS ─────────────────────────
echo "=== Step 2: Subsampling BAMs to $MIN_READS reads ==="

# Start report
{
  echo "v4_02 Subsampling Report"
  echo "========================"
  echo "Date: $(date)"
  echo "Seed: $SEED"
  echo "Target reads: $MIN_READS (from $MIN_SAMPLE)"
  echo ""
  printf "%-15s %15s %10s %15s\n" "Sample" "Original" "Fraction" "Expected"
  echo "-----------------------------------------------------------"
} > "$REPORT"

declare -A SUB_BAMS
for sid in "${SAMPLE_ORDER[@]}"; do
  bam="${BAM_FILES[$sid]}"
  out_bam="${OUTDIR}/${sid}.subsampled.bam"
  orig=${TOTAL_READS[$sid]}

  if [[ "$sid" == "$MIN_SAMPLE" ]]; then
    # This is the smallest — just copy it
    frac="1.000000"
    echo "  $sid: copying (already minimum)"
    cp "$bam" "$out_bam"
  else
    # Compute fraction: min / total
    # Use awk for floating point division
    frac=$(awk "BEGIN {printf \"%.6f\", $MIN_READS / $orig}")
    echo "  $sid: subsampling at fraction $frac ($orig -> ~$MIN_READS)"
    # samtools view -s SEED.FRAC: SEED is integer part, FRAC is decimal part
    samtools view -bs "${SEED}.${frac#0.}" "$bam" > "$out_bam"
  fi

  # Index
  samtools index "$out_bam"
  SUB_BAMS[$sid]="$out_bam"

  # Report line
  printf "%-15s %15d %10s %15d\n" "$sid" "$orig" "$frac" "$MIN_READS" >> "$REPORT"
done

echo "" >> "$REPORT"

# ── Step 3: Verify subsampled read counts ────────────────────────────
echo ""
echo "=== Step 3: Verifying subsampled BAMs ==="
{
  echo ""
  echo "Verification (actual subsampled counts):"
  printf "%-15s %15s\n" "Sample" "Subsampled"
  echo "-------------------------------"
} >> "$REPORT"

for sid in "${SAMPLE_ORDER[@]}"; do
  n=$(samtools view -c "${SUB_BAMS[$sid]}")
  echo "  $sid: $n reads"
  printf "%-15s %15d\n" "$sid" "$n" >> "$REPORT"
done

# ── Step 4: Count reads per ACR with bedtools multicov ───────────────
echo ""
echo "=== Step 4: Counting reads per ACR ==="

# Build BAM array in stable order
SUB_BAM_ARRAY=()
for sid in "${SAMPLE_ORDER[@]}"; do
  SUB_BAM_ARRAY+=("${SUB_BAMS[$sid]}")
done

echo "  BED: $BED ($(wc -l < "$BED") regions)"
echo "  BAMs: ${#SUB_BAM_ARRAY[@]}"

bedtools multicov -bams "${SUB_BAM_ARRAY[@]}" -bed "$BED" > "${OUTDIR}/multicov_raw.tsv"

# Add header
{
  printf "chr\tstart\tend"
  for sid in "${SAMPLE_ORDER[@]}"; do
    printf "\t%s" "$sid"
  done
  printf "\n"
  cat "${OUTDIR}/multicov_raw.tsv"
} > "$COUNTS_OUT"
rm -f "${OUTDIR}/multicov_raw.tsv"

NROWS=$(tail -n +2 "$COUNTS_OUT" | wc -l)
echo "  Output: $COUNTS_OUT ($NROWS ACRs x ${#SAMPLE_ORDER[@]} samples)"

# Report library sizes from subsampled counts
{
  echo ""
  echo "ACR count totals (from subsampled BAMs):"
} >> "$REPORT"

# Use awk to compute column sums
awk -F'\t' 'NR>1 {
  for(i=4; i<=NF; i++) sum[i]+=$i
}
END {
  for(i=4; i<=NF; i++) printf "  Column %d: %d\n", i-3, sum[i]
}' "$COUNTS_OUT" >> "$REPORT"

echo ""
echo "=== Done ==="
echo "  Counts: $COUNTS_OUT"
echo "  Report: $REPORT"
cat "$REPORT"
