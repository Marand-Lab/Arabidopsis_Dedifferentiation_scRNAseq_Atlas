#!/usr/bin/env python3
"""
Build the canonical ACR metadata table for v4 pipeline.

Joins:
  - Native ACR coordinates (1_ACRs/Athaliana_leaf_protoplast.mergedACRs.bed)
  - edgeR differential accessibility — original 3-rep results
  - Genomic context from GFF3 (promoter / intragenic / distal)

Changes from v2:
  - ACR classification uses FDR only (no logFC threshold) — direction from logFC sign
  - LEC2 annotation removed (not relevant for v4 analysis)
  - Outputs to v4/data/ (does not overwrite shared data/)

Output: v4/data/acr_metadata.tsv.gz
"""

import os
import re
import numpy as np
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE = os.path.join(PROJECT_ROOT, "v4")  # v4/ data dir stays in original location
ACR_BED = os.path.join(PROJECT_ROOT, "1_ACRs", "Athaliana_leaf_protoplast.mergedACRs.bed")
EDGER_2REP = os.path.join(BASE, "differential_ACRs_2rep.tsv")
EDGER_3REP = os.path.join(PROJECT_ROOT, "1_ACRs", "differential_ACRs_tests.unfiltered.txt")
GFF3 = os.path.join(PROJECT_ROOT, "genome", "At.TAIR10.60.Chr.gff3")
OUT_DIR = os.path.join(BASE, "data")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Thresholds ───────────────────────────────────────────────────────────────
FDR_THRESH = 0.05
PROMOTER_UPSTREAM = 1000  # bp upstream of TSS


# =============================================================================
# 1. Load native ACRs
# =============================================================================
print("=" * 60)
print("Step 1: Loading native ACRs")
print("=" * 60)

acrs = pd.read_csv(ACR_BED, sep="\t", header=None, names=["chr", "start", "end"])
acrs["width"] = acrs["end"] - acrs["start"]
# Build join key matching edgeR row IDs: "Chr1_2884_3224"
acrs["edger_key"] = acrs["chr"] + "_" + acrs["start"].astype(str) + "_" + acrs["end"].astype(str)
# Build canonical acr_id in lowercase: "chr1:2884-3224"
acrs["acr_id"] = acrs["chr"].str.lower() + ":" + acrs["start"].astype(str) + "-" + acrs["end"].astype(str)

print(f"  Loaded {len(acrs)} native ACRs")
print(f"  Width: min={acrs['width'].min()}, median={acrs['width'].median():.0f}, "
      f"max={acrs['width'].max()}")


# =============================================================================
# 2. Load and join edgeR results
# =============================================================================
print("\n" + "=" * 60)
print("Step 2: Joining edgeR results")
print("=" * 60)

edger = pd.read_csv(EDGER_3REP, sep="\t", index_col=0)
edger.index.name = "edger_key"
edger = edger.reset_index()

# Rename columns with edgeR_ prefix
edger = edger.rename(columns={
    "logFC": "edgeR_logFC",
    "logCPM": "edgeR_logCPM",
    "F": "edgeR_F",
    "PValue": "edgeR_pvalue",
    "fdr": "edgeR_fdr",
    "lec2": "lec2_flag",
    "size": "edgeR_size",
})

# Join
acrs = acrs.merge(edger, on="edger_key", how="left")
n_matched = acrs["edgeR_logFC"].notna().sum()
print(f"  edgeR rows: {len(edger)}")
print(f"  Matched to ACRs: {n_matched} / {len(acrs)}")

# Classify ACRs — FDR only (no logFC threshold)
# edgeR sign: positive logFC = proto-enriched, negative = leaf-enriched
acrs["acr_class"] = "stable"
acrs.loc[
    (acrs["edgeR_logFC"] > 0) & (acrs["edgeR_fdr"] < FDR_THRESH),
    "acr_class"
] = "proto_gain"
acrs.loc[
    (acrs["edgeR_logFC"] < 0) & (acrs["edgeR_fdr"] < FDR_THRESH),
    "acr_class"
] = "leaf_gain"

class_counts = acrs["acr_class"].value_counts()
print(f"  ACR classification (fdr < {FDR_THRESH}, direction from logFC sign):")
for cls in ["proto_gain", "stable", "leaf_gain"]:
    n = class_counts.get(cls, 0)
    pct = n / len(acrs) * 100
    print(f"    {cls}: {n} ({pct:.1f}%)")





# =============================================================================
# 2. Genomic context annotation (promoter / intragenic / distal)
# =============================================================================
print("\n" + "=" * 60)
print("Step 4: Annotating genomic context")
print("=" * 60)

# Parse GFF3 for gene coordinates
genes = []
with open(GFF3) as f:
    for line in f:
        if line.startswith("#"):
            continue
        fields = line.strip().split("\t")
        if len(fields) < 9 or fields[2] != "gene":
            continue
        chrom = fields[0]
        start = int(fields[3]) - 1  # convert to 0-based
        end = int(fields[4])
        strand = fields[6]
        # Extract gene ID and name
        gene_id = ""
        gene_name = ""
        for attr in fields[8].split(";"):
            if attr.startswith("ID=gene:"):
                gene_id = attr.split("ID=gene:")[1]
            elif attr.startswith("Name="):
                gene_name = attr.split("Name=")[1]
        tss = start if strand == "+" else end
        genes.append({
            "chrom": chrom, "start": start, "end": end,
            "strand": strand, "gene_id": gene_id,
            "gene_name": gene_name, "tss": tss,
        })

genes_df = pd.DataFrame(genes)
print(f"  Parsed {len(genes_df)} genes from GFF3")

# Normalize GFF3 chroms to match ACR BED naming (Chr1, not chr1)
gff3_chroms = genes_df["chrom"].unique()
acr_chroms = set(acrs["chr"].unique())
if len(acr_chroms & set(gff3_chroms)) == 0:
    # Try capitalizing first letter: chr1 -> Chr1
    chrom_map = {c: c[0].upper() + c[1:] for c in gff3_chroms}
    genes_df["chrom"] = genes_df["chrom"].map(chrom_map)
    n_shared = len(acr_chroms & set(genes_df["chrom"].unique()))
    print(f"  Chrom mapping applied (GFF3 '{gff3_chroms[0]}' -> ACR '{chrom_map[gff3_chroms[0]]}')")
    print(f"  Shared chroms after mapping: {n_shared}")

# Build interval arrays for fast annotation (searchsorted approach)
_promoter_intervals = {}
_genebody_intervals = {}
_tss_arrays = {}  # for nearest gene / distance

for chrom, g in genes_df.groupby("chrom"):
    # Promoter intervals
    prom_s = np.where(g["strand"].values == "+",
                      g["tss"].values - PROMOTER_UPSTREAM,
                      g["tss"].values)
    prom_e = np.where(g["strand"].values == "+",
                      g["tss"].values,
                      g["tss"].values + PROMOTER_UPSTREAM)
    order = np.argsort(prom_s)
    _promoter_intervals[chrom] = (prom_s[order], prom_e[order])

    # Gene body intervals
    gb_s = g["start"].values.copy()
    gb_e = g["end"].values.copy()
    order_gb = np.argsort(gb_s)
    _genebody_intervals[chrom] = (gb_s[order_gb], gb_e[order_gb])

    # TSS + gene info for nearest gene lookup
    tss_order = np.argsort(g["tss"].values)
    _tss_arrays[chrom] = {
        "tss": g["tss"].values[tss_order],
        "gene_name": g["gene_name"].values[tss_order],
        "gene_id": g["gene_id"].values[tss_order],
    }


def annotate_acrs(acrs_df):
    """Annotate each ACR with genomic context, nearest gene, and distance to TSS."""
    n = len(acrs_df)
    context = np.full(n, 2, dtype=np.int8)  # 0=Promoter, 1=Gene body, 2=Intergenic
    labels = np.array(["Promoter", "Gene body", "Intergenic"])
    nearest_gene = np.full(n, "", dtype=object)
    dist_to_tss = np.full(n, np.nan, dtype=float)

    for chrom in acrs_df["chr"].unique():
        acr_mask = acrs_df["chr"].values == chrom
        # Use ACR midpoint for context annotation
        midpoints = ((acrs_df.loc[acr_mask, "start"].values +
                       acrs_df.loc[acr_mask, "end"].values) / 2).astype(int)

        if chrom not in _promoter_intervals:
            continue

        # Promoter check
        prom_s, prom_e = _promoter_intervals[chrom]
        idx = np.searchsorted(prom_s, midpoints, side="right") - 1
        valid = (idx >= 0) & (idx < len(prom_s))
        in_prom = valid & (midpoints <= prom_e[np.clip(idx, 0, len(prom_e) - 1)])
        context[np.where(acr_mask)[0][in_prom]] = 0

        # Gene body check (only for non-promoter ACRs)
        gb_s, gb_e = _genebody_intervals[chrom]
        still_ig = context[acr_mask] == 2
        if still_ig.any():
            mp_ig = midpoints[still_ig]
            idx_g = np.searchsorted(gb_s, mp_ig, side="right") - 1
            valid_g = (idx_g >= 0) & (idx_g < len(gb_s))
            in_body = valid_g & (mp_ig <= gb_e[np.clip(idx_g, 0, len(gb_e) - 1)])
            ig_positions = np.where(acr_mask)[0][still_ig]
            context[ig_positions[in_body]] = 1

        # Nearest gene + distance to TSS
        if chrom in _tss_arrays:
            tss_arr = _tss_arrays[chrom]["tss"]
            gene_names = _tss_arrays[chrom]["gene_name"]
            gene_ids = _tss_arrays[chrom]["gene_id"]

            ins_idx = np.searchsorted(tss_arr, midpoints)
            # Compare distance to left and right neighbors
            for j, (mp, ii) in enumerate(zip(midpoints, ins_idx)):
                acr_idx = np.where(acr_mask)[0][j]
                best_dist = np.inf
                best_gene = ""
                for candidate in [ii - 1, ii]:
                    if 0 <= candidate < len(tss_arr):
                        d = abs(mp - tss_arr[candidate])
                        if d < best_dist:
                            best_dist = d
                            name = gene_names[candidate]
                            gid = gene_ids[candidate]
                            best_gene = name if name else gid
                if best_dist < np.inf:
                    nearest_gene[acr_idx] = best_gene
                    dist_to_tss[acr_idx] = best_dist

    acrs_df = acrs_df.copy()
    acrs_df["genomic_context"] = labels[context]
    acrs_df["nearest_gene"] = nearest_gene
    acrs_df["distance_to_tss"] = dist_to_tss
    return acrs_df


acrs = annotate_acrs(acrs)

context_counts = acrs["genomic_context"].value_counts()
print(f"  Genomic context distribution:")
for ctx in ["Promoter", "Gene body", "Intergenic"]:
    n = context_counts.get(ctx, 0)
    pct = n / len(acrs) * 100
    print(f"    {ctx}: {n} ({pct:.1f}%)")


# =============================================================================
# 5. Save output
# =============================================================================
print("\n" + "=" * 60)
print("Step 5: Saving ACR metadata")
print("=" * 60)

# Select and order columns
out_cols = [
    "acr_id", "chr", "start", "end", "width",
    "edgeR_logFC", "edgeR_logCPM", "edgeR_F", "edgeR_pvalue", "edgeR_fdr",
    "acr_class",
    "genomic_context", "nearest_gene", "distance_to_tss",
]
out_df = acrs[out_cols].copy()

out_path = os.path.join(OUT_DIR, "acr_metadata.tsv.gz")
out_df.to_csv(out_path, sep="\t", index=False, compression="gzip")
print(f"  Saved: {out_path}")
print(f"  Rows: {len(out_df)}")
print(f"  Columns: {list(out_df.columns)}")

# Quick summary
print(f"\n  ACR class summary:")
print(out_df["acr_class"].value_counts().to_string(header=False))
print(f"\n  Genomic context:")
print(out_df["genomic_context"].value_counts().to_string(header=False))

# =============================================================================
# 6. Generate resized ACR BED for scPrinter (uniform 2000 bp)
# =============================================================================
print("\n" + "=" * 60)
print("Step 6: Generating resized ACR BED for scPrinter")
print("=" * 60)

FIXED_WIDTH = 2000
FAI_PATH = os.path.join(PROJECT_ROOT, "genome", "At.TAIR10.dna_sm.Chr.fa.fai")

# Load chromosome sizes from FASTA index (lowercase chr1 → uppercase Chr1)
chrom_sizes = {}
with open(FAI_PATH) as fh:
    for line in fh:
        parts = line.strip().split("\t")
        cname = parts[0]
        csize = int(parts[1])
        # Store both original and capitalized keys
        chrom_sizes[cname] = csize
        chrom_sizes[cname[0].upper() + cname[1:]] = csize

# Compute resized coordinates centred on ACR midpoint
acrs["center"] = ((acrs["start"] + acrs["end"]) / 2).astype(int)
acrs["resized_start"] = acrs["center"] - FIXED_WIDTH // 2
acrs["resized_end"] = acrs["center"] + FIXED_WIDTH // 2

# Clip to chromosome boundaries
acrs["resized_start"] = acrs.apply(
    lambda r: max(0, r["resized_start"]), axis=1)
acrs["resized_end"] = acrs.apply(
    lambda r: min(chrom_sizes.get(r["chr"], 999_999_999), r["resized_end"]), axis=1)
acrs["resized_width"] = acrs["resized_end"] - acrs["resized_start"]

n_full = (acrs["resized_width"] == FIXED_WIDTH).sum()
n_clipped = len(acrs) - n_full
if n_clipped > 0:
    print(f"  WARNING: {n_clipped} ACRs too close to chromosome boundary — dropping")
    acrs = acrs[acrs["resized_width"] == FIXED_WIDTH].copy()

# Build region strings (lowercase chr to match genome/scPrinter convention)
acrs["native_str"] = (
    acrs["chr"].str.lower() + ":" +
    acrs["start"].astype(str) + "-" + acrs["end"].astype(str)
)
acrs["resized_str"] = (
    acrs["chr"].str.lower() + ":" +
    acrs["resized_start"].astype(str) + "-" + acrs["resized_end"].astype(str)
)

# Save resized BED (3 columns, Chr naming to match ACR BED convention)
resized_bed = acrs[["chr", "resized_start", "resized_end"]].copy()
resized_bed_path = os.path.join(OUT_DIR, "acr_resized_2000bp.bed")
resized_bed.to_csv(resized_bed_path, sep="\t", header=False, index=False)
print(f"  Saved resized BED: {resized_bed_path}")
print(f"    Regions: {len(resized_bed)}, all width = {FIXED_WIDTH}")

# Save native → resized mapping (for 04b coordinate lookup)
mapping = acrs[["native_str", "resized_str", "resized_start"]].copy()
mapping_path = os.path.join(OUT_DIR, "acr_native_to_resized.tsv")
mapping.to_csv(mapping_path, sep="\t", index=False)
print(f"  Saved mapping: {mapping_path}")
print(f"    Rows: {len(mapping)}")

# Cleanup temp columns
acrs.drop(columns=["center", "resized_start", "resized_end",
                    "resized_width", "native_str", "resized_str"],
          inplace=True, errors="ignore")

print("\n[DONE]")
