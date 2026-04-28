#!/usr/bin/env python3
"""
v4_qc_scale_bias.py — Scale-resolved condition bias QC

Tests whether condition differences in multiscale footprint depth are real biology
or systematic Tn5 accessibility bias (Explanation 3 from the cross-scale sign-flip
analysis). Uses the v4 merged-condition h5ads (leaf_merged, proto_merged).

Two analyses:
  1. ACR-center signal: mean FP depth at ACR center (pos 1000/2000) per scale,
     averaged over all ACRs. Condition × scale interaction at random positions
     → pure bias.
  2. Null loci: random positions within ACRs with no JASPAR motif within ±50 bp.
     Scale-resolved delta (leaf − proto) at these nulls should be ~0 if no bias.

Outputs:
  results/v4_qc_scale_bias/
    scale_bias_qc.npz       — raw arrays for downstream use
    scale_bias_qc.pdf       — 4-panel figure

Usage:
  python v4_qc_scale_bias.py
  python v4_qc_scale_bias.py --n-null 5000 --seed 42
"""
from __future__ import annotations

import argparse
import gc
import os
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Paths ─────────────────────────────────────────────────────────────────────
FP_DIR        = "v4/3_PRINT/FP"
REGIONS_BED   = "v4/data/acr_resized_2000bp.bed"       # 2000 bp resized ACRs
MOTIF_HITS    = "data/v3_merged_motif_hits.tsv.gz"              # v3 signatures, native coords
COORD_MAP     = "data/acr_native_to_resized.tsv"
OUT_DIR       = "results/v4_qc_scale_bias"

LEAF_H5   = os.path.join(FP_DIR, "leaf_merged__ALL.h5ad")
PROTO_H5  = os.path.join(FP_DIR, "proto_merged__ALL.h5ad")

REGION_WIDTH  = 2000   # ACRs are all 2000 bp
ACR_CENTER    = 1000   # 0-based index of the center position

os.makedirs(OUT_DIR, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_fp_tensor(fp_adata, region_str: str) -> np.ndarray:
    """Return (n_scales, n_positions) FP tensor for a single-obs bulk h5ad."""
    arr = np.asarray(fp_adata.obsm[region_str])
    if arr.ndim == 3:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"Unexpected FP shape for {region_str}: {arr.shape}")
    return arr


def region_str_from_row(row) -> str:
    """Build 'Chr1:100-2100' style key matching scPrinter obsm keys."""
    return f"{row.Chromosome}:{row.Start}-{row.End}"


def safe_region_str(row) -> str:
    """Lowercase version used as obsm key (scPrinter stores lowercase)."""
    return region_str_from_row(row).lower()


# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--n-null", type=int, default=5000,
                    help="Number of null loci to sample (default: 5000)")
parser.add_argument("--motif-excl-radius", type=int, default=50,
                    help="Exclude positions within this bp of any motif hit (default: 50)")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--n-acr-center", type=int, default=0,
                    help="Max ACRs for center-signal analysis (0 = all)")
args = parser.parse_args()

rng = np.random.default_rng(args.seed)
print(f"[INFO] n_null={args.n_null}, motif_excl_radius={args.motif_excl_radius}, "
      f"seed={args.seed}", flush=True)

# ── Load ACR regions ──────────────────────────────────────────────────────────
print("[LOAD] ACR regions", flush=True)
regions = pd.read_csv(
    REGIONS_BED, sep="\t", header=None, usecols=[0, 1, 2],
    names=["Chromosome", "Start", "End"]
)
regions["Chromosome"] = regions["Chromosome"].astype(str)
regions["Start"] = regions["Start"].astype(np.int64)
regions["End"] = regions["End"].astype(np.int64)
regions["resized_str"] = regions.apply(safe_region_str, axis=1)
print(f"  {len(regions):,} ACRs", flush=True)

# ── Load motif hits → build exclusion sets per ACR ───────────────────────────
# We need resized coordinates of motif centers to define null loci.
# merged_motif_hits uses native ACR coords; map via coord_map.
print("[LOAD] motif hits + coord mapping", flush=True)
coord_map = pd.read_csv(COORD_MAP, sep="\t")
coord_map["native_str"] = coord_map["native_str"].str.lower()
coord_map["resized_str"] = coord_map["resized_str"].str.lower()
native_to_resized  = dict(zip(coord_map["native_str"], coord_map["resized_str"]))
resized_start_map  = dict(zip(coord_map["resized_str"], coord_map["resized_start"].astype(int)))

# Load hits in chunks — only need motif hit_center in resized coords
motif_centers: dict[str, list[int]] = {}   # resized_str → [center_pos_in_resized, ...]

chunk_rows = 0
for chunk in pd.read_csv(
    MOTIF_HITS, sep="\t", chunksize=200_000,
    usecols=["region_str", "hit_center"],
    dtype={"region_str": str, "hit_center": np.int32},
):
    chunk_rows += len(chunk)
    chunk["region_str"] = chunk["region_str"].str.lower()
    chunk["resized_str"] = chunk["region_str"].map(native_to_resized)
    chunk = chunk.dropna(subset=["resized_str"])
    # Translate hit_center from native to resized coordinate
    for resized_str, grp in chunk.groupby("resized_str"):
        r_start = resized_start_map.get(resized_str)
        if r_start is None:
            continue
        centers_abs = grp["hit_center"].values
        offsets = centers_abs - r_start  # position within the 2000 bp window
        valid = offsets[(offsets >= 0) & (offsets < REGION_WIDTH)]
        if len(valid):
            if resized_str not in motif_centers:
                motif_centers[resized_str] = []
            motif_centers[resized_str].extend(valid.tolist())

print(f"  {chunk_rows:,} hit rows → {len(motif_centers):,} ACRs have motif hits", flush=True)

# ── Sample null loci ──────────────────────────────────────────────────────────
# For each candidate null locus: random position in a random ACR, not within
# motif_excl_radius of any motif hit.
print("[NULL] sampling null loci", flush=True)
excl = args.motif_excl_radius
margin = excl + 10   # don't sample too close to ACR edge either

region_idx_arr = np.arange(len(regions))
null_loci: list[tuple[str, int]] = []   # (resized_str, position_in_window)

max_tries = args.n_null * 20
tries = 0
while len(null_loci) < args.n_null and tries < max_tries:
    tries += 1
    idx = int(rng.integers(0, len(regions)))
    row = regions.iloc[idx]
    rstr = row.resized_str
    pos = int(rng.integers(margin, REGION_WIDTH - margin))
    # Check exclusion
    hits_in_acr = motif_centers.get(rstr, [])
    if hits_in_acr:
        dists = np.abs(np.array(hits_in_acr) - pos)
        if dists.min() < excl:
            continue
    null_loci.append((rstr, pos))

n_null_actual = len(null_loci)
print(f"  Sampled {n_null_actual:,} null loci ({tries:,} tries)", flush=True)
if n_null_actual < args.n_null * 0.5:
    print(f"  [WARN] Only {n_null_actual} null loci; motif density is high.", flush=True)

# Group null loci by ACR for efficient h5ad access
null_by_acr: dict[str, list[int]] = {}
for rstr, pos in null_loci:
    null_by_acr.setdefault(rstr, []).append(pos)
null_acr_keys = set(null_by_acr.keys())

# ACR subset for center-signal analysis
n_acr_center = args.n_acr_center if args.n_acr_center > 0 else len(regions)
acr_center_idx = np.arange(min(n_acr_center, len(regions)))
acr_center_keys = set(regions.iloc[acr_center_idx]["resized_str"].tolist())

# ── Load FP h5ads and extract values ─────────────────────────────────────────
import anndata

conditions = {"leaf": LEAF_H5, "proto": PROTO_H5}

# Output arrays — allocated after we know n_scales from first h5ad
n_scales = None
scales = None

# null: (n_null, n_scales, 2)  — axis2: 0=leaf, 1=proto
null_fp    = None
# acr_center: (n_acr_center, n_scales, 2)
acr_fp     = None

for ci, (cond, h5_path) in enumerate(conditions.items()):
    print(f"\n[FP] Loading {cond} h5ad: {h5_path}", flush=True)
    t0 = time.time()

    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"Missing h5ad: {h5_path}")

    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    fp_adata = anndata.read_h5ad(h5_path, backed="r")

    if n_scales is None:
        scales   = np.array(fp_adata.uns["scales"], dtype=np.float64)
        n_scales = len(scales)
        print(f"  n_scales={n_scales}, range={scales[0]:.0f}–{scales[-1]:.0f} bp",
              flush=True)
        null_fp  = np.full((n_null_actual, n_scales, 2), np.nan, dtype=np.float32)
        acr_fp   = np.full((len(acr_center_idx), n_scales, 2), np.nan, dtype=np.float32)

    avail = set(fp_adata.obsm.keys())

    # --- Null loci extraction ---
    print(f"  Extracting null FP ({n_null_actual:,} loci across "
          f"{len(null_acr_keys):,} ACRs)...", flush=True)
    null_idx = 0
    locus_idx: dict[str, list[tuple[int, int]]] = {}  # rstr → [(locus_global_idx, pos)]
    for i, (rstr, pos) in enumerate(null_loci):
        locus_idx.setdefault(rstr, []).append((i, pos))

    n_found_null = 0
    for rstr, loci in locus_idx.items():
        if rstr not in avail:
            continue
        tensor = load_fp_tensor(fp_adata, rstr)  # (n_scales, n_positions)
        n_found_null += 1
        for global_i, pos in loci:
            if 0 <= pos < tensor.shape[1]:
                null_fp[global_i, :, ci] = tensor[:, pos]
    print(f"    {n_found_null:,}/{len(null_acr_keys):,} null ACRs found in h5ad",
          flush=True)

    # --- ACR center extraction ---
    print(f"  Extracting ACR-center FP ({len(acr_center_idx):,} ACRs)...", flush=True)
    n_found_center = 0
    for arr_i, reg_i in enumerate(acr_center_idx):
        rstr = regions.iloc[reg_i]["resized_str"]
        if rstr not in avail:
            continue
        tensor = load_fp_tensor(fp_adata, rstr)
        if ACR_CENTER < tensor.shape[1]:
            acr_fp[arr_i, :, ci] = tensor[:, ACR_CENTER]
            n_found_center += 1
        if (arr_i + 1) % 5000 == 0:
            print(f"    {arr_i+1:,}/{len(acr_center_idx):,} ACRs...", flush=True)
    print(f"    {n_found_center:,}/{len(acr_center_idx):,} ACR centers found",
          flush=True)

    fp_adata.file.close()
    del fp_adata
    gc.collect()
    print(f"  Done {cond} in {time.time()-t0:.1f}s", flush=True)

# ── Compute summaries ─────────────────────────────────────────────────────────
print("\n[COMPUTE] Summarising...", flush=True)

# Null: drop rows with any NaN
null_valid_mask = np.all(np.isfinite(null_fp), axis=(1, 2))
null_fp_valid   = null_fp[null_valid_mask]   # (n_valid, n_scales, 2)
print(f"  Null valid: {null_fp_valid.shape[0]:,}/{n_null_actual:,}", flush=True)

null_delta       = null_fp_valid[:, :, 0] - null_fp_valid[:, :, 1]  # leaf - proto
null_mean_delta  = null_delta.mean(axis=0)   # (n_scales,)
null_sem_delta   = null_delta.std(axis=0) / np.sqrt(null_fp_valid.shape[0])
null_mean_leaf   = null_fp_valid[:, :, 0].mean(axis=0)
null_mean_proto  = null_fp_valid[:, :, 1].mean(axis=0)
null_sem_leaf    = null_fp_valid[:, :, 0].std(axis=0) / np.sqrt(null_fp_valid.shape[0])
null_sem_proto   = null_fp_valid[:, :, 1].std(axis=0) / np.sqrt(null_fp_valid.shape[0])

# Per-scale t-test: is mean delta significantly different from 0?
null_tstat = np.zeros(n_scales)
null_pval  = np.ones(n_scales)
for si in range(n_scales):
    d = null_delta[:, si]
    d = d[np.isfinite(d)]
    if len(d) > 2:
        t, p = stats.ttest_1samp(d, 0)
        null_tstat[si] = t
        null_pval[si]  = p

# ACR center: drop rows with any NaN
acr_valid_mask = np.all(np.isfinite(acr_fp), axis=(1, 2))
acr_fp_valid   = acr_fp[acr_valid_mask]
print(f"  ACR center valid: {acr_fp_valid.shape[0]:,}/{len(acr_center_idx):,}", flush=True)

acr_delta      = acr_fp_valid[:, :, 0] - acr_fp_valid[:, :, 1]  # leaf - proto
acr_mean_delta = acr_delta.mean(axis=0)
acr_sem_delta  = acr_delta.std(axis=0) / np.sqrt(acr_fp_valid.shape[0])
acr_mean_leaf  = acr_fp_valid[:, :, 0].mean(axis=0)
acr_mean_proto = acr_fp_valid[:, :, 1].mean(axis=0)
acr_sem_leaf   = acr_fp_valid[:, :, 0].std(axis=0) / np.sqrt(acr_fp_valid.shape[0])
acr_sem_proto  = acr_fp_valid[:, :, 1].std(axis=0) / np.sqrt(acr_fp_valid.shape[0])

# ── Save NPZ ──────────────────────────────────────────────────────────────────
npz_path = os.path.join(OUT_DIR, "scale_bias_qc.npz")
np.savez_compressed(
    npz_path,
    scales          = scales,
    # Null
    null_mean_leaf  = null_mean_leaf,
    null_mean_proto = null_mean_proto,
    null_mean_delta = null_mean_delta,
    null_sem_leaf   = null_sem_leaf,
    null_sem_proto  = null_sem_proto,
    null_sem_delta  = null_sem_delta,
    null_tstat      = null_tstat,
    null_pval       = null_pval,
    null_n_valid    = np.array([null_fp_valid.shape[0]]),
    # ACR center
    acr_mean_leaf   = acr_mean_leaf,
    acr_mean_proto  = acr_mean_proto,
    acr_mean_delta  = acr_mean_delta,
    acr_sem_leaf    = acr_sem_leaf,
    acr_sem_proto   = acr_sem_proto,
    acr_sem_delta   = acr_sem_delta,
    acr_n_valid     = np.array([acr_fp_valid.shape[0]]),
)
print(f"[SAVE] {npz_path}", flush=True)

# ── Figure ────────────────────────────────────────────────────────────────────
print("[PLOT] generating figure...", flush=True)

fig = plt.figure(figsize=(14, 15))
gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

ax_a = fig.add_subplot(gs[0, 0])   # ACR center: leaf vs proto mean FP per scale
ax_b = fig.add_subplot(gs[0, 1])   # ACR center: delta per scale
ax_c = fig.add_subplot(gs[1, 0])   # Null: leaf vs proto mean FP per scale
ax_d = fig.add_subplot(gs[1, 1])   # Null: delta + t-stat coloring
ax_e = fig.add_subplot(gs[2, 0])   # Zoom 2-20bp: null delta + ACR center delta
ax_f = fig.add_subplot(gs[2, 1])   # Zoom 2-20bp: -log10(p) from t-test

leaf_col  = "#2166ac"
proto_col = "#d6604d"
delta_col = "#4d9221"

# Significance threshold for colouring (Bonferroni-corrected)
alpha_bonf = 0.05 / n_scales

# ── Panel A: ACR center, leaf vs proto ──
ax_a.plot(scales, acr_mean_leaf,  color=leaf_col,  lw=1.8, label="leaf")
ax_a.plot(scales, acr_mean_proto, color=proto_col, lw=1.8, label="proto")
ax_a.fill_between(scales,
                  acr_mean_leaf  - acr_sem_leaf,
                  acr_mean_leaf  + acr_sem_leaf,
                  color=leaf_col,  alpha=0.15)
ax_a.fill_between(scales,
                  acr_mean_proto - acr_sem_proto,
                  acr_mean_proto + acr_sem_proto,
                  color=proto_col, alpha=0.15)
ax_a.set_xlabel("Scale (bp)")
ax_a.set_ylabel("Mean FP score (−log₁₀ p)")
ax_a.set_title(f"A  ACR center signal\n(n={acr_fp_valid.shape[0]:,} ACRs, pos={ACR_CENTER}bp)",
               fontsize=9)
ax_a.legend(fontsize=8)
ax_a.axvline(20, color="grey", lw=0.7, ls="--", alpha=0.5)
ax_a.text(21, ax_a.get_ylim()[0], "20 bp", fontsize=7, color="grey", va="bottom")

# ── Panel B: ACR center delta ──
ax_b.axhline(0, color="black", lw=0.8, ls="-")
ax_b.plot(scales, acr_mean_delta, color=delta_col, lw=1.8, label="leaf − proto")
ax_b.fill_between(scales,
                  acr_mean_delta - acr_sem_delta,
                  acr_mean_delta + acr_sem_delta,
                  color=delta_col, alpha=0.2)
ax_b.set_xlabel("Scale (bp)")
ax_b.set_ylabel("Mean Δ FP score (leaf − proto)")
ax_b.set_title("B  ACR center: leaf − proto delta", fontsize=9)
ax_b.axvline(20, color="grey", lw=0.7, ls="--", alpha=0.5)
# Shade sub-nucleosomal region
ax_b.axvspan(2, 20, color="gold", alpha=0.08, label="sub-nucleosomal (<20 bp)")
ax_b.legend(fontsize=8)

# ── Panel C: Null loci, leaf vs proto ──
ax_c.plot(scales, null_mean_leaf,  color=leaf_col,  lw=1.8, label="leaf")
ax_c.plot(scales, null_mean_proto, color=proto_col, lw=1.8, label="proto")
ax_c.fill_between(scales,
                  null_mean_leaf  - null_sem_leaf,
                  null_mean_leaf  + null_sem_leaf,
                  color=leaf_col,  alpha=0.15)
ax_c.fill_between(scales,
                  null_mean_proto - null_sem_proto,
                  null_mean_proto + null_sem_proto,
                  color=proto_col, alpha=0.15)
ax_c.set_xlabel("Scale (bp)")
ax_c.set_ylabel("Mean FP score (−log₁₀ p)")
ax_c.set_title(f"C  Null loci (no motif ±{args.motif_excl_radius}bp)\n"
               f"(n={null_fp_valid.shape[0]:,} loci)", fontsize=9)
ax_c.legend(fontsize=8)
ax_c.axvline(20, color="grey", lw=0.7, ls="--", alpha=0.5)

# ── Panel D: Null delta + significance coloring ──
# Color points by significance: red = Bonferroni-significant, grey = not
sig_mask = null_pval < alpha_bonf
delta_colors = np.where(sig_mask, "#b2182b", "#888888")

ax_d.axhline(0, color="black", lw=0.8, ls="-")
ax_d.plot(scales, null_mean_delta, color="#555555", lw=1.2, zorder=1)
ax_d.fill_between(scales,
                  null_mean_delta - null_sem_delta,
                  null_mean_delta + null_sem_delta,
                  color="#aaaaaa", alpha=0.35, zorder=1)
# Scatter significant points in red
if sig_mask.any():
    ax_d.scatter(scales[sig_mask], null_mean_delta[sig_mask],
                 color="#b2182b", s=18, zorder=3,
                 label=f"Bonf. sig. (p<{alpha_bonf:.1e})")
ax_d.scatter(scales[~sig_mask], null_mean_delta[~sig_mask],
             color="#888888", s=8, zorder=2, alpha=0.6, label="not sig.")
ax_d.axvspan(2, 20, color="gold", alpha=0.08, label="sub-nucleosomal")
ax_d.set_xlabel("Scale (bp)")
ax_d.set_ylabel("Mean Δ FP score (leaf − proto)")
ax_d.set_title("D  Null loci: leaf − proto delta\n(red = Bonferroni-significant bias)",
               fontsize=9)
ax_d.axvline(20, color="grey", lw=0.7, ls="--", alpha=0.5)
ax_d.legend(fontsize=8)

# ── Panel E: Zoom 2-20bp — null delta vs ACR-center delta ──
zoom = scales <= 20
zoom_scales = scales[zoom]

ax_e.axhline(0, color="black", lw=0.8, ls="-")
# Null delta
ax_e.plot(zoom_scales, null_mean_delta[zoom], color="#555555", lw=2,
          label="null loci delta", zorder=2)
ax_e.fill_between(zoom_scales,
                  null_mean_delta[zoom] - null_sem_delta[zoom],
                  null_mean_delta[zoom] + null_sem_delta[zoom],
                  color="#aaaaaa", alpha=0.35, zorder=1)
# ACR-center delta for comparison
ax_e.plot(zoom_scales, acr_mean_delta[zoom], color=delta_col, lw=2,
          ls="--", label="ACR center delta", zorder=2)
ax_e.fill_between(zoom_scales,
                  acr_mean_delta[zoom] - acr_sem_delta[zoom],
                  acr_mean_delta[zoom] + acr_sem_delta[zoom],
                  color=delta_col, alpha=0.15, zorder=1)
# Mark each scale as a dot
zoom_sig = sig_mask[zoom]
if zoom_sig.any():
    ax_e.scatter(zoom_scales[zoom_sig], null_mean_delta[zoom][zoom_sig],
                 color="#b2182b", s=30, zorder=4,
                 label=f"Bonf. sig. (p<{alpha_bonf:.1e})")
ax_e.scatter(zoom_scales[~zoom_sig], null_mean_delta[zoom][~zoom_sig],
             color="#888888", s=15, zorder=3, alpha=0.7)
ax_e.set_xlabel("Scale (bp)")
ax_e.set_ylabel("Mean Δ FP score (leaf − proto)")
ax_e.set_title("E  Zoom: sub-nucleosomal delta (2–20 bp)\n"
               "solid = null loci, dashed = ACR center", fontsize=9)
ax_e.set_xlim(1.5, 20.5)
ax_e.set_xticks(range(2, 21, 2))
ax_e.legend(fontsize=7, loc="lower left")

# ── Panel F: Zoom 2-20bp — t-test -log10(p) per scale ──
neg_log_p = -np.log10(np.clip(null_pval, 1e-300, 1.0))
bonf_line = -np.log10(alpha_bonf)

ax_f.bar(zoom_scales, neg_log_p[zoom], width=0.8, color="#4393c3", alpha=0.8,
         edgecolor="none", zorder=2)
ax_f.axhline(bonf_line, color="#b2182b", lw=1.5, ls="--", zorder=3,
             label=f"Bonferroni threshold (−log₁₀ {alpha_bonf:.1e} = {bonf_line:.1f})")
ax_f.axhline(-np.log10(0.05), color="#fdae61", lw=1.2, ls=":",
             label="nominal α = 0.05", zorder=3)
ax_f.set_xlabel("Scale (bp)")
ax_f.set_ylabel("−log₁₀(p)  from one-sample t-test")
ax_f.set_title("F  Zoom: t-test significance per scale (2–20 bp)", fontsize=9)
ax_f.set_xlim(1.5, 20.5)
ax_f.set_xticks(range(2, 21, 2))
ax_f.legend(fontsize=7, loc="upper right")
# Annotate t-statistics on bars
for si in range(len(zoom_scales)):
    s = zoom_scales[si]
    idx = np.searchsorted(scales, s)
    t_val = null_tstat[idx]
    p_val = null_pval[idx]
    if neg_log_p[idx] > 0.5:  # only label visible bars
        ax_f.text(s, neg_log_p[idx] + 0.05, f"t={t_val:.1f}",
                  ha="center", va="bottom", fontsize=6, rotation=45)

# ── Annotation ──
n_sig = sig_mask.sum()
fig.suptitle(
    f"Scale-resolved condition bias QC  |  "
    f"Null: {null_fp_valid.shape[0]:,} loci, "
    f"{n_sig}/{n_scales} scales Bonf. significant at nulls",
    fontsize=10, fontweight="bold"
)

pdf_path = os.path.join(OUT_DIR, "scale_bias_qc.pdf")
fig.savefig(pdf_path, bbox_inches="tight")
png_path = pdf_path.replace(".pdf", ".png")
fig.savefig(png_path, bbox_inches="tight", dpi=150)
plt.close(fig)
print(f"[SAVE] {pdf_path}", flush=True)
print(f"[SAVE] {png_path}", flush=True)

# ── Text summary ─────────────────────────────────────────────────────────────
summary_path = os.path.join(OUT_DIR, "scale_bias_summary.txt")
with open(summary_path, "w") as oh:
    oh.write("=== Scale-resolved condition bias QC ===\n\n")
    oh.write(f"Null loci: {null_fp_valid.shape[0]:,} (target {args.n_null:,})\n")
    oh.write(f"Motif exclusion radius: ±{args.motif_excl_radius} bp\n")
    oh.write(f"ACR center loci: {acr_fp_valid.shape[0]:,}\n")
    oh.write(f"Scales: {n_scales} ({scales[0]:.0f}–{scales[-1]:.0f} bp)\n\n")

    oh.write("--- Null loci delta (leaf − proto) ---\n")
    oh.write(f"  Overall mean delta: {null_mean_delta.mean():.5f}\n")
    oh.write(f"  Sub-nucleosomal (<20bp) mean delta: "
             f"{null_mean_delta[scales < 20].mean():.5f}\n")
    oh.write(f"  Nucleosomal (>100bp range, if present): "
             f"{null_mean_delta[scales > 80].mean():.5f}\n")
    oh.write(f"  Bonferroni-significant scales (p<{alpha_bonf:.1e}): "
             f"{n_sig}/{n_scales}\n")
    if n_sig > 0:
        sig_scales = scales[sig_mask]
        oh.write(f"  Significant scale range: {sig_scales.min():.0f}–"
                 f"{sig_scales.max():.0f} bp\n")
        oh.write(f"  Max |delta| at sig. scales: "
                 f"{np.abs(null_mean_delta[sig_mask]).max():.5f}\n")
    oh.write("\n--- ACR center delta (leaf − proto) ---\n")
    oh.write(f"  Overall mean delta: {acr_mean_delta.mean():.5f}\n")
    oh.write(f"  Sub-nucleosomal (<20bp) mean delta: "
             f"{acr_mean_delta[scales < 20].mean():.5f}\n")
    oh.write(f"  Nucleosomal (scales >80bp): "
             f"{acr_mean_delta[scales > 80].mean():.5f}\n")

    oh.write("\n--- Interpretation guide ---\n")
    oh.write(
        "If Panel D (null loci delta) is flat near 0 with no Bonf. significant scales:\n"
        "  → No systematic Tn5 bias; cross-scale sign-flips are likely real biology.\n"
        "If Panel D shows significant delta at sub-nucleosomal scales (<20 bp):\n"
        "  → Explanation 3 (Tn5 bias) cannot be excluded; treat small-scale signals\n"
        "     with caution.\n"
        "If Panel D shows significant delta only at large scales (>80 bp):\n"
        "  → Global chromatin accessibility difference between conditions affects large\n"
        "     scales; small-scale TF signals are likely genuine.\n"
        "Compare Panel B (ACR center) vs Panel D (null): if B >> D at small scales,\n"
        "  → Small-scale enrichment is motif/TF-specific, not a generic bias.\n"
    )

print(f"[SAVE] {summary_path}", flush=True)
print("\n[DONE]", flush=True)
