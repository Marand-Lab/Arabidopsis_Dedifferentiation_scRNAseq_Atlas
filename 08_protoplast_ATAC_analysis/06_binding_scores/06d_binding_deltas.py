#!/usr/bin/env python3
"""
v4_03d — FP delta distributions at bound/occupied tile positions.

For each of 9 categories (3 ACR classes x 3 overlap groups: shared, leaf-only,
proto-only), extracts multi-scale FP from both conditions, computes delta
(leaf - proto), and tests against a null distribution built from bottom-
percentile tiles (unbound/free in both conditions).

Scale ranges (from v4_03b):
  - TF scales:  4-10 bp (7 scales, averaged)
  - Nuc scales: 40-60 bp (21 scales, averaged)

Usage (SLURM):
  sbatch v4/v4_03d_binding_deltas.sh
"""

import argparse
import os
import time

import h5py
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, norm


# ── Constants ─────────────────────────────────────────────────────────────────
def pct_tag(tfbs_pct, nucbs_pct):
    """Format percentile suffix for filenames."""
    def _fmt(v):
        return str(int(v)) if v == int(v) else f"{v:.1f}"
    return f"_tf{_fmt(tfbs_pct)}_nuc{_fmt(nucbs_pct)}"


ACR_CLASSES = ["proto_gain", "stable", "leaf_gain"]
OVERLAP_GROUPS = ["shared", "leaf_only", "proto_only"]
CLASS_COLORS = {"proto_gain": "#E64B35", "stable": "#808080", "leaf_gain": "#4DBBD5"}
GROUP_COLORS = {"shared": "#7B2D8E", "leaf_only": "#4DBBD5", "proto_only": "#E64B35"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bs-dir", default="results/v4_03a_binding_scores")
    p.add_argument("--fp-dir", default="v4/3_PRINT/FP",
                   help="Directory with FP h5ads (or /tmp copy)")
    p.add_argument("--metadata", default="v4/data/acr_metadata.tsv.gz")
    p.add_argument("--mapping", default="data/acr_native_to_resized.tsv")
    p.add_argument("--outdir", default="results/v4_03d_binding_deltas")
    p.add_argument("--tfbs-pct", type=float, default=5)
    p.add_argument("--nucbs-pct", type=float, default=2)
    p.add_argument("--tf-scale-min", type=float, default=4)
    p.add_argument("--tf-scale-max", type=float, default=10)
    p.add_argument("--nuc-scale-min", type=float, default=40)
    p.add_argument("--nuc-scale-max", type=float, default=60)
    p.add_argument("--null-max-tiles", type=int, default=50000,
                   help="Max null tiles per ACR class (subsample if more)")
    p.add_argument("--native-only", action="store_true",
                   help="Restrict to tiles inside native ACR boundaries")
    return p.parse_args()


# ── Helpers ───────────────────────────────────────────────────────────────────

def build_region_to_class(metadata_path, mapping_path, region_strs):
    """Map region strings (resized coords) -> ACR class."""
    meta = pd.read_csv(metadata_path, sep="\t",
                       usecols=["chr", "start", "end", "acr_class"])
    meta["native_str"] = (meta["chr"].str.lower() + ":" +
                          meta["start"].astype(str) + "-" +
                          meta["end"].astype(str))
    mapping = pd.read_csv(mapping_path, sep="\t")
    nat_to_resized = dict(zip(mapping["native_str"], mapping["resized_str"]))
    meta["resized_str"] = meta["native_str"].map(nat_to_resized)
    resized_to_class = dict(zip(meta["resized_str"], meta["acr_class"]))
    classes = np.array([resized_to_class.get(r, "unknown") for r in region_strs])
    n_mapped = (classes != "unknown").sum()
    print(f"  Mapped {n_mapped:,}/{len(region_strs):,} regions to ACR class")
    return classes


def get_scale_indices(fp_h5, region_strs, scale_min, scale_max):
    """Get scale indices for a given bp range from the FP h5ad."""
    # Read scales from uns if available, else assume 2..100
    try:
        import json
        uns_keys = list(fp_h5["uns"].keys())
        if "scales" in uns_keys:
            scales = np.array(fp_h5["uns"]["scales"][...], dtype=np.float64)
        else:
            # Infer from first region shape
            sample_key = region_strs[0]
            n_scales = fp_h5["obsm"][sample_key].shape[1]
            scales = np.arange(n_scales) + 2
    except Exception:
        scales = np.arange(99) + 2

    idx = np.where((scales >= scale_min) & (scales <= scale_max))[0]
    return scales, idx


def extract_fp_at_tiles(fp_h5, region_strs, tile_bp, mask, scale_idx, label=""):
    """Extract FP averaged over scale_idx for tiles where mask is True.

    Returns (n_tiles,) array of mean FP across the selected scales.
    """
    obsm = fp_h5["obsm"]
    avail = set(obsm.keys())
    n_positions = int(mask.sum())

    if n_positions == 0:
        return np.array([], dtype=np.float32)

    fp_out = np.empty(n_positions, dtype=np.float32)
    t0 = time.time()
    pos_idx = 0

    for ri in range(len(region_strs)):
        tile_mask = mask[ri]
        if not tile_mask.any():
            continue

        rstr = region_strs[ri]
        if rstr not in avail:
            n_here = tile_mask.sum()
            fp_out[pos_idx:pos_idx + n_here] = np.nan
            pos_idx += n_here
            continue

        # (n_scales, 2000) — select scale_idx, then tile bp positions
        fp_arr = obsm[rstr][0]  # (n_scales, 2000)
        bp_positions = tile_bp[tile_mask]
        # Mean across selected scales at each tile position
        fp_out[pos_idx:pos_idx + len(bp_positions)] = \
            fp_arr[np.ix_(scale_idx, bp_positions)].mean(axis=0)
        pos_idx += len(bp_positions)

        if (ri + 1) % 5000 == 0 and label:
            print(f"    [{label}] {ri+1:,}/{len(region_strs):,} "
                  f"({pos_idx:,} tiles)...", flush=True)

    if label:
        print(f"  [{label}] {pos_idx:,} tiles in {time.time()-t0:.1f}s", flush=True)
    return fp_out[:pos_idx]


def build_category_masks(mask_leaf, mask_proto, region_classes):
    """Build 9 boolean masks: 3 ACR classes x 3 overlap groups.

    Returns dict of {(acr_class, group): bool_mask} where bool_mask
    has the same shape as mask_leaf (n_regions, n_tiles), True only for
    tiles in this category.
    """
    masks = {}
    for cls in ACR_CLASSES:
        cls_idx = region_classes == cls
        # Expand to tile level
        cls_mask = np.zeros_like(mask_leaf)
        cls_mask[cls_idx] = True

        ml = mask_leaf & cls_mask
        mp = mask_proto & cls_mask

        masks[(cls, "shared")] = ml & mp
        masks[(cls, "leaf_only")] = ml & ~mp
        masks[(cls, "proto_only")] = ~ml & mp
    return masks


def build_null_mask(bot_leaf, bot_proto, max_tiles, rng):
    """Build genome-wide null mask from tiles in bottom percentile of BOTH conditions.

    Bias and noise are genome-wide, not ACR-class-specific, so one pooled null.
    Subsample if needed.
    """
    null_full = bot_leaf & bot_proto
    n_null = int(null_full.sum())

    if n_null > max_tiles:
        rows, cols = np.where(null_full)
        chosen = rng.choice(len(rows), max_tiles, replace=False)
        null_sub = np.zeros_like(null_full)
        null_sub[rows[chosen], cols[chosen]] = True
        print(f"  null: {n_null:,} -> subsampled to {max_tiles:,}")
        return null_sub
    else:
        print(f"  null: {n_null:,} tiles")
        return null_full


def plot_violins(results, null_delta, score_type, scale_label, outdir, tag, suffix=""):
    """3x3 grid of violin plots: rows=ACR class, cols=overlap group.
    null_delta is a single genome-wide array."""
    CLIP = 10.0
    fig, axes = plt.subplots(3, 3, figsize=(12, 10), sharey=True)
    null_d = null_delta

    for row, cls in enumerate(ACR_CLASSES):
        for col, grp in enumerate(OVERLAP_GROUPS):
            ax = axes[row, col]
            key = (cls, grp)
            d = results.get(key, np.array([]))

            # Null reference (clipped)
            if len(null_d) > 0:
                null_clean = np.clip(null_d[np.isfinite(null_d)][:5000], -CLIP, CLIP)
                parts_null = ax.violinplot(
                    [null_clean],
                    positions=[0], widths=0.8, showmedians=True, showextrema=False)
                for pc in parts_null["bodies"]:
                    pc.set_facecolor("#D9D9D9")
                    pc.set_alpha(0.6)
                parts_null["cmedians"].set_color("#999")

            # Category (clipped)
            if len(d) > 0:
                d_clean = np.clip(d[np.isfinite(d)], -CLIP, CLIP)
                if len(d_clean) > 5000:
                    d_clean = np.random.choice(d_clean, 5000, replace=False)
                if len(d_clean) > 1:
                    parts = ax.violinplot(
                        [d_clean], positions=[1], widths=0.8,
                        showmedians=True, showextrema=False)
                    for pc in parts["bodies"]:
                        pc.set_facecolor(GROUP_COLORS[grp])
                        pc.set_alpha(0.7)
                    parts["cmedians"].set_color("k")

            ax.set_ylim(-CLIP, CLIP)
            ax.axhline(0, color="gray", ls="--", lw=0.5, alpha=0.5)

            # Stats annotation
            r = results.get(f"{key}_stats", {})
            n = r.get("n", 0)
            med = r.get("median", 0)
            p = r.get("fdr", 1)
            stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            ax.set_title(f"{grp} (n={n:,})\nmed={med:.3f} {stars}",
                         fontsize=8, color=GROUP_COLORS[grp], fontweight="bold")

            ax.set_xticks([0, 1])
            ax.set_xticklabels(["null", grp[:6]], fontsize=7)

            if col == 0:
                ax.set_ylabel(f"{cls}\ndelta (leaf-proto)", fontsize=9,
                              color=CLASS_COLORS[cls], fontweight="bold")

    fig.suptitle(f"{score_type}: FP delta at {scale_label} scales\n"
                 f"(null = genome-wide bottom-percentile tiles)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(outdir, f"fig_{tag}_delta_violins{suffix}")
    for fmt in ("pdf", "png"):
        fig.savefig(f"{path}.{fmt}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVE] {path}.pdf/.png")


def plot_summary_heatmap(all_stats, outdir, suffix=""):
    """Summary heatmap: median delta with significance, TFBS and NucBS side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, st in zip(axes, ["TFBS", "NucBS"]):
        mat = np.full((3, 3), np.nan)
        annot = np.empty((3, 3), dtype=object)

        for ri, cls in enumerate(ACR_CLASSES):
            for ci, grp in enumerate(OVERLAP_GROUPS):
                key = f"{st}_{cls}_{grp}"
                row = all_stats.get(key, {})
                med = row.get("median", np.nan)
                fdr = row.get("fdr", 1.0)
                mat[ri, ci] = med
                stars = "***" if fdr < 0.001 else "**" if fdr < 0.01 else \
                        "*" if fdr < 0.05 else ""
                annot[ri, ci] = f"{med:.3f}{stars}"

        vmax = np.nanmax(np.abs(mat)) or 0.1
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        for i in range(3):
            for j in range(3):
                ax.text(j, i, annot[i, j], ha="center", va="center",
                        fontsize=9, fontweight="bold")

        ax.set_xticks(range(3))
        ax.set_xticklabels(OVERLAP_GROUPS, fontsize=9)
        ax.set_yticks(range(3))
        ax.set_yticklabels(ACR_CLASSES, fontsize=9)
        ax.set_title(st, fontsize=12, fontweight="bold")
        plt.colorbar(im, ax=ax, shrink=0.8, label="median delta (leaf-proto)")

    fig.suptitle("FP delta summary: median per category (* FDR<0.05, ** <0.01, *** <0.001)",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(outdir, f"fig_C_summary_heatmap{suffix}")
    for fmt in ("pdf", "png"):
        fig.savefig(f"{path}.{fmt}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVE] {path}.pdf/.png")


def plot_tile_heatmaps(tile_level, all_stats, score_type, scale_label, outdir, tag, suffix=""):
    """3x3 grid of single-column heatmaps: ONLY significant tiles (FDR<0.10).
    Two blocks: sig_pos (leaf enriched, top) and sig_neg (proto enriched, bottom).
    Shared color scale normalized to [-1, 1]."""
    fig, axes = plt.subplots(3, 3, figsize=(10, 12))
    FDR_THRESH = 0.10

    # Global vmax for normalization (only from significant tiles)
    global_max = 0.01
    for cls in ACR_CLASSES:
        for grp in OVERLAP_GROUPS:
            tl = tile_level.get((score_type, cls, grp), {})
            d = tl.get("delta", np.array([]))
            fdr = tl.get("fdr", np.array([]))
            z = tl.get("z", np.array([]))
            if len(d) > 0 and len(fdr) > 0:
                sig = fdr < FDR_THRESH
                if sig.any():
                    global_max = max(global_max, np.nanpercentile(np.abs(d[sig]), 99))

    last_im = None
    for row, cls in enumerate(ACR_CLASSES):
        for col, grp in enumerate(OVERLAP_GROUPS):
            ax = axes[row, col]
            key = (score_type, cls, grp)
            tl = tile_level.get(key, {})
            z = tl.get("z", np.array([]))
            fdr = tl.get("fdr", np.array([]))
            delta = tl.get("delta", np.array([]))
            n_total = len(delta)

            if len(delta) == 0:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=9)
                ax.axis("off")
                continue

            # Significant tiles only
            sig_pos = (fdr < FDR_THRESH) & (z > 0)
            sig_neg = (fdr < FDR_THRESH) & (z < 0)

            # Sort: sig_pos descending, then sig_neg descending
            idx_pos = np.where(sig_pos)[0][np.argsort(-delta[sig_pos])]
            idx_neg = np.where(sig_neg)[0][np.argsort(-delta[sig_neg])]
            sorted_idx = np.concatenate([idx_pos, idx_neg])

            n_pos = len(idx_pos)
            n_neg = len(idx_neg)
            n_sig = n_pos + n_neg

            if n_sig == 0:
                ax.text(0.5, 0.5, f"n={n_total:,}\nno sig tiles", ha="center",
                        va="center", transform=ax.transAxes, fontsize=9,
                        color="#999")
                ax.axis("off")
                continue

            sorted_delta = delta[sorted_idx] / global_max

            last_im = ax.imshow(sorted_delta.reshape(-1, 1), aspect="auto",
                                cmap="RdBu_r", vmin=-1, vmax=1,
                                interpolation="none")

            # Block boundary
            if n_pos > 0 and n_neg > 0:
                ax.axhline(n_pos - 0.5, color="k", lw=1.0)

            # Labels: red at top, blue at bottom
            fs = 7
            if n_pos > 0:
                y_pos = n_pos / 2  # pixel coords, top half
                ax.text(1.5, y_pos, f"{n_pos:,}\nleaf",
                        fontsize=fs, color="#D62728",
                        va="center", ha="left", fontweight="bold")
            if n_neg > 0:
                y_neg = n_pos + n_neg / 2  # pixel coords, bottom half
                ax.text(1.5, y_neg, f"{n_neg:,}\nproto",
                        fontsize=fs, color="#1F77B4",
                        va="center", ha="left", fontweight="bold")

            ax.set_xticks([])
            ax.set_yticks([])

            # Title
            cat_key = f"{score_type}_{cls}_{grp}"
            cat_fdr = all_stats.get(cat_key, {}).get("fdr", 1)
            cat_med = all_stats.get(cat_key, {}).get("median", 0)
            stars = "***" if cat_fdr < 0.001 else "**" if cat_fdr < 0.01 \
                else "*" if cat_fdr < 0.05 else "ns"
            direction = "leaf enriched" if cat_med > 0 else "proto enriched"
            pct_sig = n_sig / n_total * 100 if n_total > 0 else 0
            ax.set_title(f"{grp}\n{n_sig:,}/{n_total:,} sig ({pct_sig:.1f}%) "
                         f"({stars} {direction})",
                         fontsize=7, fontweight="bold", color=GROUP_COLORS[grp])

            if col == 0:
                ax.set_ylabel(cls, fontsize=10, fontweight="bold",
                              color=CLASS_COLORS[cls])

    # Shared colorbar
    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=axes, shrink=0.6, pad=0.08,
                            label="normalized delta (leaf-proto)")
        cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
        cbar.set_ticklabels(["proto\nenriched", "", "0", "", "leaf\nenriched"])

    fig.suptitle(f"{score_type}: tile-level delta at {scale_label}\n"
                 f"(FDR<0.05 from null z-score)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 0.88, 0.95])
    path = os.path.join(outdir, f"fig_{tag}_tile_heatmap{suffix}")
    for fmt in ("pdf", "png"):
        fig.savefig(f"{path}.{fmt}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVE] {path}.pdf/.png")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    rng = np.random.default_rng(42)

    print("=== v4_03d: FP Deltas at Bound/Occupied Positions ===")
    print(f"  TFBS top/bottom {args.tfbs_pct}%, NucBS top/bottom {args.nucbs_pct}%")
    print(f"  TF scales: {args.tf_scale_min}-{args.tf_scale_max} bp")
    print(f"  Nuc scales: {args.nuc_scale_min}-{args.nuc_scale_max} bp\n")

    # ── Load binding scores ───────────────────────────────────────────────
    print("[LOAD] Binding scores...")
    leaf_bs = np.load(os.path.join(args.bs_dir, "_bs_leaf.npz"), allow_pickle=True)
    proto_bs = np.load(os.path.join(args.bs_dir, "_bs_proto.npz"), allow_pickle=True)
    region_strs = leaf_bs["region_strs"]
    tile_bp = leaf_bs["tile_bp"]
    assert np.array_equal(region_strs, proto_bs["region_strs"])
    n_regions, n_tiles = leaf_bs["TFBS_prob"].shape
    print(f"  {n_regions:,} regions x {n_tiles} tiles")

    # ── Per-condition percentile masks ────────────────────────────────────
    print("\n[MASKS] Computing per-condition percentile masks...")
    tf_cutoff = 100 - args.tfbs_pct
    nuc_cutoff = 100 - args.nucbs_pct

    # Top percentile = bound/occupied
    tf_leaf = leaf_bs["TFBS_prob"] > np.percentile(leaf_bs["TFBS_prob"], tf_cutoff)
    tf_proto = proto_bs["TFBS_prob"] > np.percentile(proto_bs["TFBS_prob"], tf_cutoff)
    nuc_leaf = leaf_bs["NucBS_prob"] > np.percentile(leaf_bs["NucBS_prob"], nuc_cutoff)
    nuc_proto = proto_bs["NucBS_prob"] > np.percentile(proto_bs["NucBS_prob"], nuc_cutoff)

    # Bottom percentile = unbound/free (for null)
    tf_unb_leaf = leaf_bs["TFBS_prob"] < np.percentile(leaf_bs["TFBS_prob"], args.tfbs_pct)
    tf_unb_proto = proto_bs["TFBS_prob"] < np.percentile(proto_bs["TFBS_prob"], args.tfbs_pct)
    nuc_free_leaf = leaf_bs["NucBS_prob"] < np.percentile(leaf_bs["NucBS_prob"], args.nucbs_pct)
    nuc_free_proto = proto_bs["NucBS_prob"] < np.percentile(proto_bs["NucBS_prob"], args.nucbs_pct)

    # Null = bottom percentile in BOTH conditions
    tf_null_both = tf_unb_leaf & tf_unb_proto
    nuc_null_both = nuc_free_leaf & nuc_free_proto

    # ── Native-only masking ───────────────────────────────────────────
    if args.native_only:
        import sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib"))
        from _tile_utils import build_native_tile_mask
        print("\n[NATIVE] Building native ACR tile mask...")
        native_mask, _ = build_native_tile_mask(
            region_strs, args.metadata, args.mapping)
        tf_leaf &= native_mask
        tf_proto &= native_mask
        nuc_leaf &= native_mask
        nuc_proto &= native_mask
        tf_null_both &= native_mask
        nuc_null_both &= native_mask

    print(f"  TFBS bound:  leaf={tf_leaf.sum():,}  proto={tf_proto.sum():,}")
    print(f"  TFBS null (bottom {args.tfbs_pct}% both): {tf_null_both.sum():,}")
    print(f"  NucBS occupied: leaf={nuc_leaf.sum():,}  proto={nuc_proto.sum():,}")
    print(f"  NucBS null (bottom {args.nucbs_pct}% both): {nuc_null_both.sum():,}")

    # ── ACR class mapping ─────────────────────────────────────────────────
    print("\n[MAP] Region -> ACR class...")
    rcls = build_region_to_class(args.metadata, args.mapping, region_strs)

    # ── Build category + null masks ───────────────────────────────────────
    print("\n[MASKS] Building 9 category masks...")
    tf_cat_masks = build_category_masks(tf_leaf, tf_proto, rcls)
    nuc_cat_masks = build_category_masks(nuc_leaf, nuc_proto, rcls)

    print("\n[MASKS] Building genome-wide null masks (bottom pct, both conditions)...")
    tf_null_mask = build_null_mask(tf_unb_leaf, tf_unb_proto,
                                    args.null_max_tiles, rng)
    nuc_null_mask = build_null_mask(nuc_free_leaf, nuc_free_proto,
                                     args.null_max_tiles, rng)

    for cls in ACR_CLASSES:
        tf_total = sum(tf_cat_masks[(cls, g)].sum() for g in OVERLAP_GROUPS)
        nuc_total = sum(nuc_cat_masks[(cls, g)].sum() for g in OVERLAP_GROUPS)
        print(f"  {cls}: TFBS {tf_total:,} category, NucBS {nuc_total:,} category")
    print(f"  Genome-wide null: TFBS {tf_null_mask.sum():,}, NucBS {nuc_null_mask.sum():,}")

    # ── Open FP h5ads ─────────────────────────────────────────────────────
    fp_leaf_path = os.path.join(args.fp_dir, "leaf_merged__ALL.h5ad")
    fp_proto_path = os.path.join(args.fp_dir, "proto_merged__ALL.h5ad")
    print(f"\n[OPEN] {fp_leaf_path}")
    fp_leaf_h5 = h5py.File(fp_leaf_path, "r")
    print(f"[OPEN] {fp_proto_path}")
    fp_proto_h5 = h5py.File(fp_proto_path, "r")

    # Get scale indices
    scales, tf_si = get_scale_indices(fp_leaf_h5, region_strs,
                                       args.tf_scale_min, args.tf_scale_max)
    _, nuc_si = get_scale_indices(fp_leaf_h5, region_strs,
                                   args.nuc_scale_min, args.nuc_scale_max)
    print(f"\n  TF scale indices: {tf_si[0]}-{tf_si[-1]} "
          f"({scales[tf_si[0]]:.0f}-{scales[tf_si[-1]]:.0f} bp, n={len(tf_si)})")
    print(f"  Nuc scale indices: {nuc_si[0]}-{nuc_si[-1]} "
          f"({scales[nuc_si[0]]:.0f}-{scales[nuc_si[-1]]:.0f} bp, n={len(nuc_si)})")

    # ── Extract FP and compute deltas ─────────────────────────────────────
    all_stats = {}
    tfbs_deltas = {}
    tfbs_null_deltas = {}
    nucbs_deltas = {}
    nucbs_null_deltas = {}

    for score_type, cat_masks, null_mask, scale_idx, scale_label in [
        ("TFBS", tf_cat_masks, tf_null_mask, tf_si, "TF 4-10bp"),
        ("NucBS", nuc_cat_masks, nuc_null_mask, nuc_si, "Nuc 40-60bp"),
    ]:
        delta_dict = tfbs_deltas if score_type == "TFBS" else nucbs_deltas
        null_ref = tfbs_null_deltas if score_type == "TFBS" else nucbs_null_deltas

        print(f"\n{'='*60}")
        print(f"[EXTRACT] {score_type} at {scale_label}")
        print(f"{'='*60}")

        # Extract genome-wide null deltas (one per score type)
        print(f"\n  --- null genome-wide ({null_mask.sum():,} tiles) ---")
        fp_l = extract_fp_at_tiles(fp_leaf_h5, region_strs, tile_bp,
                                    null_mask, scale_idx, "null_leaf")
        fp_p = extract_fp_at_tiles(fp_proto_h5, region_strs, tile_bp,
                                    null_mask, scale_idx, "null_proto")
        null_ref["genome_wide"] = fp_l - fp_p

        # Extract category deltas
        for cls in ACR_CLASSES:
            for grp in OVERLAP_GROUPS:
                key = (cls, grp)
                cm = cat_masks[key]
                n_tiles_cat = cm.sum()
                if n_tiles_cat == 0:
                    delta_dict[key] = np.array([])
                    continue

                print(f"\n  --- {cls}/{grp} ({n_tiles_cat:,} tiles) ---")
                fp_l = extract_fp_at_tiles(fp_leaf_h5, region_strs, tile_bp,
                                            cm, scale_idx, f"{cls}_{grp}_leaf")
                fp_p = extract_fp_at_tiles(fp_proto_h5, region_strs, tile_bp,
                                            cm, scale_idx, f"{cls}_{grp}_proto")
                delta_dict[key] = fp_l - fp_p

    fp_leaf_h5.close()
    fp_proto_h5.close()

    # ── Statistical tests ─────────────────────────────────────────────────
    print(f"\n[TEST] Mann-Whitney U vs null per category...")
    stat_rows = []

    for score_type, delta_dict, null_dict in [
        ("TFBS", tfbs_deltas, tfbs_null_deltas),
        ("NucBS", nucbs_deltas, nucbs_null_deltas),
    ]:
        pvals = []
        keys = []
        null_d = null_dict["genome_wide"]
        null_d_clean = null_d[np.isfinite(null_d)] if len(null_d) > 0 else np.array([])
        print(f"\n  {score_type} null: {len(null_d_clean):,} tiles, "
              f"median={np.median(null_d_clean):.4f}")

        for cls in ACR_CLASSES:
            for grp in OVERLAP_GROUPS:
                key = (cls, grp)
                d = delta_dict[key]
                d_clean = d[np.isfinite(d)] if len(d) > 0 else np.array([])

                n = len(d_clean)
                if n < 5 or len(null_d_clean) < 5:
                    pvals.append(1.0)
                    keys.append(key)
                    stat_rows.append(dict(
                        score_type=score_type, acr_class=cls, group=grp,
                        n=n, median=np.nan, mean=np.nan,
                        null_median=np.median(null_d_clean) if len(null_d_clean) > 0 else np.nan,
                        U=np.nan, pvalue=1.0, fdr=1.0, effect_size=np.nan))
                    continue

                U, p = mannwhitneyu(d_clean, null_d_clean, alternative="two-sided")
                # Rank-biserial effect size
                r_biserial = 1 - (2 * U) / (len(d_clean) * len(null_d_clean))

                pvals.append(p)
                keys.append(key)
                stat_rows.append(dict(
                    score_type=score_type, acr_class=cls, group=grp,
                    n=n, median=np.median(d_clean), mean=np.mean(d_clean),
                    null_median=np.median(null_d_clean),
                    U=U, pvalue=p, fdr=np.nan, effect_size=r_biserial))

        # BH-FDR correction
        pvals_arr = np.array(pvals)
        n_tests = len(pvals_arr)
        if n_tests > 0:
            sorted_idx = np.argsort(pvals_arr)
            fdr = np.empty(n_tests)
            for i, si in enumerate(sorted_idx):
                fdr[si] = pvals_arr[si] * n_tests / (np.searchsorted(
                    pvals_arr[sorted_idx], pvals_arr[si], side="right"))
            fdr = np.minimum.accumulate(fdr[np.argsort(np.argsort(pvals_arr))][::-1])[::-1]
            fdr = np.clip(fdr, 0, 1)

            # Write back FDR
            offset = len(stat_rows) - n_tests
            for i in range(n_tests):
                stat_rows[offset + i]["fdr"] = fdr[i]

                # Store in all_stats for heatmap
                cls, grp = keys[i]
                stat_key = f"{score_type}_{cls}_{grp}"
                all_stats[stat_key] = stat_rows[offset + i]

                # Also store in delta_dict for violin
                delta_dict[keys[i] + ("_stats",)] = stat_rows[offset + i]

    # Rekey for violin plot compatibility
    for score_type, delta_dict in [("TFBS", tfbs_deltas), ("NucBS", nucbs_deltas)]:
        for cls in ACR_CLASSES:
            for grp in OVERLAP_GROUPS:
                key = f"{score_type}_{cls}_{grp}"
                if key in all_stats:
                    delta_dict[f"{(cls, grp)}_stats"] = all_stats[key]

    # ── Print summary ─────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    for row in stat_rows:
        stars = "***" if row["fdr"] < 0.001 else "**" if row["fdr"] < 0.01 else \
                "*" if row["fdr"] < 0.05 else "ns"
        print(f"  {row['score_type']:6s} {row['acr_class']:12s} {row['group']:12s}: "
              f"n={row['n']:>7,}  med={row['median']:>7.3f}  "
              f"null_med={row['null_median']:>7.3f}  "
              f"effect={row['effect_size']:>6.3f}  fdr={row['fdr']:.2e} {stars}")

    # ── Tile-level z-score classification ─────────────────────────────────
    print(f"\n[ZSCORE] Tile-level classification from null distribution...")
    tile_level = {}  # {(score_type, cls, grp): {"z", "fdr", "class"}}

    for score_type, delta_dict, null_dict in [
        ("TFBS", tfbs_deltas, tfbs_null_deltas),
        ("NucBS", nucbs_deltas, nucbs_null_deltas),
    ]:
        null_d = null_dict["genome_wide"]
        null_clean = null_d[np.isfinite(null_d)]
        null_mean = np.mean(null_clean)
        null_sd = np.std(null_clean, ddof=1)
        print(f"\n  {score_type} null: mean={null_mean:.4f}, sd={null_sd:.4f}")

        for cls in ACR_CLASSES:
            for grp in OVERLAP_GROUPS:
                key = (score_type, cls, grp)
                d = delta_dict[(cls, grp)]
                d_clean = d[np.isfinite(d)] if len(d) > 0 else np.array([])
                n = len(d_clean)

                if n == 0 or null_sd == 0:
                    tile_level[key] = {"z": np.array([]), "fdr": np.array([]),
                                       "cls_arr": np.array([]),
                                       "n_sig_pos": 0, "n_ns": 0, "n_sig_neg": 0,
                                       "n_acrs_sig": 0}
                    continue

                # Z-scores
                z = (d_clean - null_mean) / null_sd
                # Two-sided p-values
                pvals_tile = 2 * norm.sf(np.abs(z))

                # BH-FDR within this category
                n_t = len(pvals_tile)
                order = np.argsort(pvals_tile)
                fdr_tile = np.empty(n_t)
                fdr_tile[order] = pvals_tile[order] * n_t / np.arange(1, n_t + 1)
                # Monotonicity (backward cumulative min)
                fdr_tile[order] = np.minimum.accumulate(fdr_tile[order][::-1])[::-1]
                fdr_tile = np.clip(fdr_tile, 0, 1)

                # Classify
                sig_pos = (fdr_tile < 0.10) & (z > 0)
                sig_neg = (fdr_tile < 0.10) & (z < 0)
                ns = ~sig_pos & ~sig_neg

                tile_level[key] = {
                    "z": z, "fdr": fdr_tile, "delta": d_clean,
                    "n_sig_pos": int(sig_pos.sum()),
                    "n_ns": int(ns.sum()),
                    "n_sig_neg": int(sig_neg.sum()),
                }

                stars_cat = ""
                cat_key = f"{score_type}_{cls}_{grp}"
                if cat_key in all_stats:
                    cat_fdr = all_stats[cat_key].get("fdr", 1)
                    stars_cat = "***" if cat_fdr < 0.001 else "**" if cat_fdr < 0.01 \
                        else "*" if cat_fdr < 0.05 else "ns"

                print(f"    {cls:12s} {grp:12s}: n={n:>7,}  "
                      f"sig+={sig_pos.sum():>6,}  ns={ns.sum():>7,}  "
                      f"sig-={sig_neg.sum():>6,}  (MW {stars_cat})")

        # Add tile-level counts to stat_rows
        for row in stat_rows:
            if row["score_type"] == score_type:
                key = (score_type, row["acr_class"], row["group"])
                tl = tile_level.get(key, {})
                row["n_sig_pos"] = tl.get("n_sig_pos", 0)
                row["n_ns"] = tl.get("n_ns", 0)
                row["n_sig_neg"] = tl.get("n_sig_neg", 0)

    # ── Save ──────────────────────────────────────────────────────────────
    _tag = pct_tag(args.tfbs_pct, args.nucbs_pct)
    if args.native_only:
        _tag += "_native"

    tsv_path = os.path.join(args.outdir, f"delta_summary{_tag}.tsv")
    pd.DataFrame(stat_rows).to_csv(tsv_path, sep="\t", index=False, float_format="%.6f")
    print(f"\n[SAVE] {tsv_path}")

    # Save raw deltas
    npz_data = {}
    for score_type, delta_dict, null_dict in [
        ("TFBS", tfbs_deltas, tfbs_null_deltas),
        ("NucBS", nucbs_deltas, nucbs_null_deltas),
    ]:
        npz_data[f"{score_type}_null"] = null_dict["genome_wide"]
        for cls in ACR_CLASSES:
            for grp in OVERLAP_GROUPS:
                npz_data[f"{score_type}_{cls}_{grp}"] = delta_dict[(cls, grp)]
                tl = tile_level.get((score_type, cls, grp), {})
                if len(tl.get("z", [])) > 0:
                    npz_data[f"{score_type}_{cls}_{grp}_z"] = tl["z"]
                    npz_data[f"{score_type}_{cls}_{grp}_fdr"] = tl["fdr"]
    npz_path = os.path.join(args.outdir, f"tile_deltas{_tag}.npz")
    np.savez_compressed(npz_path, **npz_data)
    print(f"[SAVE] {npz_path}")

    # ── Plots ─────────────────────────────────────────────────────────────
    print("\n[PLOT]")

    # Repackage for violin plot function
    tf_results = {}
    nuc_results = {}
    for cls in ACR_CLASSES:
        for grp in OVERLAP_GROUPS:
            key = (cls, grp)
            tf_results[key] = tfbs_deltas[key]
            nuc_results[key] = nucbs_deltas[key]
            tf_stats_key = f"TFBS_{cls}_{grp}"
            nuc_stats_key = f"NucBS_{cls}_{grp}"
            if tf_stats_key in all_stats:
                tf_results[f"{key}_stats"] = all_stats[tf_stats_key]
            if nuc_stats_key in all_stats:
                nuc_results[f"{key}_stats"] = all_stats[nuc_stats_key]

    plot_violins(tf_results, tfbs_null_deltas["genome_wide"],
                 "TFBS bound positions", "TF (4-10 bp)", args.outdir, "A_tfbs",
                 suffix=_tag)
    plot_violins(nuc_results, nucbs_null_deltas["genome_wide"],
                 "NucBS occupied regions", "Nuc (40-60 bp)", args.outdir, "B_nucbs",
                 suffix=_tag)
    plot_summary_heatmap(all_stats, args.outdir, suffix=_tag)
    plot_tile_heatmaps(tile_level, all_stats, "TFBS", "TF (4-10 bp)",
                       args.outdir, "D_tfbs", suffix=_tag)
    plot_tile_heatmaps(tile_level, all_stats, "NucBS", "Nuc (40-60 bp)",
                       args.outdir, "E_nucbs", suffix=_tag)

    print("\n[DONE]")


if __name__ == "__main__":
    main()
