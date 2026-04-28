#!/usr/bin/env python3
"""v3_08c_plot — Nucleosome-scale FP landscape around WRKY hits.

Reads wrky_nuc_profiles.npz (from v3_08c) and produces a deeptools-style
figure comparing nucleosome signal between leaf and proto at WRKY hit sites,
stratified by ACR class.

Two-step normalization:
  1. Per-hit (row): divide by row mean → removes coverage bias
  2. Per-position (column): z-score across hits → highlights relative enrichment

Layout (2 columns × 3 rows):
  ┌─ proto_gain ACRs ─┐   ┌─ leaf_gain ACRs ──┐
  │  Leaf (heatmap)    │   │  Leaf (heatmap)    │
  │  Proto (heatmap)   │   │  Proto (heatmap)   │
  │  Mean ± SD profile │   │  Mean ± SD profile │
  └────────────────────┘   └────────────────────┘

Ward clustering on leaf signal, same order applied to proto.

Usage (local):
  ~/Local_installs/miniconda3/bin/python3 -u v3_08c_plot.py
"""

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from scipy.cluster.hierarchy import linkage, leaves_list, optimal_leaf_ordering
from scipy.spatial.distance import pdist

# ── Project paths ─────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))

# Inline the few constants we need from _utils (avoids statsmodels dependency)
ACR_CLASS_COLORS = {"proto_gain": "#D64045", "stable": "#8C8C8C",
                    "leaf_gain": "#3A7D44"}
CLASS_LABELS = {"proto_gain": "Proto-gain", "stable": "Stable",
                "leaf_gain": "Leaf-gain"}


def nature_figure_defaults():
    """Set matplotlib rcParams for Nature-style figures."""
    try:
        import seaborn as sns
        sns.set_context("paper", font_scale=1.1)
    except ImportError:
        pass
    plt.rcParams.update({
        "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 7, "figure.dpi": 150,
    })

COND_COLORS = {"leaf": "#2ca02c", "proto": "#d62728"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_context_mask(hit_region_str, meta_path, coord_map_path):
    """Return boolean mask: True for Promoter/Intergenic hits (exclude Gene body).

    Parameters
    ----------
    hit_region_str : (N,) array of resized region strings (e.g. 'chr1:2054-4054')
    meta_path      : path to acr_metadata.tsv.gz (has genomic_context column)
    coord_map_path : path to acr_native_to_resized.tsv
    """
    import pandas as pd
    meta = pd.read_csv(meta_path, sep="\t")
    # Meta uses Capitalized chr (Chr1), native_str uses lowercase (chr1)
    meta["region_str"] = (meta["chr"].str.lower() + ":" +
                          meta["start"].astype(str) + "-" + meta["end"].astype(str))
    ctx_map = dict(zip(meta["region_str"], meta["genomic_context"]))

    coord_map = pd.read_csv(coord_map_path, sep="\t")
    resized_to_native = dict(zip(coord_map["resized_str"], coord_map["native_str"]))

    contexts = []
    for rs in hit_region_str:
        native = resized_to_native.get(rs, rs)
        ctx = ctx_map.get(native, "unknown")
        contexts.append(ctx)
    contexts = np.array(contexts)

    keep = (contexts == "Promoter") | (contexts == "Intergenic")
    n_total = len(contexts)
    n_keep = keep.sum()
    n_gene = (contexts == "Gene body").sum()
    print(f"  Genomic context filter: {n_total:,} total → {n_keep:,} kept "
          f"({n_gene:,} gene body excluded)", flush=True)
    return keep

def _row_scale(mat):
    """Step 1: per-hit row normalization (divide by row mean).

    Removes coverage bias so high- and low-coverage ACRs are comparable.
    Preserves position-specific signal shape.
    """
    row_mean = np.nanmean(mat, axis=1, keepdims=True)
    row_mean[row_mean == 0] = 1.0
    return mat / row_mean


def _col_zscore(mat):
    """Step 2: per-position column z-score across hits.

    Highlights positions where signal deviates from the population mean.
    Good for heatmaps; destroys mean profile (forces column mean → 0).
    """
    col_mu = np.nanmean(mat, axis=0, keepdims=True)
    col_std = np.nanstd(mat, axis=0, keepdims=True)
    col_std[col_std == 0] = 1.0
    return (mat - col_mu) / col_std


def _ward_order(mat, opt_leaf_max=2000):
    """Return Ward-ordered row indices (NaN → row mean before clustering)."""
    n = mat.shape[0]
    if n < 2:
        return np.arange(n)
    row_means = np.nanmean(mat, axis=1, keepdims=True)
    clean = np.where(np.isfinite(mat), mat, row_means)
    clean = np.nan_to_num(clean, nan=0.0)
    Z = linkage(clean, method="ward", metric="euclidean")
    if n <= opt_leaf_max:
        try:
            Z = optimal_leaf_ordering(Z, pdist(clean))
        except Exception:
            pass
    return leaves_list(Z)


def _ward_order_twolevel(nuc_delta_mat, tf_magnitude, n_clusters=None,
                         opt_leaf_max=2000):
    """Two-level ordering: Ward on nuc delta, then within-cluster sort by |TF delta|.

    Parameters
    ----------
    nuc_delta_mat : (N, P) array — row-scaled nuc-scale delta (leaf − proto)
    tf_magnitude  : (N,) array  — |TF center delta| per hit
    n_clusters    : int or None — number of flat clusters; auto if None
    """
    from scipy.cluster.hierarchy import fcluster
    n = nuc_delta_mat.shape[0]
    if n < 2:
        return np.arange(n)

    # Clean matrix for clustering
    row_means = np.nanmean(nuc_delta_mat, axis=1, keepdims=True)
    clean = np.where(np.isfinite(nuc_delta_mat), nuc_delta_mat, row_means)
    clean = np.nan_to_num(clean, nan=0.0)

    Z = linkage(clean, method="ward", metric="euclidean")
    if n <= opt_leaf_max:
        try:
            Z = optimal_leaf_ordering(Z, pdist(clean))
        except Exception:
            pass

    # Get dendrogram leaf order (global Ward structure)
    ward_leaves = leaves_list(Z)

    # Flat clusters for within-group sorting
    if n_clusters is None:
        n_clusters = max(2, min(n // 20, 10))  # ~20 hits per cluster, max 10
    labels = fcluster(Z, t=n_clusters, criterion="maxclust")

    # Walk Ward leaf order; within each contiguous cluster block, sort by |TF delta|
    final_order = []
    i = 0
    while i < n:
        # Find the contiguous block of the same cluster label in Ward order
        current_label = labels[ward_leaves[i]]
        block = []
        while i < n and labels[ward_leaves[i]] == current_label:
            block.append(ward_leaves[i])
            i += 1
        # Sort block by |TF delta| descending
        block.sort(key=lambda idx: tf_magnitude[idx], reverse=True)
        final_order.extend(block)

    return np.array(final_order)


def _draw_heatmap(ax, mat, pos_x, vmax, cmap, title, show_xlabel=False):
    """Draw a single heatmap panel."""
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(color="lightgray")
    masked = np.ma.array(mat, mask=~np.isfinite(mat))
    n_rows = mat.shape[0]
    im = ax.imshow(masked, aspect="auto", cmap=cmap_obj,
                   vmin=-vmax, vmax=vmax, rasterized=True,
                   extent=[pos_x[0], pos_x[-1], n_rows, 0],
                   interpolation="nearest")
    ax.axvline(0, color="white", lw=0.8, ls="--", alpha=0.7)
    ax.set_xlim(pos_x[0], pos_x[-1])
    ax.set_yticks([])
    ax.set_ylabel(f"n = {n_rows:,}", fontsize=7)
    ax.set_title(title, fontsize=8, fontweight="bold", pad=3)
    if show_xlabel:
        ax.set_xlabel("Distance from WRKY center (bp)", fontsize=7)
    else:
        ax.set_xticks([])
    return im


def _draw_profile(ax, leaf_mat, proto_mat, pos_x, acr_class):
    """Draw mean ± SEM profile for leaf and proto on the same axes."""
    for cond, mat, color in [("Leaf", leaf_mat, COND_COLORS["leaf"]),
                              ("Proto", proto_mat, COND_COLORS["proto"])]:
        n = np.sum(np.isfinite(mat), axis=0).clip(1)
        mu = np.nanmean(mat, axis=0)
        sem = np.nanstd(mat, axis=0) / np.sqrt(n)
        ax.plot(pos_x, mu, color=color, lw=1.2, label=cond)
        ax.fill_between(pos_x, mu - sem, mu + sem, color=color, alpha=0.25)

    ax.axvline(0, color="gray", lw=0.6, ls="--")
    ax.axhline(0, color="gray", lw=0.3, ls=":")
    ax.set_xlim(pos_x[0], pos_x[-1])
    ax.set_ylim(0, 2)
    ax.set_xticks([])
    ax.set_ylabel("Coverage-scaled FP\n(fold of hit mean)", fontsize=7)
    ax.legend(fontsize=6, loc="upper right", framealpha=0.7)
    ax.set_title(f"{CLASS_LABELS[acr_class]} — Mean ± SEM (row-scaled)",
                 fontsize=8)


def _draw_delta_profile(ax, leaf_mat, proto_mat, pos_x, acr_class):
    """Draw mean ± SEM of per-hit delta (leaf − proto) on row-scaled data."""
    delta = leaf_mat - proto_mat
    n = np.sum(np.isfinite(delta), axis=0).clip(1)
    mu = np.nanmean(delta, axis=0)
    sem = np.nanstd(delta, axis=0) / np.sqrt(n)

    ax.plot(pos_x, mu, color="black", lw=1.2)
    ax.fill_between(pos_x, mu - sem, mu + sem, color="gray", alpha=0.25)
    # Shade regions where SEM band excludes zero
    sig = (mu - sem > 0) | (mu + sem < 0)
    ax.fill_between(pos_x, mu - sem, mu + sem,
                    where=sig, color="#2166AC", alpha=0.35)

    ax.axvline(0, color="gray", lw=0.6, ls="--")
    ax.axhline(0, color="red", lw=0.6, ls=":")
    ax.set_xlim(pos_x[0], pos_x[-1])
    ax.set_ylim(-0.4, 1.0)
    ax.set_xlabel("Distance from WRKY center (bp)", fontsize=7)
    ax.set_ylabel("Δ FP\n(leaf − proto)", fontsize=7)
    ax.set_title(f"{CLASS_LABELS[acr_class]} — Delta Mean ± SEM",
                 fontsize=8)


# ── Main figure ───────────────────────────────────────────────────────────────

def make_figure(npz_path, outdir, context_mask=None):
    """Build the two-column deeptools-style nucleosome figure."""
    nature_figure_defaults()

    npz = np.load(npz_path, allow_pickle=True)
    positions_bp = npz["positions_bp"]
    leaf_raw = npz["hit_nuc_leaf_raw"]
    proto_raw = npz["hit_nuc_proto_raw"]
    hit_acr_class = np.array(npz["hit_acr_class"], dtype=str)

    # Apply genomic context filter
    if context_mask is not None:
        leaf_raw = leaf_raw[context_mask]
        proto_raw = proto_raw[context_mask]
        hit_acr_class = hit_acr_class[context_mask]

    # Position window (±250 bp around motif center)
    pos_win = (positions_bp >= -250) & (positions_bp <= 250)
    pos_x = positions_bp[pos_win]

    # Normalize — two stages kept separate
    print("  Normalizing...", flush=True)
    # Step 1 only: for profiles (preserves position-specific signal)
    leaf_s1 = _row_scale(leaf_raw[:, pos_win])
    proto_s1 = _row_scale(proto_raw[:, pos_win])
    # Step 1 + 2: for heatmaps (highlights per-hit deviations)
    leaf_norm = _col_zscore(leaf_s1)
    proto_norm = _col_zscore(proto_s1)

    # Symmetric color scale (98th percentile)
    vmax = np.nanpercentile(
        np.abs(np.concatenate([leaf_norm.ravel(), proto_norm.ravel()])), 98)
    cmap = "RdBu_r"

    acr_classes = ["proto_gain", "leaf_gain"]

    # ── Figure layout: 2 columns × 4 rows ──
    fig = plt.figure(figsize=(14, 12))
    outer = gridspec.GridSpec(1, 2, figure=fig, wspace=0.25)

    for col_idx, cls in enumerate(acr_classes):
        mask = hit_acr_class == cls
        n_hits = mask.sum()
        if n_hits == 0:
            print(f"  {cls}: 0 hits — skipped", flush=True)
            continue

        print(f"  {cls}: {n_hits:,} hits — clustering...", flush=True)

        # Step 1+2 for heatmaps
        leaf_sub = leaf_norm[mask]
        proto_sub = proto_norm[mask]
        # Step 1 only for profiles
        leaf_sub_s1 = leaf_s1[mask]
        proto_sub_s1 = proto_s1[mask]

        # Ward-cluster each condition independently
        leaf_order = _ward_order(leaf_sub)
        proto_order = _ward_order(proto_sub)
        leaf_ord = leaf_sub[leaf_order]
        proto_ord = proto_sub[proto_order]
        leaf_ord_s1 = leaf_sub_s1[leaf_order]
        proto_ord_s1 = proto_sub_s1[proto_order]

        # Inner grid: leaf heat | proto heat | profile | delta profile
        inner = gridspec.GridSpecFromSubplotSpec(
            4, 1, subplot_spec=outer[col_idx],
            height_ratios=[1.0, 1.0, 0.3, 0.3], hspace=0.10)

        ax_leaf = fig.add_subplot(inner[0])
        ax_proto = fig.add_subplot(inner[1])
        ax_prof = fig.add_subplot(inner[2])
        ax_delta = fig.add_subplot(inner[3])

        im = _draw_heatmap(
            ax_leaf, leaf_ord, pos_x, vmax, cmap,
            f"{CLASS_LABELS[cls]} ACRs — Leaf nuc-scale FP")
        _draw_heatmap(
            ax_proto, proto_ord, pos_x, vmax, cmap,
            f"{CLASS_LABELS[cls]} ACRs — Proto nuc-scale FP")
        # Profiles use step-1-only data (row-scaled, not column-z-scored)
        _draw_profile(ax_prof, leaf_sub_s1, proto_sub_s1, pos_x, cls)
        _draw_delta_profile(ax_delta, leaf_sub_s1, proto_sub_s1, pos_x, cls)

    # Shared colorbar
    cbar_ax = fig.add_axes([0.92, 0.35, 0.015, 0.3])
    cb = fig.colorbar(
        plt.cm.ScalarMappable(norm=plt.Normalize(-vmax, vmax),
                              cmap=cmap),
        cax=cbar_ax)
    cb.set_label("Normalized nuc-scale FP (z-score)", fontsize=7)

    ctx_note = " — gene body excluded" if context_mask is not None else ""
    fig.suptitle(
        f"Nucleosome-scale footprint landscape around WRKY hits{ctx_note}\n"
        "(per-hit coverage-scaled → per-position z-scored; each condition Ward-clustered independently)",
        fontsize=10, fontweight="bold", y=1.01)

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(outdir, exist_ok=True)

    pdf_path = os.path.join(outdir, "wrky_nucleosome_landscape.pdf")
    png_path = os.path.join(outdir, "wrky_nucleosome_landscape.png")
    fig.savefig(pdf_path, dpi=150, bbox_inches="tight")
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {pdf_path}", flush=True)
    print(f"  Saved: {png_path}", flush=True)

    return pdf_path


# ── Quadrant-stratified figure ────────────────────────────────────────────────

QUADRANT_ORDER = [
    ("leaf_gain",  True,  "Concordant leaf\n(upper-left, TF δ > 0)"),
    ("leaf_gain",  False, "Discordant leaf\n(lower-left, TF δ < 0)"),
    ("proto_gain", True,  "Discordant proto\n(upper-right, TF δ > 0)"),
    ("proto_gain", False, "Concordant proto\n(lower-right, TF δ < 0)"),
]


def make_quadrant_figure(npz_path, outdir, context_mask=None):
    """4-column figure: nucleosome landscape stratified by TF-scale concordance."""
    nature_figure_defaults()

    npz = np.load(npz_path, allow_pickle=True)
    positions_bp = npz["positions_bp"]
    leaf_raw = npz["hit_nuc_leaf_raw"]
    proto_raw = npz["hit_nuc_proto_raw"]
    tf_delta_raw = npz["hit_tf_delta_raw"]
    hit_acr_class = np.array(npz["hit_acr_class"], dtype=str)

    # Apply genomic context filter
    if context_mask is not None:
        leaf_raw = leaf_raw[context_mask]
        proto_raw = proto_raw[context_mask]
        tf_delta_raw = tf_delta_raw[context_mask]
        hit_acr_class = hit_acr_class[context_mask]

    # TF delta sign at center (mean of ±5 bp)
    center_win = (positions_bp >= -5) & (positions_bp <= 5)
    tf_center_delta = np.nanmean(tf_delta_raw[:, center_win], axis=1)
    tf_sign_pos = tf_center_delta > 0  # positive = leaf-enriched TF delta

    # Position window (±250 bp)
    pos_win = (positions_bp >= -250) & (positions_bp <= 250)
    pos_x = positions_bp[pos_win]

    # Normalize — two stages
    print("  [Quadrants] Normalizing...", flush=True)
    leaf_s1 = _row_scale(leaf_raw[:, pos_win])
    proto_s1 = _row_scale(proto_raw[:, pos_win])
    leaf_norm = _col_zscore(leaf_s1)
    proto_norm = _col_zscore(proto_s1)

    # Shared color scale
    vmax = np.nanpercentile(
        np.abs(np.concatenate([leaf_norm.ravel(), proto_norm.ravel()])), 98)
    cmap = "RdBu_r"

    # ── Figure layout: 4 columns × 4 rows ──
    fig = plt.figure(figsize=(24, 13))
    outer = gridspec.GridSpec(1, 4, figure=fig, wspace=0.20)

    last_im = None
    for col_idx, (cls, is_tf_pos, title) in enumerate(QUADRANT_ORDER):
        mask = (hit_acr_class == cls) & (tf_sign_pos == is_tf_pos)
        n_hits = mask.sum()

        print(f"  [Quadrants] {title.split(chr(10))[0]}: {n_hits:,} hits", flush=True)

        inner = gridspec.GridSpecFromSubplotSpec(
            4, 1, subplot_spec=outer[col_idx],
            height_ratios=[1.0, 1.0, 0.3, 0.3], hspace=0.10)

        ax_leaf = fig.add_subplot(inner[0])
        ax_proto = fig.add_subplot(inner[1])
        ax_prof = fig.add_subplot(inner[2])
        ax_delta = fig.add_subplot(inner[3])

        if n_hits < 5:
            for ax in (ax_leaf, ax_proto, ax_prof, ax_delta):
                ax.text(0.5, 0.5, f"n = {n_hits}\n(too few)",
                        ha="center", va="center", fontsize=9, color="gray",
                        transform=ax.transAxes)
                ax.set_xticks([]); ax.set_yticks([])
            ax_leaf.set_title(f"{title}\nn = {n_hits}", fontsize=8,
                              fontweight="bold")
            continue

        # Subset
        leaf_sub = leaf_norm[mask]
        proto_sub = proto_norm[mask]
        leaf_sub_s1 = leaf_s1[mask]
        proto_sub_s1 = proto_s1[mask]

        # Cluster on proto signal, apply same order to leaf
        #order = _ward_order(proto_sub)
        order = _ward_order(leaf_sub)

        last_im = _draw_heatmap(
            ax_leaf, leaf_sub[order], pos_x, vmax, cmap,
            f"{title}\nLeaf nuc-scale FP (n = {n_hits:,})")
        _draw_heatmap(
            ax_proto, proto_sub[order], pos_x, vmax, cmap,
            f"Proto nuc-scale FP")

        _draw_profile(ax_prof, leaf_sub_s1, proto_sub_s1, pos_x,
                      cls)
        _draw_delta_profile(ax_delta, leaf_sub_s1, proto_sub_s1, pos_x,
                            cls)

    # Shared colorbar
    if last_im is not None:
        cbar_ax = fig.add_axes([0.93, 0.35, 0.012, 0.3])
        cb = fig.colorbar(
            plt.cm.ScalarMappable(norm=plt.Normalize(-vmax, vmax), cmap=cmap),
            cax=cbar_ax)
        cb.set_label("Normalized nuc-scale FP (z-score)", fontsize=7)

    ctx_note = " — gene body excluded" if context_mask is not None else ""
    fig.suptitle(
        f"Nucleosome landscape around WRKY hits — stratified by TF-scale concordance{ctx_note}\n"
        "(quadrants from O2: TF δ sign × ACR class; rows Ward-clustered on proto signal)",
        fontsize=11, fontweight="bold", y=1.01)

    # ── Save ──
    os.makedirs(outdir, exist_ok=True)
    pdf_path = os.path.join(outdir, "wrky_nucleosome_quadrants.pdf")
    png_path = os.path.join(outdir, "wrky_nucleosome_quadrants.png")
    fig.savefig(pdf_path, dpi=150, bbox_inches="tight")
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {pdf_path}", flush=True)
    print(f"  Saved: {png_path}", flush=True)
    return pdf_path


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="v3_08c_plot — Nucleosome landscape around WRKY hits")
    p.add_argument("--npz",
                   default="results/v3_08_gradient_boosting/wrky_nuc_profiles.npz",
                   help="Path to wrky_nuc_profiles.npz (from v3_08c)")
    p.add_argument("--acr-metadata",
                   default="data/acr_metadata.tsv.gz")
    p.add_argument("--coord-mapping",
                   default="data/acr_native_to_resized.tsv")
    p.add_argument("--outdir",
                   default="results/v3_08_gradient_boosting/wrky_nuc_landscape")
    return p.parse_args()


def main():
    args = parse_args()

    # Resolve relative paths
    for attr in ("npz", "acr_metadata", "coord_mapping", "outdir"):
        val = getattr(args, attr)
        if not os.path.isabs(val):
            setattr(args, attr, os.path.join(BASE, val))

    print(f"Loading: {args.npz}", flush=True)

    # Build genomic context filter (exclude gene body)
    npz = np.load(args.npz, allow_pickle=True)
    hit_region_str = np.array(npz["hit_region_str"], dtype=str)
    context_mask = _load_context_mask(hit_region_str, args.acr_metadata,
                                      args.coord_mapping)

    print("\n=== Figure 1: ACR-class stratified ===", flush=True)
    make_figure(args.npz, args.outdir, context_mask=context_mask)
    print("\n=== Figure 2: TF-concordance quadrants ===", flush=True)
    make_quadrant_figure(args.npz, args.outdir, context_mask=context_mask)
    print("[DONE]", flush=True)


if __name__ == "__main__":
    main()
