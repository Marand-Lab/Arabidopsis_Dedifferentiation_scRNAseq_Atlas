#!/usr/bin/env python3
"""
v3 Step 07: Top-signature introduction — scale-resolved FP analysis.

Generalised from v2 10_wrky_introduction.py to show the top N signatures
(by |delta|) instead of WRKY-only. WRKY will naturally be among them.

Figures:
  A — Multi-panel hexbin scatter (leaf vs proto FP at ~10bp scale) for top
      signatures, split by ACR class, with OLS regression + CI.
      Multi-page: 10 signatures per page.
  B — Stacked heatmap (3 ACR-class blocks × scales × top signatures),
      color = OLS slope (β), top-3 dominant scales highlighted.
  C — Ranked bar: top N/2 per direction by |residualized mean delta|.
  D — Family boxplot: per-ACR residualized mean delta, stratified by ACR class.
  E — Family-level hexbin scatter (all ~42 families, per-hit FP at target scale).
      Multi-page: 10 families per page.
  F — Family-level stacked beta heatmap (all families × 3 ACR-class blocks).

All delta-based plots (C, D) use OLS-residualized values (confounders removed).
Scatter/heatmap plots (A, B, E, F) use per-hit FP with ACR-level confounder
correction applied to leaf and proto FP separately.

Depends on: v3_06 per-scale NPZ (chunk NPZs for per-hit data).
Output: results/v3_07_top_signatures/
"""
from __future__ import annotations

import argparse
import gc
import os
import pickle
import sys
import time
import warnings
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from scipy.cluster.hierarchy import linkage, leaves_list, optimal_leaf_ordering
from scipy.spatial.distance import pdist
from scipy.stats import linregress

from _utils import (
    ACR_CLASS_COLORS,
    PALETTE,
    load_acr_metadata,
    nature_figure_defaults,
    nature_savefig,
    residualize_features,
)

warnings.filterwarnings("ignore", category=FutureWarning)

EXCLUDE_REPS = {3}
ACTIVE_REPS = sorted({1, 2, 3} - EXCLUDE_REPS)
LEAF_IDS = [f"leaf_rep{r}" for r in ACTIVE_REPS]
PROTO_IDS = [f"proto_rep{r}" for r in ACTIVE_REPS]
ALL_IDS = LEAF_IDS + PROTO_IDS
N_LEAF = len(LEAF_IDS)

BASE = os.path.dirname(os.path.abspath(__file__))

# Display-time renaming for ambiguous family subgroup names
FAMILY_RENAME = {
    "Group A": "bZIP Group A", "Group B": "bZIP Group B",
    "Group D": "bZIP Group D", "Group G": "bZIP Group G",
    "Group H": "bZIP Group H", "Group I": "bZIP Group I",
    "Group K": "bZIP Group K", "Group S": "bZIP Group S",
    "Type II": "MADS Type II",
}


def rename_family(name):
    return FAMILY_RENAME.get(name, name)


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_signature_metadata(path):
    return pd.read_csv(path, sep="\t")


def load_fp_tensor(fp_adata, region_str):
    arr = np.asarray(fp_adata.obsm[region_str])
    if arr.ndim == 3:
        arr = arr[0]
    return arr


# ── Residualize delta NPZ (ACR-level) ───────────────────────────────────────

def residualize_delta_3d(delta_3d, acr_ids, entity_ids, acr_meta):
    """Residualize (n_acrs, n_entities, n_scales) delta matrix.

    Reshapes to 2D, applies OLS residualization from _utils, reshapes back.
    Returns residualized 3D array + the ACR IDs that survived (subset of input).
    """
    n_acrs, n_ent, n_scales = delta_3d.shape
    col_names = [f"{entity_ids[ei]}_s{si}"
                 for ei in range(n_ent) for si in range(n_scales)]
    flat = delta_3d.reshape(n_acrs, -1)
    df = pd.DataFrame(flat, index=list(acr_ids), columns=col_names)

    resid_df = residualize_features(df, acr_meta)

    # Map back to 3D
    common_acrs = resid_df.index
    resid_3d = resid_df.values.reshape(len(common_acrs), n_ent, n_scales)
    return resid_3d, np.array(common_acrs)


# ── Per-hit FP at target scale from v3_06 chunk NPZs ────────────────────────

def load_perhit_at_target_scale(chunks_dir, sig_meta, target_scale):
    """Stream v3_06 chunk NPZs, extract leaf/proto FP at target scale per hit.

    Returns (leaf_fp, proto_fp, families, region_strs, motif_ids) as 1D arrays.
    """
    mid_to_fam = dict(zip(sig_meta["signature_id"],
                           sig_meta["primary_family"]))
    all_leaf, all_proto, all_fam, all_region, all_motif = [], [], [], [], []

    chunk_paths = sorted(glob(os.path.join(chunks_dir,
                                            "per_hit_fp_chunk_*.npz")))
    if not chunk_paths:
        return None

    for ci, chunk_path in enumerate(chunk_paths):
        data = np.load(chunk_path, allow_pickle=True)
        fp = data["fp_values"]       # (n_hits, n_scales, n_samples)
        scales = data["scales"]
        scale_idx = int(np.argmin(np.abs(scales - target_scale)))

        fp_at_scale = fp[:, scale_idx, :]  # (n_hits, n_samples)
        leaf = np.nanmean(fp_at_scale[:, :N_LEAF], axis=1)
        proto = np.nanmean(fp_at_scale[:, N_LEAF:], axis=1)
        families = np.array([mid_to_fam.get(m, "Unknown")
                              for m in data["motif_ids"]])

        all_leaf.append(leaf)
        all_proto.append(proto)
        all_fam.append(families)
        all_region.append(data["region_strs"])
        all_motif.append(data["motif_ids"])
        del data, fp
        gc.collect()

        if (ci + 1) % 10 == 0:
            print(f"  Chunks loaded: {ci + 1}/{len(chunk_paths)}", flush=True)

    print(f"  All {len(chunk_paths)} chunks loaded", flush=True)
    return {
        "leaf_fp": np.concatenate(all_leaf),
        "proto_fp": np.concatenate(all_proto),
        "families": np.concatenate(all_fam),
        "region_strs": np.concatenate(all_region),
        "motif_ids": np.concatenate(all_motif),
        "target_scale": scales[scale_idx],
    }


def apply_acr_level_correction(leaf_fp, proto_fp, region_strs, acr_meta):
    """Residualize per-hit leaf/proto FP using ACR-level confounder betas.

    For each ACR, compute OLS predicted value from confounders on ACR-level
    mean FP, then subtract from each hit in that ACR.
    """
    from numpy.linalg import lstsq
    from _utils import _build_design_matrix

    # ACR-level mean FP
    unique_acrs = np.unique(region_strs)
    acr_leaf_mean = {}
    acr_proto_mean = {}
    for acr in unique_acrs:
        mask = region_strs == acr
        acr_leaf_mean[acr] = np.nanmean(leaf_fp[mask])
        acr_proto_mean[acr] = np.nanmean(proto_fp[mask])

    # Build design matrix
    acr_order = [a for a in unique_acrs if a in acr_meta.index]
    if len(acr_order) < 20:
        return leaf_fp, proto_fp  # not enough ACRs for correction

    C, valid = _build_design_matrix(acr_meta, acr_order)
    acr_order = list(C.index)
    C_arr = np.column_stack([np.ones(len(C)), C.values])

    # Fit leaf correction
    y_leaf = np.array([acr_leaf_mean.get(a, np.nan) for a in acr_order])
    finite = np.isfinite(y_leaf)
    if finite.sum() >= 20:
        beta_leaf, _, _, _ = lstsq(C_arr[finite], y_leaf[finite], rcond=None)
        pred_leaf = dict(zip(np.array(acr_order), C_arr @ beta_leaf))
    else:
        pred_leaf = {}

    # Fit proto correction
    y_proto = np.array([acr_proto_mean.get(a, np.nan) for a in acr_order])
    finite = np.isfinite(y_proto)
    if finite.sum() >= 20:
        beta_proto, _, _, _ = lstsq(C_arr[finite], y_proto[finite], rcond=None)
        pred_proto = dict(zip(np.array(acr_order), C_arr @ beta_proto))
    else:
        pred_proto = {}

    # Apply per-hit correction
    leaf_corr = leaf_fp.copy()
    proto_corr = proto_fp.copy()
    for i, acr in enumerate(region_strs):
        if acr in pred_leaf:
            leaf_corr[i] -= pred_leaf[acr] - np.nanmean(y_leaf[finite])
        if acr in pred_proto:
            proto_corr[i] -= pred_proto[acr] - np.nanmean(y_proto[finite])

    return leaf_corr, proto_corr


# ── ANCOVA helper ────────────────────────────────────────────────────────────

def _ancova_interaction_p(leaf, proto, classes, class_order):
    """F-test for slope heterogeneity: proto ~ leaf * acr_class.

    Returns the p-value for the interaction term (whether OLS slopes differ
    across ACR classes).  Uses a manual F-test comparing the reduced model
    (common slope) vs the full model (per-class slopes).
    """
    from scipy.stats import f as f_dist

    valid_set = set(class_order)
    mask = (np.isfinite(leaf) & np.isfinite(proto)
            & np.array([c in valid_set for c in classes]))
    if mask.sum() < 30:
        return np.nan

    x, y, c = leaf[mask], proto[mask], classes[mask]
    unique = [cl for cl in class_order if np.sum(c == cl) >= 10]
    if len(unique) < 2:
        return np.nan

    # Reduced model: proto ~ 1 + leaf  (common slope)
    X_r = np.column_stack([np.ones(len(x)), x])
    beta_r = np.linalg.lstsq(X_r, y, rcond=None)[0]
    ss_r = np.sum((y - X_r @ beta_r) ** 2)

    # Full model: proto ~ 1 + leaf + class_dummies + leaf:class_dummies
    cols = [np.ones(len(x)), x]
    for cl in unique[1:]:
        d = (c == cl).astype(float)
        cols.append(d)      # class intercept shift
        cols.append(d * x)  # class slope interaction
    X_f = np.column_stack(cols)
    beta_f = np.linalg.lstsq(X_f, y, rcond=None)[0]
    ss_f = np.sum((y - X_f @ beta_f) ** 2)

    df1 = len(unique) - 1   # number of interaction terms
    df2 = len(x) - X_f.shape[1]
    if df2 <= 0 or ss_f <= 0:
        return np.nan

    f_stat = ((ss_r - ss_f) / df1) / (ss_f / df2)
    return 1 - f_dist.cdf(f_stat, df1, df2)


# ── Phase 1: Extract (or load from NPZ) ─────────────────────────────────────

def load_or_extract_perscale(args, sig_hits, outdir):
    """Load per-hit per-scale FP from existing chunk NPZs or extract fresh."""
    npz_path = os.path.join(outdir, "top_sig_perscale.npz")

    if os.path.exists(npz_path) and not args.force_extract:
        print(f"[INFO] Loading existing NPZ: {npz_path}", flush=True)
        data = np.load(npz_path, allow_pickle=True)
        return {k: data[k] for k in data.files}

    # Extract from h5ad files
    import scprinter as scp

    print(f"[INFO] Extracting per-scale FP for {len(sig_hits):,} hits...",
          flush=True)

    # Load coordinate mapping
    coord_map = pd.read_csv(
        os.path.join(BASE, args.acr_coord_mapping), sep="\t")
    coord_map["native_str"] = coord_map["native_str"].str.lower()
    coord_map["resized_str"] = coord_map["resized_str"].str.lower()
    native_to_resized = dict(zip(coord_map["native_str"],
                                  coord_map["resized_str"]))
    resized_start_map = dict(zip(coord_map["resized_str"],
                                  coord_map["resized_start"]))

    # Map hits to resized coordinates
    sig_hits = sig_hits.copy()
    sig_hits["native_str"] = sig_hits["region_str"].str.replace(
        r"^Chr", "chr", regex=True)
    sig_hits["resized_str"] = sig_hits["native_str"].map(native_to_resized)
    sig_hits = sig_hits.dropna(subset=["resized_str"]).copy()
    sig_hits["resized_start"] = sig_hits["resized_str"].map(
        resized_start_map).astype(int)

    with open(os.path.join(BASE, args.genome_pkl), "rb") as f:
        genome = pickle.load(f)

    # Get scales from first h5ad
    first_h5ad = os.path.join(args.print_dir,
                               f"printer_{ALL_IDS[0]}_bulk.h5ad")
    printer = scp.load_printer(first_h5ad, genome)
    fp_key = f"FP_{ALL_IDS[0]}_ALL".replace("-", "_").replace(".", "_")
    scales = np.array(printer.footprintsadata[fp_key].uns["scales"],
                       dtype=np.float64)
    printer.close(); del printer; gc.collect()

    n_hits = len(sig_hits)
    n_scales = len(scales)
    n_samples = len(ALL_IDS)
    fp_values = np.full((n_hits, n_scales, n_samples), np.nan,
                         dtype=np.float32)

    hit_rows = sig_hits.reset_index(drop=True)
    acr_keys = set(hit_rows["resized_str"].unique())

    for si, sid in enumerate(ALL_IDS):
        h5ad_path = os.path.join(args.print_dir,
                                  f"printer_{sid}_bulk.h5ad")
        if not os.path.exists(h5ad_path):
            continue
        printer = scp.load_printer(h5ad_path, genome)
        fp_key = f"FP_{sid}_ALL".replace("-", "_").replace(".", "_")
        fp_adata = printer.footprintsadata[fp_key]
        avail = set(fp_adata.obsm.keys())

        for acr_str in acr_keys:
            if acr_str not in avail:
                continue
            tensor = load_fp_tensor(fp_adata, acr_str)
            mask = hit_rows["resized_str"] == acr_str
            for idx in hit_rows.index[mask]:
                ci = (int(hit_rows.at[idx, "hit_center"])
                      - int(hit_rows.at[idx, "resized_start"]))
                if 0 <= ci < tensor.shape[1]:
                    fp_values[idx, :, si] = tensor[:, ci]

        printer.close(); del printer, fp_adata; gc.collect()
        print(f"  {sid}: done", flush=True)

    os.makedirs(outdir, exist_ok=True)
    np.savez_compressed(
        npz_path,
        fp_values=fp_values,
        scales=scales,
        region_strs=hit_rows["resized_str"].values,
        motif_ids=hit_rows["motif_id"].values,
        sample_ids=np.array(ALL_IDS),
    )
    print(f"  Saved {npz_path}", flush=True)
    return {"fp_values": fp_values, "scales": scales,
            "region_strs": hit_rows["resized_str"].values,
            "motif_ids": hit_rows["motif_id"].values,
            "sample_ids": np.array(ALL_IDS)}


# ── Figure A: Multi-panel scatter (multi-page) ──────────────────────────────

def figure_a_scatter(fp_values, scales, region_strs, motif_ids,
                     acr_class_map, acr_meta, sig_meta, target_scale,
                     top_sigs, outdir, sigs_per_page=10):
    """Hexbin scatter of leaf vs proto FP at target scale, one row per sig."""
    nature_figure_defaults()

    scale_idx = np.argmin(np.abs(scales - target_scale))
    actual_scale = scales[scale_idx]

    leaf_fp = np.nanmean(fp_values[:, scale_idx, :N_LEAF], axis=1)
    proto_fp = np.nanmean(fp_values[:, scale_idx, N_LEAF:], axis=1)

    # Apply ACR-level confounder correction
    leaf_fp, proto_fp = apply_acr_level_correction(
        leaf_fp, proto_fp, region_strs, acr_meta)

    acr_classes = np.array([acr_class_map.get(rs, "unknown")
                             for rs in region_strs])

    class_order = ["leaf_gain", "stable", "proto_gain"]
    class_labels = {"proto_gain": "Proto-gain", "stable": "Stable",
                    "leaf_gain": "Leaf-gain"}
    class_colors = {k: PALETTE.get(k, "0.5") for k in class_order}
    dn_map = dict(zip(sig_meta["signature_id"], sig_meta["display_name"]))

    n_sigs = len(top_sigs)
    n_pages = (n_sigs + sigs_per_page - 1) // sigs_per_page

    # -- shared helper for one row of panels --
    def _draw_sig_row(axes_row, sig_id, sig_mask, row_idx=0,
                      show_col_titles=True):
        """Draw 3 class panels for one signature; return ANCOVA p."""
        for ci, cls in enumerate(class_order):
            ax = axes_row[ci]
            m = (sig_mask & (acr_classes == cls)
                 & np.isfinite(leaf_fp) & np.isfinite(proto_fp))
            x, y = leaf_fp[m], proto_fp[m]
            n_hits = m.sum()
            n_acrs = np.unique(region_strs[m]).size if n_hits > 0 else 0

            if n_hits < 10:
                ax.set_title(f"{class_labels[cls]} "
                             f"(hits={n_hits}, ACRs={n_acrs})")
                continue

            ax_max = 4.0
            ax.hexbin(x, y, gridsize=60, cmap="Greys", mincnt=1,
                      linewidths=0.2, extent=[0, ax_max, 0, ax_max])
            ax.plot([0, ax_max], [0, ax_max], "--", color="0.5",
                    lw=0.8, zorder=3)

            res = linregress(x, y)
            xfit = np.linspace(0, ax_max, 200)
            yfit = res.slope * xfit + res.intercept
            ax.plot(xfit, yfit, color=class_colors[cls], lw=1.5, zorder=4)

            ax.set_xlim(0, ax_max)
            ax.set_ylim(0, ax_max)
            ax.annotate(
                f"\u03b2={res.slope:.3f} R\u00b2={res.rvalue**2:.3f}\n"
                f"hits={n_hits:,}  ACRs={n_acrs:,}",
                xy=(0.05, 0.95), xycoords="axes fraction",
                va="top", fontsize=6, family="monospace")

            if show_col_titles:
                ax.set_title(f"{class_labels[cls]}")

        # ANCOVA interaction test across classes
        all_m = sig_mask & np.isfinite(leaf_fp) & np.isfinite(proto_fp)
        ancova_p = _ancova_interaction_p(
            leaf_fp[all_m], proto_fp[all_m],
            acr_classes[all_m], class_order)
        return ancova_p

    # -- Multi-page PDF --
    pdf_path = os.path.join(outdir, "fig_a_scatter.pdf")
    with PdfPages(pdf_path) as pdf:
        for page in range(n_pages):
            start = page * sigs_per_page
            end = min(start + sigs_per_page, n_sigs)
            page_sigs = top_sigs[start:end]
            n_rows = len(page_sigs)

            fig, axes = plt.subplots(n_rows, 3,
                                      figsize=(9, 2.8 * n_rows),
                                      squeeze=False)

            for ri, sig_id in enumerate(page_sigs):
                sig_mask = motif_ids == sig_id
                ancova_p = _draw_sig_row(
                    axes[ri], sig_id, sig_mask, row_idx=ri,
                    show_col_titles=(ri == 0))
                label = dn_map.get(sig_id, sig_id)
                if np.isfinite(ancova_p):
                    label += f"  (ANCOVA p={ancova_p:.2e})"
                axes[ri, 0].set_ylabel(label, fontsize=7)

            fig.suptitle(f"Leaf vs Proto FP at ~{actual_scale:.0f} bp "
                         f"(corrected, page {page + 1}/{n_pages})",
                         fontsize=11)
            fig.supxlabel("Leaf FP (corrected)", fontsize=9)
            fig.supylabel("Proto FP (corrected)", fontsize=9)
            plt.tight_layout()
            pdf.savefig(fig, dpi=200)
            plt.close(fig)

    # -- Individual PNGs per signature --
    ind_dir = os.path.join(outdir, "fig_a_scatter")
    os.makedirs(ind_dir, exist_ok=True)
    for sig_id in top_sigs:
        sig_mask = motif_ids == sig_id
        fig, axes = plt.subplots(1, 3, figsize=(9, 2.8), squeeze=False)
        ancova_p = _draw_sig_row(axes[0], sig_id, sig_mask)
        axes[0, 0].set_ylabel("Proto FP (corrected)", fontsize=8)
        display = dn_map.get(sig_id, sig_id)
        ancova_str = (f"  |  ANCOVA interaction p={ancova_p:.2e}"
                      if np.isfinite(ancova_p) else "")
        fig.suptitle(f"{display} \u2014 Leaf vs Proto FP at "
                     f"~{actual_scale:.0f} bp{ancova_str}",
                     fontsize=9)
        fig.supxlabel("Leaf FP (corrected)", fontsize=9)
        plt.tight_layout()
        safe_name = sig_id.replace("/", "_").replace(" ", "_")
        nature_savefig(fig, safe_name, ind_dir)
        plt.close(fig)
    print(f"  Saved Figure A: {n_pages}-page PDF + "
          f"{n_sigs} individual PNGs in fig_a_scatter/", flush=True)


# ── Figure B: Stacked beta heatmap ──────────────────────────────────────────

def figure_b_heatmap(fp_values, scales, region_strs, motif_ids,
                     acr_class_map, acr_meta, sig_meta, top_sigs, outdir):
    """Stacked heatmap: 3 ACR-class blocks × scales × top signatures."""
    nature_figure_defaults()

    leaf_fp = np.nanmean(fp_values[:, :, :N_LEAF], axis=2)
    proto_fp = np.nanmean(fp_values[:, :, N_LEAF:], axis=2)
    acr_classes = np.array([acr_class_map.get(rs, "unknown")
                             for rs in region_strs])

    dn_map = dict(zip(sig_meta["signature_id"], sig_meta["display_name"]))
    class_order = ["proto_gain", "stable", "leaf_gain"]
    n_sigs = len(top_sigs)
    n_scales = len(scales)

    # Compute OLS slope (proto ~ leaf) per (class, scale, signature)
    beta_blocks = []
    for cls in class_order:
        betas = np.full((n_sigs, n_scales), np.nan)
        for si, sig_id in enumerate(top_sigs):
            m = (motif_ids == sig_id) & (acr_classes == cls)
            for sci in range(n_scales):
                x = leaf_fp[m, sci]
                y = proto_fp[m, sci]
                valid = np.isfinite(x) & np.isfinite(y)
                if valid.sum() >= 20:
                    res = linregress(x[valid], y[valid])
                    betas[si, sci] = res.slope
        beta_blocks.append(betas)

    _plot_stacked_heatmap(beta_blocks, class_order, top_sigs,
                           dn_map, scales, outdir, "fig_b_heatmap",
                           "Scale-resolved FP regression: top signatures")


def _plot_stacked_heatmap(beta_blocks, class_order, entity_ids,
                           label_map, scales, outdir, fig_name, title):
    """Shared logic for signature-level (Fig B) and family-level (Fig F).

    Each ACR class is drawn on its own subplot so that the resulting PDF
    contains 3 independent image objects (editable in Illustrator).
    """
    n_ent = len(entity_ids)
    n_scales = len(scales)
    class_labels = {"proto_gain": "Proto-gain", "stable": "Stable",
                    "leaf_gain": "Leaf-gain"}

    # Cluster entities (Ward on mean beta across all blocks)
    mean_beta = np.nanmean(np.vstack(beta_blocks), axis=1)
    mean_beta_2d = mean_beta.reshape(3, n_ent)
    feat = np.nan_to_num(mean_beta_2d.T, nan=1.0)
    if n_ent > 2:
        dist = pdist(feat, metric="euclidean")
        Z = linkage(dist, method="ward")
        try:
            Z = optimal_leaf_ordering(Z, dist)
        except Exception:
            pass
        order = leaves_list(Z)
    else:
        order = np.arange(n_ent)

    ent_labels = [label_map.get(entity_ids[i], entity_ids[i])
                   for i in order]

    # Shared color normalization across all 3 blocks
    all_vals = np.vstack(beta_blocks)
    finite_vals = all_vals[np.isfinite(all_vals)]
    if len(finite_vals) == 0:
        return
    vmax = np.nanpercentile(np.abs(finite_vals), 95)
    norm = TwoSlopeNorm(vcenter=1.0, vmin=1.0 - vmax, vmax=1.0 + vmax)

    # 3 subplots — one per ACR class
    panel_h = max(0.3 * n_ent, 2)
    fig, axes = plt.subplots(
        3, 1, figsize=(10, panel_h * 3 + 1.5),
        gridspec_kw={"height_ratios": [n_ent] * 3, "hspace": 0.35},
        sharex=True)

    tick_positions = np.linspace(0, n_scales - 1, 6).astype(int)
    im = None

    for bi, cls in enumerate(class_order):
        ax = axes[bi]
        block = beta_blocks[bi][order]
        im = ax.imshow(block, aspect="auto", cmap="PRGn", norm=norm,
                       interpolation="nearest")

        # Entity labels on y-axis
        ax.set_yticks(range(n_ent))
        ax.set_yticklabels(ent_labels, fontsize=5)

        # ACR class label
        ax.set_title(class_labels.get(cls, cls), fontsize=9,
                     fontweight="bold")

        # Top-3 dominant scales per entity
        for si in range(n_ent):
            row = block[si]
            deviation = np.abs(row - 1.0)
            deviation = np.where(np.isfinite(deviation), deviation, -1)
            top3 = np.argsort(deviation)[-3:]
            top3 = top3[deviation[top3] > 0]
            for sci in top3:
                ax.plot(sci, si, marker="o", ms=3, mfc="none",
                        mec="k", mew=0.6, zorder=5)

        # Only bottom subplot gets x-tick labels
        if bi < 2:
            ax.tick_params(axis="x", labelbottom=False)

    axes[-1].set_xticks(tick_positions)
    axes[-1].set_xticklabels([f"{scales[i]:.0f}" for i in tick_positions])
    axes[-1].set_xlabel("Scale (bp)")

    fig.colorbar(im, ax=axes.tolist(), label="OLS slope (\u03b2)",
                 shrink=0.6, pad=0.02)
    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    nature_savefig(fig, fig_name, outdir)
    plt.close(fig)
    print(f"  Saved {fig_name}", flush=True)


# ── Figure C: Ranked bar — residualized delta ────────────────────────────────

def figure_c_ranked_bar(sig_delta_resid, sig_ids, sig_meta, outdir,
                         top_n=20):
    """Horizontal bar: top N/2 per direction sorted by |residualized delta|."""
    nature_figure_defaults()
    LEAF_COLOR = PALETTE.get("leaf_gain", "#2b8cbe")
    PROTO_COLOR = PALETTE.get("proto_gain", "#e34a33")

    mean_delta = np.nanmean(sig_delta_resid, axis=(0, 2))
    dn_map = dict(zip(sig_meta["signature_id"], sig_meta["display_name"]))

    df = pd.DataFrame({"signature_id": list(sig_ids),
                        "mean_delta": mean_delta})
    df["display_name"] = (df["signature_id"].map(dn_map)
                          .fillna(df["signature_id"]))
    df["direction"] = np.where(df["mean_delta"] > 0,
                                "Leaf-enriched", "Proto-enriched")
    df["abs_delta"] = df["mean_delta"].abs()

    n_each = max(top_n // 2, 5)
    top_leaf = df[df["direction"] == "Leaf-enriched"].nlargest(
        n_each, "abs_delta")
    top_proto = df[df["direction"] == "Proto-enriched"].nlargest(
        n_each, "abs_delta")
    plot_df = pd.concat([top_leaf, top_proto]).sort_values(
        "mean_delta", ascending=True)

    colors = [LEAF_COLOR if d == "Leaf-enriched" else PROTO_COLOR
              for d in plot_df["direction"]]

    fig, ax = plt.subplots(figsize=(7, max(4, len(plot_df) * 0.35)))
    ax.barh(range(len(plot_df)), plot_df["mean_delta"],
            color=colors, edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(plot_df["display_name"], fontsize=7)
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_xlabel("Mean residualized delta (leaf − proto)")
    ax.set_title(f"Top {n_each} signatures per direction "
                 f"(confounders removed)")
    ax.legend(handles=[
        Line2D([0], [0], color=LEAF_COLOR, lw=6, label="Leaf-enriched"),
        Line2D([0], [0], color=PROTO_COLOR, lw=6, label="Proto-enriched"),
    ], loc="lower right", fontsize=7)
    plt.tight_layout()
    nature_savefig(fig, "fig_c_ranked_bar", outdir)
    plt.close(fig)
    print("  Saved Figure C", flush=True)


# ── Figure D: Family delta boxplot — stratified by ACR class ─────────────────

def figure_d_family_delta(fam_delta_resid, family_ids, acr_ids, acr_meta,
                           outdir):
    """3-panel boxplot: per-ACR residualized mean delta, by ACR class."""
    nature_figure_defaults()

    acr_mean = np.nanmean(fam_delta_resid, axis=2)
    fam_labels = [rename_family(f) for f in family_ids]
    df_wide = pd.DataFrame(acr_mean, index=list(acr_ids),
                            columns=fam_labels)

    # Add ACR class
    if "acr_class" in acr_meta.columns:
        df_wide["acr_class"] = acr_meta.reindex(df_wide.index)["acr_class"]
    else:
        df_wide["acr_class"] = "all"

    df_long = df_wide.melt(id_vars="acr_class", var_name="family",
                            value_name="mean_delta")
    df_long = df_long.dropna(subset=["mean_delta", "acr_class"])

    # Sort families by overall median
    fam_order = (df_long.groupby("family")["mean_delta"]
                 .median().sort_values().index.tolist())

    class_order = ["proto_gain", "stable", "leaf_gain"]
    class_labels = {"proto_gain": "Proto-gain", "stable": "Stable",
                    "leaf_gain": "Leaf-gain"}

    fig, axes = plt.subplots(1, 3,
                              figsize=(24, max(5, len(fam_order) * 0.35)),
                              sharey=True)

    for ci, cls in enumerate(class_order):
        ax = axes[ci]
        sub = df_long[df_long["acr_class"] == cls]
        n_acrs = sub["family"].value_counts().iloc[0] if len(sub) > 0 else 0

        if sub.empty:
            ax.set_title(f"{class_labels[cls]} (n=0)")
            continue

        sns.boxplot(data=sub, y="family", x="mean_delta", order=fam_order,
                    color=ACR_CLASS_COLORS.get(cls, "#888888"),
                    showfliers=True,
                    flierprops=dict(marker=".", markersize=2, alpha=0.3),
                    ax=ax)
        ax.axvline(0, color="grey", ls="--", lw=0.8)
        ax.set_xlabel("Mean residualized delta", fontsize=8)
        ax.set_title(f"{class_labels[cls]} (n={n_acrs:,})", fontsize=9)
        ax.tick_params(axis="y", labelsize=6)
        if ci > 0:
            ax.set_ylabel("")

    fig.suptitle("Residualized FP Delta by TF Family × ACR Class",
                 fontsize=11)
    plt.tight_layout()
    nature_savefig(fig, "fig_d_family_delta", outdir)
    plt.close(fig)
    print("  Saved Figure D", flush=True)


# ── Figure E: Family-level per-hit hexbin scatter (multi-page) ──────────────

def figure_e_family_scatter(perhit_data, acr_class_map, acr_meta,
                             outdir, fams_per_page=10):
    """Hexbin scatter: leaf vs proto FP per hit, one row per family."""
    nature_figure_defaults()

    leaf_fp = perhit_data["leaf_fp"]
    proto_fp = perhit_data["proto_fp"]
    families = perhit_data["families"]
    region_strs = perhit_data["region_strs"]
    actual_scale = perhit_data["target_scale"]

    # Apply ACR-level correction
    leaf_fp, proto_fp = apply_acr_level_correction(
        leaf_fp, proto_fp, region_strs, acr_meta)

    acr_classes = np.array([acr_class_map.get(rs, "unknown")
                             for rs in region_strs])

    # All families sorted by median |delta|
    unique_fams = np.unique(families[families != "Unknown"])
    fam_med = {}
    for fam in unique_fams:
        m = families == fam
        d = leaf_fp[m] - proto_fp[m]
        fam_med[fam] = np.nanmedian(np.abs(d))
    fam_order = sorted(unique_fams, key=lambda f: fam_med.get(f, 0),
                        reverse=True)

    class_order = ["leaf_gain", "stable", "proto_gain"]
    class_labels = {"proto_gain": "Proto-gain", "stable": "Stable",
                    "leaf_gain": "Leaf-gain"}
    class_colors = {k: PALETTE.get(k, "0.5") for k in class_order}

    n_fams = len(fam_order)
    n_pages = (n_fams + fams_per_page - 1) // fams_per_page

    # -- shared helper for one row of family panels --
    def _draw_fam_row(axes_row, fam, fam_mask, show_col_titles=True):
        """Draw 3 class panels for one family; return ANCOVA p."""
        for ci, cls in enumerate(class_order):
            ax = axes_row[ci]
            m = (fam_mask & (acr_classes == cls)
                 & np.isfinite(leaf_fp) & np.isfinite(proto_fp))
            x, y = leaf_fp[m], proto_fp[m]
            n_hits = m.sum()
            n_acrs = np.unique(region_strs[m]).size if n_hits > 0 else 0

            if n_hits < 10:
                ax.set_title(f"{class_labels[cls]} "
                             f"(hits={n_hits}, ACRs={n_acrs})")
                continue

            ax_max = 4.0
            ax.hexbin(x, y, gridsize=60, cmap="Greys", mincnt=1,
                      linewidths=0.2, extent=[0, ax_max, 0, ax_max])
            ax.plot([0, ax_max], [0, ax_max], "--", color="0.5",
                    lw=0.8, zorder=3)

            res = linregress(x, y)
            xfit = np.linspace(0, ax_max, 200)
            yfit = res.slope * xfit + res.intercept
            ax.plot(xfit, yfit, color=class_colors[cls], lw=1.5, zorder=4)

            ax.set_xlim(0, ax_max)
            ax.set_ylim(0, ax_max)
            ax.annotate(
                f"\u03b2={res.slope:.3f} R\u00b2={res.rvalue**2:.3f}\n"
                f"hits={n_hits:,}  ACRs={n_acrs:,}",
                xy=(0.05, 0.95), xycoords="axes fraction",
                va="top", fontsize=6, family="monospace")

            if show_col_titles:
                ax.set_title(f"{class_labels[cls]}")

        # ANCOVA interaction test across classes
        all_m = fam_mask & np.isfinite(leaf_fp) & np.isfinite(proto_fp)
        ancova_p = _ancova_interaction_p(
            leaf_fp[all_m], proto_fp[all_m],
            acr_classes[all_m], class_order)
        return ancova_p

    # -- Multi-page PDF --
    pdf_path = os.path.join(outdir, "fig_e_family_scatter.pdf")
    with PdfPages(pdf_path) as pdf:
        for page in range(n_pages):
            start = page * fams_per_page
            end = min(start + fams_per_page, n_fams)
            page_fams = fam_order[start:end]
            n_rows = len(page_fams)

            fig, axes = plt.subplots(n_rows, 3,
                                      figsize=(9, 2.8 * n_rows),
                                      squeeze=False)

            for ri, fam in enumerate(page_fams):
                fam_mask = families == fam
                ancova_p = _draw_fam_row(
                    axes[ri], fam, fam_mask,
                    show_col_titles=(ri == 0))
                label = rename_family(fam)
                if np.isfinite(ancova_p):
                    label += f"  (ANCOVA p={ancova_p:.2e})"
                axes[ri, 0].set_ylabel(label, fontsize=7)

            fig.suptitle(f"Family-level FP at ~{actual_scale:.0f} bp "
                         f"(corrected, page {page + 1}/{n_pages})",
                         fontsize=11)
            fig.supxlabel("Leaf FP (corrected)", fontsize=9)
            fig.supylabel("Proto FP (corrected)", fontsize=9)
            plt.tight_layout()
            pdf.savefig(fig, dpi=200)
            plt.close(fig)

    # -- Individual PNGs per family --
    ind_dir = os.path.join(outdir, "fig_e_family_scatter")
    os.makedirs(ind_dir, exist_ok=True)
    for fam in fam_order:
        fam_mask = families == fam
        fig, axes = plt.subplots(1, 3, figsize=(9, 2.8), squeeze=False)
        ancova_p = _draw_fam_row(axes[0], fam, fam_mask)
        axes[0, 0].set_ylabel("Proto FP (corrected)", fontsize=8)
        display = rename_family(fam)
        ancova_str = (f"  |  ANCOVA interaction p={ancova_p:.2e}"
                      if np.isfinite(ancova_p) else "")
        fig.suptitle(f"{display} \u2014 Family FP at "
                     f"~{actual_scale:.0f} bp{ancova_str}",
                     fontsize=9)
        fig.supxlabel("Leaf FP (corrected)", fontsize=9)
        plt.tight_layout()
        safe_name = fam.replace("/", "_").replace(" ", "_")
        nature_savefig(fig, safe_name, ind_dir)
        plt.close(fig)
    print(f"  Saved Figure E: {n_pages}-page PDF + "
          f"{n_fams} individual PNGs in fig_e_family_scatter/", flush=True)


# ── Figure F: Family-level beta heatmap ─────────────────────────────────────

def figure_f_family_heatmap(fam_delta_resid, family_ids, acr_ids,
                             acr_class_map, scales, outdir):
    """Stacked heatmap: 3 ACR-class blocks × scales × all families."""
    nature_figure_defaults()

    acr_classes = np.array([acr_class_map.get(a, "unknown")
                             for a in acr_ids])
    label_map = {f: rename_family(f) for f in family_ids}
    class_order = ["proto_gain", "stable", "leaf_gain"]
    n_fam = len(family_ids)
    n_scales = len(scales)

    # For the beta heatmap we need per-ACR leaf/proto FP, but the family NPZ
    # only stores delta. Use delta to compute a pseudo-slope:
    # At each scale, regress delta across ACRs within each class → the slope
    # captures how delta varies with scale. Instead, use the mean delta as
    # a proxy displayed in a diverging heatmap centered at 0.
    beta_blocks = []
    for cls in class_order:
        cls_mask = acr_classes == cls
        if cls_mask.sum() == 0:
            beta_blocks.append(np.full((n_fam, n_scales), np.nan))
            continue
        # Mean residualized delta per family × scale for this ACR class
        block = np.nanmean(fam_delta_resid[cls_mask], axis=0)  # (n_fam, n_scales)
        beta_blocks.append(block)

    # Cluster families (mean across scales → (3*n_fam,) → (n_fam, 3))
    mean_per_ent = np.nanmean(np.vstack(beta_blocks), axis=1)
    feat = np.nan_to_num(mean_per_ent.reshape(3, n_fam).T, nan=0)
    if n_fam > 2:
        dist = pdist(feat, metric="euclidean")
        Z = linkage(dist, method="ward")
        try:
            Z = optimal_leaf_ordering(Z, dist)
        except Exception:
            pass
        order = leaves_list(Z)
    else:
        order = np.arange(n_fam)

    fam_labels = [label_map.get(family_ids[i], family_ids[i])
                   for i in order]

    # Shared color normalization (centered at 0 — mean deltas, not slopes)
    all_vals = np.vstack(beta_blocks)
    finite_vals = all_vals[np.isfinite(all_vals)]
    if len(finite_vals) == 0:
        return
    vmax = np.nanpercentile(np.abs(finite_vals), 95)
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)

    class_labels_disp = {"proto_gain": "Proto-gain", "stable": "Stable",
                         "leaf_gain": "Leaf-gain"}

    # 3 subplots — one per ACR class (independent objects in PDF)
    panel_h = max(0.3 * n_fam, 2)
    fig, axes = plt.subplots(
        3, 1, figsize=(10, panel_h * 3 + 1.5),
        gridspec_kw={"height_ratios": [n_fam] * 3, "hspace": 0.35},
        sharex=True)

    tick_positions = np.linspace(0, n_scales - 1, 6).astype(int)
    im = None

    for bi, cls in enumerate(class_order):
        ax = axes[bi]
        block = beta_blocks[bi][order]
        im = ax.imshow(block, aspect="auto", cmap="PRGn", norm=norm,
                       interpolation="nearest")

        ax.set_yticks(range(n_fam))
        ax.set_yticklabels(fam_labels, fontsize=5)
        ax.set_title(class_labels_disp.get(cls, cls), fontsize=9,
                     fontweight="bold")

        if bi < 2:
            ax.tick_params(axis="x", labelbottom=False)

    axes[-1].set_xticks(tick_positions)
    axes[-1].set_xticklabels([f"{scales[i]:.0f}" for i in tick_positions])
    axes[-1].set_xlabel("Scale (bp)")

    fig.colorbar(im, ax=axes.tolist(), label="Mean residualized delta",
                 shrink=0.6, pad=0.02)
    fig.suptitle("Family-level scale-resolved delta by ACR class "
                 "(confounders removed)", fontsize=11)
    plt.tight_layout()
    nature_savefig(fig, "fig_f_family_heatmap", outdir)
    plt.close(fig)
    print("  Saved Figure F", flush=True)


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="v3 Step 07: Top-signature introduction")
    p.add_argument("--genome-pkl", default="3_PRINT_bulk/At_genome_OBJ")
    p.add_argument("--print-dir", default="3_PRINT_per_rep")
    p.add_argument("--chunks-dir", default="data/v3_chunks")
    p.add_argument("--perscale-chunks-dir",
                   default="results/v3_06_perscale_fp/chunks")
    p.add_argument("--sig-metadata",
                   default="data/motif_signatures/signature_metadata.tsv")
    p.add_argument("--acr-metadata", default="data/acr_metadata.tsv.gz")
    p.add_argument("--acr-coord-mapping",
                   default="data/acr_native_to_resized.tsv")
    p.add_argument("--outdir", default="results/v3_07_top_signatures")
    p.add_argument("--top-n", type=int, default=20)
    p.add_argument("--target-scale", type=float, default=10.0)
    p.add_argument("--min-hits-per-cell", type=int, default=20)
    p.add_argument("--force-extract", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    outdir = os.path.join(BASE, args.outdir)
    os.makedirs(outdir, exist_ok=True)

    print("=" * 60, flush=True)
    print("v3_07 — Top-signature introduction", flush=True)
    print("=" * 60, flush=True)

    # Load metadata
    sig_meta = load_signature_metadata(
        os.path.join(BASE, args.sig_metadata))
    acr_meta = load_acr_metadata(
        os.path.join(BASE, args.acr_metadata))

    # Build ACR class map (resized_str → class)
    coord_map = pd.read_csv(
        os.path.join(BASE, args.acr_coord_mapping), sep="\t")
    coord_map["native_str"] = coord_map["native_str"].str.lower()
    coord_map["resized_str"] = coord_map["resized_str"].str.lower()
    acr_meta["region_str_lower"] = acr_meta["region_str"].str.lower()
    merged = coord_map.merge(
        acr_meta[["region_str_lower", "acr_class"]],
        left_on="native_str", right_on="region_str_lower", how="left")
    acr_class_map = dict(zip(merged["resized_str"], merged["acr_class"]))

    # Also set acr_meta index to resized_str for residualization
    acr_meta_resized = acr_meta.copy()
    acr_meta_resized = acr_meta_resized.merge(
        coord_map[["native_str", "resized_str"]],
        left_on="region_str_lower", right_on="native_str", how="left")
    acr_meta_resized = acr_meta_resized.set_index("resized_str")

    # Load all hits from chunks
    import gzip
    all_hits = []
    for i in range(50):
        hp = os.path.join(BASE, args.chunks_dir, f"chunk_{i:02d}",
                           "motif_hits.tsv.gz")
        if os.path.exists(hp):
            with gzip.open(hp, "rt") as f:
                df = pd.read_csv(f, sep="\t",
                                 usecols=["region_str", "motif_id",
                                           "hit_center"])
            all_hits.append(df)
    hits = pd.concat(all_hits, ignore_index=True)
    print(f"[INFO] Total hits: {len(hits):,}", flush=True)

    # Load v3_06 NPZs
    sig_npz_path = os.path.join(BASE, "results/v3_06_perscale_fp",
                                 "delta_acr_signature_scale.npz")
    fam_npz_path = os.path.join(BASE, "results/v3_06_perscale_fp",
                                 "delta_acr_family_scale.npz")
    have_npz = os.path.exists(sig_npz_path) and os.path.exists(fam_npz_path)

    if have_npz:
        sig_data = np.load(sig_npz_path, allow_pickle=True)
        fam_data = np.load(fam_npz_path, allow_pickle=True)
        sig_delta = sig_data["delta"]
        sig_ids = sig_data["signature_ids"]
        fam_delta = fam_data["delta"]
        family_ids = fam_data["family_ids"]
        acr_ids_sig = sig_data["acr_ids"]
        acr_ids_fam = fam_data["acr_ids"]
        scales_npz = sig_data["scales"]

        # ── Residualize delta matrices ──────────────────────────────────
        print("[INFO] Residualizing signature delta matrix...", flush=True)
        sig_delta_resid, acr_ids_sig_r = residualize_delta_3d(
            sig_delta, acr_ids_sig, sig_ids, acr_meta_resized)
        print(f"  {sig_delta_resid.shape[0]} ACRs after residualization",
              flush=True)

        print("[INFO] Residualizing family delta matrix...", flush=True)
        fam_delta_resid, acr_ids_fam_r = residualize_delta_3d(
            fam_delta, acr_ids_fam, family_ids, acr_meta_resized)
        print(f"  {fam_delta_resid.shape[0]} ACRs after residualization",
              flush=True)

        # Top N signatures by |residualized delta|
        mean_abs = np.nanmean(np.abs(sig_delta_resid), axis=(0, 2))
        top_idx = np.argsort(mean_abs)[::-1][:args.top_n]
        top_sigs = list(sig_ids[top_idx])
        dn = dict(zip(sig_meta["signature_id"], sig_meta["display_name"]))
        print(f"[INFO] Top {args.top_n} signatures by |resid delta|:",
              flush=True)
        for s in top_sigs:
            si = list(sig_ids).index(s)
            print(f"  {s}: {dn.get(s, s)} "
                  f"(|δ_resid|={mean_abs[si]:.4f})", flush=True)
    else:
        top_sigs = list(hits["motif_id"].value_counts()
                        .head(args.top_n).index)
        print(f"[INFO] Using top {args.top_n} by hit count "
              f"(no v3_06 NPZ)", flush=True)

    # Filter hits to top signatures + extract per-scale FP
    sig_hits = hits[hits["motif_id"].isin(top_sigs)].copy()
    print(f"[INFO] Hits for top sigs: {len(sig_hits):,}", flush=True)
    data = load_or_extract_perscale(args, sig_hits, outdir)

    # ── Figures ──────────────────────────────────────────────────────────
    print("\n[Figures]", flush=True)

    # Fig A: Signature-level scatter (corrected, multi-page)
    figure_a_scatter(
        data["fp_values"], data["scales"], data["region_strs"],
        data["motif_ids"], acr_class_map, acr_meta_resized, sig_meta,
        args.target_scale, top_sigs, outdir)

    # Fig B: Signature-level beta heatmap
    figure_b_heatmap(
        data["fp_values"], data["scales"], data["region_strs"],
        data["motif_ids"], acr_class_map, acr_meta_resized, sig_meta,
        top_sigs, outdir)

    if have_npz:
        # Fig C: Ranked bar (residualized delta)
        figure_c_ranked_bar(sig_delta_resid, sig_ids, sig_meta, outdir,
                             top_n=args.top_n)

        # Fig D: Family delta boxplot stratified by ACR class
        figure_d_family_delta(fam_delta_resid, family_ids, acr_ids_fam_r,
                               acr_meta_resized, outdir)

        # Fig E: Family-level per-hit hexbin scatter
        perscale_chunks = os.path.join(BASE, args.perscale_chunks_dir)
        if os.path.isdir(perscale_chunks):
            print("[INFO] Loading per-hit FP from v3_06 chunks...",
                  flush=True)
            perhit_data = load_perhit_at_target_scale(
                perscale_chunks, sig_meta, args.target_scale)
            if perhit_data is not None:
                figure_e_family_scatter(perhit_data, acr_class_map,
                                         acr_meta_resized, outdir)
                del perhit_data; gc.collect()
            else:
                print("[INFO] No v3_06 chunk NPZs — skipping Fig E",
                      flush=True)
        else:
            print(f"[INFO] Chunks dir not found: {perscale_chunks} "
                  f"— skipping Fig E", flush=True)

        # Fig F: Family-level beta heatmap (residualized delta)
        figure_f_family_heatmap(fam_delta_resid, family_ids,
                                 acr_ids_fam_r, acr_class_map,
                                 scales_npz, outdir)
    else:
        print("[INFO] Skipping Figs C-F — v3_06 NPZs not found",
              flush=True)

    print("\n[DONE]", flush=True)


if __name__ == "__main__":
    main()
