#!/usr/bin/env python3
"""
v3 Step 09 Visualization: SHAP interaction figures.

Loads raw interaction tensors from v3_09_shap_interactions, computes bootstrap
CIs, and produces 6 publication-quality figures per pass (all + changing).

Figures:
  A - Hexbin: main effect vs total SHAP, 3 ACR-class panels
  B - Class-stratified signed interaction heatmaps (3 class + delta panel)
  C - Top interaction pairs per class (bar + bootstrap std error bars)
  D - Performance comparison: T1/T2 (v3_08) vs v3_09 regression/classification
  E - Confusion matrices side-by-side: T1 (v3_08) vs v3_09 classification
  F - Scale hexbin: all 7381 pairs density + significant pairs overlay

Output: results/v3_09_shap_interactions/{pass}/fig_{a-f}_*.{pdf,png,svg}
        results/v3_09_shap_interactions/{pass}/significant_pairs_{pass}.tsv
        results/v3_09_shap_interactions/{pass}/bootstrap_ci_{pass}.npz
"""
from __future__ import annotations

import argparse
import os
import warnings

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm, LogNorm
from scipy.cluster.hierarchy import leaves_list, linkage, optimal_leaf_ordering
from scipy.spatial.distance import squareform

from _utils import (
    ACR_CLASS_COLORS,
    PALETTE,
    load_acr_metadata,
    nature_figure_defaults,
    nature_savefig,
)

warnings.filterwarnings("ignore", category=FutureWarning)

BASE = os.path.dirname(os.path.abspath(__file__))

ACR_CLASSES = ["proto_gain", "stable", "leaf_gain"]
ACR_CLASS_LABELS = {"proto_gain": "Proto-gain", "stable": "Stable", "leaf_gain": "Leaf-gain"}


# ── Bootstrap CI ──────────────────────────────────────────────────────────────

def bootstrap_ci(tensor, mask=None, B=500, alpha=0.05, seed=42):
    """
    Bootstrap CI for mean SHAP interaction matrix.

    Parameters
    ----------
    tensor : (n_test, n_feat, n_feat) float32
    mask   : boolean array length n_test (None = use all)
    B      : number of bootstrap resamples
    alpha  : CI level (0.05 → 95% CI)

    Returns
    -------
    lo, hi : (n_feat, n_feat) arrays (percentile bounds)
    boot_mean : (n_feat, n_feat) bootstrap mean
    """
    rng = np.random.RandomState(seed)
    sub = tensor[mask] if mask is not None else tensor
    n = sub.shape[0]
    boot_means = np.empty((B, sub.shape[1], sub.shape[2]), dtype=np.float32)
    for b in range(B):
        idx = rng.choice(n, n, replace=True)
        boot_means[b] = sub[idx].mean(axis=0)
    lo = np.percentile(boot_means, 100 * alpha / 2, axis=0)
    hi = np.percentile(boot_means, 100 * (1 - alpha / 2), axis=0)
    return lo, hi, boot_means.mean(axis=0)


def load_or_compute_bootstrap(tensor, acr_class_arr, pass_outdir, pass_label,
                               B=500, alpha=0.05, seed=42, force=False):
    """Load cached bootstrap CI or compute fresh."""
    cache = os.path.join(pass_outdir, f"bootstrap_ci_{pass_label}.npz")
    if os.path.exists(cache) and not force:
        print(f"  Loading bootstrap CI from cache...", flush=True)
        d = np.load(cache, allow_pickle=True)
        return d["lo"], d["hi"], d["boot_mean"], {
            cls: {"lo": d[f"lo_{cls}"], "hi": d[f"hi_{cls}"],
                  "boot_mean": d[f"boot_mean_{cls}"]}
            for cls in ACR_CLASSES if f"lo_{cls}" in d
        }

    print(f"  Computing bootstrap CI (B={B})...", flush=True)
    lo, hi, boot_mean = bootstrap_ci(tensor, B=B, alpha=alpha, seed=seed)

    class_ci = {}
    save_kwargs = {"lo": lo, "hi": hi, "boot_mean": boot_mean}
    for cls in ACR_CLASSES:
        mask = acr_class_arr == cls
        if mask.sum() < 10:
            continue
        lo_c, hi_c, bm_c = bootstrap_ci(tensor, mask=mask, B=B,
                                          alpha=alpha, seed=seed)
        class_ci[cls] = {"lo": lo_c, "hi": hi_c, "boot_mean": bm_c}
        save_kwargs[f"lo_{cls}"] = lo_c
        save_kwargs[f"hi_{cls}"] = hi_c
        save_kwargs[f"boot_mean_{cls}"] = bm_c

    np.savez_compressed(cache, **save_kwargs)
    return lo, hi, boot_mean, class_ci


# ── Ward clustering helper ─────────────────────────────────────────────────────

def ward_order(matrix):
    """Return leaf ordering for Ward clustering of a symmetric matrix."""
    # Use 1 - |corr| as distance, falling back to raw distance if needed
    try:
        flat = matrix[np.triu_indices_from(matrix, k=1)]
        if np.any(np.isnan(flat)):
            raise ValueError
        dist = squareform(np.abs(matrix - matrix.T).clip(0))
        np.fill_diagonal(dist, 0)
        Z = linkage(squareform(dist, checks=False), method="ward")
        order = leaves_list(optimal_leaf_ordering(Z, squareform(dist, checks=False)))
    except Exception:
        order = np.arange(matrix.shape[0])
    return order


# ── Family color map ──────────────────────────────────────────────────────────

def make_family_colors(feature_names, sig_meta):
    """Map sig_ids → family → color. Returns (family_per_feat, color_per_feat, legend_handles)."""
    fam_map = dict(zip(sig_meta["signature_id"], sig_meta["primary_family"]))
    families = [fam_map.get(f, "Unknown") for f in feature_names]
    unique_fams = sorted(set(families))
    cmap = plt.get_cmap("tab20", len(unique_fams))
    fam_colors = {f: cmap(i) for i, f in enumerate(unique_fams)}
    colors = [fam_colors[f] for f in families]
    from matplotlib.patches import Patch
    handles = [Patch(color=fam_colors[f], label=f) for f in unique_fams]
    return families, colors, handles


# ── Figure A: Main effect vs total SHAP hexbin ────────────────────────────────

def fig_a(tensor, acr_class_arr, feat_names, pass_label, pass_outdir):
    """Hexbin: main effect vs total SHAP, 3 ACR-class panels."""
    nature_figure_defaults()
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)

    # Per-sample per-feature: main effect = diagonal, total = row sum
    main_all = tensor.diagonal(axis1=1, axis2=2)   # (n_test, n_feat)
    total_all = tensor.sum(axis=2)                  # (n_test, n_feat)

    global_max = np.nanpercentile(np.abs(total_all), 99)

    for ax, cls in zip(axes, ACR_CLASSES):
        mask = acr_class_arr == cls
        n_acr = mask.sum()
        if n_acr == 0:
            ax.set_visible(False)
            continue
        main_flat = main_all[mask].ravel()
        total_flat = total_all[mask].ravel()

        # Remove extreme outliers for display
        keep = (np.abs(main_flat) < global_max) & (np.abs(total_flat) < global_max * 1.5)
        hb = ax.hexbin(main_flat[keep], total_flat[keep],
                        gridsize=60, cmap="YlOrRd", mincnt=1,
                        bins="log", linewidths=0.1)
        lim = global_max * 0.8
        ax.axline((0, 0), slope=1, color="k", lw=0.8, ls="--", alpha=0.6,
                   label="y = x (no interaction)")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim * 1.2, lim * 1.2)
        ax.set_xlabel("Main effect SHAP (diagonal)", fontsize=8)
        ax.set_ylabel("Total SHAP (main + interactions)", fontsize=8)
        color = ACR_CLASS_COLORS.get(cls, "grey")
        ax.set_title(f"{ACR_CLASS_LABELS[cls]}\n(n={n_acr} ACRs × {len(feat_names)} sigs)",
                      fontsize=9, color=color, fontweight="bold")
        plt.colorbar(hb, ax=ax, label="log₁₀(count)", pad=0.01)

    fig.suptitle(f"Main effect vs total SHAP — {pass_label}", fontsize=10)
    nature_savefig(fig, f"fig_a_main_vs_total_{pass_label}", pass_outdir)
    plt.close(fig)
    print(f"  Saved Fig A", flush=True)


# ── Figure B: Class-stratified interaction heatmaps ───────────────────────────

def fig_b(tensor, acr_class_arr, feat_names, sig_meta, lo, hi,
           pass_label, pass_outdir):
    """4-panel signed interaction heatmaps: 3 classes + proto−leaf delta."""
    nature_figure_defaults()

    # Compute class-conditional mean signed interaction (off-diagonal focus)
    class_means = {}
    for cls in ACR_CLASSES:
        mask = acr_class_arr == cls
        if mask.sum() < 5:
            class_means[cls] = np.zeros((tensor.shape[1], tensor.shape[2]))
        else:
            class_means[cls] = tensor[mask].mean(axis=0)

    delta = class_means["proto_gain"] - class_means["leaf_gain"]

    # Shared Ward ordering from overall mean
    overall = tensor.mean(axis=0)
    np.fill_diagonal(overall, 0)   # ignore diagonal for clustering
    order = ward_order(overall)

    # Significance mask (overall CI, off-diagonal)
    sig_mask = np.zeros(lo.shape, dtype=bool)
    sig_mask[(lo > 0) | (hi < 0)] = True
    np.fill_diagonal(sig_mask, False)

    panels = [
        (class_means["proto_gain"], "Proto-gain", ACR_CLASS_COLORS.get("proto_gain", "#D64045")),
        (class_means["stable"],     "Stable",     ACR_CLASS_COLORS.get("stable", "#888888")),
        (class_means["leaf_gain"],  "Leaf-gain",  ACR_CLASS_COLORS.get("leaf_gain", "#3A7D44")),
        (delta,                     "Proto − Leaf delta", "black"),
    ]

    # Global color limit from 95th percentile of off-diagonal values
    off_diag_vals = []
    for mat, _, _ in panels:
        m = mat.copy(); np.fill_diagonal(m, 0)
        off_diag_vals.append(np.abs(m).ravel())
    clim = np.percentile(np.concatenate(off_diag_vals), 95)
    clim = max(clim, 1e-6)

    # Family annotation colors
    families, fam_colors, fam_handles = make_family_colors(
        [feat_names[i] for i in order], sig_meta)

    fig, axes = plt.subplots(1, 4, figsize=(18, 6), constrained_layout=True)
    norm = TwoSlopeNorm(vmin=-clim, vcenter=0, vmax=clim)

    for ax, (mat, title, title_color) in zip(axes, panels):
        mat_ord = mat[np.ix_(order, order)].copy()
        np.fill_diagonal(mat_ord, np.nan)  # mask diagonal (shown in Fig A)

        im = ax.imshow(mat_ord, cmap="RdBu_r", norm=norm, aspect="auto",
                        interpolation="none")

        # Hatch significant pairs
        sig_ord = sig_mask[np.ix_(order, order)]
        for (r, c) in zip(*np.where(sig_ord)):
            ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1,
                                        fill=False, hatch="///",
                                        edgecolor="k", linewidth=0.2))

        n = len(order)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, fontsize=9, color=title_color, fontweight="bold")

        # Family color sidebar (left)
        sidebar = np.array([[mcolors.to_rgba(c)] for c in fam_colors])
        ax_sb = ax.inset_axes([-0.04, 0, 0.03, 1])
        ax_sb.imshow(sidebar, aspect="auto", interpolation="none")
        ax_sb.set_xticks([]); ax_sb.set_yticks([])

        plt.colorbar(im, ax=ax, label="Mean signed SHAP interaction",
                      shrink=0.8, pad=0.01)

    # Family legend in last axis
    axes[-1].legend(handles=fam_handles, loc="upper left", fontsize=5,
                     title="Family", title_fontsize=6,
                     bbox_to_anchor=(1.18, 1), frameon=False)

    fig.suptitle(f"Class-stratified SHAP interactions — {pass_label}\n"
                  f"(hatching = significant at 95% bootstrap CI)", fontsize=10)
    nature_savefig(fig, f"fig_b_class_heatmaps_{pass_label}", pass_outdir)
    plt.close(fig)
    print(f"  Saved Fig B", flush=True)


# ── Figure C: Top pairs per class ─────────────────────────────────────────────

def fig_c(tensor, acr_class_arr, feat_names, sig_meta, class_ci,
           pass_label, pass_outdir, top_n=20):
    """Bar plot of top interaction pairs per ACR class with bootstrap std."""
    nature_figure_defaults()
    fig, axes = plt.subplots(1, 3, figsize=(16, 6), constrained_layout=True)

    dn = dict(zip(sig_meta["signature_id"], sig_meta["display_name"]))
    fam_map = dict(zip(sig_meta["signature_id"], sig_meta["primary_family"]))

    for ax, cls in zip(axes, ACR_CLASSES):
        mask = acr_class_arr == cls
        n_acr = mask.sum()
        if n_acr < 5 or cls not in class_ci:
            ax.set_visible(False)
            continue

        # Mean abs interaction for this class
        sub = tensor[mask]
        mean_abs = np.mean(np.abs(sub), axis=0)
        boot_mean = class_ci[cls]["boot_mean"]

        n_feat = len(feat_names)
        rows = []
        for i in range(n_feat):
            for j in range(i + 1, n_feat):
                fam_i = fam_map.get(feat_names[i], "")
                fam_j = fam_map.get(feat_names[j], "")
                # Bootstrap std from per-class boot_mean (rough estimate)
                rows.append({
                    "label": f"{dn.get(feat_names[i], feat_names[i])}\n× "
                              f"{dn.get(feat_names[j], feat_names[j])}",
                    "mean_abs": mean_abs[i, j],
                    "same_family": fam_i == fam_j,
                    "signed": np.mean(sub[:, i, j]),
                })

        df = (pd.DataFrame(rows)
              .sort_values("mean_abs", ascending=False)
              .head(top_n)
              .reset_index(drop=True))

        colors = [ACR_CLASS_COLORS.get(cls, "grey") if not sf
                   else "lightgrey"
                   for sf in df["same_family"]]
        edgecolors = ["k" if sf else "none" for sf in df["same_family"]]

        ax.barh(range(len(df)), df["mean_abs"].values,
                 color=colors, edgecolor=edgecolors, linewidth=0.5)
        ax.set_yticks(range(len(df)))
        ax.set_yticklabels(df["label"].values, fontsize=5)
        ax.invert_yaxis()
        ax.set_xlabel("Mean |SHAP interaction|", fontsize=8)
        color = ACR_CLASS_COLORS.get(cls, "grey")
        ax.set_title(f"{ACR_CLASS_LABELS[cls]}\n(n={n_acr} ACRs)",
                      fontsize=9, color=color, fontweight="bold")

        # Legend for same vs cross family
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(color="lightgrey", edgecolor="k", label="Same family"),
                             Patch(color=color, label="Cross-family")],
                   fontsize=6, loc="lower right")

    fig.suptitle(f"Top {top_n} interaction pairs per class — {pass_label}", fontsize=10)
    nature_savefig(fig, f"fig_c_top_pairs_{pass_label}", pass_outdir)
    plt.close(fig)
    print(f"  Saved Fig C", flush=True)


# ── Figure D: Performance comparison ──────────────────────────────────────────

def fig_d(gb_dir, v09_outdir, pass_label, pass_outdir):
    """Grouped bar: R² and BA across T1/T2 (v3_08) vs v3_09 reg/clf."""
    nature_figure_defaults()

    # Load v3_08 summary
    summary_path = os.path.join(BASE, gb_dir, "model_summary.tsv")
    if not os.path.exists(summary_path):
        print(f"  Fig D: model_summary.tsv not found — skipping", flush=True)
        return
    v08 = pd.read_csv(summary_path, sep="\t")
    v08_pass = v08[v08["pass"] == pass_label]

    # Load v3_09 results
    reg_path = os.path.join(BASE, v09_outdir, pass_label, "regression_results.tsv")
    clf_path = os.path.join(BASE, v09_outdir, pass_label, "clf_results.tsv")
    if not os.path.exists(reg_path) or not os.path.exists(clf_path):
        print(f"  Fig D: v3_09 results not found — skipping", flush=True)
        return

    reg = pd.read_csv(reg_path, sep="\t")
    clf = pd.read_csv(clf_path, sep="\t")

    labels = ["T1\n(v3_08)", "T2\n(v3_08)", "v3_09\nregression", "v3_09\nclassification"]
    r2_vals = [
        v08_pass[v08_pass["tier"] == "T1"]["r2"].values[0] if not v08_pass[v08_pass["tier"] == "T1"].empty else np.nan,
        v08_pass[v08_pass["tier"] == "T2"]["r2"].values[0] if not v08_pass[v08_pass["tier"] == "T2"].empty else np.nan,
        reg["r2"].values[0],
        np.nan,
    ]
    ba_vals = [
        v08_pass[v08_pass["tier"] == "T1"]["ba"].values[0] if not v08_pass[v08_pass["tier"] == "T1"].empty else np.nan,
        v08_pass[v08_pass["tier"] == "T2"]["ba"].values[0] if not v08_pass[v08_pass["tier"] == "T2"].empty else np.nan,
        np.nan,
        clf["ba"].values[0],
    ]
    bar_colors = ["#aaaaaa", "#cccccc", PALETTE.get("leaf", "#3A7D44"),
                   PALETTE.get("proto", "#D64045")]

    fig, axes = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)
    x = np.arange(len(labels))

    for ax, vals, ylabel, title in [
        (axes[0], r2_vals, "R² (test set)", "Regression performance"),
        (axes[1], ba_vals, "Balanced accuracy (test set)", "Classification performance"),
    ]:
        bars = ax.bar(x, vals, color=bar_colors, edgecolor="k", linewidth=0.5,
                       width=0.6)
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01,
                         f"{v:.3f}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=9)
        ax.set_ylim(0, 1.05)

    fig.suptitle(f"Model performance comparison — {pass_label}", fontsize=10)
    nature_savefig(fig, f"fig_d_performance_{pass_label}", pass_outdir)
    plt.close(fig)
    print(f"  Saved Fig D", flush=True)


# ── Figure E: Confusion matrices ──────────────────────────────────────────────

def fig_e(gb_dir, v09_outdir, pass_label, pass_outdir):
    """Side-by-side confusion matrices: T1 (v3_08) vs v3_09 classification."""
    nature_figure_defaults()

    t1_path = os.path.join(BASE, gb_dir, pass_label, "confusion_matrix_tier1.npz")
    clf_path = os.path.join(BASE, v09_outdir, pass_label, "confusion_matrix_clf.npz")
    if not os.path.exists(t1_path) or not os.path.exists(clf_path):
        print(f"  Fig E: confusion matrix files not found — skipping", flush=True)
        return

    d_t1 = np.load(t1_path, allow_pickle=True)
    d_clf = np.load(clf_path, allow_pickle=True)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)
    titles = ["T1 — v3_08 (4,851 features)", "v3_09 (122 features)"]

    for ax, d, title in zip(axes, [d_t1, d_clf], titles):
        cm = d["cm"]
        classes = list(d["classes"])
        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(classes)))
        ax.set_yticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=30, ha="right", fontsize=8)
        ax.set_yticklabels(classes, fontsize=8)
        ax.set_xlabel("Predicted", fontsize=8)
        ax.set_ylabel("True", fontsize=8)
        ax.set_title(title, fontsize=9)
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j, i, f"{cm[i,j]:.2f}", ha="center", va="center",
                         fontsize=8, color="white" if cm[i, j] > 0.6 else "black")
        plt.colorbar(im, ax=ax, label="Fraction (row-normalized)", shrink=0.8)

    fig.suptitle(f"Confusion matrices — {pass_label}", fontsize=10)
    nature_savefig(fig, f"fig_e_confusion_{pass_label}", pass_outdir)
    plt.close(fig)
    print(f"  Saved Fig E", flush=True)


# ── Figure F: Scale hexbin for ALL pairs ──────────────────────────────────────

def fig_f(feat_names, sig_meta, v09_outdir, lo, hi, tensor, pass_label, pass_outdir):
    """
    Scale hexbin: background = all 7381 pair density; overlay = significant pairs
    colored by signed interaction strength.
    """
    nature_figure_defaults()

    bp_path = os.path.join(BASE, v09_outdir, pass_label, "best_scale_bp.tsv")
    if not os.path.exists(bp_path):
        print(f"  Fig F: best_scale_bp.tsv not found — skipping", flush=True)
        return

    bp_df = pd.read_csv(bp_path, sep="\t").set_index("signature_id")
    fam_map = dict(zip(sig_meta["signature_id"], sig_meta["primary_family"]))
    dn = dict(zip(sig_meta["signature_id"], sig_meta["display_name"]))

    n_feat = len(feat_names)
    mean_signed = tensor.mean(axis=0)

    # Significance mask (off-diagonal only)
    sig_mask = ((lo > 0) | (hi < 0))
    np.fill_diagonal(sig_mask, False)

    # Build all pairs
    rows = []
    for i in range(n_feat):
        for j in range(i + 1, n_feat):
            si, sj = feat_names[i], feat_names[j]
            bp_i = float(bp_df.loc[si, "scale_bp"]) if si in bp_df.index else np.nan
            bp_j = float(bp_df.loc[sj, "scale_bp"]) if sj in bp_df.index else np.nan
            is_sig = sig_mask[i, j]
            rows.append({
                "sig_i": dn.get(si, si), "sig_j": dn.get(sj, sj),
                "sig_id_i": si, "sig_id_j": sj,
                "scale_i_bp": bp_i, "scale_j_bp": bp_j,
                "mean_signed_interaction": mean_signed[i, j],
                "mean_abs_interaction": np.abs(mean_signed[i, j]),
                "ci_lo": lo[i, j], "ci_hi": hi[i, j],
                "same_family": fam_map.get(si, "") == fam_map.get(sj, ""),
                "significant": is_sig,
            })

    pairs_df = pd.DataFrame(rows)
    pairs_df.to_csv(
        os.path.join(pass_outdir, f"significant_pairs_{pass_label}.tsv"),
        sep="\t", index=False)
    n_sig = pairs_df["significant"].sum()
    print(f"  Significant pairs: {n_sig} / {len(pairs_df)}", flush=True)

    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)

    # Background: hexbin of ALL pairs
    valid = pairs_df.dropna(subset=["scale_i_bp", "scale_j_bp"])
    hb = ax.hexbin(valid["scale_i_bp"], valid["scale_j_bp"],
                    gridsize=20, cmap="Greys", mincnt=1,
                    bins="log", linewidths=0.2, alpha=0.8)
    plt.colorbar(hb, ax=ax, label="log₁₀(pair count)", shrink=0.8)

    # Overlay: significant pairs colored by signed interaction
    sig_df = pairs_df[pairs_df["significant"] & pairs_df[["scale_i_bp", "scale_j_bp"]].notna().all(axis=1)]
    if len(sig_df) > 0:
        vext = np.abs(sig_df["mean_signed_interaction"]).quantile(0.95)
        vext = max(vext, 1e-6)
        norm = TwoSlopeNorm(vmin=-vext, vcenter=0, vmax=vext)
        sc = ax.scatter(sig_df["scale_i_bp"], sig_df["scale_j_bp"],
                         c=sig_df["mean_signed_interaction"],
                         cmap="RdBu_r", norm=norm,
                         s=30, edgecolors="k", linewidths=0.3, zorder=5,
                         label=f"Significant pairs (n={len(sig_df)})")
        plt.colorbar(sc, ax=ax, label="Mean signed SHAP interaction", shrink=0.5)

    # Diagonal y=x
    lim_max = valid[["scale_i_bp", "scale_j_bp"]].max().max() * 1.05
    lim_min = valid[["scale_i_bp", "scale_j_bp"]].min().min() * 0.95
    ax.plot([lim_min, lim_max], [lim_min, lim_max], "k--", lw=0.8, alpha=0.5,
             label="y = x (same scale)")

    # Biological boundary lines
    for bp, label in [(20, "20 bp\n(TF/sub-nuc)"), (80, "80 bp\n(nuc)")]:
        ax.axvline(bp, color="steelblue", ls=":", lw=0.8, alpha=0.7)
        ax.axhline(bp, color="steelblue", ls=":", lw=0.8, alpha=0.7)
        ax.text(bp + 1, lim_max * 0.97, label, fontsize=6, color="steelblue",
                 va="top")

    ax.set_xlabel("Best scale — signature i (bp)", fontsize=9)
    ax.set_ylabel("Best scale — signature j (bp)", fontsize=9)
    ax.set_title(f"Scale distribution of all {len(pairs_df):,} pairs — {pass_label}\n"
                  f"(colored overlay: {n_sig} significant, bootstrap 95% CI excludes 0)",
                  fontsize=9)
    ax.legend(fontsize=7, loc="upper left")

    nature_savefig(fig, f"fig_f_scale_hexbin_{pass_label}", pass_outdir)
    plt.close(fig)
    print(f"  Saved Fig F", flush=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="v3 Step 09 Visualization")
    p.add_argument("--outdir", default="results/v3_09_shap_interactions")
    p.add_argument("--gb-dir", default="results/v3_08_gradient_boosting")
    p.add_argument("--sig-metadata",
                   default="data/motif_signatures/signature_metadata.tsv")
    p.add_argument("--bootstrap-b", type=int, default=500)
    p.add_argument("--bootstrap-alpha", type=float, default=0.05)
    p.add_argument("--top-n-pairs", type=int, default=20)
    p.add_argument("--force-bootstrap", action="store_true",
                   help="Recompute bootstrap CI even if cached")
    p.add_argument("--passes", nargs="+", default=["all", "changing"])
    return p.parse_args()


def main():
    args = parse_args()
    outdir = os.path.join(BASE, args.outdir)
    sig_meta = pd.read_csv(os.path.join(BASE, args.sig_metadata), sep="\t")

    print("=" * 60, flush=True)
    print("v3_09_viz — SHAP Interaction Figures", flush=True)
    print("=" * 60, flush=True)

    for pass_label in args.passes:
        pass_outdir = os.path.join(outdir, pass_label)
        if not os.path.isdir(pass_outdir):
            print(f"\n[{pass_label}] Output directory not found — skipping", flush=True)
            continue

        tensor_path = os.path.join(pass_outdir, "raw_interaction_tensor.npz")
        if not os.path.exists(tensor_path):
            print(f"\n[{pass_label}] raw_interaction_tensor.npz not found — "
                  f"run v3_09_shap_interactions.py first", flush=True)
            continue

        print(f"\n{'─'*40}", flush=True)
        print(f"Pass: {pass_label}", flush=True)

        # Load tensor
        print(f"  Loading raw interaction tensor...", flush=True)
        d = np.load(tensor_path, allow_pickle=True)
        tensor = d["interaction_values"]          # (n_test, n_feat, n_feat)
        acr_class_arr = d["acr_class"].astype(str)
        feat_names = list(d["feature_names"].astype(str))
        print(f"  Tensor shape: {tensor.shape}", flush=True)

        # Bootstrap CI
        lo, hi, boot_mean, class_ci = load_or_compute_bootstrap(
            tensor, acr_class_arr, pass_outdir, pass_label,
            B=args.bootstrap_b, alpha=args.bootstrap_alpha,
            force=args.force_bootstrap)

        # Generate figures
        print(f"  Generating figures...", flush=True)

        fig_a(tensor, acr_class_arr, feat_names, pass_label, pass_outdir)
        fig_b(tensor, acr_class_arr, feat_names, sig_meta, lo, hi,
               pass_label, pass_outdir)
        fig_c(tensor, acr_class_arr, feat_names, sig_meta, class_ci,
               pass_label, pass_outdir, top_n=args.top_n_pairs)
        fig_d(args.gb_dir, args.outdir, pass_label, pass_outdir)
        fig_e(args.gb_dir, args.outdir, pass_label, pass_outdir)
        fig_f(feat_names, sig_meta, args.outdir, lo, hi, tensor,
               pass_label, pass_outdir)

        print(f"  [{pass_label}] All figures saved", flush=True)

    print("\n[DONE]", flush=True)


if __name__ == "__main__":
    main()
