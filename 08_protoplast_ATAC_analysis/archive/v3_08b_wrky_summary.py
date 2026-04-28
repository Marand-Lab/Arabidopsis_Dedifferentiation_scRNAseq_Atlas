#!/usr/bin/env python3
"""v3_08b — WRKY Summary Figure.

Consolidates WRKY-relevant panels from v3_07, v3_08, and v3_09 into
a single multi-page PDF.  Reads cached NPZs/TSVs only — no recomputation.

Pages:
  1. Raw WRKY footprint signal (delta heatmap, family scatter embed)
  2. WRKY in the model (signed SHAP heatmap rows, SHAP profile embed)
  3. WRKY in context (asymmetry, permutation importance)
  4. WRKY interactions (concordant/contrasting families from v3_09)

Usage (local — no cluster needed):
  conda run -n scprinter-local python -u v3_08b_wrky_summary.py
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.image import imread
import seaborn as sns
from scipy import stats

# ── Project paths ────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from _utils import (nature_figure_defaults, nature_savefig,
                     ACR_CLASS_COLORS, PALETTE, residualize_features)

# Import rename helpers from v3_08
from v3_08_gradient_boosting import rename_family, FAMILY_RENAME

WRKY_FAMILY = "WRKY"
CLASS_ORDER = ["proto_gain", "stable", "leaf_gain"]
CLASS_LABELS = {"proto_gain": "Proto-gain", "stable": "Stable",
                "leaf_gain": "Leaf-gain"}


# ── Helpers ──────────────────────────────────────────────────────────────────

def _save_panel(fig_or_func, panel_dir, name):
    """Save a standalone panel as PDF + PNG.

    If fig_or_func is a Figure, save and close it.
    If callable, call it to produce a Figure, save, close.
    """
    if callable(fig_or_func):
        f = fig_or_func()
    else:
        f = fig_or_func
    for ext in ("pdf", "png"):
        path = os.path.join(panel_dir, f"{name}.{ext}")
        f.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(f)


def _embed_png(ax, path, title=None):
    """Embed a PNG into an axes, hiding ticks."""
    if os.path.exists(path):
        img = imread(path)
        ax.imshow(img, aspect="auto")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        if title:
            ax.set_title(title, fontsize=9, fontweight="bold")
        return True
    else:
        ax.text(0.5, 0.5, f"Missing:\n{os.path.basename(path)}",
                ha="center", va="center", fontsize=7, color="gray",
                transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        if title:
            ax.set_title(title, fontsize=9, fontweight="bold")
        return False


def _load_sig_meta(path):
    """Load signature metadata."""
    df = pd.read_csv(path, sep="\t")
    return df


def _family_for_sig(sig_id, sig_meta):
    """Map signature_id → renamed family."""
    fam_map = dict(zip(sig_meta["signature_id"], sig_meta["primary_family"]))
    return rename_family(fam_map.get(sig_id, "Unknown"))


def _load_shap_t1(npz_path, sig_meta):
    """Load T1 raw SHAP, parse features, aggregate to family × ACR.

    Returns: shap_by_family (n_acrs × n_fam), family_names, acr_class
    """
    if not os.path.exists(npz_path):
        return None
    npz = np.load(npz_path, allow_pickle=True)
    shap_vals = npz["shap_values"]
    feat_names = list(npz["feature_names"])
    acr_class = npz["acr_class_test"]

    if shap_vals.ndim == 3:
        shap_vals = shap_vals.mean(axis=2)

    # Parse family_sNN → (family, scale_idx)
    fam_col_map = {}
    for fi, fname in enumerate(feat_names):
        parts = fname.rsplit("_s", 1)
        if len(parts) == 2:
            try:
                fam = rename_family(parts[0])
                fam_col_map.setdefault(fam, []).append(fi)
            except ValueError:
                pass

    families = sorted(fam_col_map.keys())
    # Sum SHAP across scales per family per ACR → (n_acrs, n_fam)
    mat = np.zeros((shap_vals.shape[0], len(families)))
    for j, fam in enumerate(families):
        mat[:, j] = shap_vals[:, fam_col_map[fam]].sum(axis=1)

    return {"shap_by_family": mat, "family_names": families,
            "acr_class": acr_class}


def _wrky_shap_by_scale(npz_path, sig_meta):
    """Extract WRKY-family SHAP across scales from T1 NPZ.

    Returns: dict with shap_by_scale (n_acrs × n_scales), scales, acr_class
    """
    if not os.path.exists(npz_path):
        return None
    npz = np.load(npz_path, allow_pickle=True)
    shap_vals = npz["shap_values"]
    feat_names = list(npz["feature_names"])
    acr_class = npz["acr_class_test"]

    if shap_vals.ndim == 3:
        shap_vals = shap_vals.mean(axis=2)

    # Find WRKY features
    wrky_scales = {}
    for fi, fname in enumerate(feat_names):
        parts = fname.rsplit("_s", 1)
        if len(parts) == 2:
            fam = rename_family(parts[0])
            if fam == WRKY_FAMILY:
                try:
                    sidx = int(parts[1])
                    wrky_scales[sidx] = fi
                except ValueError:
                    pass

    if not wrky_scales:
        return None

    scale_idxs = sorted(wrky_scales.keys())
    mat = np.zeros((shap_vals.shape[0], len(scale_idxs)))
    for j, sidx in enumerate(scale_idxs):
        mat[:, j] = shap_vals[:, wrky_scales[sidx]]

    # Convert scale indices to bp (scale = index + 2)
    scales_bp = [s + 2 for s in scale_idxs]

    return {"shap_by_scale": mat, "scales_bp": scales_bp,
            "acr_class": acr_class}


# ── Page 1 helpers ───────────────────────────────────────────────────────────

def _load_wrky_zhit_from_chunks(chunk_dir, acr_meta_path, coord_map_path,
                                 scale_lo=2, scale_hi=10):
    """Load WRKY per-hit FP from v3_06 chunks, z-score per scale, average.

    For each scale in [scale_lo, scale_hi]:
      1. Collect raw FP across all WRKY hits (leaf reps, proto reps separately)
      2. Average reps → per-hit leaf FP and proto FP at that scale
      3. Z-score each across hits (mean=0, sd=1)
    Then average the z-scored values across scales per hit.

    Returns dict with leaf_z, proto_z, hit_classes arrays (one value per hit).
    """
    import glob as _glob

    wrky_sigs = {"sig_121", "sig_122"}
    chunk_files = sorted(_glob.glob(
        os.path.join(chunk_dir, "per_hit_fp_chunk_*.npz")))
    if not chunk_files:
        return None

    # Load coordinate mapping + ACR metadata
    acr_meta = pd.read_csv(acr_meta_path, sep="\t")
    acr_meta["acr_id"] = acr_meta["acr_id"].astype(str)
    native_class_map = dict(zip(acr_meta["acr_id"], acr_meta["acr_class"]))

    resized_to_native = {}
    if os.path.exists(coord_map_path):
        cm = pd.read_csv(coord_map_path, sep="\t")
        resized_to_native = dict(zip(cm["resized_str"], cm["native_str"]))

    # Pass 1: collect per-hit raw FP at each scale (leaf avg, proto avg)
    # Store as list of (leaf_fp_per_scale, proto_fp_per_scale, acr_class)
    all_leaf = []   # each: (n_scales_sel,)
    all_proto = []
    all_classes = []
    scales_ref = None
    tf_idx = None

    for cf in chunk_files:
        try:
            cnpz = np.load(cf, allow_pickle=True)
        except Exception:
            continue
        mids = cnpz.get("motif_ids")
        regions = cnpz.get("region_strs")
        fp = cnpz.get("fp_values")       # (n_hits, n_scales, n_samples)
        scales_c = cnpz.get("scales")
        if fp is None or scales_c is None or mids is None:
            continue

        if scales_ref is None:
            scales_ref = scales_c
            tf_idx = np.where((scales_c >= scale_lo) &
                              (scales_c <= scale_hi))[0]
            if len(tf_idx) == 0:
                return None

        wrky_mask = np.isin(list(mids), list(wrky_sigs))
        if wrky_mask.sum() == 0:
            continue

        fp_sel = fp[wrky_mask][:, tf_idx, :]   # (n_wrky, n_tf_scales, 4)
        regions_sel = np.array(list(regions))[wrky_mask]

        # Per-hit: average reps → (n_wrky, n_tf_scales)
        leaf_fp = np.nanmean(fp_sel[:, :, :2], axis=2)
        proto_fp = np.nanmean(fp_sel[:, :, 2:], axis=2)

        all_leaf.append(leaf_fp)
        all_proto.append(proto_fp)

        for r in regions_sel:
            native = resized_to_native.get(str(r), str(r))
            all_classes.append(native_class_map.get(native, "stable"))

    if not all_leaf:
        return None

    # Stack all hits: (N_total, n_tf_scales)
    leaf_mat = np.concatenate(all_leaf, axis=0)
    proto_mat = np.concatenate(all_proto, axis=0)
    hit_classes = np.array(all_classes)

    n_hits, n_scales_sel = leaf_mat.shape

    # Z-score per scale across hits, then average across scales
    leaf_z_scales = np.full_like(leaf_mat, np.nan)
    proto_z_scales = np.full_like(proto_mat, np.nan)

    for si in range(n_scales_sel):
        lf_col = leaf_mat[:, si]
        pr_col = proto_mat[:, si]
        # Pool leaf + proto for a common z-score reference per scale
        pooled = np.concatenate([lf_col, pr_col])
        valid = np.isfinite(pooled)
        if valid.sum() < 3:
            continue
        mu = np.nanmean(pooled[valid])
        sd = np.nanstd(pooled[valid])
        if sd < 1e-12:
            continue
        leaf_z_scales[:, si] = (lf_col - mu) / sd
        proto_z_scales[:, si] = (pr_col - mu) / sd

    # Average z-scored FP across scales per hit
    leaf_z = np.nanmean(leaf_z_scales, axis=1)
    proto_z = np.nanmean(proto_z_scales, axis=1)

    # Keep only finite hits
    valid = np.isfinite(leaf_z) & np.isfinite(proto_z)
    scales_bp = scales_ref[tf_idx] if scales_ref is not None else None

    return {"leaf_z": leaf_z[valid], "proto_z": proto_z[valid],
            "hit_classes": hit_classes[valid],
            "n_total": int(valid.sum()),
            "scales_bp": scales_bp}


# ── Page 1: Raw WRKY footprint signal ────────────────────────────────────────

def page1_raw_signal(pdf, args):
    """WRKY z-scored FP scatter + paired violin + delta heatmap."""
    nature_figure_defaults()
    fig = plt.figure(figsize=(16, 19))
    gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.40, wspace=0.3)

    # ── Load z-scored per-hit WRKY FP (scales 2-10 bp) ────────────────────
    chunk_dir = os.path.join(args.v06_dir, "chunks")
    coord_map_path = os.path.join(BASE, "data", "acr_native_to_resized.tsv")
    zhit = _load_wrky_zhit_from_chunks(
        chunk_dir, os.path.join(BASE, args.acr_metadata), coord_map_path,
        scale_lo=2, scale_hi=10)

    # ── Panel A1: Faceted hexbin — leaf_z vs proto_z, one per ACR class ──
    gs_a1 = gs[0, :].subgridspec(1, 3, wspace=0.30)
    if zhit is not None:
        from scipy.stats import spearmanr

        leaf_z = zhit["leaf_z"]
        proto_z = zhit["proto_z"]
        hit_cls = zhit["hit_classes"]

        # Shared axis limits: clip to 1st-99th percentile for tighter view
        all_vals = np.concatenate([leaf_z, proto_z])
        lim = np.percentile(np.abs(all_vals), 99) * 1.15
        gridsize = 40

        for ci, cls in enumerate(CLASS_ORDER):
            ax = fig.add_subplot(gs_a1[ci])
            mask = hit_cls == cls
            n_cls = mask.sum()
            if n_cls < 10:
                ax.text(0.5, 0.5, f"{CLASS_LABELS[cls]}\nn={n_cls}",
                        ha="center", va="center", transform=ax.transAxes)
                continue

            lf = leaf_z[mask]
            pr = proto_z[mask]

            hb = ax.hexbin(lf, pr, gridsize=gridsize, cmap="YlOrRd",
                           mincnt=1, extent=[-lim, lim, -lim, lim],
                           linewidths=0.1, rasterized=True)

            # Diagonal reference
            ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.5, alpha=0.4)

            # OLS regression line
            r, p = spearmanr(lf, pr)
            z_coef = np.polyfit(lf, pr, 1)
            x_line = np.linspace(-lim * 0.9, lim * 0.9, 100)
            ax.plot(x_line, np.polyval(z_coef, x_line), color="#333333",
                    lw=1.5, ls="-")

            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_aspect("equal")
            ax.text(0.03, 0.97,
                    f"r={r:.3f}\np={p:.1e}\nslope={z_coef[0]:.2f}",
                    transform=ax.transAxes, fontsize=6, va="top",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white",
                              alpha=0.8))
            ax.set_title(f"{CLASS_LABELS[cls]} (n={n_cls:,})",
                         fontsize=8, fontweight="bold",
                         color=ACR_CLASS_COLORS[cls])
            ax.set_xlabel("Leaf FP (z, 2-10 bp)", fontsize=7)
            if ci == 0:
                ax.set_ylabel("Proto FP (z, 2-10 bp)", fontsize=7)
            else:
                ax.set_yticklabels([])

        # Colorbar on the last panel
        plt.colorbar(hb, ax=fig.axes[-1], shrink=0.6, label="# hits",
                     pad=0.02)
    else:
        ax_a1 = fig.add_subplot(gs[0, :])
        ax_a1.text(0.5, 0.5, "Missing v3_06 chunk data",
                   ha="center", va="center", transform=ax_a1.transAxes)
    fig.text(0.5, 0.97,
             "A — WRKY leaf vs proto FP (z-scored per scale, averaged 2-10 bp)",
             ha="center", fontsize=9, fontweight="bold")

    # ── Panel A2: Paired violin — leaf_z vs proto_z by ACR class ──────────
    ax_a2 = fig.add_subplot(gs[1, :])
    if zhit is not None:
        from matplotlib.patches import Patch
        from scipy.stats import wilcoxon

        leaf_z = zhit["leaf_z"]
        proto_z = zhit["proto_z"]
        hit_cls = zhit["hit_classes"]

        positions = []
        labels = []
        annot_info = []   # collect stats for annotation after y-lim is set
        pos_idx = 0
        width = 0.35

        for cls in CLASS_ORDER:
            m = hit_cls == cls
            if m.sum() < 10:
                continue

            lf_vals = leaf_z[m]
            pr_vals = proto_z[m]

            # Leaf violin (left)
            parts_l = ax_a2.violinplot(
                [lf_vals], positions=[pos_idx - width / 2],
                widths=width, showextrema=False,
                showmedians=False, showmeans=False)
            for pc in parts_l["bodies"]:
                pc.set_facecolor("#2166AC")
                pc.set_alpha(0.5)

            # Proto violin (right)
            parts_p = ax_a2.violinplot(
                [pr_vals], positions=[pos_idx + width / 2],
                widths=width, showextrema=False,
                showmedians=False, showmeans=False)
            for pc in parts_p["bodies"]:
                pc.set_facecolor("#B2182B")
                pc.set_alpha(0.5)

            # Draw Q25, median, Q75 lines manually for both
            for vals, x_pos in [(lf_vals, pos_idx - width / 2),
                                (pr_vals, pos_idx + width / 2)]:
                q25, med, q75 = np.percentile(vals, [25, 50, 75])
                hw = width * 0.3  # half-width of horizontal lines
                # Median (thick solid)
                ax_a2.hlines(med, x_pos - hw, x_pos + hw,
                             color="black", lw=1.5, zorder=4)
                # Q25 and Q75 (thin dashed)
                ax_a2.hlines([q25, q75], x_pos - hw, x_pos + hw,
                             color="black", lw=0.8, ls="--", zorder=4)
                # Vertical whisker connecting Q25-Q75
                ax_a2.vlines(x_pos, q25, q75,
                             color="black", lw=0.6, zorder=3)

            # Bootstrap CI on mean difference (proto - leaf)
            diff = pr_vals - lf_vals
            mean_diff = np.mean(diff)
            n_boot = 10000
            rng = np.random.default_rng(42)
            boot_means = np.array([
                np.mean(rng.choice(diff, size=len(diff), replace=True))
                for _ in range(n_boot)])
            ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])
            # Cohen's d (paired)
            sd_diff = np.std(diff, ddof=1)
            cohens_d = mean_diff / sd_diff if sd_diff > 1e-12 else 0.0

            annot_info.append((pos_idx, mean_diff, ci_lo, ci_hi, cohens_d))

            positions.append(pos_idx)
            labels.append(f"{CLASS_LABELS[cls]}\n(n={m.sum():,})")
            pos_idx += 1

        # Set y-limits and add bracket annotations above each pair
        all_vals = np.concatenate([leaf_z, proto_z])
        y_lo = np.percentile(all_vals, 0.5)
        y_hi = np.percentile(all_vals, 99.5)
        y_range = y_hi - y_lo
        bracket_y = y_hi + y_range * 0.05

        for (pi, mean_diff, ci_lo, ci_hi, cohens_d) in annot_info:
            x_l = pi - width / 2
            x_r = pi + width / 2
            y_br = bracket_y
            # Bracket
            ax_a2.plot([x_l, x_l, x_r, x_r],
                       [y_br - y_range * 0.01, y_br,
                        y_br, y_br - y_range * 0.01],
                       color="black", lw=0.8)
            # Significance: CI excludes 0?
            sig_marker = "*" if (ci_lo > 0 or ci_hi < 0) else "ns"
            ax_a2.text(pi, y_br + y_range * 0.005,
                       f"\u0394mean={mean_diff:+.3f} "
                       f"[{ci_lo:+.3f}, {ci_hi:+.3f}]\n"
                       f"Cohen's d={cohens_d:+.2f} {sig_marker}",
                       ha="center", va="bottom", fontsize=5.5)

        ax_a2.set_ylim(y_lo - y_range * 0.05,
                        bracket_y + y_range * 0.18)

        ax_a2.set_xticks(positions)
        ax_a2.set_xticklabels(labels, fontsize=7)
        ax_a2.axhline(0, color="gray", lw=0.5, ls="--")
        ax_a2.set_ylabel("Z-scored FP (mean 2-10 bp)", fontsize=8)
        ax_a2.legend(handles=[
            Patch(facecolor="#2166AC", alpha=0.5, label="Leaf"),
            Patch(facecolor="#B2182B", alpha=0.5, label="Proto")],
            fontsize=7, loc="upper right")
    ax_a2.set_title("A2 — WRKY FP: leaf vs proto per ACR class "
                     "(paired per hit, z-scored)",
                     fontsize=9, fontweight="bold")

    # ── Panel B: WRKY residualized + z-scored delta heatmap ─────────────
    ax_b = fig.add_subplot(gs[2, 0])
    delta_path = os.path.join(args.v06_dir, "delta_acr_family_scale.npz")
    wrky_fi = None
    acr_classes = None
    if os.path.exists(delta_path):
        npz = np.load(delta_path, allow_pickle=True)
        delta = npz["delta"]           # (n_acrs, n_fam, n_scales)
        fam_ids = list(npz["family_ids"])
        scales = npz["scales"]
        acr_ids = list(npz["acr_ids"])

        # Load ACR metadata — v3_06 uses resized coords, metadata uses native
        acr_meta = pd.read_csv(os.path.join(BASE, args.acr_metadata),
                               sep="\t")
        acr_meta["acr_id"] = acr_meta["acr_id"].astype(str)
        coord_map_path = os.path.join(BASE, "data",
                                       "acr_native_to_resized.tsv")
        if os.path.exists(coord_map_path):
            coord_map = pd.read_csv(coord_map_path, sep="\t")
            resized_to_native = dict(zip(coord_map["resized_str"],
                                          coord_map["native_str"]))
        else:
            resized_to_native = {}

        native_class_map = dict(zip(acr_meta["acr_id"],
                                     acr_meta["acr_class"]))
        acr_classes = np.array([
            native_class_map.get(resized_to_native.get(str(a), str(a)),
                                 "stable")
            for a in acr_ids])

        # Find WRKY family index
        for i, f in enumerate(fam_ids):
            if rename_family(f) == WRKY_FAMILY:
                wrky_fi = i
                break

        if wrky_fi is not None:
            wrky_delta = delta[:, wrky_fi, :]  # (n_acrs, n_scales)

            # ── Residualize on confounders per scale ──────────────────
            # Prepare acr_meta indexed by resized str for residualize_features
            acr_meta_res = acr_meta.copy()
            # Rename columns to match _utils expectations
            if "width" in acr_meta_res.columns and "acr_width" not in acr_meta_res.columns:
                acr_meta_res["acr_width"] = acr_meta_res["width"]
            if "edgeR_logCPM" in acr_meta_res.columns and "logCPM" not in acr_meta_res.columns:
                acr_meta_res["logCPM"] = acr_meta_res["edgeR_logCPM"]
            # Map native acr_id → resized str for index alignment
            native_to_resized = {v: k for k, v in resized_to_native.items()}
            acr_meta_res["resized_str"] = acr_meta_res["acr_id"].map(
                native_to_resized)
            acr_meta_res = acr_meta_res.dropna(subset=["resized_str"])
            acr_meta_res = acr_meta_res.set_index("resized_str")

            # Build DataFrame: columns = scale indices, index = resized ACR IDs
            wrky_df = pd.DataFrame(
                wrky_delta, index=acr_ids,
                columns=[f"s{si}" for si in range(len(scales))])

            # Residualize each scale column on confounders
            wrky_resid = residualize_features(wrky_df, acr_meta_res)

            # Z-score per scale across ACRs (after residualization)
            wrky_z = wrky_resid.copy()
            for col in wrky_z.columns:
                mu = wrky_z[col].mean()
                sd = wrky_z[col].std()
                if sd > 1e-12:
                    wrky_z[col] = (wrky_z[col] - mu) / sd

            # Align ACR classes to residualized index
            acr_classes_resid = np.array([
                native_class_map.get(
                    resized_to_native.get(str(a), str(a)), "stable")
                for a in wrky_z.index])

            # Mean z-scored delta per ACR class × scale
            wrky_z_arr = wrky_z.values  # (n_acrs_valid, n_scales)
            hmap_data = []
            for cls in CLASS_ORDER:
                mask = acr_classes_resid == cls
                if mask.sum() > 0:
                    hmap_data.append(np.nanmean(wrky_z_arr[mask], axis=0))
                else:
                    hmap_data.append(np.zeros(len(scales)))

            hmap = np.array(hmap_data)
            vmax = max(abs(np.nanmin(hmap)), abs(np.nanmax(hmap)))

            im = ax_b.imshow(hmap, aspect="auto", cmap="PRGn",
                             vmin=-vmax, vmax=vmax,
                             extent=[scales[0], scales[-1], 2.5, -0.5])
            ax_b.set_yticks([0, 1, 2])
            ax_b.set_yticklabels([CLASS_LABELS[c] for c in CLASS_ORDER],
                                  fontsize=7)
            ax_b.set_xlabel("Scale (bp)", fontsize=8)
            plt.colorbar(im, ax=ax_b,
                         label="Mean z-scored delta\n(residualized, leaf−proto)",
                         shrink=0.7)
        ax_b.set_title("B — WRKY delta by ACR class × scale\n"
                        "(OLS-residualized, z-scored per scale)",
                        fontsize=9, fontweight="bold")
    else:
        ax_b.text(0.5, 0.5, "Missing v3_06 NPZ", ha="center", va="center",
                  transform=ax_b.transAxes)

    # ── Panel C: WRKY residualized z-scored delta by ACR class (violin) ──
    ax_c = fig.add_subplot(gs[2, 1])
    if os.path.exists(delta_path) and wrky_fi is not None:
        from scipy.stats import mannwhitneyu

        # Mean residualized z-scored delta across scales per ACR
        wrky_z_mean = np.nanmean(wrky_z_arr, axis=1)
        vdata = []
        for cls in CLASS_ORDER:
            mask = acr_classes_resid == cls
            vals = wrky_z_mean[mask]
            vals = vals[np.isfinite(vals)]
            vdata.append(vals if len(vals) > 0 else np.array([0.0]))

        parts = ax_c.violinplot(vdata, positions=range(len(CLASS_ORDER)),
                                showextrema=False,
                                showmedians=False, showmeans=False)
        for i, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(ACR_CLASS_COLORS[CLASS_ORDER[i]])
            pc.set_alpha(0.6)

        # Draw Q25, median, Q75 lines per violin
        for i in range(len(vdata)):
            q25, med, q75 = np.percentile(vdata[i], [25, 50, 75])
            hw = 0.15
            ax_c.hlines(med, i - hw, i + hw,
                         color="black", lw=1.5, zorder=4)
            ax_c.hlines([q25, q75], i - hw, i + hw,
                         color="black", lw=0.8, ls="--", zorder=4)
            ax_c.vlines(i, q25, q75,
                         color="black", lw=0.6, zorder=3)

        ax_c.set_xticks(range(len(CLASS_ORDER)))
        ax_c.set_xticklabels(
            [f"{CLASS_LABELS[c]}\n(n={len(vdata[i]):,})"
             for i, c in enumerate(CLASS_ORDER)], fontsize=7)
        ax_c.axhline(0, color="gray", lw=0.5, ls="--")
        ax_c.set_ylabel("Mean z-scored WRKY delta\n(residualized, all scales)",
                         fontsize=8)

        # Pairwise Mann-Whitney U tests with bracket annotations
        pairs = [(0, 1, "PG vs S"), (0, 2, "PG vs LG"), (1, 2, "S vs LG")]
        y_max = max(np.percentile(v, 99) for v in vdata if len(v) > 1)
        y_range = y_max - min(np.percentile(v, 1) for v in vdata if len(v) > 1)
        bracket_y = y_max + y_range * 0.08

        for pi, (i, j, label) in enumerate(pairs):
            if len(vdata[i]) < 5 or len(vdata[j]) < 5:
                continue
            stat, p = mannwhitneyu(vdata[i], vdata[j], alternative="two-sided")
            if p < 0.001:
                p_str = f"p={p:.1e}"
            else:
                p_str = f"p={p:.3f}"

            y_br = bracket_y + pi * y_range * 0.08
            ax_c.plot([i, i, j, j], [y_br - y_range * 0.01, y_br,
                       y_br, y_br - y_range * 0.01],
                      color="black", lw=0.8)
            ax_c.text((i + j) / 2, y_br + y_range * 0.005, p_str,
                      ha="center", va="bottom", fontsize=5.5)

        ax_c.set_ylim(-4, bracket_y + len(pairs) * y_range * 0.10)

    ax_c.set_title("C — WRKY delta distribution by ACR class\n"
                    "(OLS-residualized, z-scored)",
                    fontsize=9, fontweight="bold")

    # ── Panel B2: Per-signature residualized + z-scored delta heatmap ─────
    ax_b2 = fig.add_subplot(gs[3, 0])
    sig_delta_path = os.path.join(args.v06_dir,
                                   "delta_acr_signature_scale.npz")
    wrky_sig_ids = ["sig_121", "sig_122"]
    if os.path.exists(sig_delta_path):
        sig_npz = np.load(sig_delta_path, allow_pickle=True)
        sig_delta = sig_npz["delta"]        # (n_acrs, n_sigs, n_scales)
        sig_ids = list(sig_npz["signature_ids"])
        sig_scales = sig_npz["scales"]
        sig_acr_ids = list(sig_npz["acr_ids"])

        # Reuse ACR metadata from Panel B (same resized coords)
        if acr_classes is not None:
            # Find indices for WRKY signatures
            wrky_sig_idx = [si for si, s in enumerate(sig_ids)
                            if s in wrky_sig_ids]

            if wrky_sig_idx:
                # Prepare acr_meta for residualization (reuse from Panel B)
                acr_meta_sig = pd.read_csv(
                    os.path.join(BASE, args.acr_metadata), sep="\t")
                acr_meta_sig["acr_id"] = acr_meta_sig["acr_id"].astype(str)
                if "width" in acr_meta_sig.columns and "acr_width" not in acr_meta_sig.columns:
                    acr_meta_sig["acr_width"] = acr_meta_sig["width"]
                if "edgeR_logCPM" in acr_meta_sig.columns and "logCPM" not in acr_meta_sig.columns:
                    acr_meta_sig["logCPM"] = acr_meta_sig["edgeR_logCPM"]

                coord_map_path_s = os.path.join(
                    BASE, "data", "acr_native_to_resized.tsv")
                if os.path.exists(coord_map_path_s):
                    cm_s = pd.read_csv(coord_map_path_s, sep="\t")
                    r2n_s = dict(zip(cm_s["resized_str"],
                                      cm_s["native_str"]))
                    n2r_s = {v: k for k, v in r2n_s.items()}
                else:
                    r2n_s = {}
                    n2r_s = {}

                acr_meta_sig["resized_str"] = acr_meta_sig["acr_id"].map(
                    n2r_s)
                acr_meta_sig = acr_meta_sig.dropna(subset=["resized_str"])
                native_class_map_s = dict(zip(acr_meta_sig["acr_id"],
                                               acr_meta_sig["acr_class"]))
                acr_meta_sig = acr_meta_sig.set_index("resized_str")

                acr_classes_sig = np.array([
                    native_class_map_s.get(r2n_s.get(str(a), str(a)),
                                           "stable")
                    for a in sig_acr_ids])

                # Build stacked heatmap: rows = sig × ACR class, cols = scales
                n_wrky = len(wrky_sig_idx)
                n_cls = len(CLASS_ORDER)
                hmap_rows = []
                row_labels = []

                for wi in wrky_sig_idx:
                    sig_name = sig_ids[wi]
                    sig_data = sig_delta[:, wi, :]  # (n_acrs, n_scales)

                    # Residualize per scale
                    sig_df = pd.DataFrame(
                        sig_data, index=sig_acr_ids,
                        columns=[f"s{si}" for si in range(len(sig_scales))])
                    sig_resid = residualize_features(sig_df, acr_meta_sig)

                    # Z-score per scale
                    for col in sig_resid.columns:
                        mu = sig_resid[col].mean()
                        sd = sig_resid[col].std()
                        if sd > 1e-12:
                            sig_resid[col] = (sig_resid[col] - mu) / sd

                    sig_z_arr = sig_resid.values
                    acr_cls_resid = np.array([
                        native_class_map_s.get(
                            r2n_s.get(str(a), str(a)), "stable")
                        for a in sig_resid.index])

                    for cls in CLASS_ORDER:
                        mask = acr_cls_resid == cls
                        if mask.sum() > 0:
                            hmap_rows.append(
                                np.nanmean(sig_z_arr[mask], axis=0))
                        else:
                            hmap_rows.append(np.zeros(len(sig_scales)))
                        row_labels.append(f"{sig_name}\n{CLASS_LABELS[cls]}")

                hmap = np.array(hmap_rows)
                vmax = max(abs(np.nanmin(hmap)), abs(np.nanmax(hmap)))

                im_b2 = ax_b2.imshow(
                    hmap, aspect="auto", cmap="PRGn",
                    vmin=-vmax, vmax=vmax,
                    extent=[sig_scales[0], sig_scales[-1],
                            len(hmap_rows) - 0.5, -0.5])
                ax_b2.set_yticks(range(len(row_labels)))
                ax_b2.set_yticklabels(row_labels, fontsize=6)
                ax_b2.set_xlabel("Scale (bp)", fontsize=8)
                plt.colorbar(im_b2, ax=ax_b2,
                             label="Mean z-scored delta\n(residualized)",
                             shrink=0.7)

                # Separator lines between signatures
                for sep in range(n_cls, len(hmap_rows), n_cls):
                    ax_b2.axhline(sep - 0.5, color="white", lw=1.5)
    ax_b2.set_title("B2 — Per-signature WRKY delta by ACR class × scale\n"
                     "(OLS-residualized, z-scored per scale)",
                     fontsize=9, fontweight="bold")

    # ── Panel C2: Per-signature delta violin by ACR class ─────────────────
    ax_c2 = fig.add_subplot(gs[3, 1])
    if (os.path.exists(sig_delta_path) and wrky_sig_idx
            and acr_classes_sig is not None):
        from scipy.stats import mannwhitneyu as mwu_c2

        # One violin per signature × ACR class
        vdata_c2 = []
        vlabels_c2 = []
        vcolors_c2 = []
        pos_c2 = []
        pos_idx = 0
        sig_group_starts = []

        for wi in wrky_sig_idx:
            sig_name = sig_ids[wi]
            sig_data = sig_delta[:, wi, :]

            sig_df = pd.DataFrame(
                sig_data, index=sig_acr_ids,
                columns=[f"s{si}" for si in range(len(sig_scales))])
            sig_resid = residualize_features(sig_df, acr_meta_sig)
            for col in sig_resid.columns:
                mu = sig_resid[col].mean()
                sd = sig_resid[col].std()
                if sd > 1e-12:
                    sig_resid[col] = (sig_resid[col] - mu) / sd

            sig_z_mean = np.nanmean(sig_resid.values, axis=1)
            acr_cls_resid = np.array([
                native_class_map_s.get(
                    r2n_s.get(str(a), str(a)), "stable")
                for a in sig_resid.index])

            sig_group_starts.append(pos_idx)
            for cls in CLASS_ORDER:
                mask = acr_cls_resid == cls
                vals = sig_z_mean[mask]
                vals = vals[np.isfinite(vals)]
                vdata_c2.append(vals if len(vals) > 0 else np.array([0.0]))
                vlabels_c2.append(f"{CLASS_LABELS[cls]}")
                vcolors_c2.append(ACR_CLASS_COLORS[cls])
                pos_c2.append(pos_idx)
                pos_idx += 1
            pos_idx += 0.5  # gap between signatures

        parts_c2 = ax_c2.violinplot(
            vdata_c2, positions=pos_c2,
            showextrema=False, showmedians=False, showmeans=False)
        for i, pc in enumerate(parts_c2["bodies"]):
            pc.set_facecolor(vcolors_c2[i])
            pc.set_alpha(0.6)

        # Q25/median/Q75 lines
        for i, pos in enumerate(pos_c2):
            if len(vdata_c2[i]) < 2:
                continue
            q25, med, q75 = np.percentile(vdata_c2[i], [25, 50, 75])
            hw = 0.15
            ax_c2.hlines(med, pos - hw, pos + hw,
                         color="black", lw=1.5, zorder=4)
            ax_c2.hlines([q25, q75], pos - hw, pos + hw,
                         color="black", lw=0.8, ls="--", zorder=4)
            ax_c2.vlines(pos, q25, q75,
                         color="black", lw=0.6, zorder=3)

        ax_c2.set_xticks(pos_c2)
        ax_c2.set_xticklabels(vlabels_c2, fontsize=6, rotation=45, ha="right")
        ax_c2.axhline(0, color="gray", lw=0.5, ls="--")
        ax_c2.set_ylabel("Mean z-scored delta\n(residualized, all scales)",
                          fontsize=8)

        # Add signature name labels above each group
        for gi, wi in enumerate(wrky_sig_idx):
            group_center = np.mean(
                pos_c2[gi * n_cls: (gi + 1) * n_cls])
            ax_c2.text(group_center, ax_c2.get_ylim()[1] * 0.95,
                       sig_ids[wi], ha="center", va="top",
                       fontsize=7, fontweight="bold",
                       bbox=dict(boxstyle="round,pad=0.2", fc="lightyellow",
                                 alpha=0.8))

        # Pairwise Mann-Whitney within each signature (PG vs S, PG vs LG, S vs LG)
        pairs_c2 = [(0, 1), (0, 2), (1, 2)]
        y_max_c2 = max(np.percentile(v, 99)
                       for v in vdata_c2 if len(v) > 1)
        y_min_c2 = min(np.percentile(v, 1)
                       for v in vdata_c2 if len(v) > 1)
        y_range_c2 = y_max_c2 - y_min_c2
        bracket_base_c2 = y_max_c2 + y_range_c2 * 0.08

        for gi in range(len(wrky_sig_idx)):
            offset = gi * n_cls
            for pi, (i_rel, j_rel) in enumerate(pairs_c2):
                i_abs = offset + i_rel
                j_abs = offset + j_rel
                if (len(vdata_c2[i_abs]) < 5 or
                        len(vdata_c2[j_abs]) < 5):
                    continue
                stat, p = mwu_c2(vdata_c2[i_abs], vdata_c2[j_abs],
                                  alternative="two-sided")
                p_str = (f"p={p:.1e}" if p < 0.001
                         else f"p={p:.3f}")

                y_br = bracket_base_c2 + pi * y_range_c2 * 0.07
                xi = pos_c2[i_abs]
                xj = pos_c2[j_abs]
                ax_c2.plot([xi, xi, xj, xj],
                           [y_br - y_range_c2 * 0.01, y_br,
                            y_br, y_br - y_range_c2 * 0.01],
                           color="black", lw=0.8)
                ax_c2.text((xi + xj) / 2, y_br + y_range_c2 * 0.005,
                           p_str, ha="center", va="bottom", fontsize=5)

        ax_c2.set_ylim(-4, bracket_base_c2 + len(pairs_c2) * y_range_c2 * 0.09)

    ax_c2.set_title("C2 — Per-signature WRKY delta by ACR class\n"
                     "(OLS-residualized, z-scored)",
                     fontsize=9, fontweight="bold")

    fig.suptitle("Page 1 — WRKY Footprint Signal", fontsize=12,
                 fontweight="bold", y=1.00)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    # ── Save individual panels as PDF + PNG ──────────────────────────────
    p1dir = os.path.join(args.outdir, "page1_panels")
    os.makedirs(p1dir, exist_ok=True)
    print("    Saving individual Page 1 panels...", flush=True)

    # A1: one hexbin per ACR class
    if zhit is not None:
        from scipy.stats import spearmanr as _sp_a1
        leaf_z = zhit["leaf_z"]
        proto_z = zhit["proto_z"]
        hit_cls = zhit["hit_classes"]
        all_vals = np.concatenate([leaf_z, proto_z])
        lim = np.percentile(np.abs(all_vals), 99) * 1.15

        for ci, cls in enumerate(CLASS_ORDER):
            mask = hit_cls == cls
            if mask.sum() < 10:
                continue
            fi = plt.figure(figsize=(5, 5))
            axi = fi.add_subplot(111)
            lf, pr = leaf_z[mask], proto_z[mask]
            axi.hexbin(lf, pr, gridsize=40, cmap="YlOrRd", mincnt=1,
                       extent=[-lim, lim, -lim, lim],
                       linewidths=0.1, rasterized=True)
            axi.plot([-lim, lim], [-lim, lim], "k--", lw=0.5, alpha=0.4)
            r, p = _sp_a1(lf, pr)
            z_c = np.polyfit(lf, pr, 1)
            x_l = np.linspace(-lim * 0.9, lim * 0.9, 100)
            axi.plot(x_l, np.polyval(z_c, x_l), "k-", lw=1.5)
            axi.set_xlim(-lim, lim); axi.set_ylim(-lim, lim)
            axi.set_aspect("equal")
            axi.text(0.03, 0.97, f"r={r:.3f}, p={p:.1e}\nslope={z_c[0]:.2f}",
                     transform=axi.transAxes, fontsize=7, va="top",
                     bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8))
            axi.set_xlabel("Leaf FP (z, 2-10 bp)", fontsize=8)
            axi.set_ylabel("Proto FP (z, 2-10 bp)", fontsize=8)
            axi.set_title(f"A1 — {CLASS_LABELS[cls]} (n={mask.sum():,})",
                          fontsize=9, fontweight="bold")
            _save_panel(fi, p1dir, f"A1_{cls}")

    # A2: paired violin — save as one panel
    if zhit is not None:
        # (complex panel — save the full row as-is)
        pass  # already in combined PDF; complex to re-render individually

    # B: heatmap rows — one PDF per ACR class (shared color scale)
    if os.path.exists(delta_path) and wrky_fi is not None:
        # Compute global vmax across all ACR classes
        vmax_b = 0.0
        for cls in CLASS_ORDER:
            mask_b = acr_classes_resid == cls
            if mask_b.sum() == 0:
                continue
            row_data = np.nanmean(wrky_z_arr[mask_b], axis=0)
            vmax_b = max(vmax_b, abs(np.nanmin(row_data)),
                         abs(np.nanmax(row_data)))

        for ci, cls in enumerate(CLASS_ORDER):
            mask_b = acr_classes_resid == cls
            if mask_b.sum() == 0:
                continue
            row_data = np.nanmean(wrky_z_arr[mask_b], axis=0)
            fi = plt.figure(figsize=(8, 1.5))
            axi = fi.add_subplot(111)
            im_b = axi.imshow(row_data[np.newaxis, :], aspect="auto",
                              cmap="PRGn", vmin=-vmax_b, vmax=vmax_b,
                              extent=[scales[0], scales[-1], 0.5, -0.5])
            axi.set_yticks([0])
            axi.set_yticklabels([f"{CLASS_LABELS[cls]} (n={mask_b.sum():,})"],
                                fontsize=8)
            axi.set_xlabel("Scale (bp)", fontsize=8)
            plt.colorbar(im_b, ax=axi,
                         label="Mean z-scored delta (resid.)", shrink=0.8)
            axi.set_title(f"B — WRKY delta: {CLASS_LABELS[cls]}",
                          fontsize=9, fontweight="bold")
            _save_panel(fi, p1dir, f"B_{cls}")

    # B2: per-signature heatmap rows — one PDF per sig × ACR class (shared color scale)
    if (os.path.exists(sig_delta_path) and wrky_sig_idx
            and acr_classes_sig is not None):
        # Pre-compute residualized z-scored data for all sigs, then get global vmax
        sig_z_cache = {}   # wi → (sig_z_arr, cls_arr)
        for wi in wrky_sig_idx:
            sig_data = sig_delta[:, wi, :]
            sig_df_i = pd.DataFrame(
                sig_data, index=sig_acr_ids,
                columns=[f"s{si}" for si in range(len(sig_scales))])
            sig_resid_i = residualize_features(sig_df_i, acr_meta_sig)
            for col in sig_resid_i.columns:
                mu = sig_resid_i[col].mean()
                sd = sig_resid_i[col].std()
                if sd > 1e-12:
                    sig_resid_i[col] = (sig_resid_i[col] - mu) / sd
            sig_z_i = sig_resid_i.values
            cls_i = np.array([
                native_class_map_s.get(
                    r2n_s.get(str(a), str(a)), "stable")
                for a in sig_resid_i.index])
            sig_z_cache[wi] = (sig_z_i, cls_i)

        # Global vmax across all signatures and ACR classes
        vmax_s = 0.0
        for wi, (sig_z_i, cls_i) in sig_z_cache.items():
            for cls in CLASS_ORDER:
                mask_s = cls_i == cls
                if mask_s.sum() == 0:
                    continue
                row_data = np.nanmean(sig_z_i[mask_s], axis=0)
                vmax_s = max(vmax_s, abs(np.nanmin(row_data)),
                             abs(np.nanmax(row_data)))

        for wi in wrky_sig_idx:
            sig_name = sig_ids[wi]
            sig_z_i, cls_i = sig_z_cache[wi]

            for cls in CLASS_ORDER:
                mask_s = cls_i == cls
                if mask_s.sum() == 0:
                    continue
                row_data = np.nanmean(sig_z_i[mask_s], axis=0)
                fi = plt.figure(figsize=(8, 1.5))
                axi = fi.add_subplot(111)
                im_s = axi.imshow(
                    row_data[np.newaxis, :], aspect="auto",
                    cmap="PRGn", vmin=-vmax_s, vmax=vmax_s,
                    extent=[sig_scales[0], sig_scales[-1], 0.5, -0.5])
                axi.set_yticks([0])
                axi.set_yticklabels(
                    [f"{CLASS_LABELS[cls]} (n={mask_s.sum():,})"],
                    fontsize=8)
                axi.set_xlabel("Scale (bp)", fontsize=8)
                plt.colorbar(im_s, ax=axi,
                             label="Mean z-scored delta (resid.)",
                             shrink=0.8)
                axi.set_title(f"B2 — {sig_name}: {CLASS_LABELS[cls]}",
                              fontsize=9, fontweight="bold")
                _save_panel(fi, p1dir, f"B2_{sig_name}_{cls}")

    # C: family violin — save as one panel
    if os.path.exists(delta_path) and wrky_fi is not None:
        fi = plt.figure(figsize=(5, 5))
        axi = fi.add_subplot(111)
        from scipy.stats import mannwhitneyu as _mwu_c
        wrky_z_mean_c = np.nanmean(wrky_z_arr, axis=1)
        vdata_c = []
        for cls in CLASS_ORDER:
            mask_c = acr_classes_resid == cls
            vals_c = wrky_z_mean_c[mask_c]
            vals_c = vals_c[np.isfinite(vals_c)]
            vdata_c.append(vals_c if len(vals_c) > 0 else np.array([0.0]))
        parts_c = axi.violinplot(vdata_c, positions=range(len(CLASS_ORDER)),
                                 showextrema=False,
                                 showmedians=False, showmeans=False)
        for i, pc in enumerate(parts_c["bodies"]):
            pc.set_facecolor(ACR_CLASS_COLORS[CLASS_ORDER[i]])
            pc.set_alpha(0.6)
        for i in range(len(vdata_c)):
            if len(vdata_c[i]) < 2:
                continue
            q25, med, q75 = np.percentile(vdata_c[i], [25, 50, 75])
            hw = 0.15
            axi.hlines(med, i - hw, i + hw, color="black", lw=1.5, zorder=4)
            axi.hlines([q25, q75], i - hw, i + hw,
                       color="black", lw=0.8, ls="--", zorder=4)
            axi.vlines(i, q25, q75, color="black", lw=0.6, zorder=3)
        axi.set_xticks(range(len(CLASS_ORDER)))
        axi.set_xticklabels(
            [f"{CLASS_LABELS[c]}\n(n={len(vdata_c[i]):,})"
             for i, c in enumerate(CLASS_ORDER)], fontsize=7)
        axi.axhline(0, color="gray", lw=0.5, ls="--")
        axi.set_ylabel("Mean z-scored WRKY delta\n(residualized, all scales)",
                        fontsize=8)
        axi.set_title("C — WRKY delta by ACR class (resid., z-scored)",
                       fontsize=9, fontweight="bold")
        _save_panel(fi, p1dir, "C_violin")

    print(f"    Saved Page 1 panels to {p1dir}", flush=True)


# ── Page 2: WRKY in the model ────────────────────────────────────────────────

def _wrky_sig_shap_violin_data(npz_path, max_scale_bp=5):
    """Extract per-WRKY-signature mean SHAP (scales ≤ max_scale_bp).

    Returns dict: {sig_id: {"shap_mean": (n_test,), "acr_class": (n_test,)}}
    or None if data unavailable.
    """
    if not os.path.exists(npz_path):
        return None
    npz = np.load(npz_path, allow_pickle=True)
    shap_vals = npz["shap_values"]       # (n_test, n_features) or 3D
    feat_names = list(npz["feature_names"])
    acr_class = npz["acr_class_test"]

    if shap_vals.ndim == 3:
        shap_vals = shap_vals.mean(axis=2)

    wrky_sigs = {"sig_121", "sig_122"}
    result = {}
    for target_sig in sorted(wrky_sigs):
        # Find columns for this signature at scales ≤ max_scale_bp
        col_idxs = []
        for fi, fname in enumerate(feat_names):
            # Format: sig_NNN_sScaleIdx  (scale_bp = scale_idx + 2)
            if fname.startswith(target_sig + "_s"):
                try:
                    sidx = int(fname.split("_s")[-1])
                    scale_bp = sidx + 2
                    if scale_bp <= max_scale_bp:
                        col_idxs.append(fi)
                except ValueError:
                    pass

        if col_idxs:
            # Mean SHAP across selected scales per ACR
            mean_shap = shap_vals[:, col_idxs].mean(axis=1)
            result[target_sig] = {
                "shap_mean": mean_shap,
                "acr_class": acr_class,
            }

    # Combined: average across both sigs
    if len(result) == 2:
        combined_shap = np.mean(
            [result[s]["shap_mean"] for s in sorted(wrky_sigs)], axis=0)
        result["combined"] = {
            "shap_mean": combined_shap,
            "acr_class": acr_class,
        }

    return result if result else None


def _extract_wrky_shap_and_delta(npz_path, max_scale_tf=10, min_scale_nuc=80):
    """Extract per-ACR SHAP and delta for WRKY sigs at TF and nuc scales.

    Returns dict: {sig_id: {"shap_tf", "shap_nuc", "delta_tf", "delta_nuc",
                             "acr_class"}} plus "combined" key with pooled sigs.
    """
    if not os.path.exists(npz_path):
        return None
    npz = np.load(npz_path, allow_pickle=True)
    shap_vals = npz["shap_values"]     # (n_test, n_features)
    x_test = npz["X_test"]             # (n_test, n_features) — delta values
    feat_names = list(npz["feature_names"])
    acr_class = npz["acr_class_test"]

    if shap_vals.ndim == 3:
        shap_vals = shap_vals.mean(axis=2)

    wrky_sigs = ["sig_121", "sig_122"]
    result = {}

    for target_sig in wrky_sigs:
        tf_cols, nuc_cols = [], []
        for fi, fname in enumerate(feat_names):
            if fname.startswith(target_sig + "_s"):
                try:
                    sidx = int(fname.split("_s")[-1])
                    scale_bp = sidx + 2
                    if scale_bp <= max_scale_tf:
                        tf_cols.append(fi)
                    elif scale_bp >= min_scale_nuc:
                        nuc_cols.append(fi)
                except ValueError:
                    pass

        if tf_cols or nuc_cols:
            d = {"acr_class": acr_class}
            if tf_cols:
                d["shap_tf"] = shap_vals[:, tf_cols].mean(axis=1)
                d["delta_tf"] = x_test[:, tf_cols].mean(axis=1)
            if nuc_cols:
                d["shap_nuc"] = shap_vals[:, nuc_cols].mean(axis=1)
                d["delta_nuc"] = x_test[:, nuc_cols].mean(axis=1)
            result[target_sig] = d

    # Combined: average across both sigs
    if len(result) == 2:
        combined = {"acr_class": acr_class}
        for key in ["shap_tf", "shap_nuc", "delta_tf", "delta_nuc"]:
            arrs = [result[s][key] for s in wrky_sigs if key in result[s]]
            if arrs:
                combined[key] = np.mean(arrs, axis=0)
        result["combined"] = combined

    return result if result else None


def _plot_shap_violin(ax, shap_mean, acr_class, title, ylabel, ylim=None):
    """Draw a single SHAP violin plot (reusable for D1-D3)."""
    vdata, vlabels = [], []
    for cls in CLASS_ORDER:
        mask = acr_class == cls
        vals = shap_mean[mask]
        vals = vals[np.isfinite(vals)]
        vdata.append(vals if len(vals) > 0 else np.array([0.0]))
        vlabels.append(f"{CLASS_LABELS[cls]}\n(n={len(vals):,})")

    parts = ax.violinplot(vdata, positions=range(len(CLASS_ORDER)),
                          showextrema=False,
                          showmedians=False, showmeans=False)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(ACR_CLASS_COLORS[CLASS_ORDER[i]])
        pc.set_alpha(0.6)

    for i in range(len(vdata)):
        if len(vdata[i]) < 2:
            continue
        q25, med, q75 = np.percentile(vdata[i], [25, 50, 75])
        hw = 0.15
        ax.hlines(med, i - hw, i + hw, color="black", lw=1.5, zorder=4)
        ax.hlines([q25, q75], i - hw, i + hw,
                  color="black", lw=0.8, ls="--", zorder=4)
        ax.vlines(i, q25, q75, color="black", lw=0.6, zorder=3)

    ax.set_xticks(range(len(CLASS_ORDER)))
    ax.set_xticklabels(vlabels, fontsize=6)
    ax.axhline(0, color="gray", lw=0.5, ls="--")
    ax.set_ylabel(ylabel, fontsize=7)
    ax.set_title(title, fontsize=8, fontweight="bold")

    # Pairwise Mann-Whitney U brackets
    from scipy.stats import mannwhitneyu as _mwu_d
    pairs = [(0, 1), (0, 2), (1, 2)]
    if ylim is not None:
        y_top = ylim[1]
        y_range = ylim[1] - ylim[0]
    else:
        y_top = max(np.percentile(v, 99) for v in vdata if len(v) > 1)
        y_range = y_top - min(np.percentile(v, 1)
                              for v in vdata if len(v) > 1)
    bracket_y = y_top - y_range * 0.05

    for pi, (i, j) in enumerate(pairs):
        if len(vdata[i]) < 5 or len(vdata[j]) < 5:
            continue
        _, p = _mwu_d(vdata[i], vdata[j], alternative="two-sided")
        p_str = f"p={p:.1e}" if p < 0.001 else f"p={p:.3f}"

        y_br = bracket_y - pi * y_range * 0.07
        ax.plot([i, i, j, j],
                [y_br + y_range * 0.01, y_br,
                 y_br, y_br + y_range * 0.01],
                color="black", lw=0.8)
        ax.text((i + j) / 2, y_br - y_range * 0.005, p_str,
                ha="center", va="top", fontsize=5)

    if ylim is not None:
        ax.set_ylim(ylim)


def _plot_shap_vs_delta_hexbin(ax, shap_arr, delta_arr, acr_class, title,
                               vmax_count=10, shared_xlim=None,
                               shared_ylim=None):
    """Draw SHAP vs delta hexbin colored by density."""
    valid = np.isfinite(shap_arr) & np.isfinite(delta_arr)
    if valid.sum() < 20:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                transform=ax.transAxes, fontsize=7)
        ax.set_title(title, fontsize=7, fontweight="bold")
        return

    x = delta_arr[valid]
    y = shap_arr[valid]

    # Use shared limits if provided, else auto from data
    x_lim = shared_xlim if shared_xlim else np.percentile(np.abs(x), 99) * 1.15
    y_lim = shared_ylim if shared_ylim else np.percentile(np.abs(y), 99) * 1.15

    hb = ax.hexbin(x, y, gridsize=30, cmap="Purples", mincnt=1,
                   vmax=vmax_count,
                   extent=[-x_lim, x_lim, -y_lim, y_lim],
                   linewidths=0.1, rasterized=True)

    # Reference lines
    ax.axhline(0, color="gray", lw=0.4, ls="--")
    ax.axvline(0, color="gray", lw=0.4, ls="--")

    # OLS + Spearman
    from scipy.stats import spearmanr
    r, p = spearmanr(x, y)
    z_coef = np.polyfit(x, y, 1)
    x_line = np.linspace(-x_lim * 0.9, x_lim * 0.9, 50)
    ax.plot(x_line, np.polyval(z_coef, x_line), "k-", lw=1.2)

    ax.text(0.03, 0.97, f"r={r:.3f}\np={p:.1e}\nn={valid.sum():,}",
            transform=ax.transAxes, fontsize=5.5, va="top",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8))

    ax.set_xlabel("Delta (leaf−proto)", fontsize=6)
    ax.set_ylabel("Signed SHAP", fontsize=6)
    ax.set_title(title, fontsize=7, fontweight="bold")
    ax.tick_params(labelsize=5)
    plt.colorbar(hb, ax=ax, shrink=0.6, pad=0.02)


def page2_model(pdf, args, sig_meta):
    """WRKY SHAP violins + SHAP-vs-delta hexbins + scale heatmaps."""
    nature_figure_defaults()

    # Output directory for individual panels
    p2dir = os.path.join(args.outdir, "page2_panels")
    os.makedirs(p2dir, exist_ok=True)

    fig = plt.figure(figsize=(16, 28))
    gs = gridspec.GridSpec(5, 3, figure=fig, hspace=0.35, wspace=0.35)

    # ── Load SHAP + delta data ────────────────────────────────────────────
    t2_npz_path = os.path.join(args.v08_dir, "all", "raw_shap_tier2.npz")
    shap_delta = _extract_wrky_shap_and_delta(t2_npz_path)

    # Also load the simple violin data (≤5 bp)
    wrky_shap_data = _wrky_sig_shap_violin_data(t2_npz_path, max_scale_bp=5)

    # Panel ordering: sig_121, sig_122, combined
    panel_keys = ["sig_121", "sig_122", "combined"]
    panel_labels = {}
    for sig_id in ["sig_121", "sig_122"]:
        sig_row = sig_meta[sig_meta["signature_id"] == sig_id]
        if not sig_row.empty:
            disp = sig_row.iloc[0].get("display_name", sig_id)
            panel_labels[sig_id] = f"{sig_id} ({disp})"
        else:
            panel_labels[sig_id] = sig_id
    panel_labels["combined"] = "WRKY combined"

    # ── Row 1: D1, D2, D3 — SHAP violins (scales ≤ 5 bp) ────────────────
    d_ylim = (-0.015, 0.015)  # shared y-limits for all D violins

    for ci, key in enumerate(panel_keys):
        ax = fig.add_subplot(gs[0, ci])
        if wrky_shap_data is not None and key in wrky_shap_data:
            sdata = wrky_shap_data[key]
            _plot_shap_violin(
                ax, sdata["shap_mean"], sdata["acr_class"],
                title=f"D{ci+1} — {panel_labels[key]}",
                ylabel="Mean signed SHAP\n(scales ≤ 5 bp)",
                ylim=d_ylim)
        elif key == "combined" and wrky_shap_data is not None:
            all_shap = np.mean([wrky_shap_data[s]["shap_mean"]
                                for s in ["sig_121", "sig_122"]
                                if s in wrky_shap_data], axis=0)
            acr_cls = list(wrky_shap_data.values())[0]["acr_class"]
            _plot_shap_violin(
                ax, all_shap, acr_cls,
                title=f"D{ci+1} — {panel_labels[key]}",
                ylabel="Mean signed SHAP\n(scales ≤ 5 bp)",
                ylim=d_ylim)
        else:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(f"D{ci+1} — {panel_labels[key]}",
                         fontsize=8, fontweight="bold")

        # Save individual panel
        if wrky_shap_data is not None and key in wrky_shap_data:
            fi = plt.figure(figsize=(5, 4))
            axi = fi.add_subplot(111)
            _plot_shap_violin(
                axi, wrky_shap_data[key]["shap_mean"],
                wrky_shap_data[key]["acr_class"],
                title=f"D{ci+1} — {panel_labels[key]}",
                ylabel="Mean signed SHAP (scales ≤ 5 bp)",
                ylim=d_ylim)
            _save_panel(fi, p2dir, f"D{ci+1}_{key}_violin")

    # ── Compute shared axis limits across all hexbins ───────────────────
    # Collect all delta/SHAP values to determine shared limits per scale group
    shared_lims = {}  # sg_key → (x_lim, y_lim)
    if shap_delta is not None:
        for sg_key in ["tf", "nuc"]:
            all_x, all_y = [], []
            for key in panel_keys:
                if key not in shap_delta:
                    continue
                d = shap_delta[key]
                dk, sk = f"delta_{sg_key}", f"shap_{sg_key}"
                if dk in d and sk in d:
                    ac = d["acr_class"]
                    for cls in CLASS_ORDER:
                        m = ac == cls
                        xv = d[dk][m]
                        yv = d[sk][m]
                        v = np.isfinite(xv) & np.isfinite(yv)
                        all_x.append(xv[v])
                        all_y.append(yv[v])
            if all_x:
                all_x = np.concatenate(all_x)
                all_y = np.concatenate(all_y)
                shared_lims[sg_key] = (
                    np.percentile(np.abs(all_x), 99) * 1.15,
                    np.percentile(np.abs(all_y), 99) * 1.15)

    # ── Row 2: TF-scale hexbins (3 sigs × 3 ACR classes) ─────────────────
    sg_key, sg_label = "tf", "TF (≤10 bp)"
    sx, sy = shared_lims.get(sg_key, (None, None))
    for ci, key in enumerate(panel_keys):
        for ri, cls in enumerate(CLASS_ORDER):
            ax = fig.add_subplot(gs[1, ci]) if ri == 0 else None
            # Use subgridspec for 3 ACR classes within each column
        pass  # replaced below

    # Actually use a subgridspec for rows 2-3
    # Row 2: TF scale — 3 cols (sigs) × 3 sub-rows (ACR classes)
    for ci, key in enumerate(panel_keys):
        gs_tf = gs[1, ci].subgridspec(3, 1, hspace=0.45)
        for ri, cls in enumerate(CLASS_ORDER):
            ax = fig.add_subplot(gs_tf[ri])
            shap_key = f"shap_tf"
            delta_key = f"delta_tf"

            if (shap_delta is not None and key in shap_delta
                    and shap_key in shap_delta[key]):
                d = shap_delta[key]
                m = d["acr_class"] == cls
                if m.sum() >= 20:
                    _plot_shap_vs_delta_hexbin(
                        ax, d[shap_key][m], d[delta_key][m],
                        d["acr_class"][m],
                        title=f"{panel_labels[key]}\n{CLASS_LABELS[cls]} (n={m.sum():,})",
                        shared_xlim=sx, shared_ylim=sy)
                else:
                    ax.text(0.5, 0.5, f"n={m.sum()}", ha="center",
                            va="center", transform=ax.transAxes, fontsize=7)
                    ax.set_title(f"{panel_labels[key]}\n{CLASS_LABELS[cls]}",
                                 fontsize=7)
            else:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=7)

            # Save individual panel
            if (shap_delta is not None and key in shap_delta
                    and f"shap_tf" in shap_delta[key] and m.sum() >= 20):
                fi = plt.figure(figsize=(5, 4))
                axi = fi.add_subplot(111)
                _plot_shap_vs_delta_hexbin(
                    axi, d[shap_key][m], d[delta_key][m],
                    d["acr_class"][m],
                    title=f"{panel_labels[key]} — TF — {CLASS_LABELS[cls]}",
                    shared_xlim=sx, shared_ylim=sy)
                _save_panel(fi, p2dir, f"hex_{key}_tf_{cls}")

    fig.text(0.5, 0.63, "SHAP vs Delta — TF scale (≤10 bp)",
             ha="center", fontsize=9, fontweight="bold")

    # ── Row 3: Nuc-scale hexbins (3 sigs × 3 ACR classes) ────────────────
    sg_key, sg_label = "nuc", "Nuc (≥80 bp)"
    sx, sy = shared_lims.get(sg_key, (None, None))
    for ci, key in enumerate(panel_keys):
        gs_nuc = gs[2, ci].subgridspec(3, 1, hspace=0.45)
        for ri, cls in enumerate(CLASS_ORDER):
            ax = fig.add_subplot(gs_nuc[ri])
            shap_key = f"shap_nuc"
            delta_key = f"delta_nuc"

            if (shap_delta is not None and key in shap_delta
                    and shap_key in shap_delta[key]):
                d = shap_delta[key]
                m = d["acr_class"] == cls
                if m.sum() >= 20:
                    _plot_shap_vs_delta_hexbin(
                        ax, d[shap_key][m], d[delta_key][m],
                        d["acr_class"][m],
                        title=f"{panel_labels[key]}\n{CLASS_LABELS[cls]} (n={m.sum():,})",
                        shared_xlim=sx, shared_ylim=sy)
                else:
                    ax.text(0.5, 0.5, f"n={m.sum()}", ha="center",
                            va="center", transform=ax.transAxes, fontsize=7)
                    ax.set_title(f"{panel_labels[key]}\n{CLASS_LABELS[cls]}",
                                 fontsize=7)
            else:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=7)

            # Save individual panel
            if (shap_delta is not None and key in shap_delta
                    and f"shap_nuc" in shap_delta[key] and m.sum() >= 20):
                fi = plt.figure(figsize=(5, 4))
                axi = fi.add_subplot(111)
                _plot_shap_vs_delta_hexbin(
                    axi, d[shap_key][m], d[delta_key][m],
                    d["acr_class"][m],
                    title=f"{panel_labels[key]} — Nuc — {CLASS_LABELS[cls]}",
                    shared_xlim=sx, shared_ylim=sy)
                _save_panel(fi, p2dir, f"hex_{key}_nuc_{cls}")

    fig.text(0.5, 0.42, "SHAP vs Delta — Nucleosome scale (≥80 bp)",
             ha="center", fontsize=9, fontweight="bold")

    # ── Row 4: E — WRKY signed SHAP × scale (regression) ─────────────────
    ax_e = fig.add_subplot(gs[3, :])
    wrky_data = _wrky_shap_by_scale(
        os.path.join(args.v08_dir, "all", "raw_shap_tier1.npz"), sig_meta)
    if wrky_data is not None:
        shap_by_scale = wrky_data["shap_by_scale"]
        scales_bp = wrky_data["scales_bp"]
        acr_class = wrky_data["acr_class"]

        hmap_data = []
        for cls in CLASS_ORDER:
            mask = acr_class == cls
            if mask.sum() > 0:
                hmap_data.append(shap_by_scale[mask].mean(axis=0))
            else:
                hmap_data.append(np.zeros(len(scales_bp)))

        hmap = np.array(hmap_data)
        vmax = max(abs(np.nanmin(hmap)), abs(np.nanmax(hmap)))
        if vmax < 1e-10:
            vmax = 1e-4

        im = ax_e.imshow(hmap, aspect="auto", cmap="RdBu_r",
                         vmin=-vmax, vmax=vmax,
                         extent=[scales_bp[0], scales_bp[-1], 2.5, -0.5])
        ax_e.set_yticks([0, 1, 2])
        ax_e.set_yticklabels(
            [f"{CLASS_LABELS[c]} (n={int((acr_class==c).sum()):,})"
             for c in CLASS_ORDER], fontsize=7)
        ax_e.set_xlabel("Scale (bp)", fontsize=8)
        plt.colorbar(im, ax=ax_e, label="Mean signed SHAP", shrink=0.6)
    ax_e.set_title("E — WRKY signed SHAP × scale × ACR class (regression)",
                    fontsize=9, fontweight="bold")

    # Save E: full + per-row
    if wrky_data is not None:
        # Full heatmap
        fi = plt.figure(figsize=(10, 3))
        axi = fi.add_subplot(111)
        im_i = axi.imshow(hmap, aspect="auto", cmap="RdBu_r",
                          vmin=-vmax, vmax=vmax,
                          extent=[scales_bp[0], scales_bp[-1], 2.5, -0.5])
        axi.set_yticks([0, 1, 2])
        axi.set_yticklabels(
            [f"{CLASS_LABELS[c]} (n={int((acr_class==c).sum()):,})"
             for c in CLASS_ORDER], fontsize=7)
        axi.set_xlabel("Scale (bp)", fontsize=8)
        plt.colorbar(im_i, ax=axi, label="Mean signed SHAP", shrink=0.6)
        axi.set_title("E — WRKY signed SHAP × scale (regression)",
                       fontsize=9, fontweight="bold")
        _save_panel(fi, p2dir, "E_shap_heatmap_reg")
        # Per-row
        for ri, cls in enumerate(CLASS_ORDER):
            fi = plt.figure(figsize=(10, 1.5))
            axi = fi.add_subplot(111)
            im_r = axi.imshow(hmap[ri:ri+1], aspect="auto", cmap="RdBu_r",
                              vmin=-vmax, vmax=vmax,
                              extent=[scales_bp[0], scales_bp[-1], 0.5, -0.5])
            axi.set_yticks([0])
            axi.set_yticklabels(
                [f"{CLASS_LABELS[cls]} (n={int((acr_class==cls).sum()):,})"],
                fontsize=8)
            axi.set_xlabel("Scale (bp)", fontsize=8)
            plt.colorbar(im_r, ax=axi, label="Mean signed SHAP", shrink=0.8)
            axi.set_title(f"E — WRKY SHAP (reg): {CLASS_LABELS[cls]}",
                          fontsize=9, fontweight="bold")
            _save_panel(fi, p2dir, f"E_reg_{cls}")

    # ── Row 5: F — WRKY signed SHAP × scale (classification) ─────────────
    ax_f = fig.add_subplot(gs[4, :])
    wrky_clf = _wrky_shap_by_scale(
        os.path.join(args.v08_dir, "all", "raw_shap_tier1_clf.npz"), sig_meta)
    if wrky_clf is not None:
        shap_by_scale = wrky_clf["shap_by_scale"]
        scales_bp = wrky_clf["scales_bp"]
        acr_class = wrky_clf["acr_class"]

        hmap_data = []
        for cls in CLASS_ORDER:
            mask = acr_class == cls
            if mask.sum() > 0:
                hmap_data.append(shap_by_scale[mask].mean(axis=0))
            else:
                hmap_data.append(np.zeros(len(scales_bp)))

        hmap = np.array(hmap_data)
        vmax = max(abs(np.nanmin(hmap)), abs(np.nanmax(hmap)))
        if vmax < 1e-10:
            vmax = 1e-4

        im = ax_f.imshow(hmap, aspect="auto", cmap="RdBu_r",
                         vmin=-vmax, vmax=vmax,
                         extent=[scales_bp[0], scales_bp[-1], 2.5, -0.5])
        ax_f.set_yticks([0, 1, 2])
        ax_f.set_yticklabels(
            [f"{CLASS_LABELS[c]} (n={int((acr_class==c).sum()):,})"
             for c in CLASS_ORDER], fontsize=7)
        ax_f.set_xlabel("Scale (bp)", fontsize=8)
        plt.colorbar(im, ax=ax_f, label="Mean signed SHAP", shrink=0.6)
    ax_f.set_title("F — WRKY signed SHAP × scale × ACR class (classification)",
                    fontsize=9, fontweight="bold")

    # Save F: full + per-row
    if wrky_clf is not None:
        fi = plt.figure(figsize=(10, 3))
        axi = fi.add_subplot(111)
        im_i = axi.imshow(hmap, aspect="auto", cmap="RdBu_r",
                          vmin=-vmax, vmax=vmax,
                          extent=[scales_bp[0], scales_bp[-1], 2.5, -0.5])
        axi.set_yticks([0, 1, 2])
        axi.set_yticklabels(
            [f"{CLASS_LABELS[c]} (n={int((acr_class==c).sum()):,})"
             for c in CLASS_ORDER], fontsize=7)
        axi.set_xlabel("Scale (bp)", fontsize=8)
        plt.colorbar(im_i, ax=axi, label="Mean signed SHAP", shrink=0.6)
        axi.set_title("F — WRKY signed SHAP × scale (classification)",
                       fontsize=9, fontweight="bold")
        _save_panel(fi, p2dir, "F_shap_heatmap_clf")
        # Per-row
        for ri, cls in enumerate(CLASS_ORDER):
            fi = plt.figure(figsize=(10, 1.5))
            axi = fi.add_subplot(111)
            im_r = axi.imshow(hmap[ri:ri+1], aspect="auto", cmap="RdBu_r",
                              vmin=-vmax, vmax=vmax,
                              extent=[scales_bp[0], scales_bp[-1], 0.5, -0.5])
            axi.set_yticks([0])
            axi.set_yticklabels(
                [f"{CLASS_LABELS[cls]} (n={int((acr_class==cls).sum()):,})"],
                fontsize=8)
            axi.set_xlabel("Scale (bp)", fontsize=8)
            plt.colorbar(im_r, ax=axi, label="Mean signed SHAP", shrink=0.8)
            axi.set_title(f"F — WRKY SHAP (clf): {CLASS_LABELS[cls]}",
                          fontsize=9, fontweight="bold")
            _save_panel(fi, p2dir, f"F_clf_{cls}")

    fig.suptitle("Page 2 — WRKY in the Gradient Boosting Model",
                 fontsize=12, fontweight="bold", y=1.00)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved Page 2 panels to {p2dir}", flush=True)


# ── Page 3: WRKY in context ──────────────────────────────────────────────────

def page3_context(pdf, args, sig_meta):
    """Asymmetry analysis + permutation importance."""
    nature_figure_defaults()
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    # Load T1 SHAP aggregated by family
    t1_data = _load_shap_t1(
        os.path.join(args.v08_dir, "all", "raw_shap_tier1.npz"), sig_meta)

    # Panel G: Signed SHAP grouped bar (top families + WRKY)
    ax_g = fig.add_subplot(gs[0, :])
    if t1_data is not None:
        shap_fam = t1_data["shap_by_family"]
        fam_names = t1_data["family_names"]
        acr_class = t1_data["acr_class"]

        rows = []
        for j, fam in enumerate(fam_names):
            for cls in CLASS_ORDER:
                mask = acr_class == cls
                if mask.sum() > 0:
                    rows.append({"family": fam, "acr_class": cls,
                                 "mean_shap": float(shap_fam[mask, j].mean())})
        df = pd.DataFrame(rows)

        # Top 15 by max |SHAP| + always include WRKY
        fam_max = (df.groupby("family")["mean_shap"]
                   .apply(lambda x: x.abs().max())
                   .sort_values(ascending=False))
        top_fams = fam_max.head(15).index.tolist()
        if WRKY_FAMILY not in top_fams:
            top_fams.append(WRKY_FAMILY)

        # Sort by proto-gain SHAP
        proto_shap = (df[df["acr_class"] == "proto_gain"]
                      .set_index("family")["mean_shap"])
        top_fams_sorted = sorted(top_fams, key=lambda f: proto_shap.get(f, 0))

        df_top = df[df["family"].isin(top_fams_sorted)]
        n_top = len(top_fams_sorted)
        y_pos = np.arange(n_top)
        bar_h = 0.25

        for ci, cls in enumerate(CLASS_ORDER):
            sub = df_top[df_top["acr_class"] == cls].set_index("family")
            vals = [sub.loc[f, "mean_shap"] if f in sub.index else 0
                    for f in top_fams_sorted]
            color = ACR_CLASS_COLORS.get(cls, "#888888")
            ax_g.barh(y_pos + (ci - 1) * bar_h, vals, height=bar_h,
                      color=color, edgecolor="white", linewidth=0.3,
                      label=CLASS_LABELS[cls])

        ax_g.set_yticks(y_pos)
        ylabels = []
        for f in top_fams_sorted:
            label = f"**{f}**" if f == WRKY_FAMILY else f
            ylabels.append(f)
        ax_g.set_yticklabels(ylabels, fontsize=6)
        # Bold WRKY label
        for tick_label in ax_g.get_yticklabels():
            if tick_label.get_text() == WRKY_FAMILY:
                tick_label.set_fontweight("bold")
                tick_label.set_color("#8B0000")

        ax_g.axvline(0, color="black", lw=0.5)
        ax_g.set_xlabel("Mean signed SHAP (sum across scales)", fontsize=8)
        ax_g.legend(fontsize=7, loc="lower right")
    ax_g.set_title("G — Per-family signed SHAP by ACR class",
                    fontsize=9, fontweight="bold")

    # Panel H: Asymmetry scatter (proto-gain SHAP vs leaf-gain SHAP)
    ax_h = fig.add_subplot(gs[1, 0])
    if t1_data is not None:
        fam_names = t1_data["family_names"]
        acr_class = t1_data["acr_class"]
        shap_fam = t1_data["shap_by_family"]

        x_vals, y_vals = [], []
        for j, fam in enumerate(fam_names):
            pg_mask = acr_class == "proto_gain"
            lg_mask = acr_class == "leaf_gain"
            x_vals.append(float(shap_fam[pg_mask, j].mean()) if pg_mask.sum() > 0 else 0)
            y_vals.append(float(shap_fam[lg_mask, j].mean()) if lg_mask.sum() > 0 else 0)

        x_vals = np.array(x_vals)
        y_vals = np.array(y_vals)

        ax_h.scatter(x_vals, y_vals, s=30, alpha=0.6, c="#555555",
                     edgecolors="black", linewidth=0.3, zorder=3)

        # Highlight WRKY
        for j, fam in enumerate(fam_names):
            if fam == WRKY_FAMILY:
                ax_h.scatter(x_vals[j], y_vals[j], s=80, c="#8B0000",
                             edgecolors="black", linewidth=0.8, zorder=5,
                             marker="*")

        lim = max(np.abs(x_vals).max(), np.abs(y_vals).max()) * 1.3
        ax_h.plot([-lim, lim], [lim, -lim], "k--", lw=0.5, alpha=0.4)
        ax_h.axhline(0, color="gray", lw=0.3)
        ax_h.axvline(0, color="gray", lw=0.3)
        ax_h.set_xlim(-lim, lim)
        ax_h.set_ylim(-lim, lim)
        ax_h.set_aspect("equal")

        # Label all families
        try:
            from adjustText import adjust_text
            texts = []
            for j, fam in enumerate(fam_names):
                c = "#8B0000" if fam == WRKY_FAMILY else "black"
                w = "bold" if fam == WRKY_FAMILY else "normal"
                texts.append(ax_h.text(x_vals[j], y_vals[j], fam,
                                       fontsize=5, color=c, fontweight=w))
            adjust_text(texts, ax=ax_h, force_text=(0.3, 0.3),
                        arrowprops=dict(arrowstyle="-", color="gray",
                                        lw=0.3, alpha=0.4))
        except ImportError:
            for j, fam in enumerate(fam_names):
                ax_h.annotate(fam, (x_vals[j], y_vals[j]), fontsize=4)

        ax_h.set_xlabel("Mean signed SHAP at Proto-gain ACRs", fontsize=7)
        ax_h.set_ylabel("Mean signed SHAP at Leaf-gain ACRs", fontsize=7)
    ax_h.set_title("H — SHAP asymmetry: proto-gain vs leaf-gain",
                    fontsize=9, fontweight="bold")

    # Panel I: Permutation importance (WRKY highlighted)
    ax_i = fig.add_subplot(gs[1, 1])
    perm_path = os.path.join(args.v08_dir, "all",
                              "family_permutation_importance.tsv")
    if os.path.exists(perm_path):
        perm = pd.read_csv(perm_path, sep="\t")
        perm["family"] = perm["family"].map(rename_family)
        perm = perm.sort_values("perm_importance_mean", ascending=True)
        n_fam = len(perm)

        colors = ["#8B0000" if f == WRKY_FAMILY else "#4682B4"
                  for f in perm["family"]]
        ax_i.barh(range(n_fam), perm["perm_importance_mean"].values,
                  color=colors, edgecolor="black", linewidth=0.3)
        ax_i.set_yticks(range(n_fam))
        ax_i.set_yticklabels(perm["family"].values, fontsize=5)
        for tick_label in ax_i.get_yticklabels():
            if tick_label.get_text() == WRKY_FAMILY:
                tick_label.set_fontweight("bold")
                tick_label.set_color("#8B0000")
        ax_i.set_xlabel("Permutation importance (R² drop)", fontsize=7)
    ax_i.set_title("I — Family permutation importance (WRKY highlighted)",
                    fontsize=9, fontweight="bold")

    fig.suptitle("Page 3 — WRKY in Context", fontsize=12,
                 fontweight="bold", y=0.98)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ── Page 4: Concordant & contrasting families (v3_08 + v3_09) ────────────────

def page4_interactions(pdf, args, sig_meta):
    """Families concordant/contrasting with WRKY from SHAP correlations + v3_09."""
    nature_figure_defaults()
    fig = plt.figure(figsize=(16, 14))
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.3)

    # ── Compute per-ACR family SHAP correlation with WRKY ────────────────
    t1_data = _load_shap_t1(
        os.path.join(args.v08_dir, "all", "raw_shap_tier1.npz"), sig_meta)

    corr_df = None
    if t1_data is not None:
        shap_fam = t1_data["shap_by_family"]
        fam_names = t1_data["family_names"]
        acr_class = t1_data["acr_class"]

        wrky_j = None
        for j, f in enumerate(fam_names):
            if f == WRKY_FAMILY:
                wrky_j = j
                break

        if wrky_j is not None:
            wrky_shap = shap_fam[:, wrky_j]
            corr_rows = []
            for j, fam in enumerate(fam_names):
                if fam == WRKY_FAMILY:
                    continue
                r, p = stats.spearmanr(wrky_shap, shap_fam[:, j])
                corr_rows.append({"family": fam, "spearman_r": r,
                                  "pvalue": p})
            corr_df = pd.DataFrame(corr_rows).sort_values(
                "spearman_r", ascending=False)

    # Panel J: Bar chart of SHAP correlation with WRKY (all families)
    ax_j = fig.add_subplot(gs[0, :])
    if corr_df is not None:
        corr_sorted = corr_df.sort_values("spearman_r", ascending=True)
        n = len(corr_sorted)
        colors = []
        for r in corr_sorted["spearman_r"]:
            if r > 0.1:
                colors.append("#D64045")  # concordant
            elif r < -0.1:
                colors.append("#3A7D44")  # contrasting
            else:
                colors.append("#AAAAAA")  # neutral
        ax_j.barh(range(n), corr_sorted["spearman_r"].values,
                  color=colors, edgecolor="black", linewidth=0.3)
        ax_j.set_yticks(range(n))
        ax_j.set_yticklabels(corr_sorted["family"].values, fontsize=5.5)
        ax_j.axvline(0, color="black", lw=0.5)
        ax_j.set_xlabel("Spearman r (per-ACR signed SHAP vs WRKY)", fontsize=8)
        ax_j.set_title("J — Family SHAP correlation with WRKY across ACRs\n"
                        "Red = concordant (same direction), "
                        "Green = contrasting (opposite direction)",
                        fontsize=9, fontweight="bold")

    # Panel K: Top concordant families — class-stratified SHAP comparison
    ax_k = fig.add_subplot(gs[1, 0])
    if corr_df is not None and t1_data is not None:
        top_conc = corr_df.nlargest(5, "spearman_r")["family"].tolist()
        families_to_plot = [WRKY_FAMILY] + top_conc

        bar_data = []
        for fam in families_to_plot:
            j = fam_names.index(fam) if fam in fam_names else None
            if j is None:
                continue
            for cls in CLASS_ORDER:
                mask = acr_class == cls
                if mask.sum() > 0:
                    bar_data.append({
                        "family": fam, "acr_class": cls,
                        "mean_shap": float(shap_fam[mask, j].mean())})

        if bar_data:
            bdf = pd.DataFrame(bar_data)
            fam_list = families_to_plot
            y_pos = np.arange(len(fam_list))
            bar_h = 0.25
            for ci, cls in enumerate(CLASS_ORDER):
                sub = bdf[bdf["acr_class"] == cls].set_index("family")
                vals = [sub.loc[f, "mean_shap"] if f in sub.index else 0
                        for f in fam_list]
                ax_k.barh(y_pos + (ci - 1) * bar_h, vals, height=bar_h,
                          color=ACR_CLASS_COLORS[cls], edgecolor="white",
                          linewidth=0.3, label=CLASS_LABELS[cls] if ci < 3 else "")
            ax_k.set_yticks(y_pos)
            ax_k.set_yticklabels(fam_list, fontsize=7)
            for tick_label in ax_k.get_yticklabels():
                if tick_label.get_text() == WRKY_FAMILY:
                    tick_label.set_fontweight("bold")
                    tick_label.set_color("#8B0000")
            ax_k.axvline(0, color="black", lw=0.5)
            ax_k.legend(fontsize=6)
            ax_k.set_xlabel("Mean signed SHAP", fontsize=7)
    ax_k.set_title("K — Top concordant families (+ WRKY)",
                    fontsize=9, fontweight="bold")

    # Panel L: Top contrasting families — class-stratified SHAP comparison
    ax_l = fig.add_subplot(gs[1, 1])
    if corr_df is not None and t1_data is not None:
        top_contrast = corr_df.nsmallest(5, "spearman_r")["family"].tolist()
        families_to_plot = [WRKY_FAMILY] + top_contrast

        bar_data = []
        for fam in families_to_plot:
            j = fam_names.index(fam) if fam in fam_names else None
            if j is None:
                continue
            for cls in CLASS_ORDER:
                mask = acr_class == cls
                if mask.sum() > 0:
                    bar_data.append({
                        "family": fam, "acr_class": cls,
                        "mean_shap": float(shap_fam[mask, j].mean())})

        if bar_data:
            bdf = pd.DataFrame(bar_data)
            fam_list = families_to_plot
            y_pos = np.arange(len(fam_list))
            bar_h = 0.25
            for ci, cls in enumerate(CLASS_ORDER):
                sub = bdf[bdf["acr_class"] == cls].set_index("family")
                vals = [sub.loc[f, "mean_shap"] if f in sub.index else 0
                        for f in fam_list]
                ax_l.barh(y_pos + (ci - 1) * bar_h, vals, height=bar_h,
                          color=ACR_CLASS_COLORS[cls], edgecolor="white",
                          linewidth=0.3, label=CLASS_LABELS[cls] if ci < 3 else "")
            ax_l.set_yticks(y_pos)
            ax_l.set_yticklabels(fam_list, fontsize=7)
            for tick_label in ax_l.get_yticklabels():
                if tick_label.get_text() == WRKY_FAMILY:
                    tick_label.set_fontweight("bold")
                    tick_label.set_color("#8B0000")
            ax_l.axvline(0, color="black", lw=0.5)
            ax_l.legend(fontsize=6)
            ax_l.set_xlabel("Mean signed SHAP", fontsize=7)
    ax_l.set_title("L — Top contrasting families (+ WRKY)",
                    fontsize=9, fontweight="bold")

    # Panel M: v3_09 WRKY pairwise interactions — top interacting signatures
    ax_m = fig.add_subplot(gs[2, :])
    top_int_path = os.path.join(args.v09_dir, "all", "top_interactions.tsv")
    if os.path.exists(top_int_path):
        ti = pd.read_csv(top_int_path, sep="\t")
        # Get all pairs involving WRKY
        wrky_pairs = ti[(ti["family_i"] == "WRKY") |
                        (ti["family_j"] == "WRKY")].copy()

        if not wrky_pairs.empty:
            # Create partner label
            def partner_label(row):
                if row["family_i"] == "WRKY":
                    return f"{row['sig_j']} ({row['family_j']})"
                return f"{row['sig_i']} ({row['family_i']})"

            wrky_pairs["partner"] = wrky_pairs.apply(partner_label, axis=1)
            wrky_top = wrky_pairs.nlargest(20, "mean_abs_interaction")
            wrky_top = wrky_top.sort_values("mean_abs_interaction",
                                             ascending=True)

            n = len(wrky_top)
            colors = ["#D64045" if v > 0 else "#3A7D44"
                      for v in wrky_top["mean_signed_interaction"]]
            ax_m.barh(range(n),
                      wrky_top["mean_signed_interaction"].values,
                      color=colors, edgecolor="black", linewidth=0.3)
            ax_m.set_yticks(range(n))
            ax_m.set_yticklabels(wrky_top["partner"].values, fontsize=6)
            ax_m.axvline(0, color="black", lw=0.5)
            ax_m.set_xlabel("Mean signed SHAP interaction with WRKY", fontsize=8)

            # Add |interaction| on bars
            for i, (_, row) in enumerate(wrky_top.iterrows()):
                ax_m.text(row["mean_signed_interaction"], i,
                          f"  |{row['mean_abs_interaction']:.4f}|",
                          va="center", fontsize=5, alpha=0.7)
    ax_m.set_title("M — Top WRKY pairwise interactions (v3_09)\n"
                    "Red = collaborative (+), Green = competitive (−)",
                    fontsize=9, fontweight="bold")

    fig.suptitle("Page 4 — WRKY Interactions: Concordant & Contrasting Families",
                 fontsize=12, fontweight="bold", y=0.99)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ── Page 5: Hypothesis Testing ────────────────────────────────────────────────

def _load_acr_meta(args):
    """Load ACR metadata with native↔resized mapping."""
    meta = pd.read_csv(os.path.join(BASE, args.acr_metadata), sep="\t")
    meta["acr_id"] = meta["acr_id"].astype(str)
    # Build resized → native mapping
    coord_path = os.path.join(BASE, "data", "acr_native_to_resized.tsv")
    if os.path.exists(coord_path):
        cm = pd.read_csv(coord_path, sep="\t")
        meta.attrs["resized_to_native"] = dict(zip(cm["resized_str"], cm["native_str"]))
        meta.attrs["native_to_resized"] = dict(zip(cm["native_str"], cm["resized_str"]))
    else:
        meta.attrs["resized_to_native"] = {}
        meta.attrs["native_to_resized"] = {}
    return meta


def _wrky_shap_mean_tf_scales(npz_path, sig_meta, scale_lo=2, scale_hi=10):
    """Extract mean WRKY SHAP across TF scales (2-10 bp) per test ACR.

    Returns (wrky_shap, acr_class, acr_ids_test) or (None, None, None).
    """
    if not os.path.exists(npz_path):
        return None, None, None
    npz = np.load(npz_path, allow_pickle=True)
    shap_vals = npz["shap_values"]
    feat_names = list(npz["feature_names"])
    acr_class = npz["acr_class_test"]
    acr_ids = npz.get("acr_ids_test")

    if shap_vals.ndim == 3:
        shap_vals = shap_vals.mean(axis=2)

    # Find WRKY features within scale range
    wrky_cols = []
    for fi, fname in enumerate(feat_names):
        if fname.startswith("WRKY_s"):
            try:
                sidx = int(fname.split("_s")[1])
                scale_bp = sidx + 2
                if scale_lo <= scale_bp <= scale_hi:
                    wrky_cols.append(fi)
            except ValueError:
                pass

    if not wrky_cols:
        return None, None, None

    # Mean SHAP across selected scales per ACR
    wrky_shap = shap_vals[:, wrky_cols].mean(axis=1)
    return wrky_shap, acr_class, acr_ids


def page5_hypothesis_testing(pdf, args, sig_meta):
    """Test specific WRKY compaction predictions.

    Revised: violin plots per logFC bin, 98th-pctile clipping, companion
    abs-value panels, SD ribbons on partner mediation.
    """
    nature_figure_defaults()
    fig = plt.figure(figsize=(18, 20))
    gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.45, wspace=0.3)

    acr_meta = _load_acr_meta(args)
    native_class_map = dict(zip(acr_meta["acr_id"], acr_meta["acr_class"]))
    native_logfc_map = dict(zip(acr_meta["acr_id"], acr_meta["edgeR_logFC"]))
    resized_to_native = acr_meta.attrs.get("resized_to_native", {})

    # Load T1 SHAP
    t1_npz_path = os.path.join(args.v08_dir, "all", "raw_shap_tier1.npz")
    t1_data = _load_shap_t1(t1_npz_path, sig_meta)
    acr_ids_test = None
    if os.path.exists(t1_npz_path):
        _t1 = np.load(t1_npz_path, allow_pickle=True)
        acr_ids_test = _t1.get("acr_ids_test")

    # Z-score logFC reference (shared by panels N and O)
    logfc_all_vals = np.array(list(native_logfc_map.values()))
    lfc_mu = np.nanmean(logfc_all_vals)
    lfc_sd = np.nanstd(logfc_all_vals)

    # ── Panel N: WRKY SHAP per signature vs z-scored logFC (stable ACRs) ───
    # Panel N: 2×2 grid — rows = scale group (TF / Nuc), cols = signature
    from matplotlib.gridspec import GridSpecFromSubplotSpec
    from scipy.stats import spearmanr as _sr2
    gs_n = GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[0, 0],
                                   wspace=0.35, hspace=0.45)
    # axes[scale_row][sig_col]
    n_axes = [[fig.add_subplot(gs_n[r, c]) for c in range(2)] for r in range(2)]

    t2_npz_path_n = os.path.join(args.v08_dir, "all", "raw_shap_tier2.npz")
    if os.path.exists(t2_npz_path_n):
        npz2 = np.load(t2_npz_path_n, allow_pickle=True)
        shap2 = npz2["shap_values"]          # (n_test, n_features)
        feat2 = list(npz2["feature_names"])
        acr_class2 = npz2["acr_class_test"]
        acr_ids2 = npz2.get("acr_ids_test")

        if shap2.ndim == 3:
            shap2 = shap2.mean(axis=2)

        # Map test ACR IDs → z-scored logFC
        if acr_ids2 is not None:
            logfc_test2 = np.array([
                native_logfc_map.get(
                    resized_to_native.get(str(aid), str(aid)), np.nan)
                for aid in acr_ids2])
        else:
            logfc_test2 = np.full(shap2.shape[0], np.nan)
        z_logfc2 = (logfc_test2 - lfc_mu) / lfc_sd

        stable_mask2 = (acr_class2 == "stable") & np.isfinite(z_logfc2)

        # Scale groups: feature name encodes scale as sig_id_s{idx}, scale_bp = idx+2
        scale_groups = [
            ("TF (<10 bp)",  lambda idx: idx + 2 <= 10,  0),
            ("Nuc (>80 bp)", lambda idx: idx + 2 >= 80,  1),
        ]

        for sig_col, sig_id in enumerate(["sig_121", "sig_122"]):
            for scale_label, scale_fn, scale_row in scale_groups:
                ax_n = n_axes[scale_row][sig_col]

                # Select feature columns for this sig + scale group
                sig_cols = []
                for fi, fn in enumerate(feat2):
                    if fn.startswith(f"{sig_id}_s"):
                        try:
                            idx = int(fn.split("_s")[1])
                            if scale_fn(idx):
                                sig_cols.append(fi)
                        except ValueError:
                            pass

                if not sig_cols:
                    ax_n.set_title(f"N — {sig_id}\n{scale_label}\n(no cols)",
                                   fontsize=7, fontweight="bold")
                    continue

                sig_shap = shap2[:, sig_cols].mean(axis=1)

                lf = z_logfc2[stable_mask2]
                ss = sig_shap[stable_mask2]
                valid_n = np.isfinite(ss)
                lf, ss = lf[valid_n], ss[valid_n]

                # Winsorize to 99th percentile of |value|
                y_clip = np.percentile(np.abs(ss), 99)
                ss_w = np.clip(ss, -0.0003, 0.0001)
                x_clip = np.percentile(np.abs(lf), 99)
                lf_w = np.clip(lf, -3, 3)

                if len(lf_w) > 10:
                    hb = ax_n.hexbin(lf_w, ss_w, gridsize=25, cmap="YlOrRd",
                                     mincnt=1, linewidths=0.2)
                    plt.colorbar(hb, ax=ax_n, label="count", shrink=0.8)

                    r, p = _sr2(lf, ss)   # Spearman on original
                    z_fit = np.polyfit(lf_w, ss_w, 1)
                    x_ln = np.linspace(lf_w.min(), lf_w.max(), 100)
                    ax_n.plot(x_ln, np.polyval(z_fit, x_ln), "k-", lw=1.2)
                    p_str = f"p={p:.2f}" if p >= 0.001 else f"p={p:.1e}"
                    ax_n.text(0.05, 0.95, f"r={r:.3f}\n{p_str}",
                              transform=ax_n.transAxes, fontsize=6, va="top")

                ax_n.axhline(0, color="gray", lw=0.5, ls="--")
                ax_n.axvline(0, color="gray", lw=0.3, ls=":")
                ax_n.set_xlabel("z-scored logFC\n(stable ACRs)", fontsize=7)
                ax_n.set_ylabel("Mean signed SHAP", fontsize=7)
                ax_n.set_title(f"N — {sig_id} {scale_label}\nvs z-logFC (stable)",
                               fontsize=7, fontweight="bold")

    # ── Panel O: z-scored delta (TF + nuc scales) vs z-scored logFC ──────
    # Initialise variables shared with O2 concordance panel
    delta_tf = delta_nuc = None
    valid_o = acr_cls_06 = np.array([])
    wrky_fi = None

    # Split gs[0,1] into two sub-panels: O_TF (<10 bp) and O_nuc (>80 bp)
    gs_o = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0, 1], wspace=0.35)
    ax_o_tf  = fig.add_subplot(gs_o[0, 0])
    ax_o_nuc = fig.add_subplot(gs_o[0, 1])

    delta_path = os.path.join(args.v06_dir, "delta_acr_family_scale.npz")
    if os.path.exists(delta_path):
        npz06 = np.load(delta_path, allow_pickle=True)
        delta06 = npz06["delta"]
        fam_ids_06 = list(npz06["family_ids"])
        acr_ids_06 = list(npz06["acr_ids"])
        scales_06 = npz06["scales"]

        wrky_fi = None
        for i, f in enumerate(fam_ids_06):
            if rename_family(f) == WRKY_FAMILY:
                wrky_fi = i
                break

        if wrky_fi is not None:
            wrky_delta06 = delta06[:, wrky_fi, :]  # (n_acrs, n_scales)

            # Residualize + z-score per scale (same pipeline as page 1)
            acr_meta_res = acr_meta.copy()
            if "width" in acr_meta_res.columns and "acr_width" not in acr_meta_res.columns:
                acr_meta_res["acr_width"] = acr_meta_res["width"]
            if "edgeR_logCPM" in acr_meta_res.columns and "logCPM" not in acr_meta_res.columns:
                acr_meta_res["logCPM"] = acr_meta_res["edgeR_logCPM"]
            native_to_resized = {v: k for k, v in resized_to_native.items()}
            acr_meta_res["resized_str"] = acr_meta_res["acr_id"].map(native_to_resized)
            acr_meta_res = acr_meta_res.dropna(subset=["resized_str"]).set_index("resized_str")

            wrky_df06 = pd.DataFrame(
                wrky_delta06, index=acr_ids_06,
                columns=[f"s{si}" for si in range(len(scales_06))])
            wrky_resid06 = residualize_features(wrky_df06, acr_meta_res)
            wrky_z06 = wrky_resid06.copy()
            for col in wrky_z06.columns:
                mu = wrky_z06[col].mean(); sd = wrky_z06[col].std()
                if sd > 1e-12:
                    wrky_z06[col] = (wrky_z06[col] - mu) / sd

            # TF (<10 bp) and nuc (>80 bp) scale masks
            tf_mask06  = scales_06 <= 10
            nuc_mask06 = scales_06 >= 80
            tf_cols   = [f"s{si}" for si in range(len(scales_06)) if tf_mask06[si]]
            nuc_cols  = [f"s{si}" for si in range(len(scales_06)) if nuc_mask06[si]]

            delta_tf  = wrky_z06[tf_cols].mean(axis=1).values  if tf_cols  else None
            delta_nuc = wrky_z06[nuc_cols].mean(axis=1).values if nuc_cols else None

            # z-scored logFC for these ACRs
            acr_cls_06 = np.array([
                native_class_map.get(
                    resized_to_native.get(str(a), str(a)), "stable")
                for a in wrky_z06.index])
            logfc_06 = np.array([
                native_logfc_map.get(
                    resized_to_native.get(str(a), str(a)), np.nan)
                for a in wrky_z06.index])
            z_logfc_06 = (logfc_06 - lfc_mu) / lfc_sd

            from scipy.stats import spearmanr as _sr
            for ax_o, d_arr, scale_label in [
                    (ax_o_tf,  delta_tf,  "TF scale (<10 bp)"),
                    (ax_o_nuc, delta_nuc, "Nuc scale (>80 bp)")]:
                if d_arr is None:
                    continue
                valid_o = np.isfinite(z_logfc_06) & np.isfinite(d_arr)

                # Winsorize y to 99th percentile of |delta|
                y_clip_o = np.percentile(np.abs(d_arr[valid_o]), 99)
                d_arr_w = np.where(valid_o,
                                   np.clip(d_arr, -1.5, 1.5),
                                   np.nan)
                # Winsorize x to 99th percentile of |z_logfc|
                x_clip_o = np.percentile(np.abs(z_logfc_06[valid_o]), 99)
                x_arr_w = np.clip(z_logfc_06, -3, 3)

                for cls in CLASS_ORDER:
                    m = valid_o & (acr_cls_06 == cls)
                    if m.sum() < 5:
                        continue
                    ax_o.scatter(x_arr_w[m], d_arr_w[m], s=2, alpha=0.12,
                                 color=ACR_CLASS_COLORS[cls],
                                 label=CLASS_LABELS[cls], rasterized=True)
                if valid_o.sum() > 10:
                    r, p = _sr(z_logfc_06[valid_o], d_arr[valid_o])  # Spearman on original
                    valid_w = valid_o & np.isfinite(d_arr_w)
                    z_fit = np.polyfit(x_arr_w[valid_w], d_arr_w[valid_w], 1)
                    x_ln = np.linspace(x_arr_w[valid_w].min(),
                                       x_arr_w[valid_w].max(), 100)
                    ax_o.plot(x_ln, np.polyval(z_fit, x_ln), "k-", lw=1.5)
                    ax_o.text(0.05, 0.95, f"r={r:.3f}\np={p:.1e}",
                              transform=ax_o.transAxes, fontsize=7, va="top")
                ax_o.axhline(0, color="gray", lw=0.3, ls="--")
                ax_o.axvline(0, color="gray", lw=0.3, ls=":")
                ax_o.set_xlabel("z-scored edgeR logFC", fontsize=8)
                ax_o.set_ylabel("Mean z-scored WRKY delta\n(resid., leaf−proto)",
                                fontsize=8)
                ax_o.set_title(f"O — {scale_label}\nvs z-scored logFC",
                               fontsize=8, fontweight="bold")
                ax_o.legend(fontsize=5, loc="lower right", markerscale=4)
    # fallback titles if data missing
    for ax_o, lbl in [(ax_o_tf, "TF scale (<10 bp)"), (ax_o_nuc, "Nuc scale (>80 bp)")]:
        if not ax_o.get_title():
            ax_o.set_title(f"O — {lbl}\nvs z-scored logFC",
                           fontsize=8, fontweight="bold")

    # ── Panel O2: ±delta sign proportions (nuc vs TF scale) ─────────────
    ax_o2 = fig.add_subplot(gs[2, 0])

    # Quadrant colors: full saturation = concordant, light = discordant
    # leaf_gain=#3A7D44 (green), proto_gain=#D64045 (red)
    # Stacking always: delta<0 at bottom (lower quadrant), delta>0 at top (upper quadrant)
    # TF scale concordant: leaf-gain→upper-left (delta>0), proto-gain→lower-right (delta<0)
    # Nuc scale concordant: leaf-gain→lower-left (delta<0), proto-gain→upper-right (delta>0)
    _QUAD_TF = {
        # (class, segment) → (color, quadrant_label, is_concordant)
        ("leaf_gain",  "lower"): ("#A0C8A6", "lower-left",    False),
        ("leaf_gain",  "upper"): ("#3A7D44", "upper-left ✓",  True),
        ("proto_gain", "lower"): ("#D64045", "lower-right ✓", True),
        ("proto_gain", "upper"): ("#F0A8A9", "upper-right",   False),
    }
    _QUAD_NUC = {
        ("leaf_gain",  "lower"): ("#3A7D44", "lower-left ✓",  True),
        ("leaf_gain",  "upper"): ("#A0C8A6", "upper-left",    False),
        ("proto_gain", "lower"): ("#F0A8A9", "lower-right",   False),
        ("proto_gain", "upper"): ("#D64045", "upper-right ✓", True),
    }
    _LIGHT_COLS = {"#A0C8A6", "#F0A8A9"}  # light colors → dark text

    def _draw_o2_sign(ax, dt, dn, v_o, cls_arr):
        from scipy.stats import fisher_exact as _fe
        from matplotlib.patches import Patch as _Patch
        # leaf-gain bars on left (matches scatter left side), proto-gain on right
        cls_order = [("leaf_gain",  CLASS_LABELS["leaf_gain"]),
                     ("proto_gain", CLASS_LABELS["proto_gain"])]
        scale_list = [("TF (<10 bp)", dt, _QUAD_TF),
                      ("Nuc (>80 bp)", dn, _QUAD_NUC)]
        x_pos = 0
        xtick_pos, xtick_lbl, conting = [], [], {}
        for scale_label, d_arr, qcols in scale_list:
            group_start = x_pos
            for cls, cls_label in cls_order:
                m = v_o & (cls_arr == cls) & np.isfinite(d_arr)
                if m.sum() < 5:
                    x_pos += 1; continue
                n_pos   = int((m & (d_arr > 0)).sum())
                n_neg   = int((m & (d_arr < 0)).sum())
                n_total = n_pos + n_neg
                pct_neg = 100.0 * n_neg / n_total
                pct_pos = 100.0 * n_pos / n_total

                col_lo, lbl_lo, _ = qcols[(cls, "lower")]
                col_hi, lbl_hi, _ = qcols[(cls, "upper")]

                # Bottom = delta<0 (lower quadrant), top = delta>0 (upper quadrant)
                ax.bar(x_pos, pct_neg, color=col_lo,
                       width=0.7, edgecolor="k", lw=0.5)
                ax.bar(x_pos, pct_pos, bottom=pct_neg, color=col_hi,
                       width=0.7, edgecolor="k", lw=0.5)

                # Labels inside segments: quadrant name + %
                tc_lo = "#333333" if col_lo in _LIGHT_COLS else "white"
                tc_hi = "#333333" if col_hi in _LIGHT_COLS else "white"
                if pct_neg > 12:
                    ax.text(x_pos, pct_neg / 2,
                            f"{lbl_lo}\n{pct_neg:.0f}%",
                            ha="center", va="center", fontsize=5.5,
                            fontweight="bold", color=tc_lo)
                if pct_pos > 12:
                    ax.text(x_pos, pct_neg + pct_pos / 2,
                            f"{lbl_hi}\n{pct_pos:.0f}%",
                            ha="center", va="center", fontsize=5.5,
                            fontweight="bold", color=tc_hi)

                ax.text(x_pos, 101, f"n={n_total:,}", ha="center",
                        va="bottom", fontsize=5.5)
                conting.setdefault(scale_label, {})[cls] = [n_pos, n_neg]
                xtick_pos.append(x_pos)
                xtick_lbl.append(f"{cls_label}\n{scale_label}")
                x_pos += 1
            x_pos += 0.5

            if ("leaf_gain" in conting.get(scale_label, {}) and
                    "proto_gain" in conting.get(scale_label, {})):
                t = conting[scale_label]
                table = [[t["leaf_gain"][0],  t["leaf_gain"][1]],
                         [t["proto_gain"][0], t["proto_gain"][1]]]
                _, fp = _fe(table)
                mid_x = (group_start + x_pos - 0.5) / 2
                ax.annotate(f"Fisher p={fp:.2e}", xy=(mid_x, 96),
                            ha="center", va="top", fontsize=6,
                            bbox=dict(boxstyle="round,pad=0.2",
                                      fc="white", alpha=0.8, lw=0.5))

        ax.axhline(50, color="black", lw=0.8, ls="--", alpha=0.4)
        ax.set_ylim(0, 108)
        ax.set_xticks(xtick_pos)
        ax.set_xticklabels(xtick_lbl, fontsize=6.5)
        ax.set_ylabel("% of WRKY hits", fontsize=8)
        ax.legend(handles=[
            _Patch(facecolor="#3A7D44", label="Leaf-gain concordant"),
            _Patch(facecolor="#A0C8A6", label="Leaf-gain discordant"),
            _Patch(facecolor="#D64045", label="Proto-gain concordant"),
            _Patch(facecolor="#F0A8A9", label="Proto-gain discordant"),
        ], fontsize=6, loc="lower right", ncol=2)
        ax.set_title("O2 — WRKY delta sign by ACR class\n"
                     "(quadrant labels match scatter plot; ✓ = concordant)",
                     fontsize=8, fontweight="bold")

    if (os.path.exists(delta_path) and wrky_fi is not None
            and delta_tf is not None and delta_nuc is not None):
        _draw_o2_sign(ax_o2, delta_tf, delta_nuc, valid_o, acr_cls_06)
    else:
        ax_o2.set_title("O2 — data unavailable", fontsize=8)

    # ── Panel P: Squelching — leaf vs proto FP separately ────────────────
    ax_p = fig.add_subplot(gs[1, 0])
    chunk_dir = os.path.join(args.v06_dir, "chunks")
    if os.path.isdir(chunk_dir):
        import glob
        chunk_files = sorted(glob.glob(
            os.path.join(chunk_dir, "per_hit_fp_chunk_*.npz")))

        # Collect WRKY per-hit leaf/proto FP at TF scale
        wrky_sigs = {"sig_121", "sig_122"}
        leaf_fps, proto_fps, hit_classes = [], [], []
        resized_to_native = acr_meta.attrs.get("resized_to_native", {})

        for cf in chunk_files[:50]:  # all chunks
            try:
                cnpz = np.load(cf, allow_pickle=True)
            except Exception:
                continue
            mids = cnpz.get("motif_ids")
            if mids is None:
                continue
            mids = list(mids)
            regions = list(cnpz.get("region_strs", []))
            fp = cnpz.get("fp_values")  # (n_hits, n_scales, n_samples)
            scales_c = cnpz.get("scales")
            if fp is None or scales_c is None:
                continue

            # TF scale: ~5-10 bp
            tf_idx = np.where((scales_c >= 5) & (scales_c <= 10))[0]
            if len(tf_idx) == 0:
                continue

            wrky_mask = np.isin(mids, list(wrky_sigs))
            if wrky_mask.sum() == 0:
                continue

            fp_tf = np.nanmean(fp[wrky_mask][:, tf_idx, :], axis=1)
            # samples: [leaf_rep1, leaf_rep2, proto_rep1, proto_rep2]
            leaf_fp = np.nanmean(fp_tf[:, :2], axis=1)
            proto_fp = np.nanmean(fp_tf[:, 2:], axis=1)

            # Map regions to ACR class
            for i, r in enumerate(np.array(regions)[wrky_mask]):
                native = resized_to_native.get(str(r), str(r))
                cls = native_class_map.get(native, "stable")
                hit_classes.append(cls)

            leaf_fps.extend(leaf_fp.tolist())
            proto_fps.extend(proto_fp.tolist())

        if len(leaf_fps) > 100:
            leaf_fps = np.array(leaf_fps)
            proto_fps = np.array(proto_fps)
            hit_classes = np.array(hit_classes)

            # Filter finite
            valid = np.isfinite(leaf_fps) & np.isfinite(proto_fps)

            positions = []
            vdata_leaf = []
            vdata_proto = []
            labels = []
            pos_idx = 0
            for cls in CLASS_ORDER:
                m = valid & (hit_classes == cls)
                if m.sum() < 10:
                    continue
                vdata_leaf.append(leaf_fps[m])
                vdata_proto.append(proto_fps[m])
                positions.append(pos_idx)
                labels.append(f"{CLASS_LABELS[cls]}\n(n={m.sum():,})")
                pos_idx += 1

            if positions:
                width = 0.35
                for i, pos in enumerate(positions):
                    parts_l = ax_p.violinplot(
                        [vdata_leaf[i]], positions=[pos - width / 2],
                        widths=width, showextrema=False, showmedians=True)
                    for pc in parts_l["bodies"]:
                        pc.set_facecolor("#2166AC")
                        pc.set_alpha(0.6)
                    parts_l["cmedians"].set_color("black")

                    parts_p = ax_p.violinplot(
                        [vdata_proto[i]], positions=[pos + width / 2],
                        widths=width, showextrema=False, showmedians=True)
                    for pc in parts_p["bodies"]:
                        pc.set_facecolor("#B2182B")
                        pc.set_alpha(0.6)
                    parts_p["cmedians"].set_color("black")

                ax_p.set_xticks(positions)
                ax_p.set_xticklabels(labels, fontsize=7)
                # Manual legend
                from matplotlib.patches import Patch
                ax_p.legend(handles=[
                    Patch(facecolor="#2166AC", alpha=0.6, label="Leaf FP"),
                    Patch(facecolor="#B2182B", alpha=0.6, label="Proto FP")],
                    fontsize=7)
                ax_p.set_ylabel("WRKY FP depth (5-10 bp scale)", fontsize=8)
    ax_p.set_title("P — WRKY FP depth: leaf vs proto (squelching test)\n"
                    "If squelching: proto lower despite higher WRKY expression",
                    fontsize=9, fontweight="bold")

    # ── Panel Q: Partner TF mediation ────────────────────────────────────
    ax_q = fig.add_subplot(gs[1, 1])
    if t1_data is not None and acr_ids_test is not None:
        fam_names = t1_data["family_names"]
        shap_fam = t1_data["shap_by_family"]
        acr_class = t1_data["acr_class"]

        logfc_test = np.array([
            native_logfc_map.get(
                resized_to_native.get(str(aid), str(aid)), np.nan)
            for aid in acr_ids_test])
        stable_mask = (acr_class == "stable") & np.isfinite(logfc_test)

        # Families to compare: WRKY + top concordant
        compare_fams = [WRKY_FAMILY, "NAC", "ERF/DREB", "FRS/FRF", "bZIP"]
        colors_q = ["#8B0000", "#2166AC", "#4DAF4A", "#FF7F00", "#984EA3"]

        from scipy.stats import spearmanr
        for fi, fam in enumerate(compare_fams):
            if fam not in fam_names:
                continue
            j = fam_names.index(fam)
            fam_shap = shap_fam[stable_mask, j]
            lf = logfc_test[stable_mask]
            valid = np.isfinite(fam_shap) & np.isfinite(lf)
            if valid.sum() < 20:
                continue
            r, p = spearmanr(lf[valid], fam_shap[valid])
            z = np.polyfit(lf[valid], fam_shap[valid], 1)
            x_line = np.linspace(lf[valid].min(), lf[valid].max(), 50)
            lw = 2.5 if fam == WRKY_FAMILY else 1.2
            ax_q.plot(x_line, np.polyval(z, x_line),
                      color=colors_q[fi], lw=lw,
                      label=f"{fam} (r={r:.2f})")

        ax_q.axhline(0, color="gray", lw=0.3, ls="--")
        ax_q.set_xlabel("edgeR logFC (stable ACRs)", fontsize=8)
        ax_q.set_ylabel("Signed SHAP (sum across scales)", fontsize=8)
        ax_q.legend(fontsize=6)
    ax_q.set_title("Q — Partner mediation: SHAP-logFC slopes\n"
                    "(similar slope = concordant prediction pattern)",
                    fontsize=9, fontweight="bold")

    # ── Save individual Page 5 panels ────────────────────────────────────
    p5dir = os.path.join(args.outdir, "page5_panels")
    os.makedirs(p5dir, exist_ok=True)
    print("    Saving individual Page 5 panels...", flush=True)

    # ── N panels: SHAP vs z-logFC hexbins (shared x/y limits) ────────────
    if os.path.exists(t2_npz_path_n):
        # Compute shared clip limits across all sig × scale combinations
        all_lf_w, all_ss_w = [], []
        for sig_id in ["sig_121", "sig_122"]:
            sig_cols = [fi for fi, fn in enumerate(feat2)
                        if fn.startswith(f"{sig_id}_s")]
            if not sig_cols:
                continue
            sig_shap = shap2[:, sig_cols].mean(axis=1)
            valid_n = np.isfinite(z_logfc2) & np.isfinite(sig_shap) & stable_mask2
            lf = z_logfc2[valid_n]; ss = sig_shap[valid_n]
            all_lf_w.append(np.clip(lf, -np.percentile(np.abs(lf), 99),
                                         np.percentile(np.abs(lf), 99)))
            all_ss_w.append(np.clip(ss, -np.percentile(np.abs(ss), 99),
                                         np.percentile(np.abs(ss), 99)))

        if all_lf_w:
            lf_all = np.concatenate(all_lf_w)
            ss_all = np.concatenate(all_ss_w)
            x_lim_n = (lf_all.min(), lf_all.max())
            y_lim_n = (ss_all.min(), ss_all.max())
            vmax_n = max(np.percentile(np.abs(lf_all), 98),
                         np.percentile(np.abs(ss_all), 98))

            scale_groups_p5 = [
                ("TF (<10 bp)",  lambda idx: idx + 2 <= 10),
                ("Nuc (>80 bp)", lambda idx: idx + 2 >= 80),
            ]
            for sig_id in ["sig_121", "sig_122"]:
                sig_cols_all = [fi for fi, fn in enumerate(feat2)
                                if fn.startswith(f"{sig_id}_s")]
                if not sig_cols_all:
                    continue
                for scale_label, scale_fn in scale_groups_p5:
                    sig_cols = [fi for fi, fn in enumerate(feat2)
                                if fn.startswith(f"{sig_id}_s")
                                and scale_fn(int(fn.split("_s")[1]))]
                    if not sig_cols:
                        continue
                    sig_shap = shap2[:, sig_cols].mean(axis=1)
                    valid_n = (np.isfinite(z_logfc2) & np.isfinite(sig_shap)
                               & stable_mask2)
                    lf = z_logfc2[valid_n]; ss = sig_shap[valid_n]
                    lf_w = np.clip(lf, x_lim_n[0], x_lim_n[1])
                    ss_w = np.clip(ss, y_lim_n[0], y_lim_n[1])

                    fi = plt.figure(figsize=(4, 4))
                    axi = fi.add_subplot(111)
                    if len(lf_w) > 10:
                        hb = axi.hexbin(lf_w, ss_w, gridsize=25,
                                        cmap="YlOrRd", mincnt=1,
                                        linewidths=0.2,
                                        extent=[x_lim_n[0], x_lim_n[1],
                                                y_lim_n[0], y_lim_n[1]])
                        plt.colorbar(hb, ax=axi, label="count", shrink=0.8)
                        from scipy.stats import spearmanr as _sr_p5
                        r, p = _sr_p5(lf, ss)
                        z_fit = np.polyfit(lf_w, ss_w, 1)
                        x_ln = np.linspace(x_lim_n[0], x_lim_n[1], 100)
                        axi.plot(x_ln, np.polyval(z_fit, x_ln), "k-", lw=1.2)
                        p_str = f"p={p:.2f}" if p >= 0.001 else f"p={p:.1e}"
                        axi.text(0.05, 0.95, f"r={r:.3f}\n{p_str}",
                                 transform=axi.transAxes, fontsize=7, va="top")
                    axi.set_xlim(x_lim_n); axi.set_ylim(y_lim_n)
                    axi.axhline(0, color="gray", lw=0.5, ls="--")
                    axi.axvline(0, color="gray", lw=0.3, ls=":")
                    axi.set_xlabel("z-scored logFC (stable ACRs)", fontsize=8)
                    axi.set_ylabel("Mean signed WRKY SHAP", fontsize=8)
                    axi.set_title(f"N — {sig_id} {scale_label}\nvs z-logFC (stable)",
                                  fontsize=8, fontweight="bold")
                    safe = scale_label.replace(" ", "_").replace("/", "").replace("<","lt").replace(">","gt")
                    _save_panel(fi, p5dir, f"N_{sig_id}_{safe}")

    # ── O panels: z-delta vs z-logFC scatter (shared x/y limits) ─────────
    if delta_tf is not None and delta_nuc is not None:
        # Shared x limits (same z-logFC for both)
        valid_both = (np.isfinite(z_logfc_06) &
                      (np.isfinite(delta_tf) | np.isfinite(delta_nuc)))
        x_clip_o = np.percentile(np.abs(z_logfc_06[valid_both]), 99)
        # Shared y limits across both scale groups
        y_vals = []
        for d_arr in [delta_tf, delta_nuc]:
            v = np.isfinite(d_arr) & valid_both
            if v.sum() > 0:
                yc = np.percentile(np.abs(d_arr[v]), 99)
                y_vals.append(yc)
        y_clip_o = max(y_vals) if y_vals else 2.0
        xlim_o = (-x_clip_o, x_clip_o)
        ylim_o = (-y_clip_o, y_clip_o)

        from scipy.stats import spearmanr as _sr_o
        for d_arr, scale_label, fname in [
                (delta_tf,  "TF scale (<10 bp)", "O_TF"),
                (delta_nuc, "Nuc scale (>80 bp)", "O_Nuc")]:
            valid_o_i = np.isfinite(z_logfc_06) & np.isfinite(d_arr)
            x_w = np.clip(z_logfc_06, -x_clip_o, x_clip_o)
            d_w = np.where(valid_o_i,
                           np.clip(d_arr, -y_clip_o, y_clip_o), np.nan)

            fi = plt.figure(figsize=(5, 4))
            axi = fi.add_subplot(111)
            for cls in CLASS_ORDER:
                m = valid_o_i & (acr_cls_06 == cls)
                if m.sum() < 5:
                    continue
                axi.scatter(x_w[m], d_w[m], s=2, alpha=0.12,
                            color=ACR_CLASS_COLORS[cls],
                            label=CLASS_LABELS[cls], rasterized=True)
            if valid_o_i.sum() > 10:
                r, p = _sr_o(z_logfc_06[valid_o_i], d_arr[valid_o_i])
                valid_w = valid_o_i & np.isfinite(d_w)
                z_fit = np.polyfit(x_w[valid_w], d_w[valid_w], 1)
                x_ln = np.linspace(xlim_o[0], xlim_o[1], 100)
                axi.plot(x_ln, np.polyval(z_fit, x_ln), "k-", lw=1.5)
                axi.text(0.05, 0.95, f"r={r:.3f}\np={p:.1e}",
                         transform=axi.transAxes, fontsize=7, va="top")
            axi.set_xlim(xlim_o); axi.set_ylim(ylim_o)
            axi.axhline(0, color="gray", lw=0.3, ls="--")
            axi.axvline(0, color="gray", lw=0.3, ls=":")
            axi.set_xlabel("z-scored edgeR logFC", fontsize=8)
            axi.set_ylabel("Mean z-scored WRKY delta\n(resid., leaf−proto)",
                           fontsize=8)
            axi.legend(fontsize=6, loc="lower right", markerscale=3)
            axi.set_title(f"O — {scale_label}\nvs z-scored logFC",
                          fontsize=8, fontweight="bold")
            _save_panel(fi, p5dir, fname)

    # ── O2: ±delta sign proportions (re-uses _draw_o2_sign helper) ───────
    if delta_tf is not None and delta_nuc is not None:
        fi = plt.figure(figsize=(6, 4))
        axi = fi.add_subplot(111)
        _draw_o2_sign(axi, delta_tf, delta_nuc, valid_o, acr_cls_06)
        _save_panel(fi, p5dir, "O2_concordance")

    print(f"    Saved Page 5 panels to {p5dir}", flush=True)

    fig.suptitle("Page 5 — Hypothesis Testing: WRKY Compaction Mechanism",
                 fontsize=12, fontweight="bold", y=0.99)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ── Page 6: Nucleosome-WRKY Spatial Analysis ─────────────────────────────────

def _ward_order(mat, opt_leaf_max=2000):
    """Return Ward-ordered row indices for matrix mat (NaN→row mean before clustering)."""
    from scipy.cluster.hierarchy import linkage, leaves_list, optimal_leaf_ordering
    from scipy.spatial.distance import pdist
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


def page6_nucleosome(pdf, args, sig_meta):
    """DeepTools-style nucleosome heatmap around WRKY hits (leaf top, proto bottom)."""
    from scipy.stats import gaussian_kde
    nature_figure_defaults()

    nuc_npz_path = os.path.join(args.v08_dir, "wrky_nuc_profiles.npz")
    if not os.path.exists(nuc_npz_path):
        print(f"  [Page 6] wrky_nuc_profiles.npz not found — page skipped", flush=True)
        return

    # ── Load ─────────────────────────────────────────────────────────────
    npz = np.load(nuc_npz_path, allow_pickle=True)
    positions_bp     = npz["positions_bp"]           # (1001,) -500..+500
    hit_nuc_leaf_raw  = npz["hit_nuc_leaf_raw"]      # (N, 1001) raw FP depth
    hit_nuc_proto_raw = npz["hit_nuc_proto_raw"]     # (N, 1001) raw FP depth
    # Z-scored used only for the density peak (center signal summary)
    hit_nuc_leaf_z   = npz["hit_nuc_leaf_z"]         # (N, 1001) z-scored
    hit_nuc_proto_z  = npz["hit_nuc_proto_z"]        # (N, 1001) z-scored
    hit_acr_class    = np.array(npz["hit_acr_class"], dtype=str)

    # Position window: full ±500 bp (raw arrays are 99.5% finite)
    pos_win = (positions_bp >= -500) & (positions_bp <= 500)
    pos_x   = positions_bp[pos_win]         # (1001,)

    # Colormap: sequential for raw FP depth (non-negative values 0–~1)
    VMAX   = np.nanpercentile(
        np.concatenate([hit_nuc_leaf_raw.ravel(), hit_nuc_proto_raw.ravel()]),
        98)
    CMAP   = "Blues"
    BLOCKS = ["proto_gain", "leaf_gain"]    # top block first

    # ── Build sorted matrices per block (Ward on leaf raw profile) ───────
    print("  [Page 6] Clustering WRKY hits...", flush=True)
    leaf_blocks, proto_blocks = [], []
    block_n = {}
    for cls in BLOCKS:
        mask = hit_acr_class == cls
        n    = mask.sum()
        block_n[cls] = n
        if n == 0:
            leaf_blocks.append(np.empty((0, pos_win.sum())))
            proto_blocks.append(np.empty((0, pos_win.sum())))
            continue
        lmat = hit_nuc_leaf_raw[mask][:, pos_win]
        pmat = hit_nuc_proto_raw[mask][:, pos_win]
        order = _ward_order(lmat, opt_leaf_max=2000)
        leaf_blocks.append(lmat[order])
        proto_blocks.append(pmat[order])
        print(f"    {cls}: {n:,} hits", flush=True)

    leaf_heat  = np.vstack(leaf_blocks)   # (M, n_pos)
    proto_heat = np.vstack(proto_blocks)  # (M, n_pos)
    n_proto = block_n["proto_gain"]
    n_leaf  = block_n["leaf_gain"]
    M       = n_proto + n_leaf

    if M == 0:
        print("  [Page 6] No proto_gain/leaf_gain hits found — page skipped.", flush=True)
        return

    # ── Figure layout ─────────────────────────────────────────────────────
    # 5 rows: avg_leaf | heatmap_leaf | avg_proto | heatmap_proto | density
    fig = plt.figure(figsize=(8, 20))
    gs  = gridspec.GridSpec(5, 1, figure=fig, hspace=0.04,
                            height_ratios=[0.10, 1.0, 0.10, 1.0, 0.22])

    def _avg_profile_ax(ax, mat, title_str):
        """Draw mean ± SEM profile lines per ACR class block onto ax."""
        row_start = 0
        for cls in BLOCKS:
            n = block_n[cls]
            if n == 0:
                row_start += n
                continue
            prof = np.nanmean(mat[row_start: row_start + n], axis=0)
            sem  = np.nanstd( mat[row_start: row_start + n], axis=0) / np.sqrt(n)
            ax.plot(pos_x, prof, color=ACR_CLASS_COLORS[cls], lw=1.2,
                    label=f"{CLASS_LABELS[cls]} (n={n:,})")
            ax.fill_between(pos_x, prof - sem, prof + sem,
                            color=ACR_CLASS_COLORS[cls], alpha=0.20)
            row_start += n
        ax.axvline(0, color="gray", lw=0.6, ls="--")
        ax.axhline(0, color="gray", lw=0.3, ls=":")
        ax.set_xlim(pos_x[0], pos_x[-1])
        ax.set_xticks([])
        ax.set_ylabel("Mean z-score", fontsize=7)
        ax.legend(fontsize=6, loc="upper right", framealpha=0.7)
        ax.set_title(title_str, fontsize=9, fontweight="bold", pad=3)

    def _heatmap_ax(ax, mat, show_xlabel=False):
        """Draw deeptools-style heatmap with block boundary + labels."""
        cmap_obj = plt.get_cmap(CMAP).copy()
        cmap_obj.set_bad(color="lightgray")
        masked = np.ma.array(mat, mask=~np.isfinite(mat))
        im = ax.imshow(masked, aspect="auto", cmap=cmap_obj,
                       vmin=-VMAX, vmax=VMAX, rasterized=True,
                       extent=[pos_x[0], pos_x[-1], M, 0],
                       interpolation="nearest")
        ax.axvline(0, color="white", lw=0.6, ls="--", alpha=0.7)
        if n_proto > 0 and n_leaf > 0:
            ax.axhline(n_proto, color="black", lw=1.2)
        # Block labels on left margin
        for cls, y_mid in [("proto_gain", n_proto / 2),
                            ("leaf_gain",  n_proto + n_leaf / 2)]:
            if block_n[cls] > 0:
                ax.text(pos_x[0] + 5, y_mid, CLASS_LABELS[cls],
                        fontsize=6, color="white", va="center",
                        fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.15", fc="black",
                                  alpha=0.45, lw=0))
        ax.set_xlim(pos_x[0], pos_x[-1])
        ax.set_yticks([])
        ax.set_ylabel("WRKY hits", fontsize=8)
        if show_xlabel:
            ax.set_xlabel("Distance from WRKY hit center (bp)", fontsize=8)
        else:
            ax.set_xticks([])
        return im

    # ── Leaf: average profile + heatmap ──────────────────────────────────
    _avg_profile_ax(fig.add_subplot(gs[0]), leaf_heat,
                    "Leaf — nucleosome-scale FP around WRKY hits (80–100 bp, z-scored)")
    ax_hl = fig.add_subplot(gs[1])
    im    = _heatmap_ax(ax_hl, leaf_heat)

    # ── Proto: average profile + heatmap ─────────────────────────────────
    _avg_profile_ax(fig.add_subplot(gs[2]), proto_heat,
                    "Proto — nucleosome-scale FP around WRKY hits (80–100 bp, z-scored)")
    _heatmap_ax(fig.add_subplot(gs[3]), proto_heat, show_xlabel=True)

    # Shared colorbar attached to both heatmaps
    fig.colorbar(im, ax=[fig.axes[1], fig.axes[3]],
                 label="Nuc-scale FP (z-score)", fraction=0.025,
                 pad=0.02, aspect=50)

    # ── Density: KDE of nucleosome peak position per hit (raw arrays) ────
    ax_d  = fig.add_subplot(gs[4])
    x_kde = np.linspace(-500, 500, 800)

    for cls in BLOCKS:
        mask = hit_acr_class == cls
        if mask.sum() < 10:
            continue
        color = ACR_CLASS_COLORS[cls]
        label = CLASS_LABELS[cls]
        # peak position = argmax of raw nuc profile in ±300 bp window
        search_win = (positions_bp >= -300) & (positions_bp <= 300)
        lmat_s = hit_nuc_leaf_raw[mask][:, search_win]
        pmat_s = hit_nuc_proto_raw[mask][:, search_win]
        pos_s  = positions_bp[search_win]

        peak_l = pos_s[np.nanargmax(lmat_s, axis=1)]
        peak_p = pos_s[np.nanargmax(pmat_s, axis=1)]

        ax_d.plot(x_kde, gaussian_kde(peak_l, bw_method=0.12)(x_kde),
                  color=color, lw=1.4, ls="-",  label=f"{label} — leaf")
        ax_d.plot(x_kde, gaussian_kde(peak_p, bw_method=0.12)(x_kde),
                  color=color, lw=1.4, ls="--", label=f"{label} — proto")

    ax_d.axvline(0, color="gray", lw=0.6, ls="--")
    ax_d.set_xlim(-500, 500)
    ax_d.set_xlabel("Nucleosome peak position relative to WRKY hit center (bp)",
                    fontsize=8)
    ax_d.set_ylabel("Density", fontsize=8)
    ax_d.legend(fontsize=6, ncol=2, framealpha=0.7)
    ax_d.set_title(
        "Nucleosome peak position distribution (search window ±300 bp)\n"
        "Solid = leaf  |  dashed = proto",
        fontsize=8, fontweight="bold")

    fig.suptitle("Page 6 — WRKY nucleosome landscape (deeptools-style)",
                 fontsize=11, fontweight="bold", y=0.995)

    # ── Save individual panels ────────────────────────────────────────────
    panel_dir = os.path.join(args.outdir, "page6_panels")
    os.makedirs(panel_dir, exist_ok=True)
    panel_axes = {
        "avg_leaf":   fig.axes[0],
        "heatmap_leaf":  fig.axes[1],
        "avg_proto":  fig.axes[2],
        "heatmap_proto": fig.axes[3],
        "density":    ax_d,
    }
    for pname, pax in panel_axes.items():
        ext_fig, ext_ax = plt.subplots(figsize=pax.get_figure().get_size_inches())
        ext_ax.remove()
        # Save the whole figure with only this axis visible using bbox
        bbox = pax.get_tightbbox(fig.canvas.get_renderer())
        if bbox is not None:
            for fmt in ("pdf", "png"):
                fig.savefig(os.path.join(panel_dir, f"page6_{pname}.{fmt}"),
                            bbox_inches=bbox.transformed(
                                fig.dpi_scale_trans.inverted()),
                            dpi=150)
        plt.close(ext_fig)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="v3_08b — WRKY Summary Figure")
    p.add_argument("--v06-dir",
                   default="results/v3_06_perscale_fp")
    p.add_argument("--v07-dir",
                   default="results/v3_07_top_signatures")
    p.add_argument("--v08-dir",
                   default="results/v3_08_gradient_boosting")
    p.add_argument("--v09-dir",
                   default="results/v3_09_shap_interactions")
    p.add_argument("--sig-metadata",
                   default="data/motif_signatures/signature_metadata.tsv")
    p.add_argument("--acr-metadata", default="data/acr_metadata.tsv.gz")
    p.add_argument("--outdir",
                   default="results/v3_08_gradient_boosting")
    return p.parse_args()


def main():
    args = parse_args()

    # Resolve relative paths
    for attr in ("v06_dir", "v07_dir", "v08_dir", "v09_dir",
                 "sig_metadata", "acr_metadata", "outdir"):
        val = getattr(args, attr)
        if not os.path.isabs(val):
            setattr(args, attr, os.path.join(BASE, val))

    os.makedirs(args.outdir, exist_ok=True)
    sig_meta = _load_sig_meta(args.sig_metadata)

    pdf_path = os.path.join(args.outdir, "wrky_summary.pdf")
    print(f"Generating WRKY summary → {pdf_path}", flush=True)

    nature_figure_defaults()

    with PdfPages(pdf_path) as pdf:
        print("  Page 1: Raw WRKY signal...", flush=True)
        page1_raw_signal(pdf, args)

        print("  Page 2: WRKY in the model...", flush=True)
        page2_model(pdf, args, sig_meta)

        print("  Page 3: WRKY in context...", flush=True)
        page3_context(pdf, args, sig_meta)

        print("  Page 4: WRKY interactions...", flush=True)
        page4_interactions(pdf, args, sig_meta)

        print("  Page 5: Hypothesis testing...", flush=True)
        page5_hypothesis_testing(pdf, args, sig_meta)

        print("  Page 6: Nucleosome analysis...", flush=True)
        page6_nucleosome(pdf, args, sig_meta)

    print(f"[DONE] {pdf_path} (6 pages)", flush=True)


if __name__ == "__main__":
    main()
