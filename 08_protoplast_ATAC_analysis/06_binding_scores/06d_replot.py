#!/usr/bin/env python3
"""
v4_03d_replot — Re-plot violins + tile heatmaps from saved NPZ (no SLURM needed).

Usage (local):
  /opt/anaconda3/bin/python3 -u v4/v4_03d_replot.py
  /opt/anaconda3/bin/python3 -u v4/v4_03d_replot.py --tfbs-pct 10 --nucbs-pct 5
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ACR_CLASSES = ["proto_gain", "stable", "leaf_gain"]
OVERLAP_GROUPS = ["shared", "leaf_only", "proto_only"]
CLASS_COLORS = {"proto_gain": "#E64B35", "stable": "#808080", "leaf_gain": "#4DBBD5"}
GROUP_COLORS = {"shared": "#7B2D8E", "leaf_only": "#D62728", "proto_only": "#1F77B4"}

INDIR = "results/v4_03d_binding_deltas"
Z_THRESH = 1.0


def pct_tag(tfbs_pct, nucbs_pct):
    def _fmt(v):
        return str(int(v)) if v == int(v) else f"{v:.1f}"
    return f"_tf{_fmt(tfbs_pct)}_nuc{_fmt(nucbs_pct)}"


def find_npz_and_tsv(indir, tag):
    """Find NPZ and TSV files, checking both indir and parent directory."""
    npz_path = os.path.join(indir, f"tile_deltas{tag}.npz")
    tsv_path = os.path.join(indir, f"delta_summary{tag}.tsv")
    # NPZ/TSV may be in parent dir (replot uses separate outdir from extraction)
    parent = os.path.dirname(indir.rstrip("/"))
    if not os.path.exists(npz_path):
        npz_path = os.path.join(parent, f"tile_deltas{tag}.npz")
    if not os.path.exists(tsv_path):
        tsv_path = os.path.join(parent, f"delta_summary{tag}.tsv")
    return npz_path, tsv_path


def plot_tile_heatmap(npz, stats_df, score_type, scale_label, outdir, tag, suffix=""):
    """3x3 grid, only significant tiles (|z| >= Z_THRESH), colorbar on right."""
    fig, axes = plt.subplots(3, 3, figsize=(9, 12),
                             gridspec_kw={"wspace": 0.15, "hspace": 0.35})

    # Global vmax from significant tiles across all 9 panels
    global_max = 0.01
    for cls in ACR_CLASSES:
        for grp in OVERLAP_GROUPS:
            key = f"{score_type}_{cls}_{grp}"
            if key not in npz:
                continue
            delta = npz[key]
            z = npz.get(f"{key}_z", np.array([]))
            if len(z) > 0:
                sig = np.abs(z) >= Z_THRESH
                if sig.any():
                    global_max = max(global_max, np.nanpercentile(np.abs(delta[sig]), 99))

    last_im = None
    for row, cls in enumerate(ACR_CLASSES):
        for col, grp in enumerate(OVERLAP_GROUPS):
            ax = axes[row, col]
            key = f"{score_type}_{cls}_{grp}"
            delta = npz.get(key, np.array([]))
            z = npz.get(f"{key}_z", np.array([]))
            n_total = len(delta)

            if n_total == 0 or len(z) == 0:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=9)
                ax.axis("off")
                continue

            sig_pos = z >= Z_THRESH
            sig_neg = z <= -Z_THRESH

            idx_pos = np.where(sig_pos)[0][np.argsort(-delta[sig_pos])]
            idx_neg = np.where(sig_neg)[0][np.argsort(-delta[sig_neg])]
            sorted_idx = np.concatenate([idx_pos, idx_neg])

            n_pos = len(idx_pos)
            n_neg = len(idx_neg)
            n_sig = n_pos + n_neg

            if n_sig == 0:
                ax.text(0.5, 0.5, f"n={n_total:,}\nno sig tiles",
                        ha="center", va="center", transform=ax.transAxes,
                        fontsize=8, color="#999")
                ax.axis("off")
                continue

            sorted_delta = delta[sorted_idx] / global_max
            last_im = ax.imshow(sorted_delta.reshape(-1, 1), aspect="auto",
                                cmap="RdBu_r", vmin=-1, vmax=1,
                                interpolation="none")

            # Block boundary
            if n_pos > 0 and n_neg > 0:
                ax.axhline(n_pos - 0.5, color="k", lw=1.0)

            # Labels inside heatmap, centered horizontally
            fs = 7
            if n_pos > 0:
                y_frac_pos = 1 - (n_pos / 2) / n_sig
                ax.text(0.5, y_frac_pos, f"{n_pos:,}\nleaf",
                        fontsize=fs, color="#D62728",
                        va="center", ha="center", fontweight="bold",
                        transform=ax.transAxes, clip_on=False)
            if n_neg > 0:
                y_frac_neg = 1 - (n_pos + n_neg / 2) / n_sig
                ax.text(0.5, y_frac_neg, f"{n_neg:,}\nproto",
                        fontsize=fs, color="#1F77B4",
                        va="center", ha="center", fontweight="bold",
                        transform=ax.transAxes, clip_on=False)

            ax.set_xticks([])
            ax.set_yticks([])

            # Title from stats
            mask = ((stats_df["score_type"] == score_type) &
                    (stats_df["acr_class"] == cls) &
                    (stats_df["group"] == grp))
            if mask.any():
                r = stats_df[mask].iloc[0]
                cat_fdr = r.get("fdr", 1)
                cat_med = r.get("median", 0)
            else:
                cat_fdr, cat_med = 1.0, 0.0

            stars = "***" if cat_fdr < 0.001 else "**" if cat_fdr < 0.01 \
                else "*" if cat_fdr < 0.05 else "ns"
            direction = "leaf enriched" if cat_med > 0 else "proto enriched"
            pct = n_sig / n_total * 100
            ax.set_title(f"{grp}\n{n_sig:,}/{n_total:,} sig ({pct:.1f}%)\n"
                         f"MW: {stars} {direction}",
                         fontsize=7, fontweight="bold", color=GROUP_COLORS[grp])

            if col == 0:
                ax.set_ylabel(cls, fontsize=10, fontweight="bold",
                              color=CLASS_COLORS[cls])

    # Colorbar on the right
    if last_im is not None:
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.65])
        cbar = fig.colorbar(last_im, cax=cbar_ax)
        cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
        cbar.set_ticklabels(["proto\nenriched", "", "0", "", "leaf\nenriched"])
        cbar.set_label("normalized delta (leaf-proto)", fontsize=9)

    fig.suptitle(f"{score_type}: significant tile deltas at {scale_label}\n"
                 f"(|z| >= {Z_THRESH} from null distribution)",
                 fontsize=12, fontweight="bold", y=0.98)
    fig.subplots_adjust(left=0.08, right=0.88, top=0.92, bottom=0.03)

    path = os.path.join(outdir, f"fig_{tag}_tile_heatmap{suffix}")
    for fmt in ("pdf", "png"):
        fig.savefig(f"{path}.{fmt}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVE] {path}.pdf/.png")


def plot_violins(npz, stats_df, score_type, scale_label, outdir, tag, suffix=""):
    """3x3 violin grid: category vs genome-wide null, clipped to [-10, 10]."""
    CLIP = 10.0
    fig, axes = plt.subplots(3, 3, figsize=(12, 10), sharey=True)

    null_d = npz.get(f"{score_type}_null", np.array([]))
    null_clean = np.clip(null_d[np.isfinite(null_d)][:5000], -CLIP, CLIP)

    for row, cls in enumerate(ACR_CLASSES):
        for col, grp in enumerate(OVERLAP_GROUPS):
            ax = axes[row, col]
            key = f"{score_type}_{cls}_{grp}"
            d = npz.get(key, np.array([]))

            # Null
            if len(null_clean) > 1:
                vp = ax.violinplot([null_clean], positions=[0], widths=0.8,
                                   showmedians=True, showextrema=False)
                for pc in vp["bodies"]:
                    pc.set_facecolor("#D9D9D9"); pc.set_alpha(0.6)
                vp["cmedians"].set_color("#999")

            # Category
            if len(d) > 0:
                d_clean = np.clip(d[np.isfinite(d)], -CLIP, CLIP)
                if len(d_clean) > 5000:
                    d_clean = np.random.choice(d_clean, 5000, replace=False)
                if len(d_clean) > 1:
                    vp = ax.violinplot([d_clean], positions=[1], widths=0.8,
                                       showmedians=True, showextrema=False)
                    for pc in vp["bodies"]:
                        pc.set_facecolor(GROUP_COLORS[grp]); pc.set_alpha(0.7)
                    vp["cmedians"].set_color("k")

            ax.set_ylim(-CLIP, CLIP)
            ax.axhline(0, color="gray", ls="--", lw=0.5, alpha=0.5)

            # Stats
            mask = ((stats_df["score_type"] == score_type) &
                    (stats_df["acr_class"] == cls) &
                    (stats_df["group"] == grp))
            if mask.any():
                r = stats_df[mask].iloc[0]
                n = int(r.get("n", 0))
                med = r.get("median", 0)
                fdr = r.get("fdr", 1)
            else:
                n, med, fdr = 0, 0, 1
            stars = "***" if fdr < 0.001 else "**" if fdr < 0.01 else \
                    "*" if fdr < 0.05 else "ns"
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tfbs-pct", type=float, default=5)
    p.add_argument("--nucbs-pct", type=float, default=2)
    p.add_argument("--native-only", action="store_true",
                   help="Load native-only results (tile_deltas_*_native.npz)")
    p.add_argument("--outdir",
                   help="Output directory for plots (default: same as input)")
    args = p.parse_args()

    _tag = pct_tag(args.tfbs_pct, args.nucbs_pct)
    if args.native_only:
        _tag += "_native"
    print(f"=== v4_03d_replot ({_tag}) ===\n")

    npz_path, tsv_path = find_npz_and_tsv(INDIR, _tag)
    if not os.path.exists(npz_path):
        print(f"  ERROR: NPZ not found: {npz_path}")
        return
    print(f"  NPZ: {npz_path}")
    print(f"  TSV: {tsv_path}")

    outdir = args.outdir if args.outdir else INDIR
    os.makedirs(outdir, exist_ok=True)

    npz = np.load(npz_path, allow_pickle=True)
    stats = pd.read_csv(tsv_path, sep="\t")
    print(f"  Output: {outdir}")
    print(f"  |z| threshold: {Z_THRESH}")

    plot_violins(npz, stats, "TFBS", "TF (4-10 bp)", outdir, "A_tfbs", suffix=_tag)
    plot_violins(npz, stats, "NucBS", "Nuc (40-60 bp)", outdir, "B_nucbs", suffix=_tag)
    plot_tile_heatmap(npz, stats, "TFBS", "TF (4-10 bp)", outdir, "D_tfbs", suffix=_tag)
    plot_tile_heatmap(npz, stats, "NucBS", "Nuc (40-60 bp)", outdir, "E_nucbs", suffix=_tag)

    print("\n[DONE]")


if __name__ == "__main__":
    main()
