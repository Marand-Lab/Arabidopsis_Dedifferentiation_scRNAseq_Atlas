#!/usr/bin/env python3
"""
v4_03c — Overlap of TFBS / NucBS between leaf and proto.

Uses unified probability thresholds (same cutoff for both conditions).
For each tile (22,581 regions x 180 tiles), classifies as:
  shared / leaf-only / proto-only / neither
Stratified by ACR class.

Defaults: TFBS > 0.5, NucBS > 0.65

Usage:
  /opt/anaconda3/bin/python3 -u v4/v4_03c_binding_overlap.py
  /opt/anaconda3/bin/python3 -u v4/v4_03c_binding_overlap.py --tfbs-threshold 0.3
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import pandas as pd
from scipy.stats import hypergeom


def pct_tag(tfbs_pct, nucbs_pct):
    """Format percentile suffix for filenames, e.g. '_tf5_nuc2'."""
    def _fmt(v):
        return str(int(v)) if v == int(v) else f"{v:.1f}"
    return f"_tf{_fmt(tfbs_pct)}_nuc{_fmt(nucbs_pct)}"


ACR_CLASSES = ["proto_gain", "stable", "leaf_gain"]
CLASS_COLORS = {"proto_gain": "#E64B35", "stable": "#808080", "leaf_gain": "#4DBBD5"}
OVERLAP_COLORS = {
    "shared": "#7B2D8E",
    "leaf_only": "#D62728",
    "proto_only": "#1F77B4",
    "neither": "#D9D9D9",
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bs-dir", default="results/v4_03a_binding_scores")
    p.add_argument("--metadata", default="v4/data/acr_metadata.tsv.gz")
    p.add_argument("--mapping", default="data/acr_native_to_resized.tsv")
    p.add_argument("--outdir", default="results/v4_03c_binding_overlap")
    p.add_argument("--tfbs-pct", type=float, default=5,
                   help="Top percentile for TFBS bound (default: 5)")
    p.add_argument("--nucbs-pct", type=float, default=2,
                   help="Top percentile for NucBS occupied (default: 2)")
    p.add_argument("--native-only", action="store_true",
                   help="Restrict to tiles inside native ACR boundaries")
    return p.parse_args()


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
    for c in ACR_CLASSES:
        print(f"    {c}: {(classes == c).sum():,}")
    return classes


def compute_overlap(mask_leaf, mask_proto, region_classes):
    """Per-ACR-class tile-level overlap with hypergeometric test.

    Universe = active tiles only (bound/occupied in >= 1 condition).
    N = shared + leaf_only + proto_only.
    Hypergeometric: given N active tiles, K from leaf, n from proto,
    is the overlap x more or less than expected by chance?

    Expected overlap: E[x] = K * n / N.
    fold > 1 → conserved binding; fold < 1 → condition-specific binding.
    """
    results = {}
    for cls in ACR_CLASSES:
        idx = region_classes == cls
        ml, mp = mask_leaf[idx], mask_proto[idx]
        shared = int((ml & mp).sum())
        leaf_only = int((ml & ~mp).sum())
        proto_only = int((~ml & mp).sum())

        N = shared + leaf_only + proto_only        # active-tile universe
        K = shared + leaf_only                    # bound in leaf
        n = shared + proto_only                   # bound in proto
        x = shared

        if N > 0 and K > 0 and n > 0:
            expected = K * n / N
            fold_enrich = x / expected if expected > 0 else np.inf
            # Two-sided: test enrichment (sf) or depletion (cdf)
            if x >= expected:
                pvalue = float(hypergeom.sf(x - 1, N, K, n))  # enrichment
            else:
                pvalue = float(hypergeom.cdf(x, N, K, n))     # depletion
        else:
            expected = 0
            pvalue = 1.0
            fold_enrich = 0.0

        results[cls] = dict(shared=shared, leaf_only=leaf_only,
                            proto_only=proto_only, active_total=N,
                            n_leaf=K, n_proto=n,
                            expected=expected, fold_enrich=fold_enrich,
                            hyper_p=pvalue,
                            n_regions=int(idx.sum()))
    return results


def jaccard(s):
    active = s["shared"] + s["leaf_only"] + s["proto_only"]
    return s["shared"] / active if active > 0 else 0.0


def _draw_venn2(ax, left, overlap, right):
    """Simple 2-circle Venn."""
    total = left + overlap + right
    if total == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes)
        return

    r_l = np.sqrt((left + overlap) / total) * 0.28
    r_r = np.sqrt((right + overlap) / total) * 0.28
    jacc = overlap / total
    sep = (r_l + r_r) * (1 - 0.7 * jacc)

    cx_l, cx_r, cy = 0.5 - sep / 2, 0.5 + sep / 2, 0.48
    ax.add_patch(Circle((cx_l, cy), r_l, fc="#D62728", alpha=0.45, ec="k", lw=0.8))
    ax.add_patch(Circle((cx_r, cy), r_r, fc="#1F77B4", alpha=0.45, ec="k", lw=0.8))

    fs = 8
    ax.text(cx_l - r_l * 0.4, cy, f"{left:,}",
            ha="center", va="center", fontsize=fs, fontweight="bold")
    ax.text((cx_l + cx_r) / 2, cy, f"{overlap:,}",
            ha="center", va="center", fontsize=fs, fontweight="bold", color="#7B2D8E")
    ax.text(cx_r + r_r * 0.4, cy, f"{right:,}",
            ha="center", va="center", fontsize=fs, fontweight="bold")

    ax.text(cx_l - r_l * 0.3, cy + r_l + 0.06, "leaf",
            ha="center", fontsize=8, color="#D62728", fontweight="bold")
    ax.text(cx_r + r_r * 0.3, cy + r_r + 0.06, "proto",
            ha="center", fontsize=8, color="#1F77B4", fontweight="bold")

    size_l, size_r = left + overlap, right + overlap
    j = overlap / (left + overlap + right) if (left + overlap + right) > 0 else 0
    ax.text(0.5, 0.05,
            f"leaf: {size_l:,}  |  proto: {size_r:,}  |  Jaccard: {j:.2f}",
            ha="center", fontsize=7, transform=ax.transAxes, color="#555")

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect("equal"); ax.axis("off")


def plot_venn(overlap_stats, score_label, threshold, outdir, tag, tile_noun="tiles",
             suffix=""):
    """Venn diagrams: 1 row x 3 ACR classes."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    for i, cls in enumerate(ACR_CLASSES):
        s = overlap_stats[cls]
        _draw_venn2(axes[i], s["leaf_only"], s["shared"], s["proto_only"])
        axes[i].set_title(f"{cls}\n({s['n_regions']:,} ACRs, {s['active_total']:,} {tile_noun})",
                          fontsize=10, fontweight="bold", color=CLASS_COLORS[cls])
        # Hypergeometric annotation
        p = s["hyper_p"]
        p_str = "p < 1e-300" if p == 0 else f"p = {p:.1e}"
        axes[i].text(0.5, -0.02,
                     f"fold = {s['fold_enrich']:.2f}x  (obs={s['shared']:,}, "
                     f"exp={s['expected']:.0f}),  {p_str}",
                     ha="center", fontsize=7, transform=axes[i].transAxes,
                     color="#333", style="italic")

    fig.suptitle(f"{score_label}: leaf vs proto overlap  (threshold > {threshold})",
                 fontsize=13, fontweight="bold", y=1.05)
    fig.tight_layout()
    path = os.path.join(outdir, f"fig_{tag}_venn{suffix}")
    for fmt in ("pdf", "png"):
        fig.savefig(f"{path}.{fmt}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVE] {path}.pdf/.png")


def plot_stacked_bar(tfbs_stats, nucbs_stats, tfbs_label, nucbs_label, outdir,
                     tile_nouns=("bound", "occupied"), suffix=""):
    """Stacked bar: overlap fractions among bound/occupied tiles only."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    cats = ["shared", "leaf_only", "proto_only"]
    cat_labels = ["shared", "leaf only", "proto only"]

    for ax_i, (stats, label, tn) in enumerate([
        (tfbs_stats, tfbs_label, tile_nouns[0]),
        (nucbs_stats, nucbs_label, tile_nouns[1]),
    ]):
        x = np.arange(len(ACR_CLASSES))
        bottoms = np.zeros(len(ACR_CLASSES))
        for cat, cl in zip(cats, cat_labels):
            fracs = np.array([stats[c][cat] / stats[c]["active_total"] * 100
                              if stats[c]["active_total"] > 0 else 0
                              for c in ACR_CLASSES])
            axes[ax_i].bar(x, fracs, 0.6, bottom=bottoms,
                           color=OVERLAP_COLORS[cat], label=cl,
                           edgecolor="w", linewidth=0.5)
            for j, (f, b) in enumerate(zip(fracs, bottoms)):
                if f > 3:
                    axes[ax_i].text(x[j], b + f / 2, f"{f:.1f}%",
                                    ha="center", va="center", fontsize=7,
                                    fontweight="bold")
            bottoms += fracs

        axes[ax_i].set_xticks(x)
        axes[ax_i].set_xticklabels(
            [f"{c}\n({stats[c]['active_total']:,} {tn})"
             for c in ACR_CLASSES], fontsize=9)
        axes[ax_i].set_ylabel(f"% of {tn} tiles" if ax_i == 0 else "")
        axes[ax_i].set_title(label, fontsize=11, fontweight="bold")
        axes[ax_i].set_ylim(0, 105)

    axes[1].legend(loc="upper right", fontsize=9, framealpha=0.9)
    fig.suptitle("Tile overlap: leaf vs proto (top 2% per condition)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(outdir, f"fig_C_stacked_bar{suffix}")
    for fmt in ("pdf", "png"):
        fig.savefig(f"{path}.{fmt}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVE] {path}.pdf/.png")


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    tf_pct = args.tfbs_pct
    nuc_pct = args.nucbs_pct
    tf_cutoff = 100 - tf_pct   # top 5% → P95
    nuc_cutoff = 100 - nuc_pct  # top 2% → P98
    print(f"=== v4_03c: Binding Overlap ===")
    print(f"  TFBS: top {tf_pct}% (P{tf_cutoff}),  NucBS: top {nuc_pct}% (P{nuc_cutoff})\n")

    # ── Load ──────────────────────────────────────────────────────────────
    leaf = np.load(os.path.join(args.bs_dir, "_bs_leaf.npz"), allow_pickle=True)
    proto = np.load(os.path.join(args.bs_dir, "_bs_proto.npz"), allow_pickle=True)
    region_strs = leaf["region_strs"]
    assert np.array_equal(region_strs, proto["region_strs"])
    print(f"  {len(region_strs):,} regions x {leaf['TFBS_prob'].shape[1]} tiles")

    # ── Per-condition percentile thresholds ────────────────────────────────
    tf_t_leaf = np.percentile(leaf["TFBS_prob"], tf_cutoff)
    tf_t_proto = np.percentile(proto["TFBS_prob"], tf_cutoff)
    nuc_t_leaf = np.percentile(leaf["NucBS_prob"], nuc_cutoff)
    nuc_t_proto = np.percentile(proto["NucBS_prob"], nuc_cutoff)

    tf_leaf = leaf["TFBS_prob"] > tf_t_leaf
    tf_proto = proto["TFBS_prob"] > tf_t_proto
    nuc_leaf = leaf["NucBS_prob"] > nuc_t_leaf
    nuc_proto = proto["NucBS_prob"] > nuc_t_proto

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

    print(f"\n  TFBS top {tf_pct}%: leaf>{tf_t_leaf:.4f} ({tf_leaf.sum():,})  "
          f"proto>{tf_t_proto:.4f} ({tf_proto.sum():,})")
    print(f"  NucBS top {nuc_pct}%: leaf>{nuc_t_leaf:.4f} ({nuc_leaf.sum():,})  "
          f"proto>{nuc_t_proto:.4f} ({nuc_proto.sum():,})")

    # ── ACR class mapping ─────────────────────────────────────────────────
    print("\n[MAP] Region -> ACR class...")
    rcls = build_region_to_class(args.metadata, args.mapping, region_strs)

    # ── Tile-level overlap ────────────────────────────────────────────────
    print(f"\n[OVERLAP] TFBS (top {tf_pct}%):")
    tfbs_stats = compute_overlap(tf_leaf, tf_proto, rcls)
    for cls in ACR_CLASSES:
        s = tfbs_stats[cls]
        print(f"  {cls:12s}: shared={s['shared']:>7,}  leaf_only={s['leaf_only']:>7,}  "
              f"proto_only={s['proto_only']:>7,}  "
              f"active={s['active_total']:>7,}  "
              f"J={jaccard(s):.3f}  fold={s['fold_enrich']:.1f}x  "
              f"p={s['hyper_p']:.2e}")

    print(f"\n[OVERLAP] NucBS (top {nuc_pct}%):")
    nucbs_stats = compute_overlap(nuc_leaf, nuc_proto, rcls)
    for cls in ACR_CLASSES:
        s = nucbs_stats[cls]
        print(f"  {cls:12s}: shared={s['shared']:>7,}  leaf_only={s['leaf_only']:>7,}  "
              f"proto_only={s['proto_only']:>7,}  "
              f"active={s['active_total']:>7,}  "
              f"J={jaccard(s):.3f}  fold={s['fold_enrich']:.1f}x  "
              f"p={s['hyper_p']:.2e}")

    # ── Region-level overlap ──────────────────────────────────────────────
    print("\n[OVERLAP] Region-level (>= 1 bound/occupied tile):")
    for label, ml, mp in [("TFBS", tf_leaf, tf_proto),
                           ("NucBS", nuc_leaf, nuc_proto)]:
        rl, rp = ml.any(axis=1), mp.any(axis=1)
        print(f"  {label}:")
        for cls in ACR_CLASSES:
            idx = rcls == cls
            n = idx.sum()
            both = (rl[idx] & rp[idx]).sum()
            lo = (rl[idx] & ~rp[idx]).sum()
            po = (~rl[idx] & rp[idx]).sum()
            nei = (~rl[idx] & ~rp[idx]).sum()
            print(f"    {cls:12s}: both={both:>5,}  leaf={lo:>5,}  "
                  f"proto={po:>5,}  neither={nei:>5,}  (of {n:,})")

    # ── Save TSV ──────────────────────────────────────────────────────────
    rows = []
    for st, stats, pct in [("TFBS", tfbs_stats, tf_pct),
                            ("NucBS", nucbs_stats, nuc_pct)]:
        for cls in ACR_CLASSES:
            s = stats[cls]
            rows.append(dict(score_type=f"{st}_top{pct}pct", top_pct=pct,
                             acr_class=cls, n_regions=s["n_regions"],
                             active_total=s["active_total"],
                             n_leaf=s["n_leaf"], n_proto=s["n_proto"],
                             shared=s["shared"], leaf_only=s["leaf_only"],
                             proto_only=s["proto_only"],
                             jaccard=jaccard(s),
                             expected_overlap=s["expected"],
                             fold_enrichment=s["fold_enrich"],
                             hypergeom_p=s["hyper_p"]))
    _tag = pct_tag(tf_pct, nuc_pct)
    if args.native_only:
        _tag += "_native"
    tsv_path = os.path.join(args.outdir, f"overlap_summary{_tag}.tsv")
    pd.DataFrame(rows).to_csv(tsv_path, sep="\t", index=False, float_format="%.4f")
    print(f"\n[SAVE] {tsv_path}")

    # ── Plots ─────────────────────────────────────────────────────────────
    print("\n[PLOT]")
    plot_venn(tfbs_stats, "TFBS bound positions", f"top {tf_pct}%",
             args.outdir, "A_tfbs", tile_noun="bound", suffix=_tag)
    plot_venn(nucbs_stats, "NucBS occupied regions", f"top {nuc_pct}%",
             args.outdir, "B_nucbs", tile_noun="occupied", suffix=_tag)
    plot_stacked_bar(tfbs_stats, nucbs_stats,
                     f"TFBS top {tf_pct}%", f"NucBS top {nuc_pct}%", args.outdir,
                     suffix=_tag)

    print("\n[DONE]")


if __name__ == "__main__":
    main()
