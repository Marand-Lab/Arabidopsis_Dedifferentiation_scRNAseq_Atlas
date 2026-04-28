#!/usr/bin/env python3
"""
v4_03b — Characterize FP signal at predicted binding / non-binding sites.

Uses per-condition percentile thresholds on TFBS/NucBS probabilities from
v4_03a (top/bottom 5% for TFBS, top/bottom 2% for NucBS).
Extracts the FP value at all 99 scales at each position and asks:
  - Is the binding score correlated with any specific FP scale?
  - What does the FP profile look like at bound vs unbound sites?

Same analysis for NucBS occupied vs free.

Figures per condition:
  Fig A: Per-scale Spearman r(binding_score, FP) — TFBS and NucBS
  Fig B: FP violin per scale at bound vs unbound / occupied vs free

Usage:
  python -u v4_03b_bs_fp_correlation.py --condition leaf
  python -u v4_03b_bs_fp_correlation.py --condition proto
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
from scipy import stats

# ── Constants ────────────────────────────────────────────────────────────────
REGION_WIDTH = 2000


def pct_tag(tfbs_pct, nucbs_pct):
    """Format percentile suffix for filenames, e.g. '_tf5_nuc2'."""
    def _fmt(v):
        return str(int(v)) if v == int(v) else f"{v:.1f}"
    return f"_tf{_fmt(tfbs_pct)}_nuc{_fmt(nucbs_pct)}"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--condition", required=True, choices=["leaf", "proto"])
    p.add_argument("--bs-dir", default="results/v4_03a_binding_scores",
                   help="Directory with _bs_{cond}.npz from v4_03a")
    p.add_argument("--fp-dir", default="v4/3_PRINT/FP",
                   help="Directory with FP h5ads")
    p.add_argument("--outdir", default="results/v4_03b_bs_fp_correlation")
    p.add_argument("--tfbs-pct", type=float, default=5,
                   help="Top/bottom percentile for TFBS bound/unbound (default: 5)")
    p.add_argument("--nucbs-pct", type=float, default=2,
                   help="Top/bottom percentile for NucBS occupied/free (default: 2)")
    p.add_argument("--native-only", action="store_true",
                   help="Restrict to tiles inside native ACR boundaries")
    p.add_argument("--metadata", default="v4/data/acr_metadata.tsv.gz")
    p.add_argument("--mapping", default="data/acr_native_to_resized.tsv")
    return p.parse_args()


def get_fp_scales(fp_path, sample_key):
    """Read FP scale values from h5ad uns."""
    import anndata
    adata = anndata.read_h5ad(fp_path, backed="r")
    if "scales" in adata.uns:
        scales = np.array(adata.uns["scales"], dtype=np.float64)
    else:
        sample = np.asarray(adata.obsm[sample_key])
        n_scales = sample.shape[1]
        scales = np.arange(n_scales) + 2
    adata.file.close()
    return scales


def extract_fp_for_mask(fp_h5, region_strs, tile_bp, bs_mask, label=""):
    """Extract FP at all scales for positions where bs_mask is True.

    Parameters
    ----------
    fp_h5 : open h5py.File
    region_strs : array of region keys (n_regions,)
    tile_bp : array of bp positions (n_tiles,)
    bs_mask : bool array (n_regions, n_tiles), True at positions to extract

    Returns
    -------
    fp_values : (n_positions, n_scales)
    bs_scores : not returned here — caller indexes separately
    """
    obsm = fp_h5["obsm"]
    avail = set(obsm.keys())

    # Get n_scales from first region
    sample_key = region_strs[0]
    sample_shape = obsm[sample_key].shape  # (1, n_scales, 2000)
    n_scales = sample_shape[1]

    # Count total positions
    n_positions = int(bs_mask.sum())
    print(f"  [{label}] {n_positions:,} positions to extract, {n_scales} scales",
          flush=True)

    fp_out = np.empty((n_positions, n_scales), dtype=np.float32)

    t0 = time.time()
    pos_idx = 0
    n_regions = len(region_strs)

    for ri in range(n_regions):
        tile_mask = bs_mask[ri]  # (n_tiles,)
        if not tile_mask.any():
            continue

        rstr = region_strs[ri]
        if rstr not in avail:
            # Fill with NaN for missing regions
            n_tiles_here = tile_mask.sum()
            fp_out[pos_idx:pos_idx + n_tiles_here] = np.nan
            pos_idx += n_tiles_here
            continue

        # Load FP for this region: (1, n_scales, 2000)
        fp_arr = obsm[rstr][0]  # (n_scales, 2000)

        # Get bp positions for True tiles
        bp_positions = tile_bp[tile_mask]
        n_tiles_here = len(bp_positions)

        # Extract FP at those bp positions: (n_scales, n_tiles_here) → transpose
        fp_out[pos_idx:pos_idx + n_tiles_here] = fp_arr[:, bp_positions].T

        pos_idx += n_tiles_here

        if (ri + 1) % 5000 == 0:
            print(f"    [{label}] {ri+1:,}/{n_regions:,} regions "
                  f"({pos_idx:,} positions)...", flush=True)

    elapsed = time.time() - t0
    print(f"  [{label}] Extracted {pos_idx:,} positions in {elapsed:.1f}s",
          flush=True)

    return fp_out[:pos_idx]


def compute_per_scale_corr(bs_scores, fp_values, n_scales, label=""):
    """Spearman r(binding_score, FP) per scale across all positions."""
    corr = np.full(n_scales, np.nan)
    pval = np.full(n_scales, np.nan)

    for si in range(n_scales):
        fp = fp_values[:, si]
        mask = np.isfinite(bs_scores) & np.isfinite(fp)
        n_valid = mask.sum()
        if n_valid < 10:
            continue
        r, p = stats.spearmanr(bs_scores[mask], fp[mask])
        corr[si] = r
        pval[si] = p

    valid = np.isfinite(corr)
    print(f"  [{label}] {valid.sum()}/{n_scales} scales with valid r, "
          f"max |r|={np.nanmax(np.abs(corr)):.4f}", flush=True)
    return corr, pval


def plot_fig_a(corr_dict, scales_bp, cond, outdir, thresholds=None, suffix=""):
    """Fig A: Per-scale correlation (line plot), all 4 sets on one figure."""
    th = thresholds or {}
    fig, axes = plt.subplots(2, 1, figsize=(7, 4), sharex=True)

    # Top: TFBS
    ax = axes[0]
    ax.plot(scales_bp, corr_dict["tfbs_bound"], "o-", ms=3, lw=1.2,
            color="#2CA02C", label=f"TFBS bound (>{th.get('tfbs', '?')})")
    ax.plot(scales_bp, corr_dict["tfbs_unbound"], "o-", ms=3, lw=1.2,
            color="#FF7F0E", label=f"TFBS unbound (<{th.get('tfbs_unb', '?')})")
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.set_ylabel("Spearman r(TFBS, FP)")
    ax.set_title(f"TFBS × FP correlation by scale — {cond}")
    ax.legend(fontsize=9)

    # Bottom: NucBS
    ax = axes[1]
    ax.plot(scales_bp, corr_dict["nucbs_occ"], "o-", ms=3, lw=1.2,
            color="#2CA02C", label=f"NucBS occupied (>{th.get('nucbs', '?')})")
    ax.plot(scales_bp, corr_dict["nucbs_free"], "o-", ms=3, lw=1.2,
            color="#FF7F0E", label=f"NucBS free (<{th.get('nucbs_free', '?')})")
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("FP scale (bp)")
    ax.set_ylabel("Spearman r(NucBS, FP)")
    ax.set_title(f"NucBS × FP correlation by scale — {cond}")
    ax.legend(fontsize=9)

    fig.tight_layout()
    path = os.path.join(outdir, f"fig_A_bs_fp_corr_{cond}{suffix}")
    for fmt in ("pdf", "png"):
        fig.savefig(f"{path}.{fmt}", dpi=200)
    plt.close(fig)
    print(f"[SAVE] {path}.pdf/.png")


def plot_fig_b(fp_bound, fp_unbound, scales_bp, score_label, cond, outdir,
               bound_label="bound", unbound_label="unbound",
               bound_pct=95, unbound_pct=5,
               fp_clip=10.0, suffix=""):
    """Fig B: Violin of FP values per scale, bound vs unbound overlay."""
    # Subsample scales for readability
    step = max(1, len(scales_bp) // 25)
    scale_indices = np.arange(0, len(scales_bp), step)
    fig, ax = plt.subplots(figsize=(7, 3))
    width = scales_bp[1] - scales_bp[0] if len(scales_bp) > 1 else 1
    half = width * 0.2

    for si in scale_indices:
        pos = float(scales_bp[si])

        # Bound (right side)
        vals_b = fp_bound[:, si]
        vals_b = vals_b[np.isfinite(vals_b)]
        vals_b = np.clip(vals_b, None, fp_clip)
        if len(vals_b) > 5000:
            vals_b = np.random.choice(vals_b, 5000, replace=False)
        if len(vals_b) > 1:
            vp = ax.violinplot([vals_b], positions=[pos + half],
                               widths=width * 0.75, showmedians=True,
                               showextrema=False)
            for pc in vp["bodies"]:
                pc.set_facecolor("#2CA02C")
                pc.set_alpha(0.5)
            vp["cmedians"].set_color("#2CA02C")

        # Unbound (left side)
        vals_u = fp_unbound[:, si]
        vals_u = vals_u[np.isfinite(vals_u)]
        vals_u = np.clip(vals_u, None, fp_clip)
        if len(vals_u) > 5000:
            vals_u = np.random.choice(vals_u, 5000, replace=False)
        if len(vals_u) > 1:
            vp = ax.violinplot([vals_u], positions=[pos - half],
                               widths=width * 0.75, showmedians=True,
                               showextrema=False)
            for pc in vp["bodies"]:
                pc.set_facecolor("#FF7F0E")
                pc.set_alpha(0.5)
            vp["cmedians"].set_color("#FF7F0E")

    # Legend patches
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor="#2CA02C", alpha=0.5, label=f"{bound_label} (P{bound_pct})"),
        Patch(facecolor="#FF7F0E", alpha=0.5, label=f"{unbound_label} (P{unbound_pct})"),
    ], fontsize=9)

    ax.set_xlabel("FP scale (bp)")
    ax.set_ylabel("FP score (-log10 p)")
    ax.set_title(f"{score_label}: FP distribution at {bound_label} vs "
                 f"{unbound_label} sites — {cond}")

    fig.tight_layout()
    tag = score_label.lower().replace(" ", "_")
    path = os.path.join(outdir, f"fig_B_{tag}_fp_violin_{cond}{suffix}")
    for fmt in ("pdf", "png"):
        fig.savefig(f"{path}.{fmt}", dpi=200)
    plt.close(fig)
    print(f"[SAVE] {path}.pdf/.png")


def main():
    args = parse_args()
    cond = args.condition
    os.makedirs(args.outdir, exist_ok=True)

    print(f"=== v4_03b: BS–FP Correlation — {cond} ===\n")

    # ── Load binding scores from v4_03a ──────────────────────────────────
    bs_path = os.path.join(args.bs_dir, f"_bs_{cond}.npz")
    print(f"[LOAD] {bs_path}")
    bs = np.load(bs_path, allow_pickle=True)
    region_strs = bs["region_strs"]
    TFBS_prob = bs["TFBS_prob"]       # (n_regions, 180)
    NucBS_prob = bs["NucBS_prob"]     # (n_regions, 180)
    tile_bp = bs["tile_bp"]           # (180,)
    n_regions = len(region_strs)

    # Per-condition percentile thresholds (consistent with v4_03c)
    tf_top = 100 - args.tfbs_pct
    tf_bot = args.tfbs_pct
    nuc_top = 100 - args.nucbs_pct
    nuc_bot = args.nucbs_pct

    tf_bound_thresh = np.percentile(TFBS_prob, tf_top)
    tf_unbound_thresh = np.percentile(TFBS_prob, tf_bot)
    nuc_occ_thresh = np.percentile(NucBS_prob, nuc_top)
    nuc_free_thresh = np.percentile(NucBS_prob, nuc_bot)

    tf_bound = TFBS_prob > tf_bound_thresh
    tf_unbound = TFBS_prob < tf_unbound_thresh
    nuc_occupied = NucBS_prob > nuc_occ_thresh
    nuc_free = NucBS_prob < nuc_free_thresh

    print(f"  {n_regions:,} regions, {len(tile_bp)} tiles")

    # ── Native-only masking ───────────────────────────────────────────
    if args.native_only:
        import sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib"))
        from _tile_utils import build_native_tile_mask
        print("\n[NATIVE] Building native ACR tile mask...")
        native_mask, _ = build_native_tile_mask(
            region_strs, args.metadata, args.mapping)
        tf_bound &= native_mask
        tf_unbound &= native_mask
        nuc_occupied &= native_mask
        nuc_free &= native_mask

    print(f"  TFBS bound (P{tf_top} >{tf_bound_thresh:.4f}): {tf_bound.sum():,}, "
          f"unbound (P{tf_bot} <{tf_unbound_thresh:.4f}): {tf_unbound.sum():,}")
    print(f"  NucBS occupied (P{nuc_top} >{nuc_occ_thresh:.4f}): {nuc_occupied.sum():,}, "
          f"free (P{nuc_bot} <{nuc_free_thresh:.4f}): {nuc_free.sum():,}")

    # ── Get FP scales ────────────────────────────────────────────────────
    fp_path = os.path.join(args.fp_dir, f"{cond}_merged__ALL.h5ad")
    scales_bp = get_fp_scales(fp_path, region_strs[0])
    n_scales = len(scales_bp)
    print(f"\n  FP: {n_scales} scales ({scales_bp[0]:.0f}–{scales_bp[-1]:.0f} bp)")

    # ── Open FP h5ad ─────────────────────────────────────────────────────
    print(f"\n[EXTRACT] Opening {fp_path}")
    fp_h5 = h5py.File(fp_path, "r")

    # ── Extract FP at each position set ──────────────────────────────────
    print("\n--- TFBS bound positions ---")
    fp_tfbs_bound = extract_fp_for_mask(
        fp_h5, region_strs, tile_bp, tf_bound, "TFBS_bound")
    tfbs_bound_scores = TFBS_prob[tf_bound]

    print("\n--- TFBS unbound positions ---")
    fp_tfbs_unbound = extract_fp_for_mask(
        fp_h5, region_strs, tile_bp, tf_unbound, "TFBS_unbound")
    tfbs_unbound_scores = TFBS_prob[tf_unbound]

    print("\n--- NucBS occupied positions ---")
    fp_nucbs_occ = extract_fp_for_mask(
        fp_h5, region_strs, tile_bp, nuc_occupied, "NucBS_occ")
    nucbs_occ_scores = NucBS_prob[nuc_occupied]

    print("\n--- NucBS free positions ---")
    fp_nucbs_free = extract_fp_for_mask(
        fp_h5, region_strs, tile_bp, nuc_free, "NucBS_free")
    nucbs_free_scores = NucBS_prob[nuc_free]

    fp_h5.close()

    # ── Per-scale correlations ───────────────────────────────────────────
    print("\n[CORR] Per-scale Spearman r(binding_score, FP)...")

    corr_tfbs_bound, pval_tfbs_bound = compute_per_scale_corr(
        tfbs_bound_scores, fp_tfbs_bound, n_scales, "TFBS_bound")
    corr_tfbs_unbound, pval_tfbs_unbound = compute_per_scale_corr(
        tfbs_unbound_scores, fp_tfbs_unbound, n_scales, "TFBS_unbound")
    corr_nucbs_occ, pval_nucbs_occ = compute_per_scale_corr(
        nucbs_occ_scores, fp_nucbs_occ, n_scales, "NucBS_occ")
    corr_nucbs_free, pval_nucbs_free = compute_per_scale_corr(
        nucbs_free_scores, fp_nucbs_free, n_scales, "NucBS_free")

    # ── Save ─────────────────────────────────────────────────────────────
    _tag = pct_tag(args.tfbs_pct, args.nucbs_pct)
    if args.native_only:
        _tag += "_native"
    npz_path = os.path.join(args.outdir, f"corr_{cond}{_tag}.npz")
    np.savez_compressed(
        npz_path,
        scales_bp=scales_bp,
        corr_tfbs_bound=corr_tfbs_bound,
        corr_tfbs_unbound=corr_tfbs_unbound,
        corr_nucbs_occ=corr_nucbs_occ,
        corr_nucbs_free=corr_nucbs_free,
        pval_tfbs_bound=pval_tfbs_bound,
        pval_tfbs_unbound=pval_tfbs_unbound,
        pval_nucbs_occ=pval_nucbs_occ,
        pval_nucbs_free=pval_nucbs_free,
    )
    print(f"\n[SAVE] {npz_path}")

    # ── Summary ──────────────────────────────────────────────────────────
    for label, corr in [("TFBS_bound", corr_tfbs_bound),
                        ("TFBS_unbound", corr_tfbs_unbound),
                        ("NucBS_occ", corr_nucbs_occ),
                        ("NucBS_free", corr_nucbs_free)]:
        valid = corr[np.isfinite(corr)]
        if len(valid) == 0:
            continue
        best_si = np.nanargmax(np.abs(corr))
        print(f"  {label}: max |r|={np.abs(corr[best_si]):.4f} "
              f"at scale {scales_bp[best_si]:.0f}bp, "
              f"mean |r|={np.nanmean(np.abs(valid)):.4f}")

    # ── Plot ─────────────────────────────────────────────────────────────
    print(f"\n[PLOT] Generating figures...")

    plot_fig_a(
        {"tfbs_bound": corr_tfbs_bound,
         "tfbs_unbound": corr_tfbs_unbound,
         "nucbs_occ": corr_nucbs_occ,
         "nucbs_free": corr_nucbs_free},
        scales_bp, cond, args.outdir,
        thresholds={"tfbs": f"P{tf_top}={tf_bound_thresh:.3f}",
                    "tfbs_unb": f"P{tf_bot}={tf_unbound_thresh:.3f}",
                    "nucbs": f"P{nuc_top}={nuc_occ_thresh:.3f}",
                    "nucbs_free": f"P{nuc_bot}={nuc_free_thresh:.3f}"},
        suffix=_tag)

    plot_fig_b(fp_tfbs_bound, fp_tfbs_unbound, scales_bp,
               "TFBS", cond, args.outdir,
               bound_label="bound", unbound_label="unbound",
               bound_pct=tf_top, unbound_pct=tf_bot, suffix=_tag)

    plot_fig_b(fp_nucbs_occ, fp_nucbs_free, scales_bp,
               "NucBS", cond, args.outdir,
               bound_label="occupied", unbound_label="free",
               bound_pct=nuc_top, unbound_pct=nuc_bot, suffix=_tag)

    print(f"\n[DONE] {cond} complete.", flush=True)


if __name__ == "__main__":
    main()
