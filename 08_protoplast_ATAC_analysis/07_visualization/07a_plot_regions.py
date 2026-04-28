#!/usr/bin/env python3
"""
v4_plot_regions — Per-ACR multiscale FP viewer with TFBS tile overlay.

Two-phase design:
  1. EXTRACT (SLURM): Load FP h5ads + printer h5ads, save per-region data to NPZ
  2. PLOT (local): Load NPZ, render figures. No h5ad access needed.

Sentinel: if the NPZ for a title already exists, extraction is skipped.
Use --force-extract to re-extract.

Panels (top to bottom):
  0. Metadata banner (ACR class, logFC, gene, genomic context)
  1. Leaf FP heatmap (scale x position)
  2. Proto FP heatmap (scale x position)
  3. Tn5 insertion profile (leaf + proto overlaid, 9bp smoothed)
  4. TFBS tile strip (leaf row + proto row, colored by bound threshold)

Usage:
  # Extract + plot on SLURM (from tile table)
  sbatch v4/v4_plot_regions.sh \
      --tile-table results/v4_03c_binding_overlap/acr_tile_table_tf5_nuc5.tsv \
      --filter-score TFBS --filter-class leaf_gain --filter-group leaf_only \
      --top-n 10 --title "TFBS_leaf_only_leaf_gain"

  # Plot only from cached NPZ (local, no h5ad needed)
  /opt/anaconda3/bin/python3 -u v4/v4_plot_regions.py \
      --plot-only --title "TFBS_leaf_only_leaf_gain"

  # Plot with zoom to native ACR boundaries
  /opt/anaconda3/bin/python3 -u v4/v4_plot_regions.py \
      --plot-only --zoom-native --title "TFBS_leaf_only_leaf_gain"

  # Force re-extraction
  /opt/anaconda3/envs/scprinter-local/bin/python -u v4/v4_plot_regions.py \
      --regions "chr1:2054-4054" --title "example" --force-extract
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy.sparse import issparse


# ── Constants ──────────────────────────────────────────────────────────────
CLASS_COLORS = {
    "proto_gain": "#E64B35", "stable": "#808080", "leaf_gain": "#4DBBD5",
    "unknown": "#D9D9D9",
}
TILE_SIZE = 10
CONTEXT_RADIUS = 100
N_TILES = 180


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # Region selection (mutually exclusive sources)
    g = p.add_argument_group("Region selection")
    g.add_argument("--regions", nargs="+",
                   help="Region IDs (resized coords, e.g. chr1:2054-4054)")
    g.add_argument("--region-file",
                   help="File with one acr_id (resized coord) per line")
    g.add_argument("--tile-table",
                   help="TSV from v4_03c_acr_tile_table.py")
    g.add_argument("--filter-score", default="TFBS",
                   help="score_type filter for tile-table (default: TFBS)")
    g.add_argument("--filter-class",
                   help="acr_class filter (e.g. leaf_gain)")
    g.add_argument("--filter-group",
                   help="overlap_group filter (e.g. leaf_only)")
    g.add_argument("--top-n", type=int, default=5,
                   help="Number of top regions from tile-table (by n_tiles)")

    # Data paths chr1:26642945-26644945
    p.add_argument("--fp-dir", default="v4/3_PRINT/FP")
    p.add_argument("--tfbs-dir", default="v4/3_PRINT/TFBS")
    p.add_argument("--nucbs-dir", default="v4/3_PRINT/NucBS")
    p.add_argument("--printer-dir", default="v4/3_PRINT")
    p.add_argument("--bs-dir", default="results/v4_03a_binding_scores")
    p.add_argument("--metadata", default="v4/data/acr_metadata.tsv.gz")
    p.add_argument("--mapping", default="data/acr_native_to_resized.tsv")

    # Display options
    p.add_argument("--title", default="v4_region_viewer")
    p.add_argument("--tfbs-pct", type=float, default=5,
                   help="TFBS top percentile for tile coloring (default: 5)")
    p.add_argument("--fp-vmax", type=float, default=None,
                   help="FP heatmap color max (auto if not set)")
    p.add_argument("--scale-min", type=float, default=2)
    p.add_argument("--scale-max", type=float, default=100)
    p.add_argument("--outdir", default="results/v4_region_viewer")

    # Phase control
    p.add_argument("--plot-only", action="store_true",
                   help="Skip extraction, load cached NPZ and plot only")
    p.add_argument("--force-extract", action="store_true",
                   help="Force re-extraction even if NPZ exists")
    p.add_argument("--zoom-native", action="store_true",
                   help="Zoom x-axis to native ACR boundaries (+ small padding)")
    p.add_argument("--zoom-pad", type=int, default=200,
                   help="Padding (bp) around native ACR when --zoom-native (default: 200)")

    return p.parse_args()


def load_metadata(metadata_path, mapping_path):
    """Load ACR metadata with resized coordinate keys."""
    meta = pd.read_csv(metadata_path, sep="\t")
    col_map = {"edgeR_logFC": "logFC", "edgeR_fdr": "fdr", "edgeR_logCPM": "logCPM",
               "width": "acr_width"}
    meta = meta.rename(columns={k: v for k, v in col_map.items() if k in meta.columns})
    meta["native_str"] = (meta["chr"].str.lower() + ":" +
                          meta["start"].astype(str) + "-" +
                          meta["end"].astype(str))
    mapping = pd.read_csv(mapping_path, sep="\t")
    nat_to_resized = dict(zip(mapping["native_str"], mapping["resized_str"]))
    meta["resized_str"] = meta["native_str"].map(nat_to_resized)
    return meta.set_index("resized_str")


def select_regions(args):
    """Return list of resized-coord region IDs."""
    if args.regions:
        return args.regions
    if args.region_file:
        with open(args.region_file) as f:
            return [line.strip() for line in f if line.strip()]
    if args.tile_table:
        df = pd.read_csv(args.tile_table, sep="\t")
        mask = df["score_type"] == args.filter_score
        if args.filter_class:
            mask &= df["acr_class"] == args.filter_class
        if args.filter_group:
            mask &= df["overlap_group"] == args.filter_group
        sub = df[mask].nlargest(args.top_n, "n_tiles")
        print(f"[SELECT] {len(sub)} regions from tile table "
              f"({args.filter_score}/{args.filter_class}/{args.filter_group})")
        return sub["acr_id"].tolist()
    print("ERROR: Provide --regions, --region-file, or --tile-table")
    sys.exit(1)


def load_fp_region(fp_adata, region_key):
    """Load FP tensor for one region: shape (n_scales, 2000)."""
    # Keys may use underscore or colon separator
    for sep in [":", "_"]:
        k = region_key.replace(":", sep).replace("-", "_" if sep == "_" else "-")
        if k in fp_adata.obsm:
            arr = np.asarray(fp_adata.obsm[k])
            if arr.ndim == 3:
                return arr[0]  # (1, n_scales, 2000) -> (n_scales, 2000)
            return arr
    # Try with underscore for the colon
    k2 = region_key.replace(":", "_").replace("-", "_")
    if k2 in fp_adata.obsm:
        arr = np.asarray(fp_adata.obsm[k2])
        if arr.ndim == 3:
            return arr[0]
        return arr
    return None


def load_insertion_region(printer_adata, chrom, start, end):
    """Extract Tn5 insertion counts for a genomic window."""
    key = f"insertion_{chrom}"
    if key not in printer_adata.obsm:
        return None
    ins = printer_adata.obsm[key]
    if issparse(ins):
        row = ins[0, start:end].toarray().ravel()
    else:
        row = np.asarray(ins[0, start:end]).ravel()
    return row.astype(float)


def tile_bp_positions():
    """Tile center positions within the 2000bp window."""
    return np.arange(N_TILES) * TILE_SIZE + (TILE_SIZE // 2 + CONTEXT_RADIUS)


def plot_one_region(region_id, meta_row, fp_leaf, fp_proto, ins_leaf, ins_proto,
                    tfbs_leaf_prob, tfbs_proto_prob, tfbs_threshold_l, tfbs_threshold_p,
                    nucbs_leaf_prob, nucbs_proto_prob, nucbs_threshold_l, nucbs_threshold_p,
                    scales, outdir, title_prefix, fp_vmax,
                    zoom_native=False, zoom_pad=200):
    """Plot a single ACR region."""

    n_scales = len(scales)
    scale_mask = np.ones(n_scales, dtype=bool)  # all scales

    # Parse genomic coords from region_id (resized 2000bp window)
    parts = region_id.replace(":", "_").replace("-", "_").split("_")
    chrom_str = parts[0]
    reg_start = int(parts[1])
    reg_end = int(parts[2])

    # Native ACR boundaries (original peak, before resizing to 2000bp)
    native_start = meta_row.get("start", None) if meta_row is not None else None
    native_end = meta_row.get("end", None) if meta_row is not None else None
    has_native = native_start is not None and native_end is not None
    if has_native:
        native_start = int(native_start)
        native_end = int(native_end)

    # X-axis display range: zoom to native ACR or show full 2000bp
    if zoom_native and has_native and native_start is not None and native_end is not None:
        x_lo = max(reg_start, int(native_start) - zoom_pad)
        x_hi = min(reg_end, int(native_end) + zoom_pad)
    else:
        x_lo = reg_start
        x_hi = reg_end

    def _draw_native_boundaries(ax, yspan=True):
        """Draw vertical dashed lines at native ACR boundaries on a data panel."""
        if not has_native:
            return
        for pos, label in [(native_start, ""), (native_end, "")]:
            ax.axvline(pos, color="k", ls="--", lw=0.8, alpha=0.6, zorder=5)
        # Light shading outside native ACR
        ax.axvspan(reg_start, native_start, alpha=0.06, color="grey", zorder=0)
        ax.axvspan(native_end, reg_end, alpha=0.06, color="grey", zorder=0)

    # ── Figure layout ──────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(5, 1, height_ratios=[0.4, 3, 3, 1.5, 0.6],
                           hspace=0.35)

    # Shared x-axis formatter: suppress scientific notation offset
    from matplotlib.ticker import FuncFormatter, MaxNLocator
    def _genomic_fmt(x, pos):
        return f"{int(x):,}"

    # Color normalization for FP
    if fp_leaf is not None and fp_proto is not None:
        if fp_vmax is None:
            fp_vmax_auto = max(np.nanpercentile(fp_leaf, 99),
                               np.nanpercentile(fp_proto, 99))
            fp_vmax_auto = max(fp_vmax_auto, 0.1)
        else:
            fp_vmax_auto = fp_vmax
    else:
        fp_vmax_auto = 1.0

    # Metadata for title
    acr_class = meta_row.get("acr_class", "unknown") if meta_row is not None else "unknown"
    logFC = meta_row.get("logFC", np.nan) if meta_row is not None else np.nan
    fdr = meta_row.get("fdr", np.nan) if meta_row is not None else np.nan
    gene = meta_row.get("nearest_gene", "") if meta_row is not None else ""
    context = meta_row.get("genomic_context", "") if meta_row is not None else ""
    native = meta_row.get("native_str", region_id) if meta_row is not None else region_id

    # ── Panel 0: Metadata banner ───────────────────────────────────────────
    ax_meta = fig.add_subplot(gs[0])
    ax_meta.set_xlim(0, 1)
    ax_meta.set_ylim(0, 1)
    ax_meta.axis("off")

    cls_color = CLASS_COLORS.get(acr_class, "#999")
    banner = (f"{native}  ({region_id})")
    ax_meta.text(0.0, 0.7, banner, fontsize=11, fontweight="bold",
                 va="center", family="monospace")

    info_parts = []
    if acr_class != "unknown":
        info_parts.append(f"Class: {acr_class}")
    if not np.isnan(logFC):
        info_parts.append(f"logFC: {logFC:+.2f}")
    if not np.isnan(fdr):
        info_parts.append(f"FDR: {fdr:.1e}")
    if context:
        info_parts.append(f"{context}")
    if gene:
        info_parts.append(f"Gene: {gene}")
    if has_native:
        info_parts.append(f"ACR: {native_end - native_start} bp")
    info_str = "  |  ".join(info_parts)
    ax_meta.text(0.0, 0.2, info_str, fontsize=9, va="center", color="#444")

    # Color bar for class
    ax_meta.add_patch(Rectangle((0.92, 0.1), 0.07, 0.8,
                                fc=cls_color, ec="k", lw=0.5))
    ax_meta.text(0.955, 0.5, acr_class.replace("_", "\n"), fontsize=7,
                 ha="center", va="center", fontweight="bold", color="white")

    # Genomic position axis (bp)
    xbp = np.arange(2000) + reg_start

    # ── Panel 1 & 2: FP heatmaps ──────────────────────────────────────────
    for panel_idx, (fp_data, cond_label) in enumerate(
            [(fp_leaf, "Leaf FP"), (fp_proto, "Proto FP")], start=1):
        ax = fig.add_subplot(gs[panel_idx])
        if fp_data is not None:
            im = ax.imshow(fp_data[scale_mask], aspect="auto",
                           origin="lower", cmap="YlOrRd",
                           vmin=0, vmax=fp_vmax_auto,
                           extent=[reg_start, reg_start + 2000,
                                   scales[0], scales[-1]])
            ax.set_ylabel("Scale (bp)", fontsize=9)
            # Y-axis ticks at meaningful scales
            yticks = [s for s in [5, 10, 20, 50, 80, 100] if scales[0] <= s <= scales[-1]]
            ax.set_yticks(yticks)
            ax.set_yticklabels([str(s) for s in yticks], fontsize=8)
            if panel_idx == 1:
                cax = ax.inset_axes([1.02, 0.1, 0.015, 0.8])
                cb = fig.colorbar(im, cax=cax)
                cb.set_label("-log10(p)", fontsize=8)
        else:
            ax.text(0.5, 0.5, f"{cond_label}: no data", ha="center",
                    va="center", transform=ax.transAxes, fontsize=10, color="red")
        _draw_native_boundaries(ax)
        ax.set_xlim(x_lo, x_hi)
        ax.xaxis.set_major_formatter(FuncFormatter(_genomic_fmt))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
        ax.set_title(cond_label, fontsize=10, fontweight="bold", loc="left")
        if panel_idx == 1:
            ax.set_xticklabels([])

    # ── Panel 3: Tn5 insertion profile ─────────────────────────────────────
    ax_ins = fig.add_subplot(gs[3])
    if ins_leaf is not None and ins_proto is not None:
        # Smooth Tn5 insertions: ±4bp kernel (Tn5 insertion site width)
        kernel = np.ones(9) / 9  # 9bp window centered on insertion
        ins_leaf_sm = np.convolve(ins_leaf, kernel, mode="same")
        ins_proto_sm = np.convolve(ins_proto, kernel, mode="same")
        xbp_ins = np.arange(len(ins_leaf)) + reg_start
        ax_ins.fill_between(xbp_ins, ins_leaf_sm, alpha=0.4, color="#D62728",
                            label="Leaf", linewidth=0)
        ax_ins.fill_between(xbp_ins, ins_proto_sm, alpha=0.4, color="#1F77B4",
                            label="Proto", linewidth=0)
        ax_ins.plot(xbp_ins, ins_leaf_sm, color="#D62728", lw=0.5, alpha=0.7)
        ax_ins.plot(xbp_ins, ins_proto_sm, color="#1F77B4", lw=0.5, alpha=0.7)
        ax_ins.legend(fontsize=8, loc="upper right", framealpha=0.8)
        ax_ins.set_ylabel("Tn5 insertions", fontsize=9)
    else:
        ax_ins.text(0.5, 0.5, "Insertion data not available", ha="center",
                    va="center", transform=ax_ins.transAxes, color="red")
    _draw_native_boundaries(ax_ins)
    ax_ins.set_title("Tn5 insertion profile", fontsize=10,
                     fontweight="bold", loc="left")
    ax_ins.set_xlim(x_lo, x_hi)
    ax_ins.xaxis.set_major_formatter(FuncFormatter(_genomic_fmt))
    ax_ins.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))

    # ── Panel 4: TFBS tile strip ───────────────────────────────────────────
    ax_tiles = fig.add_subplot(gs[4])
    tile_bp = tile_bp_positions() + reg_start  # genomic coords

    if tfbs_leaf_prob is not None and tfbs_proto_prob is not None:
        # NucBS: rectangles (wider, nucleosome-scale)
        # Two rows: leaf (top), proto (bottom)
        for row_idx, (probs, label, ybase) in enumerate([
            (nucbs_leaf_prob, "Leaf NucBS", 0.55),
            (nucbs_proto_prob, "Proto NucBS", 0.05),
        ]):
            if probs is None:
                continue
            nuc_thresh = nucbs_threshold_l if row_idx == 0 else nucbs_threshold_p
            for j in range(N_TILES):
                bp = tile_bp[j]
                is_occ = probs[j] > nuc_thresh
                if is_occ:
                    color = "#D62728" if row_idx == 0 else "#1F77B4"
                    alpha = min(0.3 + 0.7 * probs[j], 1.0)
                else:
                    color = "#E0E0E0"
                    alpha = 0.3
                ax_tiles.add_patch(Rectangle(
                    (bp - TILE_SIZE / 2, ybase), TILE_SIZE, 0.4,
                    fc=color, alpha=alpha, ec="none"))
            ax_tiles.text(x_lo - 5, ybase + 0.2, label, fontsize=7,
                          ha="right", va="center", fontweight="bold")

        # TFBS: triangles (point-like, TF binding)
        for j in range(N_TILES):
            bp = tile_bp[j]
            if tfbs_leaf_prob[j] > tfbs_threshold_l:
                ax_tiles.plot(bp, 0.98, "v", color="#D62728",
                              markersize=4, alpha=0.8)
            if tfbs_proto_prob[j] > tfbs_threshold_p:
                ax_tiles.plot(bp, -0.02, "^", color="#1F77B4",
                              markersize=4, alpha=0.8)

        # Native ACR boundaries on tile strip
        _draw_native_boundaries(ax_tiles)

        ax_tiles.set_ylim(-0.1, 1.1)
        ax_tiles.set_yticks([])
        n_nuc_l = int(nucbs_leaf_prob[nucbs_leaf_prob > nucbs_threshold_l].shape[0]) if nucbs_leaf_prob is not None else 0
        n_nuc_p = int(nucbs_proto_prob[nucbs_proto_prob > nucbs_threshold_p].shape[0]) if nucbs_proto_prob is not None else 0
        n_tf_l = int((tfbs_leaf_prob > tfbs_threshold_l).sum())
        n_tf_p = int((tfbs_proto_prob > tfbs_threshold_p).sum())
        ax_tiles.set_title(
            f"NucBS tiles (L:{n_nuc_l} P:{n_nuc_p})  |  "
            f"TFBS ▼▲ (L:{n_tf_l} P:{n_tf_p})",
            fontsize=9, fontweight="bold", loc="left")
    else:
        ax_tiles.text(0.5, 0.5, "TFBS data not available", ha="center",
                      va="center", transform=ax_tiles.transAxes, color="red")
        ax_tiles.set_yticks([])

    ax_tiles.set_xlim(x_lo, x_hi)
    ax_tiles.xaxis.set_major_formatter(FuncFormatter(_genomic_fmt))
    ax_tiles.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
    ax_tiles.set_xlabel(f"Genomic position ({chrom_str})", fontsize=9)

    # ── Save ───────────────────────────────────────────────────────────────
    safe_id = region_id.replace(":", "_").replace("-", "_")
    fname = f"{title_prefix}_{safe_id}"
    for fmt in ("pdf", "png"):
        fig.savefig(os.path.join(outdir, f"{fname}.{fmt}"),
                    dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [SAVE] {fname}.pdf/.png")


def _npz_path(outdir, title):
    """Path to the cached extraction NPZ."""
    return os.path.join(outdir, f"{title}_extracted.npz")


def extract_regions(args, regions, meta):
    """Extract FP, Tn5, TFBS/NucBS data for selected regions → NPZ."""
    import anndata

    # Binding score NPZs (fast)
    bs_leaf = np.load(os.path.join(args.bs_dir, "_bs_leaf.npz"), allow_pickle=True)
    bs_proto = np.load(os.path.join(args.bs_dir, "_bs_proto.npz"), allow_pickle=True)
    bs_regions = list(bs_leaf["region_strs"])
    bs_idx = {r: i for i, r in enumerate(bs_regions)}

    tfbs_cutoff = 100 - args.tfbs_pct
    tfbs_thresh_l = float(np.percentile(bs_leaf["TFBS_prob"], tfbs_cutoff))
    tfbs_thresh_p = float(np.percentile(bs_proto["TFBS_prob"], tfbs_cutoff))
    nucbs_thresh_l = float(np.percentile(bs_leaf["NucBS_prob"], tfbs_cutoff))
    nucbs_thresh_p = float(np.percentile(bs_proto["NucBS_prob"], tfbs_cutoff))
    print(f"[TFBS] thresholds: leaf > {tfbs_thresh_l:.4f}, proto > {tfbs_thresh_p:.4f}")
    print(f"[NucBS] thresholds: leaf > {nucbs_thresh_l:.4f}, proto > {nucbs_thresh_p:.4f}")

    # FP h5ads (backed)
    print("[FP] Loading FP h5ads (backed)...")
    fp_leaf_ad = anndata.read_h5ad(
        os.path.join(args.fp_dir, "leaf_merged__ALL.h5ad"), backed="r")
    fp_proto_ad = anndata.read_h5ad(
        os.path.join(args.fp_dir, "proto_merged__ALL.h5ad"), backed="r")
    if "scales" in fp_leaf_ad.uns:
        scales = np.array(fp_leaf_ad.uns["scales"])
    else:
        scales = np.arange(2, 101)
    print(f"  Scales: {len(scales)} ({scales[0]}–{scales[-1]} bp)")

    # Printer h5ads for insertions
    print("[INS] Loading printer h5ads for Tn5 insertions...")
    printer_leaf = anndata.read_h5ad(
        os.path.join(args.printer_dir, "printer_leaf_merged_bulk.h5ad"), backed="r")
    printer_proto = anndata.read_h5ad(
        os.path.join(args.printer_dir, "printer_proto_merged_bulk.h5ad"), backed="r")

    # Extract per region
    print(f"\n[EXTRACT] {len(regions)} regions...")
    data = {}
    for region_id in regions:
        print(f"  {region_id}")
        rec = {}

        # Metadata
        if region_id in meta.index:
            m = meta.loc[region_id]
            for col in ["acr_class", "logFC", "fdr", "nearest_gene",
                        "genomic_context", "native_str", "start", "end"]:
                if col in m.index:
                    rec[f"meta_{col}"] = m[col]

        # FP
        fp_l = load_fp_region(fp_leaf_ad, region_id)
        fp_p = load_fp_region(fp_proto_ad, region_id)
        if fp_l is not None:
            rec["fp_leaf"] = fp_l
        if fp_p is not None:
            rec["fp_proto"] = fp_p

        # Tn5 insertions
        parts = region_id.replace(":", "_").replace("-", "_").split("_")
        chrom = parts[0]
        reg_start, reg_end = int(parts[1]), int(parts[2])
        ins_l = load_insertion_region(printer_leaf, chrom, reg_start, reg_end)
        ins_p = load_insertion_region(printer_proto, chrom, reg_start, reg_end)
        if ins_l is not None:
            rec["ins_leaf"] = ins_l
        if ins_p is not None:
            rec["ins_proto"] = ins_p

        # TFBS / NucBS
        idx = bs_idx.get(region_id)
        if idx is not None:
            rec["tfbs_leaf"] = bs_leaf["TFBS_prob"][idx]
            rec["tfbs_proto"] = bs_proto["TFBS_prob"][idx]
            rec["nucbs_leaf"] = bs_leaf["NucBS_prob"][idx]
            rec["nucbs_proto"] = bs_proto["NucBS_prob"][idx]

        data[region_id] = rec

    # Save to NPZ
    npz_out = _npz_path(args.outdir, args.title)
    save_dict = {
        "regions": np.array(regions),
        "scales": scales,
        "tfbs_thresh_l": tfbs_thresh_l,
        "tfbs_thresh_p": tfbs_thresh_p,
        "nucbs_thresh_l": nucbs_thresh_l,
        "nucbs_thresh_p": nucbs_thresh_p,
        "tfbs_pct": args.tfbs_pct,
    }
    for i, region_id in enumerate(regions):
        rec = data[region_id]
        for key, val in rec.items():
            if isinstance(val, np.ndarray):
                save_dict[f"r{i}_{key}"] = val
            else:
                save_dict[f"r{i}_{key}"] = np.array(val)
    np.savez_compressed(npz_out, **save_dict)
    sz_mb = os.path.getsize(npz_out) / 1e6
    print(f"\n[SAVE] {npz_out} ({sz_mb:.1f} MB, {len(regions)} regions)")
    return npz_out


def load_extracted(npz_path):
    """Load cached extraction NPZ → (regions, scales, thresholds, per-region data)."""
    npz = np.load(npz_path, allow_pickle=True)
    regions = list(npz["regions"])
    scales = npz["scales"]
    tfbs_thresh_l = float(npz["tfbs_thresh_l"])
    tfbs_thresh_p = float(npz["tfbs_thresh_p"])
    nucbs_thresh_l = float(npz["nucbs_thresh_l"]) if "nucbs_thresh_l" in npz else None
    nucbs_thresh_p = float(npz["nucbs_thresh_p"]) if "nucbs_thresh_p" in npz else None

    data = {}
    for i, region_id in enumerate(regions):
        rec = {}
        prefix = f"r{i}_"
        for key in npz.files:
            if key.startswith(prefix):
                field = key[len(prefix):]
                val = npz[key]
                # Scalar metadata stored as 0-d arrays
                if val.ndim == 0:
                    rec[field] = val.item()
                else:
                    rec[field] = val
        data[region_id] = rec

    return regions, scales, tfbs_thresh_l, tfbs_thresh_p, nucbs_thresh_l, nucbs_thresh_p, data


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    npz_file = _npz_path(args.outdir, args.title)

    # ── Phase 1: Extract (or skip if sentinel exists) ──────────────────────
    if args.plot_only:
        if not os.path.exists(npz_file):
            print(f"ERROR: --plot-only but NPZ not found: {npz_file}")
            sys.exit(1)
        print(f"[PLOT-ONLY] Loading cached {npz_file}")
    elif os.path.exists(npz_file) and not args.force_extract:
        print(f"[SENTINEL] NPZ exists, skipping extraction: {npz_file}")
        print(f"  Use --force-extract to re-extract")
    else:
        # Need to select regions and extract
        regions = select_regions(args)
        print(f"\n=== v4 Region Viewer: EXTRACT {len(regions)} regions ===")
        print(f"  Title: {args.title}")
        print(f"  TFBS top {args.tfbs_pct}%\n")

        meta = load_metadata(args.metadata, args.mapping)
        print(f"[META] {len(meta)} ACRs loaded")
        extract_regions(args, regions, meta)

    # ── Phase 2: Plot from NPZ ─────────────────────────────────────────────
    (regions, scales, tfbs_thresh_l, tfbs_thresh_p,
     nucbs_thresh_l, nucbs_thresh_p, data) = load_extracted(npz_file)
    print(f"\n[PLOT] Rendering {len(regions)} regions...")
    if args.zoom_native:
        print(f"  Zoom: native ACR ± {args.zoom_pad} bp")

    for region_id in regions:
        print(f"\n  Region: {region_id}")
        rec = data[region_id]

        # Reconstruct meta_row dict from stored metadata
        meta_row = {}
        for key, val in rec.items():
            if key.startswith("meta_"):
                meta_row[key[5:]] = val

        plot_one_region(
            region_id, meta_row,
            rec.get("fp_leaf"), rec.get("fp_proto"),
            rec.get("ins_leaf"), rec.get("ins_proto"),
            rec.get("tfbs_leaf"), rec.get("tfbs_proto"),
            tfbs_thresh_l, tfbs_thresh_p,
            rec.get("nucbs_leaf"), rec.get("nucbs_proto"),
            nucbs_thresh_l, nucbs_thresh_p,
            scales, args.outdir, args.title, args.fp_vmax,
            zoom_native=args.zoom_native, zoom_pad=args.zoom_pad,
        )

    print(f"\n[DONE] {len(regions)} regions saved to {args.outdir}/")


if __name__ == "__main__":
    main()
