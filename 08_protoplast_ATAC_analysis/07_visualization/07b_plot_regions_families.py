#!/usr/bin/env python3
"""
v4_03f — Per-ACR region viewer annotated by enriched TF families.

For a given Venn-diagram category (score_type × acr_class × overlap_group):
  1. Reads v4_03e enrichment results to find significant TF families
  2. Loads v3 motif hits and intersects with active tiles
  3. Selects all ACRs where at least one enriched family has a hit at an active tile
  4. Generates per-ACR PDF/PNG with family motif annotation strips

Two-phase design (same as v4_plot_regions):
  EXTRACT (SLURM): reads h5ad files → saves sentinel NPZ
  PLOT (local): reads NPZ → renders figures

Panels (top to bottom):
  0. Metadata banner
  1. Leaf FP heatmap (scale × position)
  2. Proto FP heatmap (scale × position)
  3. Tn5 insertion profile (smoothed)
  4. Family motif annotation strips (one row per enriched family with hits)
  5. TFBS / NucBS tile strip (original v4_plot_regions Panel 4)

Usage:
  # Extract + plot on SLURM
  sbatch v4/v4_03f_plot_regions_families.sh \\
      --score-type TFBS --acr-class leaf_gain --overlap-group leaf_only

  # Plot only from cached NPZ (local)
  /opt/anaconda3/bin/python3 -u v4/v4_03f_plot_regions_families.py \\
      --plot-only --score-type TFBS --acr-class leaf_gain --overlap-group leaf_only

  # With zoom
  /opt/anaconda3/bin/python3 -u v4/v4_03f_plot_regions_families.py \\
      --plot-only --zoom-native --score-type TFBS --acr-class leaf_gain --overlap-group leaf_only
"""

import argparse
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

# ── Add v4 dir to path for sibling imports ────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_LIB_DIR = os.path.join(os.path.dirname(_SCRIPT_DIR), "lib")
for _d in (_SCRIPT_DIR, _LIB_DIR):
    if _d not in sys.path:
        sys.path.insert(0, _d)

# ── Constants ─────────────────────────────────────────────────────────────
CLASS_COLORS = {
    "proto_gain": "#E64B35", "stable": "#808080", "leaf_gain": "#4DBBD5",
    "unknown": "#D9D9D9",
}
TILE_SIZE = 10
CONTEXT_RADIUS = 100
N_TILES = 180
TILE_BP = np.arange(N_TILES) * TILE_SIZE + (TILE_SIZE // 2 + CONTEXT_RADIUS)
# [105, 115, ..., 1895]

MAX_FAMILY_ROWS = 8  # cap family strips per region


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # Category selection
    g = p.add_argument_group("Category selection")
    g.add_argument("--score-type", required=True, choices=["TFBS", "NucBS"],
                   help="Score type for enrichment lookup")
    g.add_argument("--acr-class", required=True,
                   choices=["proto_gain", "stable", "leaf_gain"])
    g.add_argument("--overlap-group", required=True,
                   choices=["shared", "leaf_only", "proto_only"])

    # Enrichment filtering
    g2 = p.add_argument_group("Enrichment filtering")
    g2.add_argument("--direction", default="both",
                    choices=["leaf_enriched", "proto_enriched", "both"],
                    help="Direction filter for enriched families (default: both)")
    g2.add_argument("--fdr-threshold", type=float, default=0.05,
                    help="FDR cutoff for enriched families (default: 0.05)")
    g2.add_argument("--max-total", type=int, default=0,
                    help="Max ACRs to plot (0 = unlimited)")

    # Data paths
    g3 = p.add_argument_group("Data paths")
    g3.add_argument("--enrichment-tsv", default=None,
                    help="v4_03e enrichment TSV (auto-detected if not set)")
    g3.add_argument("--bs-dir", default="results/v4_03a_binding_scores")
    g3.add_argument("--delta-dir", default="results/v4_03d_binding_deltas")
    g3.add_argument("--fp-dir", default="v4/3_PRINT/FP")
    g3.add_argument("--printer-dir", default="v4/3_PRINT")
    g3.add_argument("--metadata", default="v4/data/acr_metadata.tsv.gz")
    g3.add_argument("--mapping", default="data/acr_native_to_resized.tsv")
    g3.add_argument("--v3-chunks", default="data/v3_chunks")
    g3.add_argument("--sig-metadata",
                    default="data/motif_signatures/signature_metadata.tsv")
    g3.add_argument("--enrichment-dir",
                    default="results/v4_03e_family_enrichment")

    # Display
    g4 = p.add_argument_group("Display")
    g4.add_argument("--tfbs-pct", type=float, default=5)
    g4.add_argument("--nucbs-pct", type=float, default=2)
    g4.add_argument("--zoom-native", action="store_true")
    g4.add_argument("--zoom-pad", type=int, default=200)
    g4.add_argument("--fp-vmax", type=float, default=None)
    g4.add_argument("--native-only", action="store_true",
                    help="Restrict to tiles inside native ACR boundaries")

    # Phase control
    g5 = p.add_argument_group("Phase control")
    g5.add_argument("--plot-only", action="store_true")
    g5.add_argument("--force-extract", action="store_true")
    g5.add_argument("--outdir", default="results/v4_03f_region_families")

    return p.parse_args()


# ── Helper: percentile tag (same as v4_03e) ──────────────────────────────
def _pct_tag(tfbs_pct, nucbs_pct):
    def _fmt(v):
        return str(int(v)) if v == int(v) else f"{v:.1f}"
    return f"_tf{_fmt(tfbs_pct)}_nuc{_fmt(nucbs_pct)}"


def _title_from_args(args):
    return f"{args.score_type}_{args.acr_class}_{args.overlap_group}"


# ── Load enriched families from v4_03e ────────────────────────────────────
def load_enriched_families(enrichment_tsv, score_type, acr_class,
                           overlap_group, direction, fdr_threshold):
    """Return DataFrame of enriched families (cols: family, OR, fdr)."""
    df = pd.read_csv(enrichment_tsv, sep="\t")
    mask = ((df["score_type"] == score_type) &
            (df["acr_class"] == acr_class) &
            (df["group"] == overlap_group) &
            (df["fdr"] < fdr_threshold))
    if direction != "both":
        mask &= (df["direction"] == direction)
    result = df.loc[mask, ["family", "OR", "fdr", "direction"]].copy()
    result = result.sort_values("OR", ascending=False)
    # Deduplicate family (keep best OR across directions)
    result = result.drop_duplicates(subset=["family"], keep="first")
    return result.reset_index(drop=True)


# ── Build active tile masks ───────────────────────────────────────────────
def _build_masks(bs_leaf, bs_proto, score_type, tfbs_pct, nucbs_pct,
                 native_mask=None):
    """Build per-condition boolean masks from binding score NPZs.
    Returns (leaf_mask, proto_mask, thresh_l, thresh_p) for the target score type,
    plus the same for the other score type (for Panel 5 tile strip)."""
    tf_cutoff = 100 - tfbs_pct
    nuc_cutoff = 100 - nucbs_pct

    tf_thresh_l = float(np.percentile(bs_leaf["TFBS_prob"], tf_cutoff))
    tf_thresh_p = float(np.percentile(bs_proto["TFBS_prob"], tf_cutoff))
    nuc_thresh_l = float(np.percentile(bs_leaf["NucBS_prob"], nuc_cutoff))
    nuc_thresh_p = float(np.percentile(bs_proto["NucBS_prob"], nuc_cutoff))

    tf_leaf = bs_leaf["TFBS_prob"] > tf_thresh_l
    tf_proto = bs_proto["TFBS_prob"] > tf_thresh_p
    nuc_leaf = bs_leaf["NucBS_prob"] > nuc_thresh_l
    nuc_proto = bs_proto["NucBS_prob"] > nuc_thresh_p

    if native_mask is not None:
        tf_leaf = tf_leaf & native_mask
        tf_proto = tf_proto & native_mask
        nuc_leaf = nuc_leaf & native_mask
        nuc_proto = nuc_proto & native_mask

    if score_type == "TFBS":
        return (tf_leaf, tf_proto, tf_thresh_l, tf_thresh_p,
                nuc_thresh_l, nuc_thresh_p)
    else:
        return (nuc_leaf, nuc_proto, nuc_thresh_l, nuc_thresh_p,
                tf_thresh_l, tf_thresh_p)


def _overlap_mask(leaf_mask, proto_mask, overlap_group):
    """Combine condition masks into overlap group boolean mask."""
    if overlap_group == "shared":
        return leaf_mask & proto_mask
    elif overlap_group == "leaf_only":
        return leaf_mask & ~proto_mask
    elif overlap_group == "proto_only":
        return ~leaf_mask & proto_mask
    raise ValueError(f"Unknown overlap_group: {overlap_group}")


# ── ACR class mapping ─────────────────────────────────────────────────────
def _build_region_to_class(metadata_path, mapping_path, region_strs):
    """Map resized region strings → ACR class."""
    meta = pd.read_csv(metadata_path, sep="\t",
                       usecols=["chr", "start", "end", "acr_class"])
    meta["native_str"] = (meta["chr"].str.lower() + ":" +
                          meta["start"].astype(str) + "-" +
                          meta["end"].astype(str))
    mapping = pd.read_csv(mapping_path, sep="\t")
    nat_to_resized = dict(zip(mapping["native_str"], mapping["resized_str"]))
    meta["resized_str"] = meta["native_str"].map(nat_to_resized)
    r2c = dict(zip(meta["resized_str"], meta["acr_class"]))
    return np.array([r2c.get(r, "unknown") for r in region_strs])


# ── Load motif hits (from v4_03e pattern) ─────────────────────────────────
def _load_motif_hits(v3_chunks_dir, mapping_path, sig_metadata_path,
                     target_families=None):
    """Load v3 signature hits, map to resized tile indices.

    Returns DataFrame: resized_str, tile_idx, motif_id, family, hit_bp_resized
    """
    mapping = pd.read_csv(mapping_path, sep="\t")
    nat_to_resized = dict(zip(mapping["native_str"], mapping["resized_str"]))
    resized_starts = dict(zip(
        mapping["resized_str"],
        mapping["resized_str"].str.split(":").str[1]
        .str.split("-").str[0].astype(int)))

    sig_meta = pd.read_csv(sig_metadata_path, sep="\t",
                           usecols=["signature_id", "primary_family"])
    sig_to_family = dict(zip(sig_meta["signature_id"],
                             sig_meta["primary_family"]))

    chunk_files = sorted(glob.glob(
        os.path.join(v3_chunks_dir, "chunk_*/motif_hits.tsv.gz")))
    print(f"  Loading {len(chunk_files)} chunk files...")

    frames = []
    for i, f in enumerate(chunk_files):
        df = pd.read_csv(f, sep="\t",
                         usecols=["region_str", "motif_id", "hit_center"])
        df["resized_str"] = df["region_str"].map(nat_to_resized)
        df = df.dropna(subset=["resized_str"])
        df["family"] = df["motif_id"].map(sig_to_family)
        df = df.dropna(subset=["family"])
        if target_families is not None:
            df = df[df["family"].isin(target_families)]
        if df.empty:
            continue
        df["resized_start"] = df["resized_str"].map(resized_starts)
        df["hit_bp_resized"] = df["hit_center"] - df["resized_start"]
        tile_half = TILE_SIZE // 2
        df["tile_idx"] = ((df["hit_bp_resized"] - TILE_BP[0] + tile_half)
                          // TILE_SIZE).astype(int)
        df = df[(df["tile_idx"] >= 0) & (df["tile_idx"] < N_TILES)]
        frames.append(df[["resized_str", "tile_idx", "motif_id", "family",
                          "hit_bp_resized"]])
        if (i + 1) % 10 == 0:
            print(f"    {i+1}/{len(chunk_files)} chunks...", flush=True)

    if not frames:
        return pd.DataFrame(columns=["resized_str", "tile_idx", "motif_id",
                                     "family", "hit_bp_resized"])
    hits = pd.concat(frames, ignore_index=True)
    print(f"  Hits (enriched families): {len(hits):,}")
    return hits


# ── Select ACRs with family hits at active tiles ─────────────────────────
def select_acrs_with_family_hits(region_strs, region_classes, acr_class,
                                 overlap_mask, hits_df, enriched_families,
                                 max_total=0):
    """Find ACRs where at least one enriched family has a hit at an active tile.

    Returns:
        selected_regions: list of region IDs
        region_family_hits: {region_id: [(family, motif_id, hit_bp_resized), ...]}
    """
    # Build set of (resized_str, tile_idx) for active tiles in category
    cls_idx = region_classes == acr_class
    cat_mask = overlap_mask & cls_idx[:, None]

    active_keys = set()
    cat_rows, cat_cols = np.where(cat_mask)
    for r, c in zip(cat_rows, cat_cols):
        active_keys.add((region_strs[r], int(c)))

    if not active_keys:
        return [], {}

    # Filter hits to those at active tiles AND from enriched families
    fam_set = set(enriched_families)
    qualified = hits_df[
        hits_df["family"].isin(fam_set)
    ].copy()

    # Keep only hits at active tile positions
    qualified["key"] = list(zip(qualified["resized_str"],
                                qualified["tile_idx"]))
    qualified = qualified[qualified["key"].isin(active_keys)]

    if qualified.empty:
        return [], {}

    # Build per-region hit map
    region_family_hits = {}
    for _, row in qualified.iterrows():
        rid = row["resized_str"]
        if rid not in region_family_hits:
            region_family_hits[rid] = []
        region_family_hits[rid].append(
            (row["family"], row["motif_id"], int(row["hit_bp_resized"])))

    selected = sorted(region_family_hits.keys())

    if max_total > 0 and len(selected) > max_total:
        # Prioritize regions with more family hits
        selected.sort(key=lambda r: -len(region_family_hits[r]))
        selected = selected[:max_total]
        region_family_hits = {r: region_family_hits[r] for r in selected}

    return selected, region_family_hits


# ── Metadata loader (from v4_plot_regions pattern) ────────────────────────
def _load_metadata(metadata_path, mapping_path):
    """Load ACR metadata indexed by resized coordinate string."""
    meta = pd.read_csv(metadata_path, sep="\t")
    mapping = pd.read_csv(mapping_path, sep="\t")
    nat_to_resized = dict(zip(mapping["native_str"], mapping["resized_str"]))
    meta["native_str"] = (meta["chr"].str.lower() + ":" +
                          meta["start"].astype(str) + "-" +
                          meta["end"].astype(str))
    meta["resized_str"] = meta["native_str"].map(nat_to_resized)
    meta = meta.dropna(subset=["resized_str"])
    meta = meta.set_index("resized_str")
    return meta


# ── FP / insertion loaders (from v4_plot_regions pattern) ─────────────────
def _load_fp_region(fp_adata, region_id):
    """Extract FP tensor (n_scales, 2000) from backed h5ad."""
    key = region_id.replace("-", "_")
    if key not in fp_adata.obsm:
        key = region_id
    if key not in fp_adata.obsm:
        return None
    arr = np.asarray(fp_adata.obsm[key])
    if arr.ndim == 3:
        arr = arr[0]  # (1, n_scales, 2000) → (n_scales, 2000)
    return arr


def _load_insertion_region(printer_adata, chrom, start, end, cache=None):
    """Extract Tn5 insertion counts for a genomic window.

    Uses optional cache dict to avoid repeated sparse→dense conversion
    for the same chromosome (each is ~30M×8 bytes = 240MB).
    """
    try:
        from scipy.sparse import issparse
        key = f"insertion_{chrom}"
        if key not in printer_adata.obsm:
            return None
        if cache is not None and key in cache:
            arr = cache[key]
        else:
            arr = printer_adata.obsm[key]
            if issparse(arr):
                arr = arr.toarray()
            arr = np.asarray(arr).ravel()
            if cache is not None:
                cache[key] = arr
        return arr[start:end] if end <= len(arr) else None
    except Exception:
        return None


def _tile_bp_positions():
    """Tile center positions within 2000bp window."""
    return np.arange(N_TILES) * TILE_SIZE + (TILE_SIZE // 2 + CONTEXT_RADIUS)


# ── Extract ───────────────────────────────────────────────────────────────
def extract_regions(args, regions, meta, region_family_hits,
                    enriched_fam_df, family_colors):
    """Extract FP, Tn5, TFBS/NucBS + family hit data → NPZ."""
    import anndata

    # Binding score NPZs
    bs_leaf = np.load(os.path.join(args.bs_dir, "_bs_leaf.npz"),
                      allow_pickle=True)
    bs_proto = np.load(os.path.join(args.bs_dir, "_bs_proto.npz"),
                       allow_pickle=True)
    bs_regions = list(bs_leaf["region_strs"])
    bs_idx = {r: i for i, r in enumerate(bs_regions)}

    tfbs_thresh_l = float(np.percentile(
        bs_leaf["TFBS_prob"], 100 - args.tfbs_pct))
    tfbs_thresh_p = float(np.percentile(
        bs_proto["TFBS_prob"], 100 - args.tfbs_pct))
    nucbs_thresh_l = float(np.percentile(
        bs_leaf["NucBS_prob"], 100 - args.nucbs_pct))
    nucbs_thresh_p = float(np.percentile(
        bs_proto["NucBS_prob"], 100 - args.nucbs_pct))
    print(f"[TFBS] thresholds: L>{tfbs_thresh_l:.4f}, P>{tfbs_thresh_p:.4f}")
    print(f"[NucBS] thresholds: L>{nucbs_thresh_l:.4f}, P>{nucbs_thresh_p:.4f}")

    # FP h5ads
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

    # Printer h5ads for Tn5 insertions
    print("[INS] Loading printer h5ads...")
    printer_leaf = anndata.read_h5ad(
        os.path.join(args.printer_dir, "printer_leaf_merged_bulk.h5ad"),
        backed="r")
    printer_proto = anndata.read_h5ad(
        os.path.join(args.printer_dir, "printer_proto_merged_bulk.h5ad"),
        backed="r")

    # Extract per region
    print(f"\n[EXTRACT] {len(regions)} regions...")
    save_dict = {
        "regions": np.array(regions),
        "scales": scales,
        "tfbs_thresh_l": tfbs_thresh_l,
        "tfbs_thresh_p": tfbs_thresh_p,
        "nucbs_thresh_l": nucbs_thresh_l,
        "nucbs_thresh_p": nucbs_thresh_p,
        "tfbs_pct": args.tfbs_pct,
        "enriched_families": np.array(list(enriched_fam_df["family"])),
        "enriched_families_or": np.array(list(enriched_fam_df["OR"])),
        "enriched_families_fdr": np.array(list(enriched_fam_df["fdr"])),
        "family_colors": np.array(family_colors),
    }

    # Insertion caches to avoid repeated sparse→dense conversion per chrom
    ins_cache_leaf = {}
    ins_cache_proto = {}

    for i, region_id in enumerate(regions):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"  {i+1}/{len(regions)}: {region_id}", flush=True)

        prefix = f"r{i}_"

        # Metadata
        if region_id in meta.index:
            m = meta.loc[region_id]
            for col in ["acr_class", "logFC", "fdr", "nearest_gene",
                        "genomic_context", "native_str", "start", "end"]:
                if col in m.index:
                    save_dict[f"{prefix}meta_{col}"] = np.array(m[col])

        # FP (float32 to save memory)
        fp_l = _load_fp_region(fp_leaf_ad, region_id)
        fp_p = _load_fp_region(fp_proto_ad, region_id)
        if fp_l is not None:
            save_dict[f"{prefix}fp_leaf"] = fp_l.astype(np.float32)
        if fp_p is not None:
            save_dict[f"{prefix}fp_proto"] = fp_p.astype(np.float32)

        # Tn5 insertions (cached per chromosome)
        parts = region_id.replace(":", "_").replace("-", "_").split("_")
        chrom = parts[0]
        reg_start, reg_end = int(parts[1]), int(parts[2])
        ins_l = _load_insertion_region(printer_leaf, chrom, reg_start,
                                       reg_end, cache=ins_cache_leaf)
        ins_p = _load_insertion_region(printer_proto, chrom, reg_start,
                                       reg_end, cache=ins_cache_proto)
        if ins_l is not None:
            save_dict[f"{prefix}ins_leaf"] = ins_l
        if ins_p is not None:
            save_dict[f"{prefix}ins_proto"] = ins_p

        # TFBS / NucBS (copy single row, float32)
        idx = bs_idx.get(region_id)
        if idx is not None:
            save_dict[f"{prefix}tfbs_leaf"] = \
                bs_leaf["TFBS_prob"][idx].astype(np.float32)
            save_dict[f"{prefix}tfbs_proto"] = \
                bs_proto["TFBS_prob"][idx].astype(np.float32)
            save_dict[f"{prefix}nucbs_leaf"] = \
                bs_leaf["NucBS_prob"][idx].astype(np.float32)
            save_dict[f"{prefix}nucbs_proto"] = \
                bs_proto["NucBS_prob"][idx].astype(np.float32)

        # Family hits for this region
        fhits = region_family_hits.get(region_id, [])
        if fhits:
            save_dict[f"{prefix}fhit_families"] = np.array(
                [h[0] for h in fhits])
            save_dict[f"{prefix}fhit_motif_ids"] = np.array(
                [h[1] for h in fhits])
            save_dict[f"{prefix}fhit_bp"] = np.array(
                [h[2] for h in fhits])

    # Release insertion caches before NPZ compression
    del ins_cache_leaf, ins_cache_proto

    title = _title_from_args(args)
    cat_dir = os.path.join(args.outdir, title)
    os.makedirs(cat_dir, exist_ok=True)
    npz_out = os.path.join(cat_dir, f"{title}_extracted.npz")
    np.savez_compressed(npz_out, **save_dict)
    sz_mb = os.path.getsize(npz_out) / 1e6
    print(f"\n[SAVE] {npz_out} ({sz_mb:.1f} MB, {len(regions)} regions)")
    return npz_out


# ── Load extracted ────────────────────────────────────────────────────────
def load_extracted(npz_path):
    """Load sentinel NPZ → regions, scales, thresholds, data, family info."""
    npz = np.load(npz_path, allow_pickle=True)
    regions = list(npz["regions"])
    scales = npz["scales"]
    tfbs_thresh_l = float(npz["tfbs_thresh_l"])
    tfbs_thresh_p = float(npz["tfbs_thresh_p"])
    nucbs_thresh_l = float(npz.get("nucbs_thresh_l", 0))
    nucbs_thresh_p = float(npz.get("nucbs_thresh_p", 0))
    enriched_families = list(npz.get("enriched_families", []))
    family_colors = list(npz.get("family_colors", []))
    enriched_or = list(npz.get("enriched_families_or", []))

    data = {}
    for i, region_id in enumerate(regions):
        rec = {}
        prefix = f"r{i}_"
        for key in npz.files:
            if key.startswith(prefix):
                field = key[len(prefix):]
                val = npz[key]
                if val.ndim == 0:
                    rec[field] = val.item()
                else:
                    rec[field] = val
        data[region_id] = rec

    return (regions, scales, tfbs_thresh_l, tfbs_thresh_p,
            nucbs_thresh_l, nucbs_thresh_p,
            enriched_families, family_colors, enriched_or, data)


# ── Plot one region ───────────────────────────────────────────────────────
def plot_one_region_with_families(
        region_id, meta_row, fp_leaf, fp_proto, ins_leaf, ins_proto,
        tfbs_leaf_prob, tfbs_proto_prob, tfbs_thresh_l, tfbs_thresh_p,
        nucbs_leaf_prob, nucbs_proto_prob, nucbs_thresh_l, nucbs_thresh_p,
        scales, family_hits_in_region, all_enriched_families, family_color_map,
        outdir, title_prefix, fp_vmax, zoom_native=False, zoom_pad=200):
    """Plot single ACR: Panels 0-3 (standard) + Panel 4 (families) + Panel 5 (tiles)."""

    from matplotlib.ticker import FuncFormatter, MaxNLocator

    # Parse genomic coords
    parts = region_id.replace(":", "_").replace("-", "_").split("_")
    chrom_str = parts[0]
    reg_start = int(parts[1])
    reg_end = int(parts[2])

    # Native ACR boundaries
    native_start = meta_row.get("start") if meta_row else None
    native_end = meta_row.get("end") if meta_row else None
    has_native = native_start is not None and native_end is not None
    if has_native:
        native_start = int(native_start)
        native_end = int(native_end)

    # X-axis range
    if zoom_native and has_native:
        x_lo = max(reg_start, native_start - zoom_pad)
        x_hi = min(reg_end, native_end + zoom_pad)
    else:
        x_lo, x_hi = reg_start, reg_end

    def _draw_native(ax):
        if not has_native:
            return
        for pos in (native_start, native_end):
            ax.axvline(pos, color="k", ls="--", lw=0.8, alpha=0.6, zorder=5)
        ax.axvspan(reg_start, native_start, alpha=0.06, color="grey", zorder=0)
        ax.axvspan(native_end, reg_end, alpha=0.06, color="grey", zorder=0)

    def _genomic_fmt(x, pos):
        return f"{int(x):,}"

    # Determine families present in this region
    families_here = sorted(set(h[0] for h in family_hits_in_region),
                           key=lambda f: -(family_color_map.get(f, (0,))[1]
                                           if isinstance(family_color_map.get(f), tuple)
                                           else 0))
    # Re-sort by enrichment OR (highest first)
    fam_or = {}
    for f in all_enriched_families:
        if f in family_color_map:
            fam_or[f] = family_color_map[f][1] if isinstance(family_color_map[f], tuple) else 0
    families_here = sorted(families_here, key=lambda f: -fam_or.get(f, 0))

    n_fam = min(len(families_here), MAX_FAMILY_ROWS)
    extra_fam = len(families_here) - n_fam
    families_here = families_here[:n_fam]

    # Build per-family hit positions (genomic bp)
    fam_hit_bps = {f: [] for f in families_here}
    for fam, motif_id, bp_resized in family_hits_in_region:
        if fam in fam_hit_bps:
            fam_hit_bps[fam].append(bp_resized + reg_start)

    # ── Figure layout ─────────────────────────────────────────────────
    panel4_h = max(0.6, 0.3 * n_fam)
    height_ratios = [0.4, 3, 3, 1.5, panel4_h, 0.6]
    fig_h = 10 + max(0, (n_fam - 2) * 0.4)
    fig = plt.figure(figsize=(14, fig_h))
    gs = gridspec.GridSpec(6, 1, height_ratios=height_ratios, hspace=0.35)

    # FP color normalization
    if fp_leaf is not None and fp_proto is not None:
        if fp_vmax is None:
            fp_vmax_auto = max(np.nanpercentile(fp_leaf, 99),
                               np.nanpercentile(fp_proto, 99))
            fp_vmax_auto = max(fp_vmax_auto, 0.1)
        else:
            fp_vmax_auto = fp_vmax
    else:
        fp_vmax_auto = 1.0

    # Metadata
    acr_class = meta_row.get("acr_class", "unknown") if meta_row else "unknown"
    logFC = meta_row.get("logFC", np.nan) if meta_row else np.nan
    fdr = meta_row.get("fdr", np.nan) if meta_row else np.nan
    gene = meta_row.get("nearest_gene", "") if meta_row else ""
    context = meta_row.get("genomic_context", "") if meta_row else ""
    native = meta_row.get("native_str", region_id) if meta_row else region_id

    # ── Panel 0: Metadata banner ──────────────────────────────────────
    ax_meta = fig.add_subplot(gs[0])
    ax_meta.set_xlim(0, 1); ax_meta.set_ylim(0, 1)
    ax_meta.axis("off")

    cls_color = CLASS_COLORS.get(acr_class, "#999")
    ax_meta.text(0.0, 0.7, f"{native}  ({region_id})",
                 fontsize=11, fontweight="bold", va="center",
                 family="monospace")
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
    info_parts.append(f"Families: {len(families_here)}")
    ax_meta.text(0.0, 0.2, "  |  ".join(info_parts),
                 fontsize=9, va="center", color="#444")
    ax_meta.add_patch(Rectangle((0.92, 0.1), 0.07, 0.8,
                                fc=cls_color, ec="k", lw=0.5))
    ax_meta.text(0.955, 0.5, acr_class.replace("_", "\n"), fontsize=7,
                 ha="center", va="center", fontweight="bold", color="white")

    xbp = np.arange(2000) + reg_start

    # ── Panels 1-2: FP heatmaps ──────────────────────────────────────
    for panel_idx, (fp_data, cond_label) in enumerate(
            [(fp_leaf, "Leaf FP"), (fp_proto, "Proto FP")], start=1):
        ax = fig.add_subplot(gs[panel_idx])
        if fp_data is not None:
            im = ax.imshow(fp_data, aspect="auto", origin="lower",
                           cmap="Blues", vmin=0, vmax=fp_vmax_auto,
                           extent=[reg_start, reg_start + 2000,
                                   scales[0], scales[-1]])
            ax.set_ylabel("Scale (bp)", fontsize=9)
            yticks = [s for s in [5, 10, 20, 50, 80, 100]
                      if scales[0] <= s <= scales[-1]]
            ax.set_yticks(yticks)
            ax.set_yticklabels([str(s) for s in yticks], fontsize=8)
            if panel_idx == 1:
                cax = ax.inset_axes([1.02, 0.1, 0.015, 0.8])
                cb = fig.colorbar(im, cax=cax)
                cb.set_label("-log10(p)", fontsize=8)
        else:
            ax.text(0.5, 0.5, f"{cond_label}: no data", ha="center",
                    va="center", transform=ax.transAxes, fontsize=10,
                    color="red")
        _draw_native(ax)
        ax.set_xlim(x_lo, x_hi)
        ax.xaxis.set_major_formatter(FuncFormatter(_genomic_fmt))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
        ax.set_title(cond_label, fontsize=10, fontweight="bold", loc="left")
        if panel_idx == 1:
            ax.set_xticklabels([])

    # ── Panel 3: Tn5 insertion profile ────────────────────────────────
    ax_ins = fig.add_subplot(gs[3])
    if ins_leaf is not None and ins_proto is not None:
        kernel = np.ones(9) / 9
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
    _draw_native(ax_ins)
    ax_ins.set_title("Tn5 insertion profile", fontsize=10,
                     fontweight="bold", loc="left")
    ax_ins.set_xlim(x_lo, x_hi)
    ax_ins.xaxis.set_major_formatter(FuncFormatter(_genomic_fmt))
    ax_ins.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))

    # ── Panel 4: Family motif annotation strips ───────────────────────
    ax_fam = fig.add_subplot(gs[4])
    tile_bp = _tile_bp_positions() + reg_start

    if n_fam > 0:
        for row_idx, fam in enumerate(families_here):
            y_center = n_fam - row_idx - 0.5
            color = family_color_map.get(fam, ("#888888", 0))[0] \
                if isinstance(family_color_map.get(fam), tuple) \
                else family_color_map.get(fam, "#888888")

            # Background: faint rects at active tile positions
            # (we don't have the full active mask in NPZ, so just show
            # a subtle row background)
            ax_fam.axhspan(y_center - 0.4, y_center + 0.4,
                           alpha=0.04, color=color, zorder=0)
            # Thin separator line
            if row_idx > 0:
                ax_fam.axhline(n_fam - row_idx, color="#E0E0E0",
                               lw=0.5, zorder=1)

            # Motif hit markers (exact positions)
            bps = fam_hit_bps.get(fam, [])
            if bps:
                ax_fam.plot(bps, [y_center] * len(bps), "v",
                            color=color, markersize=6, alpha=0.85, zorder=5)

            # Family label
            ax_fam.text(x_lo - (x_hi - x_lo) * 0.005, y_center, fam,
                        fontsize=7, ha="right", va="center",
                        fontweight="bold", color=color, clip_on=False)

        ax_fam.set_ylim(0, n_fam)
        ax_fam.set_yticks([])
        title_str = f"Enriched family motif hits ({n_fam} families)"
        if extra_fam > 0:
            title_str += f"  [{extra_fam} more not shown]"
        ax_fam.set_title(title_str, fontsize=9, fontweight="bold", loc="left")
    else:
        ax_fam.text(0.5, 0.5, "No enriched family hits in this region",
                    ha="center", va="center", transform=ax_fam.transAxes,
                    color="grey")
        ax_fam.set_yticks([])

    _draw_native(ax_fam)
    ax_fam.set_xlim(x_lo, x_hi)
    ax_fam.xaxis.set_major_formatter(FuncFormatter(_genomic_fmt))
    ax_fam.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
    ax_fam.set_xticklabels([])

    # ── Panel 5: Original TFBS/NucBS tile strip ───────────────────────
    ax_tiles = fig.add_subplot(gs[5])
    if tfbs_leaf_prob is not None and tfbs_proto_prob is not None:
        # NucBS rectangles (two rows)
        for row_idx, (probs, label, ybase) in enumerate([
            (nucbs_leaf_prob, "Leaf NucBS", 0.55),
            (nucbs_proto_prob, "Proto NucBS", 0.05),
        ]):
            if probs is None:
                continue
            nuc_thresh = nucbs_thresh_l if row_idx == 0 else nucbs_thresh_p
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

        # TFBS triangles
        for j in range(N_TILES):
            bp = tile_bp[j]
            if tfbs_leaf_prob[j] > tfbs_thresh_l:
                ax_tiles.plot(bp, 0.98, "v", color="#D62728",
                              markersize=4, alpha=0.8)
            if tfbs_proto_prob[j] > tfbs_thresh_p:
                ax_tiles.plot(bp, -0.02, "^", color="#1F77B4",
                              markersize=4, alpha=0.8)

        _draw_native(ax_tiles)
        ax_tiles.set_ylim(-0.1, 1.1)
        ax_tiles.set_yticks([])
        n_nuc_l = int((nucbs_leaf_prob > nucbs_thresh_l).sum()) if nucbs_leaf_prob is not None else 0
        n_nuc_p = int((nucbs_proto_prob > nucbs_thresh_p).sum()) if nucbs_proto_prob is not None else 0
        n_tf_l = int((tfbs_leaf_prob > tfbs_thresh_l).sum())
        n_tf_p = int((tfbs_proto_prob > tfbs_thresh_p).sum())
        ax_tiles.set_title(
            f"NucBS tiles (L:{n_nuc_l} P:{n_nuc_p})  |  "
            f"TFBS ▼▲ (L:{n_tf_l} P:{n_tf_p})",
            fontsize=9, fontweight="bold", loc="left")
    else:
        ax_tiles.text(0.5, 0.5, "Tile data not available", ha="center",
                      va="center", transform=ax_tiles.transAxes, color="red")
        ax_tiles.set_yticks([])

    ax_tiles.set_xlim(x_lo, x_hi)
    ax_tiles.xaxis.set_major_formatter(FuncFormatter(_genomic_fmt))
    ax_tiles.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
    ax_tiles.set_xlabel(f"Genomic position ({chrom_str})", fontsize=9)

    # ── Save ──────────────────────────────────────────────────────────
    safe_id = region_id.replace(":", "_").replace("-", "_")
    fname = f"{title_prefix}_{safe_id}"
    for fmt in ("pdf", "png"):
        fig.savefig(os.path.join(outdir, f"{fname}.{fmt}"),
                    dpi=200, bbox_inches="tight")
    plt.close(fig)
    return fname


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    title = _title_from_args(args)
    cat_dir = os.path.join(args.outdir, title)
    os.makedirs(cat_dir, exist_ok=True)
    npz_file = os.path.join(cat_dir, f"{title}_extracted.npz")

    print(f"=== v4_03f: Family Region Viewer ===")
    print(f"  Category: {title}")
    print(f"  Direction: {args.direction}")
    print(f"  FDR threshold: {args.fdr_threshold}")

    # ── Phase 0: Family & ACR selection ───────────────────────────────
    # Auto-detect enrichment TSV
    if args.enrichment_tsv is None:
        tag = _pct_tag(args.tfbs_pct, args.nucbs_pct)
        if args.native_only:
            tag += "_native"
        candidates = [
            os.path.join(args.enrichment_dir, f"enrichment_results{tag}.tsv"),
            os.path.join(args.enrichment_dir, "enrichment_results.tsv"),
        ]
        for c in candidates:
            if os.path.exists(c):
                args.enrichment_tsv = c
                break
        if args.enrichment_tsv is None:
            print(f"ERROR: No enrichment TSV found. Tried: {candidates}")
            sys.exit(1)
    print(f"  Enrichment TSV: {args.enrichment_tsv}")

    # Load enriched families
    enriched_df = load_enriched_families(
        args.enrichment_tsv, args.score_type, args.acr_class,
        args.overlap_group, args.direction, args.fdr_threshold)

    if enriched_df.empty:
        print(f"\n  No enriched families found (FDR < {args.fdr_threshold}).")
        print("  Try a higher --fdr-threshold or different category.")
        sys.exit(0)

    print(f"\n  Enriched families ({len(enriched_df)}):")
    for _, row in enriched_df.iterrows():
        print(f"    {row['family']:20s}  OR={row['OR']:.2f}  "
              f"FDR={row['fdr']:.1e}  ({row['direction']})")

    # Assign colors (consistent across all plots in this category)
    cmap = matplotlib.colormaps.get_cmap("tab20")
    family_color_map = {}
    for i, (_, row) in enumerate(enriched_df.iterrows()):
        hex_color = matplotlib.colors.rgb2hex(cmap(i % 20))
        family_color_map[row["family"]] = (hex_color, row["OR"])
    family_colors = [family_color_map[f][0] for f in enriched_df["family"]]

    if not args.plot_only:
        # Load binding scores + build masks
        print("\n[LOAD] Binding scores...")
        bs_leaf = np.load(os.path.join(args.bs_dir, "_bs_leaf.npz"),
                          allow_pickle=True)
        bs_proto = np.load(os.path.join(args.bs_dir, "_bs_proto.npz"),
                           allow_pickle=True)
        region_strs = np.array(bs_leaf["region_strs"])

        # Native-only masking
        native_mask = None
        if args.native_only:
            from _tile_utils import build_native_tile_mask
            print("[NATIVE] Building native tile mask...")
            native_mask, _ = build_native_tile_mask(
                region_strs, args.metadata, args.mapping)

        # Build active masks for target score type
        tf_cutoff = 100 - args.tfbs_pct
        nuc_cutoff = 100 - args.nucbs_pct

        if args.score_type == "TFBS":
            thresh_l = float(np.percentile(bs_leaf["TFBS_prob"], tf_cutoff))
            thresh_p = float(np.percentile(bs_proto["TFBS_prob"], tf_cutoff))
            leaf_mask = bs_leaf["TFBS_prob"] > thresh_l
            proto_mask = bs_proto["TFBS_prob"] > thresh_p
        else:
            thresh_l = float(np.percentile(bs_leaf["NucBS_prob"], nuc_cutoff))
            thresh_p = float(np.percentile(bs_proto["NucBS_prob"], nuc_cutoff))
            leaf_mask = bs_leaf["NucBS_prob"] > thresh_l
            proto_mask = bs_proto["NucBS_prob"] > thresh_p

        if native_mask is not None:
            leaf_mask = leaf_mask & native_mask
            proto_mask = proto_mask & native_mask

        overlap = _overlap_mask(leaf_mask, proto_mask, args.overlap_group)
        print(f"  {args.score_type} {args.overlap_group} active tiles: "
              f"{overlap.sum():,}")

        # ACR class mapping
        print("\n[MAP] ACR classes...")
        rcls = _build_region_to_class(args.metadata, args.mapping,
                                      region_strs)

        # Load motif hits (filtered to enriched families)
        print("\n[LOAD] Motif hits for enriched families...")
        target_fams = set(enriched_df["family"])
        hits = _load_motif_hits(args.v3_chunks, args.mapping,
                                args.sig_metadata, target_fams)

        # Select ACRs
        print("\n[SELECT] ACRs with family hits at active tiles...")
        selected, region_family_hits = select_acrs_with_family_hits(
            region_strs, rcls, args.acr_class, overlap, hits,
            target_fams, args.max_total)

        print(f"  Selected: {len(selected)} ACRs")
        if not selected:
            print("  No qualifying ACRs found.")
            sys.exit(0)

        # Per-family counts
        fam_counts = {}
        for rid, fhits in region_family_hits.items():
            for fam, _, _ in fhits:
                fam_counts[fam] = fam_counts.get(fam, 0) + 1
        print("  Per-family ACR counts:")
        for fam in enriched_df["family"]:
            if fam in fam_counts:
                print(f"    {fam:20s}: {fam_counts[fam]} ACRs")

        # Save summary TSV
        summary_rows = []
        for rid in selected:
            fhits = region_family_hits[rid]
            fams = sorted(set(h[0] for h in fhits))
            summary_rows.append({
                "acr_id": rid,
                "acr_class": args.acr_class,
                "n_family_hits": len(fhits),
                "n_families": len(fams),
                "families": ";".join(fams),
            })
        summary_df = pd.DataFrame(summary_rows)
        summary_path = os.path.join(cat_dir, f"{title}_summary.tsv")
        summary_df.to_csv(summary_path, sep="\t", index=False)
        print(f"  Summary: {summary_path}")

    # ── Phase 1: Extract or skip ──────────────────────────────────────
    if args.plot_only:
        if not os.path.exists(npz_file):
            print(f"ERROR: --plot-only but NPZ not found: {npz_file}")
            sys.exit(1)
        print(f"\n[PLOT-ONLY] Loading cached {npz_file}")
    elif os.path.exists(npz_file) and not args.force_extract:
        print(f"\n[SENTINEL] NPZ exists, skipping extraction: {npz_file}")
        print(f"  Use --force-extract to re-extract")
    else:
        print(f"\n[EXTRACT] {len(selected)} regions to NPZ...")
        meta = _load_metadata(args.metadata, args.mapping)
        extract_regions(args, selected, meta, region_family_hits,
                        enriched_df, family_colors)

    # ── Phase 2: Plot ─────────────────────────────────────────────────
    (regions, scales, tfbs_thresh_l, tfbs_thresh_p,
     nucbs_thresh_l, nucbs_thresh_p,
     enriched_families, fam_colors, enriched_or,
     data) = load_extracted(npz_file)

    # Rebuild color map from NPZ data
    family_color_map = {}
    for i, fam in enumerate(enriched_families):
        c = fam_colors[i] if i < len(fam_colors) else "#888888"
        or_val = enriched_or[i] if i < len(enriched_or) else 0
        family_color_map[fam] = (c, or_val)

    print(f"\n[PLOT] Rendering {len(regions)} regions...")
    if args.zoom_native:
        print(f"  Zoom: native ACR ± {args.zoom_pad} bp")

    n_plotted = 0
    for region_id in regions:
        rec = data[region_id]

        # Reconstruct meta_row
        meta_row = {}
        for key, val in rec.items():
            if key.startswith("meta_"):
                meta_row[key[5:]] = val

        # Reconstruct family hits
        fhit_fams = rec.get("fhit_families", np.array([]))
        fhit_mids = rec.get("fhit_motif_ids", np.array([]))
        fhit_bps = rec.get("fhit_bp", np.array([]))
        family_hits = list(zip(fhit_fams, fhit_mids, fhit_bps))

        fname = plot_one_region_with_families(
            region_id, meta_row,
            rec.get("fp_leaf"), rec.get("fp_proto"),
            rec.get("ins_leaf"), rec.get("ins_proto"),
            rec.get("tfbs_leaf"), rec.get("tfbs_proto"),
            tfbs_thresh_l, tfbs_thresh_p,
            rec.get("nucbs_leaf"), rec.get("nucbs_proto"),
            nucbs_thresh_l, nucbs_thresh_p,
            scales, family_hits, enriched_families, family_color_map,
            cat_dir, title, args.fp_vmax,
            zoom_native=args.zoom_native, zoom_pad=args.zoom_pad,
        )
        n_plotted += 1
        if n_plotted % 20 == 0:
            print(f"  {n_plotted}/{len(regions)} plotted...")

    print(f"\n[DONE] {n_plotted} regions saved to {cat_dir}/")


if __name__ == "__main__":
    main()
