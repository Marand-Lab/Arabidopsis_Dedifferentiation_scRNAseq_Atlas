#!/usr/bin/env python3
"""v3_08c — Extract spatial nucleosome profiles around WRKY hits.

Reads per-rep h5ad FP tensors and extracts nucleosome-scale (>80 bp)
and TF-scale (2-10 bp) spatial profiles centered on WRKY motif hits.

Preprocessing per scale (before averaging across scales):
  1. OLS-residualize on confounders {log_width, logCPM, genomic_context}
  2. Z-score across hits (pooled leaf + proto reference)
  3. Average z-scored values across scales per condition

Output NPZ arrays (consumed by v3_08b_wrky_summary.py Page 6):
  - Per-hit ±500 bp windows: leaf, proto, delta (nuc + TF scales)
  - Per-ACR full 2000 bp profiles: leaf, proto, delta (nuc scale)
  - Raw (un-preprocessed) per-hit windows for reference

Must run on cluster (h5ad access).

Usage:
  conda activate scprinter-cpu
  python -u v3_08c_wrky_nucleosome_extract.py
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import glob
import gc

BASE = os.path.dirname(os.path.abspath(__file__))

# ── Config ───────────────────────────────────────────────────────────────────
EXCLUDE_REPS = {3}
ACTIVE_REPS = sorted({1, 2, 3} - EXCLUDE_REPS)
LEAF_IDS = [f"leaf_rep{r}" for r in ACTIVE_REPS]
PROTO_IDS = [f"proto_rep{r}" for r in ACTIVE_REPS]
ALL_IDS = LEAF_IDS + PROTO_IDS
N_LEAF = len(LEAF_IDS)

WRKY_SIGS = {"sig_121", "sig_122"}
NUC_SCALE_MIN = 80   # bp
TF_SCALE_MIN = 2
TF_SCALE_MAX = 10
WINDOW_HALF = 500    # ±500 bp around hit center
REGION_WIDTH = 2000


def parse_args():
    p = argparse.ArgumentParser(
        description="v3_08c: Extract nucleosome profiles around WRKY hits")
    p.add_argument("--print-dir", default="3_PRINT_per_rep")
    p.add_argument("--genome-obj", default="3_PRINT_bulk/At_genome_OBJ")
    p.add_argument("--chunk-dir", default="data/v3_chunks")
    p.add_argument("--coord-mapping",
                   default="data/acr_native_to_resized.tsv")
    p.add_argument("--acr-metadata", default="data/acr_metadata.tsv.gz")
    p.add_argument("--outdir", default="results/v3_08_gradient_boosting")
    p.add_argument("--tmp-dir", default="/tmp")
    return p.parse_args()


def load_wrky_hits(chunk_dir):
    """Load all WRKY hit positions from v3 chunks."""
    chunk_dirs = sorted(glob.glob(os.path.join(chunk_dir, "chunk_*")))
    all_hits = []
    for cd in chunk_dirs:
        hit_path = os.path.join(cd, "motif_hits.tsv.gz")
        if not os.path.exists(hit_path):
            continue
        df = pd.read_csv(hit_path, sep="\t")
        wrky = df[df["motif_id"].isin(WRKY_SIGS)].copy()
        if not wrky.empty:
            all_hits.append(wrky)
    if not all_hits:
        return pd.DataFrame()
    hits = pd.concat(all_hits, ignore_index=True)
    print(f"  Loaded {len(hits):,} WRKY hits from {len(chunk_dirs)} chunks",
          flush=True)
    return hits


def copy_h5ad_to_tmp(print_dir, tmp_dir):
    """Copy h5ad files to local /tmp for fast I/O."""
    import shutil
    paths = {}
    for sid in ALL_IDS:
        src = os.path.join(print_dir, f"printer_{sid}_bulk.h5ad")
        dst = os.path.join(tmp_dir, f"printer_{sid}_bulk.h5ad")
        if os.path.exists(dst):
            print(f"  [tmp] {sid} already cached", flush=True)
        elif os.path.exists(src):
            print(f"  [tmp] Copying {sid}...", flush=True)
            shutil.copy2(src, dst)
        else:
            print(f"  [WARN] Missing: {src}", flush=True)
            continue
        paths[sid] = dst
    return paths


def _build_hit_confounder_matrix(acr_ids, acr_confounders):
    """Build per-hit confounder design matrix (n_hits × n_conf).

    Called ONCE per scale group, then reused across all scales and positions.
    Returns (C_full, finite_rows) where C_full includes intercept column.
    """
    conf_cols = acr_confounders.columns.tolist()
    hit_conf = np.full((len(acr_ids), len(conf_cols)), np.nan)
    # Vectorised lookup via pandas reindex
    idx_series = acr_confounders.reindex(acr_ids)
    hit_conf = idx_series.values.astype(float)
    finite_rows = np.all(np.isfinite(hit_conf), axis=1)
    C_full = np.column_stack([np.ones(len(acr_ids)), hit_conf])
    return C_full, finite_rows


def _residualize_matrix(values_mat, C_full, finite_rows):
    """OLS-residualize a 2-D matrix (n_hits × n_positions) in one lstsq call.

    Replaces the old per-column loop: instead of n_positions separate lstsq
    fits, one call with RHS = (n_hits_finite × n_positions) handles all
    positions at once.  ~1000× faster for the nuc-scale phase.

    Parameters
    ----------
    values_mat : (n_hits, n_positions) float array
    C_full     : (n_hits, n_conf+1) design matrix with intercept
    finite_rows: (n_hits,) bool — rows with valid confounder values

    Returns residualized matrix, same shape (NaN where input was NaN).
    """
    from numpy.linalg import lstsq

    result = values_mat.copy()
    C_fin = C_full[finite_rows]          # (n_fin, n_conf+1)
    Y_fin = values_mat[finite_rows]      # (n_fin, n_positions)

    # Mask positions that are entirely non-finite
    col_finite = np.isfinite(Y_fin).any(axis=0)
    if col_finite.sum() == 0:
        return result

    beta, _, _, _ = lstsq(C_fin[:, :], Y_fin[:, col_finite], rcond=None)
    pred_full = C_full @ beta            # (n_hits, n_cols_finite)
    result[np.ix_(finite_rows, col_finite)] = (
        Y_fin[:, col_finite] - pred_full[finite_rows])
    return result


def _build_confounder_df(acr_meta, coord_map):
    """Build confounder DataFrame indexed by resized_str."""
    meta = acr_meta.copy()
    meta["acr_id"] = meta["acr_id"].astype(str)

    # Rename to match conventions
    if "width" in meta.columns and "acr_width" not in meta.columns:
        meta["acr_width"] = meta["width"]
    if "edgeR_logCPM" in meta.columns and "logCPM" not in meta.columns:
        meta["logCPM"] = meta["edgeR_logCPM"]

    meta["log_width"] = np.log1p(meta["acr_width"])

    # Map native → resized
    n2r = dict(zip(coord_map["native_str"], coord_map["resized_str"]))
    meta["resized_str"] = meta["acr_id"].map(n2r)
    meta = meta.dropna(subset=["resized_str"])
    meta = meta.set_index("resized_str")

    # Build confounder columns
    cols = ["log_width"]
    if "logCPM" in meta.columns:
        cols.append("logCPM")
    if "genomic_context" in meta.columns:
        dummies = pd.get_dummies(meta["genomic_context"], prefix="gc",
                                 drop_first=True)
        meta = pd.concat([meta, dummies], axis=1)
        cols.extend(dummies.columns.tolist())

    return meta[cols].astype(float)


def extract_profiles(hits, h5ad_paths, genome_obj_path, coord_map,
                     acr_meta, scales):
    """Extract spatial profiles from h5ad for all WRKY hits.

    Returns dict with per-hit and per-ACR profiles:
    - Raw: per-condition and delta
    - Preprocessed (residualized + z-scored): per-condition and delta
    """
    import scprinter as scp
    import pickle

    # Load genome object
    with open(genome_obj_path, "rb") as f:
        genome = pickle.load(f)

    # Build coordinate mappings
    native_to_resized = dict(zip(coord_map["native_str"],
                                  coord_map["resized_str"]))
    resized_start_map = {}
    for _, row in coord_map.iterrows():
        resized_start_map[row["resized_str"]] = int(
            row["resized_str"].split(":")[1].split("-")[0])

    # ACR class mapping (native → class)
    class_map = dict(zip(acr_meta["acr_id"].astype(str),
                          acr_meta["acr_class"]))

    # Scale indices
    nuc_scale_idx = np.where(scales >= NUC_SCALE_MIN)[0]
    tf_scale_idx = np.where((scales >= TF_SCALE_MIN) &
                             (scales <= TF_SCALE_MAX))[0]

    # Map hits to resized coordinates
    hits = hits.copy()
    hits["native_str"] = hits["region_str"].str.replace(
        r"^Chr", "chr", regex=True)
    hits["resized_str"] = hits["native_str"].map(native_to_resized)
    hits = hits.dropna(subset=["resized_str"])
    hits["resized_start"] = hits["resized_str"].map(resized_start_map)
    hits = hits.dropna(subset=["resized_start"])
    hits["resized_start"] = hits["resized_start"].astype(int)
    hits["center_idx"] = hits["hit_center"].astype(int) - hits["resized_start"]

    # Filter valid positions
    hits = hits[(hits["center_idx"] >= 0) &
                (hits["center_idx"] < REGION_WIDTH)]

    # Get ACR class per hit
    hits["acr_class"] = hits["native_str"].map(class_map).fillna("stable")

    # Unique ACR regions to process
    unique_regions = hits["resized_str"].unique()
    print(f"  {len(hits):,} valid WRKY hits in {len(unique_regions):,} ACRs",
          flush=True)

    # ── Phase 1: Extract per-sample per-scale FP from h5ad ──────────────
    # sample_profiles[sid][region_str] = arr (n_scales, 2000)
    sample_profiles = {}
    for sid, h5ad_path in h5ad_paths.items():
        print(f"  Loading {sid}...", flush=True)
        os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
        printer = scp.load_printer(h5ad_path, genome)
        fp_key = f"FP_{sid}_ALL".replace("-", "_").replace(".", "_")
        fp_adata = printer.footprintsadata[fp_key]

        avail_keys = set(fp_adata.obsm.keys())
        sid_profiles = {}

        for region_str in unique_regions:
            if region_str not in avail_keys:
                continue
            try:
                arr = np.asarray(fp_adata.obsm[region_str])
                if arr.ndim == 3:
                    arr = arr[0]  # (n_scales, 2000)
                sid_profiles[region_str] = arr.astype(np.float32)
            except Exception:
                continue

        sample_profiles[sid] = sid_profiles
        print(f"    Extracted {len(sid_profiles):,} regions", flush=True)
        del printer, fp_adata
        gc.collect()

    # ── Phase 2: Build per-hit per-scale per-condition arrays ────────────
    hits_reset = hits.reset_index(drop=True)
    n_hits = len(hits_reset)
    n_all_scales = len(scales)
    window_size = 2 * WINDOW_HALF + 1

    print("  Building per-hit per-scale arrays...", flush=True)

    # Per-hit, per-scale, per-condition centered windows
    # Shape: (n_hits, n_scales, window_size)
    # We process nuc and TF scale groups separately to save memory
    for scale_group, scale_idx, group_name in [
            ("nuc", nuc_scale_idx, "nucleosome"),
            ("tf", tf_scale_idx, "TF")]:

        n_sg = len(scale_idx)
        print(f"  Processing {group_name} scales ({n_sg} scales)...",
              flush=True)

        # Per-hit: (n_hits, n_sg) per condition — FP at hit center position
        # For spatial profiles: (n_hits, n_sg, window_size) per condition
        hit_leaf_windows = np.full((n_hits, n_sg, window_size),
                                    np.nan, np.float32)
        hit_proto_windows = np.full((n_hits, n_sg, window_size),
                                     np.nan, np.float32)

        for i, row in hits_reset.iterrows():
            region_str = row["resized_str"]
            center = int(row["center_idx"])

            src_start = max(0, center - WINDOW_HALF)
            src_end = min(REGION_WIDTH, center + WINDOW_HALF + 1)
            dst_start = src_start - (center - WINDOW_HALF)
            dst_end = dst_start + (src_end - src_start)

            # Average across leaf reps
            leaf_arrs = []
            for sid in LEAF_IDS:
                arr = sample_profiles.get(sid, {}).get(region_str)
                if arr is not None:
                    leaf_arrs.append(arr[scale_idx, src_start:src_end])
            if leaf_arrs:
                leaf_avg = np.nanmean(leaf_arrs, axis=0)  # (n_sg, width)
                hit_leaf_windows[i, :, dst_start:dst_end] = leaf_avg

            # Average across proto reps
            proto_arrs = []
            for sid in PROTO_IDS:
                arr = sample_profiles.get(sid, {}).get(region_str)
                if arr is not None:
                    proto_arrs.append(arr[scale_idx, src_start:src_end])
            if proto_arrs:
                proto_avg = np.nanmean(proto_arrs, axis=0)
                hit_proto_windows[i, :, dst_start:dst_end] = proto_avg

        # ── Phase 3: Residualize + z-score per scale (vectorised) ────────
        # Build confounder matrix ONCE — reused across all scales × positions.
        # Old code: 21 scales × 1000 positions × per-hit loop = ~264M iters.
        # New code: 21 lstsq calls, each (n_hits × n_positions) at once.
        print(f"  Residualizing + z-scoring {group_name}...", flush=True)
        conf_df = _build_confounder_df(acr_meta, coord_map)
        hit_acr_ids = hits_reset["resized_str"].values
        C_full, finite_rows = _build_hit_confounder_matrix(hit_acr_ids,
                                                            conf_df)

        hit_leaf_z = np.full_like(hit_leaf_windows, np.nan)
        hit_proto_z = np.full_like(hit_proto_windows, np.nan)

        for si in range(n_sg):
            # Residualize all positions for this scale in one lstsq call
            lf_resid = _residualize_matrix(hit_leaf_windows[:, si, :],
                                            C_full, finite_rows)
            pr_resid = _residualize_matrix(hit_proto_windows[:, si, :],
                                            C_full, finite_rows)

            # Z-score pooled per position
            pooled = np.concatenate([lf_resid, pr_resid], axis=0)
            mu = np.nanmean(pooled, axis=0)           # (n_positions,)
            sd = np.nanstd(pooled, axis=0)
            sd_safe = np.where(sd < 1e-12, np.nan, sd)

            hit_leaf_z[:, si, :] = (lf_resid - mu) / sd_safe
            hit_proto_z[:, si, :] = (pr_resid - mu) / sd_safe

        # Average across scales → (n_hits, window_size)
        raw_leaf = np.nanmean(hit_leaf_windows, axis=1)
        raw_proto = np.nanmean(hit_proto_windows, axis=1)
        raw_delta = raw_leaf - raw_proto

        z_leaf = np.nanmean(hit_leaf_z, axis=1)
        z_proto = np.nanmean(hit_proto_z, axis=1)
        z_delta = z_leaf - z_proto

        if scale_group == "nuc":
            nuc_raw_leaf, nuc_raw_proto, nuc_raw_delta = (
                raw_leaf, raw_proto, raw_delta)
            nuc_z_leaf, nuc_z_proto, nuc_z_delta = (
                z_leaf, z_proto, z_delta)
        else:
            tf_raw_leaf, tf_raw_proto, tf_raw_delta = (
                raw_leaf, raw_proto, raw_delta)
            tf_z_leaf, tf_z_proto, tf_z_delta = (
                z_leaf, z_proto, z_delta)

        del hit_leaf_windows, hit_proto_windows
        del hit_leaf_z, hit_proto_z
        gc.collect()

    # ── Phase 4: Per-ACR full profiles (nuc scale, raw + z-scored) ───────
    print("  Building per-ACR profiles...", flush=True)
    acr_ids_list = list(unique_regions)
    n_acrs = len(acr_ids_list)
    n_nuc = len(nuc_scale_idx)

    acr_leaf_raw = np.full((n_acrs, REGION_WIDTH), np.nan, np.float32)
    acr_proto_raw = np.full((n_acrs, REGION_WIDTH), np.nan, np.float32)

    for ai, region_str in enumerate(acr_ids_list):
        leaf_arrs = []
        proto_arrs = []
        for sid in LEAF_IDS:
            arr = sample_profiles.get(sid, {}).get(region_str)
            if arr is not None:
                leaf_arrs.append(
                    np.nanmean(arr[nuc_scale_idx, :], axis=0))
        for sid in PROTO_IDS:
            arr = sample_profiles.get(sid, {}).get(region_str)
            if arr is not None:
                proto_arrs.append(
                    np.nanmean(arr[nuc_scale_idx, :], axis=0))
        if leaf_arrs:
            acr_leaf_raw[ai] = np.nanmean(leaf_arrs, axis=0)
        if proto_arrs:
            acr_proto_raw[ai] = np.nanmean(proto_arrs, axis=0)

    acr_delta_raw = acr_leaf_raw - acr_proto_raw

    resized_to_native = dict(zip(coord_map["resized_str"],
                                  coord_map["native_str"]))
    acr_classes = np.array([
        class_map.get(resized_to_native.get(r, r), "stable")
        for r in acr_ids_list])

    # Hit metadata
    hit_acr_classes = hits_reset["acr_class"].values
    hit_center_idxs = hits_reset["center_idx"].values.astype(int)
    hit_region_strs = hits_reset["resized_str"].values
    hit_sig_ids = hits_reset["motif_id"].values
    positions = np.arange(-WINDOW_HALF, WINDOW_HALF + 1)

    return {
        # Per-hit nucleosome-scale windows (±500 bp)
        "hit_nuc_leaf_raw": nuc_raw_leaf,
        "hit_nuc_proto_raw": nuc_raw_proto,
        "hit_nuc_delta_raw": nuc_raw_delta,
        "hit_nuc_leaf_z": nuc_z_leaf,
        "hit_nuc_proto_z": nuc_z_proto,
        "hit_nuc_delta_z": nuc_z_delta,
        # Per-hit TF-scale windows (±500 bp)
        "hit_tf_leaf_raw": tf_raw_leaf,
        "hit_tf_proto_raw": tf_raw_proto,
        "hit_tf_delta_raw": tf_raw_delta,
        "hit_tf_leaf_z": tf_z_leaf,
        "hit_tf_proto_z": tf_z_proto,
        "hit_tf_delta_z": tf_z_delta,
        # Per-ACR full nuc profiles (2000 bp)
        "acr_nuc_leaf_raw": acr_leaf_raw,
        "acr_nuc_proto_raw": acr_proto_raw,
        "acr_nuc_delta_raw": acr_delta_raw,
        # Metadata
        "hit_acr_class": hit_acr_classes,
        "hit_center_idx": hit_center_idxs,
        "hit_region_str": hit_region_strs,
        "hit_sig_id": hit_sig_ids,
        "positions_bp": positions,
        "acr_ids": np.array(acr_ids_list),
        "acr_class": acr_classes,
        "nuc_scales_bp": scales[nuc_scale_idx],
        "tf_scales_bp": scales[tf_scale_idx],
    }


def main():
    args = parse_args()

    # Resolve paths
    for attr in ("print_dir", "genome_obj", "chunk_dir", "coord_mapping",
                 "acr_metadata", "outdir", "tmp_dir"):
        val = getattr(args, attr)
        if not os.path.isabs(val):
            setattr(args, attr, os.path.join(BASE, val))

    os.makedirs(args.outdir, exist_ok=True)
    out_path = os.path.join(args.outdir, "wrky_nuc_profiles.npz")

    print("=" * 60, flush=True)
    print("v3_08c — WRKY Nucleosome Profile Extraction", flush=True)
    print("  (with per-condition + residualized + z-scored output)")
    print("=" * 60, flush=True)

    # Load WRKY hits
    hits = load_wrky_hits(args.chunk_dir)
    if hits.empty:
        print("[ERROR] No WRKY hits found", flush=True)
        sys.exit(1)

    # Load coordinate mapping
    coord_map = pd.read_csv(args.coord_mapping, sep="\t")

    # Load ACR metadata
    acr_meta = pd.read_csv(args.acr_metadata, sep="\t")
    acr_meta["acr_id"] = acr_meta["acr_id"].astype(str)

    # Copy h5ad to tmp
    h5ad_paths = copy_h5ad_to_tmp(args.print_dir, args.tmp_dir)
    if not h5ad_paths:
        print("[ERROR] No h5ad files available", flush=True)
        sys.exit(1)

    # Scales
    scales = np.arange(2, 101, dtype=np.float64)

    # Extract profiles
    result = extract_profiles(hits, h5ad_paths, args.genome_obj,
                               coord_map, acr_meta, scales)

    # Save NPZ
    np.savez_compressed(out_path, **result)
    fsize = os.path.getsize(out_path) / 1e6
    print(f"\n[DONE] Saved {out_path} ({fsize:.1f} MB)", flush=True)
    print(f"  hit_nuc_leaf_z:  {result['hit_nuc_leaf_z'].shape}", flush=True)
    print(f"  hit_tf_leaf_z:   {result['hit_tf_leaf_z'].shape}", flush=True)
    print(f"  acr_nuc_leaf_raw: {result['acr_nuc_leaf_raw'].shape}",
          flush=True)
    n_valid = np.isfinite(
        result["hit_nuc_leaf_z"]).any(axis=1).sum()
    print(f"  Valid hit profiles: {n_valid:,}/{result['hit_nuc_leaf_z'].shape[0]:,}",
          flush=True)


if __name__ == "__main__":
    main()
