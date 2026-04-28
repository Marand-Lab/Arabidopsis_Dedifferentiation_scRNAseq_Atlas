#!/usr/bin/env python3
"""
v3 Step 05: Per-replicate FP band extraction at motif hit centers.

Three modes:
  --extract --condition leaf|proto  → per-chunk FP extraction (array job)
  --merge-conditions               → per-chunk leaf+proto merge + deltas
  --merge-chunks                   → concatenate all chunks

Adapts v2 04b + 04b_merge + 04c into a single script for the v3 pipeline.

Input:
  - data/v3_chunks/chunk_NN/motif_hits.tsv.gz (from v3_04)
  - 3_PRINT_per_rep/printer_{sample}_bulk.h5ad (6 files)
  - data/acr_native_to_resized.tsv
  - data/library_sizes.tsv

Output:
  - data/v3_chunks/chunk_NN/motif_hits_fpband_{leaf|proto}.tsv.gz (extract)
  - data/v3_chunks/chunk_NN/motif_hits_fpband_per_rep.tsv.gz (merge-conditions)
  - data/v3_merged_motif_hits_fpband_per_rep.tsv.gz (merge-chunks)
"""

from __future__ import annotations

import os
import warnings

warnings.filterwarnings("ignore", message=r".*pkg_resources.*", category=UserWarning)
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import argparse
import gzip
import pickle
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# ── FP extraction functions ──────────────────────────────────────────────────

def make_dynamic_windows(scales: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scale-adaptive center/flank window sizes."""
    scales = np.asarray(scales, dtype=float)
    s = (scales - scales.min()) / (scales.max() - scales.min() + 1e-12)
    center_bp = np.rint(8 + 10 * s).astype(int)
    flank_l2 = np.rint(25 + 60 * s).astype(int)
    flank_w = 50
    flank_l1 = flank_l2 + flank_w
    flank_l2 = np.maximum(flank_l2, center_bp + 2)
    flank_l1 = np.maximum(flank_l1, flank_l2 + 10)
    return center_bp, flank_l1, flank_l2


def fp_depth_dynamic(fp_scales_pos, center_idx, center_bps, flank_l1s, flank_l2s):
    """Compute FP depth and flanking signal per scale."""
    n_scales, n_pos = fp_scales_pos.shape
    depths = np.full(n_scales, np.nan, dtype=float)
    flanks = np.full(n_scales, np.nan, dtype=float)
    for si in range(n_scales):
        c_bp = int(center_bps[si])
        l1 = int(flank_l1s[si])
        l2 = int(flank_l2s[si])

        def safe_mean(lo, hi):
            lo_idx = max(0, lo)
            hi_idx = min(n_pos, hi)
            if hi_idx <= lo_idx:
                return np.nan
            return float(np.nanmean(fp_scales_pos[si, lo_idx:hi_idx]))

        center_val = safe_mean(center_idx - c_bp, center_idx + c_bp + 1)
        f1_val = safe_mean(center_idx - l1, center_idx - l2)
        f2_val = safe_mean(center_idx + l2, center_idx + l1)

        if not (np.isfinite(center_val) and np.isfinite(f1_val) and np.isfinite(f2_val)):
            continue
        flank_mean = (f1_val + f2_val) / 2.0
        flanks[si] = flank_mean
        depths[si] = flank_mean - center_val
    return depths, flanks


def make_band_masks(scales, edges):
    """Create 3-band masks from scale breakpoints."""
    scales = np.asarray(scales, float)
    e1, e2 = float(edges[0]), float(edges[1])
    m1 = scales < e1
    m2 = (scales >= e1) & (scales <= e2)
    m3 = scales > e2
    return [m1, m2, m3]


def load_fp_tensor(fp_adata, region_str: str) -> np.ndarray:
    arr = np.asarray(fp_adata.obsm[region_str])
    if arr.ndim == 3:
        arr = arr[0]
    return arr


def extract_band_depths_for_rep(
    hits: pd.DataFrame,
    fp_adata,
    scales: np.ndarray,
    edges: List[float],
    sample_id: str,
    native_to_resized: Dict[str, Tuple[str, int]] | None = None,
) -> Dict[str, np.ndarray]:
    """Extract per-hit band depths and flanking signal for a single replicate."""
    band_masks = make_band_masks(scales, edges)
    center_bp_arr, flank_l1_arr, flank_l2_arr = make_dynamic_windows(scales)

    n = len(hits)
    bands = {f"{sample_id}_band{bi+1}": np.full(n, np.nan, float) for bi in range(3)}
    flank_bands = {f"{sample_id}_flank{bi+1}": np.full(n, np.nan, float) for bi in range(3)}
    fp_cache = {}

    t0 = time.time()
    kept = 0
    for i, h in enumerate(hits.itertuples(index=False)):
        if native_to_resized is not None:
            mapped = native_to_resized.get(h.region_str)
            if mapped is None:
                continue
            rs, resized_start = mapped
        else:
            rs = h.region_str
            resized_start = int(h.Start)

        if rs not in fp_adata.obsm:
            continue
        if rs not in fp_cache:
            fp_cache[rs] = load_fp_tensor(fp_adata, rs)
        fp_t = fp_cache[rs]
        n_scales, n_pos = fp_t.shape
        center_idx = int(h.hit_center) - resized_start
        if center_idx < 0 or center_idx >= n_pos:
            continue

        depth, flank = fp_depth_dynamic(fp_t, center_idx, center_bp_arr, flank_l1_arr, flank_l2_arr)
        valid = np.isfinite(depth) & np.isfinite(flank)

        any_band = False
        for bi, mask in enumerate(band_masks):
            m = mask & valid
            if not np.any(m):
                continue
            bands[f"{sample_id}_band{bi+1}"][i] = float(np.mean(depth[m]))
            flank_bands[f"{sample_id}_flank{bi+1}"][i] = float(np.mean(flank[m]))
            any_band = True
        if any_band:
            kept += 1

        if (i + 1) % 200_000 == 0 or (i + 1) == n:
            dt = time.time() - t0
            rate = (i + 1) / max(dt, 1e-9)
            print(f"  [{sample_id}] {i+1:,}/{n:,} ({(i+1)/n:.1%}) "
                  f"{rate:,.0f} hits/s kept={kept:,}", flush=True)

    bands.update(flank_bands)
    return bands


# ── Library-size normalization ────────────────────────────────────────────────

def load_size_factors(lib_sizes_path: str) -> dict[str, float]:
    if not os.path.exists(lib_sizes_path):
        return {}
    lib_df = pd.read_csv(lib_sizes_path, sep="\t")
    if "sample_id" not in lib_df.columns or "size_factor" not in lib_df.columns:
        return {}
    return dict(zip(lib_df["sample_id"], lib_df["size_factor"]))


# ── Mode 1: Extract ──────────────────────────────────────────────────────────

def do_extract(args):
    """Extract FP for one condition (leaf or proto) in one chunk."""
    chunk_dir = Path(args.chunk_dir)
    hits_path = chunk_dir / "motif_hits.tsv.gz"
    if not hits_path.exists():
        raise FileNotFoundError(f"Missing: {hits_path}")

    edges = [float(x) for x in args.band_edges.split(",")]

    # Load genome
    with open(args.genome_pkl, "rb") as f:
        genome = pickle.load(f)

    # Load coordinate mapping
    native_to_resized = None
    if os.path.exists(args.mapping):
        mdf = pd.read_csv(args.mapping, sep="\t")
        native_to_resized = dict(zip(
            mdf["native_str"],
            zip(mdf["resized_str"], mdf["resized_start"].astype(int)),
        ))
        print(f"[INFO] Loaded coordinate mapping: {len(native_to_resized):,} regions",
              flush=True)

    # Load hits
    print(f"[INFO] Loading hits from {hits_path}", flush=True)
    with gzip.open(hits_path, "rt") as f:
        hits = pd.read_csv(f, sep="\t")
    print(f"[INFO] Loaded {len(hits):,} hits", flush=True)

    import scprinter as scp

    all_sample_ids = ["leaf_rep1", "leaf_rep2", "leaf_rep3",
                      "proto_rep1", "proto_rep2", "proto_rep3"]
    if args.condition == "leaf":
        sample_ids = [s for s in all_sample_ids if s.startswith("leaf_")]
    else:
        sample_ids = [s for s in all_sample_ids if s.startswith("proto_")]

    all_band_cols = {}
    for sid in sample_ids:
        h5ad_path = os.path.join(args.print_dir, f"printer_{sid}_bulk.h5ad")
        if not os.path.exists(h5ad_path):
            print(f"[WARN] Missing {h5ad_path}, skipping {sid}", flush=True)
            for bi in range(3):
                all_band_cols[f"{sid}_band{bi+1}"] = np.full(len(hits), np.nan)
                all_band_cols[f"{sid}_flank{bi+1}"] = np.full(len(hits), np.nan)
            continue

        print(f"\n[INFO] Processing {sid}: {h5ad_path}", flush=True)
        printer = scp.load_printer(h5ad_path, genome)
        fp_key = f"FP_{sid}_ALL".replace("-", "_").replace(".", "_")
        fp_adata = printer.footprintsadata[fp_key]
        scales = np.asarray(fp_adata.uns["scales"], dtype=float)

        band_cols = extract_band_depths_for_rep(
            hits, fp_adata, scales, edges, sid,
            native_to_resized=native_to_resized,
        )
        all_band_cols.update(band_cols)

        printer.close()
        import gc; gc.collect()

    for col, arr in all_band_cols.items():
        hits[col] = arr

    out_name = f"motif_hits_fpband_{args.condition}.tsv.gz"
    out_path = chunk_dir / out_name
    with gzip.open(out_path, "wt") as f:
        hits.to_csv(f, sep="\t", index=False)
    print(f"\n[DONE] Saved {out_path} ({len(hits):,} hits, "
          f"{len(all_band_cols)} new columns)", flush=True)


# ── Mode 2: Merge conditions ─────────────────────────────────────────────────

def do_merge_conditions(args):
    """Merge leaf + proto intermediates for all chunks, compute deltas."""
    sf = load_size_factors(args.library_sizes)
    if sf:
        print(f"[INFO] Loaded size factors for {len(sf)} replicates")
    else:
        print("[WARN] No library-size factors; using unnormalized depths")

    min_flank = args.min_flank

    for i in range(50):
        chunk_dir = Path(f"data/v3_chunks/chunk_{i:02d}")
        leaf_path = chunk_dir / "motif_hits_fpband_leaf.tsv.gz"
        proto_path = chunk_dir / "motif_hits_fpband_proto.tsv.gz"
        out_path = chunk_dir / "motif_hits_fpband_per_rep.tsv.gz"

        if not leaf_path.exists() or not proto_path.exists():
            print(f"[SKIP] chunk_{i:02d}: missing intermediates")
            continue

        print(f"\n[INFO] Merging chunk_{i:02d} ...", flush=True)

        with gzip.open(leaf_path, "rt") as f:
            df = pd.read_csv(f, sep="\t")
        with gzip.open(proto_path, "rt") as f:
            df_proto = pd.read_csv(f, sep="\t")

        assert len(df) == len(df_proto), f"Hit count mismatch in chunk_{i:02d}"

        # Add proto-only columns
        leaf_cols = set(df.columns)
        for col in df_proto.columns:
            if col not in leaf_cols:
                df[col] = df_proto[col].values

        # Library-size normalization
        if sf:
            for sid, s in sf.items():
                for bi in range(1, 4):
                    for suffix in [f"_band{bi}", f"_flank{bi}"]:
                        col = f"{sid}{suffix}"
                        if col in df.columns:
                            df[col] = df[col].values / s

        # Compute deltas
        for rep in [1, 2, 3]:
            for bi in range(1, 4):
                lv = df.get(f"leaf_rep{rep}_band{bi}", pd.Series(np.nan, index=df.index)).values
                pv = df.get(f"proto_rep{rep}_band{bi}", pd.Series(np.nan, index=df.index)).values
                df[f"delta_rep{rep}_band{bi}"] = lv - pv

                lf = df.get(f"leaf_rep{rep}_flank{bi}", pd.Series(np.nan, index=df.index)).values
                pf = df.get(f"proto_rep{rep}_flank{bi}", pd.Series(np.nan, index=df.index)).values
                leaf_frac = np.where(lf >= min_flank, lv / lf, np.nan)
                proto_frac = np.where(pf >= min_flank, pv / pf, np.nan)
                df[f"delta_frac_rep{rep}_band{bi}"] = leaf_frac - proto_frac

        with gzip.open(out_path, "wt") as f:
            df.to_csv(f, sep="\t", index=False)
        norm_tag = " (lib-norm)" if sf else ""
        print(f"  [DONE] {out_path.name}: {len(df):,} hits{norm_tag}", flush=True)


# ── Mode 3: Merge chunks ─────────────────────────────────────────────────────

def do_merge_chunks(args):
    """Concatenate per-chunk outputs into one merged file."""
    frames = []
    hits_frames = []

    for i in range(50):
        chunk_dir = Path(f"data/v3_chunks/chunk_{i:02d}")
        fp_path = chunk_dir / "motif_hits_fpband_per_rep.tsv.gz"
        hits_path = chunk_dir / "motif_hits.tsv.gz"

        if not fp_path.exists():
            print(f"[WARN] Missing {fp_path}, skipping", flush=True)
            continue

        with gzip.open(fp_path, "rt") as f:
            df = pd.read_csv(f, sep="\t")
        cid = f"{i:02d}"
        df["chunk_id"] = cid
        if "hit_id" in df.columns:
            df["hit_uid"] = df["chunk_id"] + ":" + df["hit_id"].astype(str)
        else:
            df["hit_uid"] = df["chunk_id"] + ":" + df.index.astype(str)
        frames.append(df)
        print(f"  chunk_{cid}: {len(df):,} hits", flush=True)

        if hits_path.exists():
            with gzip.open(hits_path, "rt") as f:
                hdf = pd.read_csv(f, sep="\t")
            hdf["chunk_id"] = cid
            if "hit_id" in hdf.columns:
                hdf["hit_uid"] = hdf["chunk_id"] + ":" + hdf["hit_id"].astype(str)
            else:
                hdf["hit_uid"] = hdf["chunk_id"] + ":" + hdf.index.astype(str)
            hits_frames.append(hdf)

    if not frames:
        raise RuntimeError("No chunk outputs found to merge.")

    merged = pd.concat(frames, ignore_index=True)
    print(f"[INFO] Total merged: {len(merged):,} hits", flush=True)

    outdir = Path("data")
    outdir.mkdir(exist_ok=True)

    # Save merged hits (scan-only, no FP)
    if hits_frames:
        merged_hits = pd.concat(hits_frames, ignore_index=True)
        hits_out = outdir / "v3_merged_motif_hits.tsv.gz"
        with gzip.open(hits_out, "wt") as f:
            merged_hits.to_csv(f, sep="\t", index=False)
        print(f"[INFO] Saved merged hits: {hits_out} ({len(merged_hits):,})", flush=True)

    # Save merged FP band data
    out_path = outdir / "v3_merged_motif_hits_fpband_per_rep.tsv.gz"
    with gzip.open(out_path, "wt") as f:
        merged.to_csv(f, sep="\t", index=False)
    print(f"[INFO] Saved: {out_path} ({len(merged):,})", flush=True)

    # Sanity check
    sids = ["leaf_rep1", "leaf_rep2", "leaf_rep3",
            "proto_rep1", "proto_rep2", "proto_rep3"]
    expected = ([f"{s}_band{b}" for s in sids for b in [1,2,3]]
                + [f"{s}_flank{b}" for s in sids for b in [1,2,3]]
                + [f"delta_rep{r}_band{b}" for r in [1,2,3] for b in [1,2,3]]
                + [f"delta_frac_rep{r}_band{b}" for r in [1,2,3] for b in [1,2,3]])
    present = sum(1 for c in expected if c in merged.columns)
    print(f"[INFO] Per-rep columns: {present}/{len(expected)}", flush=True)


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="v3 Step 05: Per-replicate FP band extraction",
    )
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--extract", action="store_true",
                      help="Extract FP for one condition in one chunk")
    mode.add_argument("--merge-conditions", action="store_true",
                      help="Merge leaf+proto intermediates for all chunks")
    mode.add_argument("--merge-chunks", action="store_true",
                      help="Concatenate all chunks into final merged file")

    # Extract-mode args
    p.add_argument("--chunk-dir", type=str,
                   help="Chunk directory (extract mode)")
    p.add_argument("--condition", choices=["leaf", "proto"],
                   help="Condition to extract (extract mode)")
    p.add_argument("--print-dir", default="3_PRINT_per_rep")
    p.add_argument("--genome-pkl", default="3_PRINT_bulk/At_genome_OBJ")
    p.add_argument("--mapping", default="data/acr_native_to_resized.tsv")
    p.add_argument("--band-edges", default="20,50")

    # Merge args
    p.add_argument("--min-flank", type=float, default=1.0)
    p.add_argument("--library-sizes", default="data/library_sizes.tsv")

    return p.parse_args()


def main():
    args = parse_args()
    if args.extract:
        if not args.chunk_dir or not args.condition:
            raise ValueError("--extract requires --chunk-dir and --condition")
        do_extract(args)
    elif args.merge_conditions:
        do_merge_conditions(args)
    elif args.merge_chunks:
        do_merge_chunks(args)


if __name__ == "__main__":
    main()
