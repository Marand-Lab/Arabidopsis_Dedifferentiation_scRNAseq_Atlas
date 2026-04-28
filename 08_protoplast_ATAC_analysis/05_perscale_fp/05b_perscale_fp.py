#!/usr/bin/env python3
"""
v3 Step 06: Per-scale FP extraction for all motif signatures.

For each signature hit centre, extracts FP value at every scale (2–100 bp)
from per-rep h5ad files.

Three modes:
  --chunk-id NN : Phase 1 — extract one chunk (SLURM array task)
  --merge       : Phase 2 — merge chunk NPZs + aggregate to delta matrices
  Neither       : Single-job mode (full pipeline)

Output:
  Phase 1: results/v3_06_perscale_fp/chunks/per_hit_fp_chunk_NN.npz
  Phase 2: results/v3_06_perscale_fp/delta_acr_signature_scale.npz  (n_acrs, N_sigs, n_scales)
           results/v3_06_perscale_fp/delta_acr_family_scale.npz     (n_acrs, ~34, n_scales)

Adapts v2 15a_extract_perscale_fp.py for v3 signatures.
"""
from __future__ import annotations

import argparse
import gc
import glob as globmod
import os
import pickle
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Replicate exclusion (rep3 label-swap — see REPs_README.md) ───────────────
EXCLUDE_REPS = {3}
ACTIVE_REPS = sorted({1, 2, 3} - EXCLUDE_REPS)

LEAF_IDS = [f"leaf_rep{r}" for r in ACTIVE_REPS]
PROTO_IDS = [f"proto_rep{r}" for r in ACTIVE_REPS]
ALL_IDS = LEAF_IDS + PROTO_IDS

BASE = os.path.dirname(os.path.abspath(__file__))


# ── Signature metadata loader ────────────────────────────────────────────────

def load_signature_metadata(path: str) -> pd.DataFrame:
    """Load signature_metadata.tsv with signature_id, display_name, primary_family."""
    df = pd.read_csv(path, sep="\t")
    return df


# ── Phase 1: Per-hit per-scale extraction ────────────────────────────────────

def load_fp_tensor(fp_adata, region_str: str) -> np.ndarray:
    arr = np.asarray(fp_adata.obsm[region_str])
    if arr.ndim == 3:
        arr = arr[0]
    return arr


def load_hits(hits_path: str, coord_mapping_path: str):
    """Load motif hits with native → resized coordinate mapping."""
    coord_map = pd.read_csv(coord_mapping_path, sep="\t")
    coord_map["native_str"] = coord_map["native_str"].str.lower()
    coord_map["resized_str"] = coord_map["resized_str"].str.lower()
    native_to_resized = dict(zip(coord_map["native_str"],
                                  coord_map["resized_str"]))
    resized_start_map = dict(zip(coord_map["resized_str"],
                                  coord_map["resized_start"]))

    usecols = ["region_str", "motif_id", "hit_center"]
    chunks = []
    for chunk in pd.read_csv(hits_path, sep="\t", chunksize=500_000,
                             usecols=usecols):
        chunk["native_str"] = chunk["region_str"].str.replace(
            r"^Chr", "chr", regex=True)
        chunk["resized_str"] = chunk["native_str"].map(native_to_resized)
        chunk = chunk.dropna(subset=["resized_str"])
        chunk["resized_start"] = chunk["resized_str"].map(resized_start_map)
        chunk = chunk.dropna(subset=["resized_start"])
        chunk["resized_start"] = chunk["resized_start"].astype(int)
        chunk["hit_center"] = chunk["hit_center"].astype(int)
        chunks.append(chunk[["resized_str", "motif_id", "hit_center",
                             "resized_start"]].copy())

    if not chunks:
        print("[ERROR] No hits found!", flush=True)
        sys.exit(1)

    all_hits = pd.concat(chunks, ignore_index=True)
    del chunks; gc.collect()
    print(f"  Total hits: {len(all_hits):,}", flush=True)
    print(f"  Unique ACRs: {all_hits['resized_str'].nunique():,}", flush=True)
    print(f"  Unique signatures: {all_hits['motif_id'].nunique()}", flush=True)
    return all_hits


def extract_perscale(print_dir, genome, all_hits, npz_path):
    """Extract per-scale FP at every hit centre from h5ad files."""
    import scprinter as scp

    acr_keys = set(all_hits["resized_str"].unique())
    n_hits_total = len(all_hits)
    hit_rows = all_hits.reset_index(drop=True)

    # Determine scales from first h5ad
    first_h5ad = os.path.join(print_dir, f"printer_{ALL_IDS[0]}_bulk.h5ad")
    printer = scp.load_printer(first_h5ad, genome)
    fp_key = f"FP_{ALL_IDS[0]}_ALL".replace("-", "_").replace(".", "_")
    fp_adata = printer.footprintsadata[fp_key]
    scales = np.array(fp_adata.uns["scales"], dtype=np.float64)
    n_scales = len(scales)
    print(f"  Scales: {n_scales} ({scales[0]:.0f}–{scales[-1]:.0f} bp)", flush=True)
    printer.close(); del printer, fp_adata; gc.collect()

    n_samples = len(ALL_IDS)
    fp_values = np.full((n_hits_total, n_scales, n_samples), np.nan,
                        dtype=np.float32)
    print(f"  fp_values shape: {fp_values.shape} "
          f"({fp_values.nbytes / 1e9:.1f} GB)", flush=True)

    for si, sid in enumerate(ALL_IDS):
        h5ad_path = os.path.join(print_dir, f"printer_{sid}_bulk.h5ad")
        print(f"\n  Loading {sid}...", flush=True)
        t0 = time.time()

        if not os.path.exists(h5ad_path):
            print(f"    [WARN] {h5ad_path} not found — skipping", flush=True)
            continue

        os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
        printer = scp.load_printer(h5ad_path, genome)
        fp_key = f"FP_{sid}_ALL".replace("-", "_").replace(".", "_")
        fp_adata = printer.footprintsadata[fp_key]
        avail_keys = set(fp_adata.obsm.keys())
        n_found = 0

        for acr_str in acr_keys:
            if acr_str not in avail_keys:
                continue
            tensor = load_fp_tensor(fp_adata, acr_str)
            n_positions = tensor.shape[1]
            mask = hit_rows["resized_str"] == acr_str
            for idx in hit_rows.index[mask]:
                center_idx = int(hit_rows.at[idx, "hit_center"]) - int(hit_rows.at[idx, "resized_start"])
                if 0 <= center_idx < n_positions:
                    fp_values[idx, :, si] = tensor[:, center_idx]
            n_found += 1
            if n_found % 5000 == 0:
                print(f"    {sid}: {n_found:,}/{len(acr_keys):,} ACRs...", flush=True)

        dt = time.time() - t0
        print(f"    {sid}: {n_found:,}/{len(acr_keys):,} ACRs [{dt:.1f}s]", flush=True)
        printer.close(); del printer, fp_adata; gc.collect()

    os.makedirs(os.path.dirname(npz_path), exist_ok=True)
    np.savez_compressed(
        npz_path,
        fp_values=fp_values,
        scales=scales,
        region_strs=hit_rows["resized_str"].values,
        motif_ids=hit_rows["motif_id"].values,
        sample_ids=np.array(ALL_IDS),
    )
    sz = os.path.getsize(npz_path) / 1e9
    print(f"  Saved {npz_path} ({sz:.2f} GB)", flush=True)


# ── Chunk merge ──────────────────────────────────────────────────────────────

def merge_chunk_npzs(outdir, n_chunks=50):
    chunks_dir = os.path.join(outdir, "chunks")
    chunk_files = sorted(globmod.glob(
        os.path.join(chunks_dir, "per_hit_fp_chunk_*.npz")))

    if len(chunk_files) != n_chunks:
        print(f"[ERROR] Expected {n_chunks} chunk NPZs, found "
              f"{len(chunk_files)}", flush=True)
        sys.exit(1)

    all_fp, all_regions, all_motifs = [], [], []
    ref_scales = ref_samples = None

    for i, f in enumerate(chunk_files):
        data = np.load(f, allow_pickle=True)
        all_fp.append(data["fp_values"])
        all_regions.append(data["region_strs"])
        all_motifs.append(data["motif_ids"])
        if ref_scales is None:
            ref_scales = data["scales"]
            ref_samples = data["sample_ids"]
        print(f"    Chunk {i:02d}: {data['fp_values'].shape[0]:,} hits", flush=True)
        data.close()

    merged = {
        "fp_values": np.concatenate(all_fp, axis=0),
        "scales": ref_scales,
        "region_strs": np.concatenate(all_regions),
        "motif_ids": np.concatenate(all_motifs),
        "sample_ids": ref_samples,
    }
    print(f"  Merged: {merged['fp_values'].shape[0]:,} hits", flush=True)
    del all_fp, all_regions, all_motifs; gc.collect()
    return merged


# ── Phase 2: Aggregate to delta matrices ─────────────────────────────────────

def aggregate_to_delta_matrices(merged_data, sig_meta, outdir):
    """Compute ACR × signature × scale and ACR × family × scale delta matrices."""
    print("\n[Phase 2] Aggregating to delta matrices...", flush=True)
    t0 = time.time()

    fp_values = merged_data["fp_values"]
    scales = merged_data["scales"]
    region_strs = merged_data["region_strs"]
    motif_ids = merged_data["motif_ids"]

    n_hits, n_scales, n_samples = fp_values.shape
    n_leaf = len(LEAF_IDS)

    # Per-hit delta: leaf_mean - proto_mean
    leaf_fp = np.nanmean(fp_values[:, :, :n_leaf], axis=2)
    proto_fp = np.nanmean(fp_values[:, :, n_leaf:], axis=2)
    hit_delta = leaf_fp - proto_fp
    del fp_values, leaf_fp, proto_fp; gc.collect()

    # Build signature → family mapping
    mid_to_fam = dict(zip(sig_meta["signature_id"], sig_meta["primary_family"]))
    hit_families = np.array([mid_to_fam.get(m, "Unknown") for m in motif_ids])

    unique_acrs = np.unique(region_strs)
    unique_sigs = np.unique(motif_ids)
    unique_families = np.unique(hit_families[hit_families != "Unknown"])
    n_acrs = len(unique_acrs)
    n_sigs = len(unique_sigs)
    n_families = len(unique_families)

    acr_idx = {a: i for i, a in enumerate(unique_acrs)}
    sig_idx = {s: i for i, s in enumerate(unique_sigs)}
    fam_idx = {f: i for i, f in enumerate(unique_families)}

    print(f"  ACRs: {n_acrs:,}, Signatures: {n_sigs}, Families: {n_families}",
          flush=True)

    # ── Family-level aggregation ─────────────────────────────────────────
    print("  Aggregating family × scale...", flush=True)
    fam_sum = np.zeros((n_acrs, n_families, n_scales), dtype=np.float64)
    fam_count = np.zeros((n_acrs, n_families), dtype=np.int32)

    for i in range(n_hits):
        ai = acr_idx.get(region_strs[i])
        fi = fam_idx.get(hit_families[i])
        if ai is None or fi is None:
            continue
        valid = ~np.isnan(hit_delta[i])
        fam_sum[ai, fi, valid] += hit_delta[i, valid]
        fam_count[ai, fi] += 1
        if (i + 1) % 500_000 == 0:
            print(f"    {i + 1:,}/{n_hits:,}...", flush=True)

    fam_delta = np.full((n_acrs, n_families, n_scales), np.nan, dtype=np.float32)
    for fi in range(n_families):
        for ai in range(n_acrs):
            if fam_count[ai, fi] > 0:
                fam_delta[ai, fi] = fam_sum[ai, fi] / fam_count[ai, fi]

    fam_npz = os.path.join(outdir, "delta_acr_family_scale.npz")
    np.savez_compressed(fam_npz, delta=fam_delta, acr_ids=unique_acrs,
                        family_ids=unique_families, scales=scales)
    print(f"  Saved {fam_npz} ({os.path.getsize(fam_npz)/1e6:.1f} MB)", flush=True)
    del fam_sum, fam_count, fam_delta; gc.collect()

    # ── Signature-level aggregation ──────────────────────────────────────
    print("  Aggregating signature × scale...", flush=True)
    sig_sum = np.zeros((n_acrs, n_sigs, n_scales), dtype=np.float64)
    sig_count = np.zeros((n_acrs, n_sigs), dtype=np.int32)

    for i in range(n_hits):
        ai = acr_idx.get(region_strs[i])
        si = sig_idx.get(motif_ids[i])
        if ai is None or si is None:
            continue
        valid = ~np.isnan(hit_delta[i])
        sig_sum[ai, si, valid] += hit_delta[i, valid]
        sig_count[ai, si] += 1
        if (i + 1) % 500_000 == 0:
            print(f"    {i + 1:,}/{n_hits:,}...", flush=True)

    sig_delta = np.full((n_acrs, n_sigs, n_scales), np.nan, dtype=np.float32)
    for si in range(n_sigs):
        for ai in range(n_acrs):
            if sig_count[ai, si] > 0:
                sig_delta[ai, si] = sig_sum[ai, si] / sig_count[ai, si]

    sig_npz = os.path.join(outdir, "delta_acr_signature_scale.npz")
    np.savez_compressed(sig_npz, delta=sig_delta, acr_ids=unique_acrs,
                        signature_ids=unique_sigs, scales=scales)
    print(f"  Saved {sig_npz} ({os.path.getsize(sig_npz)/1e9:.2f} GB)", flush=True)
    del sig_sum, sig_count, sig_delta; gc.collect()

    dt = time.time() - t0
    print(f"  Phase 2 complete [{dt:.1f}s]", flush=True)


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="v3 Step 06: Per-scale FP extraction for signatures")
    p.add_argument("--genome-pkl", default="3_PRINT_bulk/At_genome_OBJ")
    p.add_argument("--print-dir", default="3_PRINT_per_rep")
    p.add_argument("--chunks-dir", default="data/v3_chunks")
    p.add_argument("--sig-metadata",
                   default="data/motif_signatures/signature_metadata.tsv")
    p.add_argument("--acr-coord-mapping",
                   default="data/acr_native_to_resized.tsv")
    p.add_argument("--outdir", default="results/v3_06_perscale_fp")
    p.add_argument("--force-extract", action="store_true")

    p.add_argument("--chunk-id", type=int, default=None,
                   help="Chunk index for array job (0-based)")
    p.add_argument("--n-chunks", type=int, default=20)

    p.add_argument("--merge", action="store_true",
                   help="Merge chunk NPZs and aggregate")
    p.add_argument("--save-merged-npz", action="store_true")

    return p.parse_args()


def main():
    args = parse_args()
    outdir = os.path.join(BASE, args.outdir)
    os.makedirs(outdir, exist_ok=True)

    print("=" * 60, flush=True)
    print("v3_06 — Per-scale FP extraction for signatures", flush=True)
    print("=" * 60, flush=True)

    # Mode 1: Array chunk extraction
    if args.chunk_id is not None:
        chunk_str = f"{args.chunk_id:02d}"
        print(f"\n[CHUNK MODE] chunk {chunk_str}", flush=True)

        hits_path = os.path.join(BASE, args.chunks_dir,
                                 f"chunk_{chunk_str}", "motif_hits.tsv.gz")
        if not os.path.exists(hits_path):
            print(f"[ERROR] {hits_path} not found", flush=True)
            sys.exit(1)

        all_hits = load_hits(hits_path,
                             os.path.join(BASE, args.acr_coord_mapping))

        npz_path = os.path.join(outdir, "chunks",
                                f"per_hit_fp_chunk_{chunk_str}.npz")

        if os.path.exists(npz_path) and not args.force_extract:
            data = np.load(npz_path, allow_pickle=True)
            print(f"  Existing NPZ: {data['fp_values'].shape}", flush=True)
            data.close()
        else:
            with open(os.path.join(BASE, args.genome_pkl), "rb") as f:
                genome = pickle.load(f)
            extract_perscale(args.print_dir, genome, all_hits, npz_path)
            del genome; gc.collect()

        print(f"\nChunk {chunk_str} complete.", flush=True)
        return

    # Mode 2: Merge + aggregate
    if args.merge:
        print(f"\n[MERGE MODE] {args.n_chunks} chunks", flush=True)

        sig_meta = load_signature_metadata(
            os.path.join(BASE, args.sig_metadata))
        print(f"  Signatures: {len(sig_meta)}, "
              f"Families: {sig_meta['primary_family'].nunique()}", flush=True)

        merged = merge_chunk_npzs(outdir, n_chunks=args.n_chunks)

        if args.save_merged_npz:
            merged_path = os.path.join(outdir, "per_hit_fp.npz")
            np.savez_compressed(merged_path, **merged)
            print(f"  Saved merged NPZ", flush=True)

        aggregate_to_delta_matrices(merged, sig_meta, outdir)
        del merged; gc.collect()
        return

    # Mode 3: Single-job
    print("\n[SINGLE-JOB MODE]", flush=True)
    sig_meta = load_signature_metadata(
        os.path.join(BASE, args.sig_metadata))

    # Load all hits
    hits_path = os.path.join(BASE, "data", "v3_merged_motif_hits.tsv.gz")
    all_hits = load_hits(hits_path,
                         os.path.join(BASE, args.acr_coord_mapping))

    npz_path = os.path.join(outdir, "per_hit_fp.npz")
    if os.path.exists(npz_path) and not args.force_extract:
        print(f"  Existing NPZ: {npz_path}", flush=True)
    else:
        with open(os.path.join(BASE, args.genome_pkl), "rb") as f:
            genome = pickle.load(f)
        extract_perscale(args.print_dir, genome, all_hits, npz_path)
        del genome; gc.collect()

    data = np.load(npz_path, allow_pickle=True)
    merged = {k: data[k] for k in data.files}
    data.close()
    aggregate_to_delta_matrices(merged, sig_meta, outdir)


if __name__ == "__main__":
    main()
