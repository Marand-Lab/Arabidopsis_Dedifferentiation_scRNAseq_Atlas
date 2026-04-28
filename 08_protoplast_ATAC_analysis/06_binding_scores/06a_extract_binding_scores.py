#!/usr/bin/env python3
"""
v4_03a — Extract TFBS & NucBS binding scores from scPrinter h5ad files.

Reads the TFBS and NucBS h5ads for one condition, applies sigmoid to NucBS
(which lacks output activation in the pretrained model), classifies each
tile position as bound/unbound (TF) or occupied/free (nucleosome), and
saves a per-condition NPZ.

TFBS/NucBS h5ads store 180 tile values per 2000bp region:
  - contextRadius=100, tileSize=10
  - tile centers at bp positions: 105, 115, 125, ..., 1895
  - TFBS model: 6 scales [10, 20, 30, 50, 80, 100], sigmoid output → [0,1]
  - NucBS model: 5 scales [10, 20, 30, 50, 80], raw linear output → unbounded

Usage:
  python -u v4_03a_extract_binding_scores.py --condition leaf
  python -u v4_03a_extract_binding_scores.py --condition proto
"""

import argparse
import os
import sys
import time

import h5py
import numpy as np

# ── Constants ────────────────────────────────────────────────────────────────
CONTEXT_RADIUS = 100
TILE_SIZE = 10
N_TILES = 180  # (2000 - 2*CONTEXT_RADIUS) / TILE_SIZE
TILE_BP = np.arange(N_TILES) * TILE_SIZE + CONTEXT_RADIUS + TILE_SIZE // 2
# = [105, 115, 125, ..., 1895]


def sigmoid(x):
    """Numerically stable sigmoid."""
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--condition", required=True, choices=["leaf", "proto"])
    p.add_argument("--tfbs-dir", default="v4/3_PRINT/TFBS")
    p.add_argument("--nucbs-dir", default="v4/3_PRINT/NucBS")
    p.add_argument("--outdir", default="results/v4_03a_binding_scores")
    p.add_argument("--tfbs-bound-percentile", type=float, default=95,
                   help="TFBS percentile above which → 'bound'")
    p.add_argument("--tfbs-unbound-percentile", type=float, default=5,
                   help="TFBS percentile below which → 'unbound'")
    p.add_argument("--nucbs-occ-percentile", type=float, default=95,
                   help="NucBS sigmoid percentile above which → 'occupied'")
    p.add_argument("--nucbs-free-percentile", type=float, default=5,
                   help="NucBS sigmoid percentile below which → 'free'")
    return p.parse_args()


def main():
    args = parse_args()
    cond = args.condition
    os.makedirs(args.outdir, exist_ok=True)

    tfbs_path = os.path.join(args.tfbs_dir, f"{cond}_merged__ALL.h5ad")
    nucbs_path = os.path.join(args.nucbs_dir, f"{cond}_merged__ALL.h5ad")

    print(f"=== v4_03a: Extract Binding Scores — {cond} ===")
    print(f"TFBS:  {tfbs_path}")
    print(f"NucBS: {nucbs_path}")
    print(f"Tile mapping: {N_TILES} tiles, bp = [{TILE_BP[0]}, {TILE_BP[1]}, ..., {TILE_BP[-1]}]")
    print(flush=True)

    # ── Open h5ads ───────────────────────────────────────────────────────
    t0 = time.time()
    f_tfbs = h5py.File(tfbs_path, "r")
    f_nucbs = h5py.File(nucbs_path, "r")

    tfbs_keys = list(f_tfbs["obsm"].keys())
    nucbs_keys = list(f_nucbs["obsm"].keys())

    # Verify key sets match
    if set(tfbs_keys) != set(nucbs_keys):
        tfbs_only = set(tfbs_keys) - set(nucbs_keys)
        nucbs_only = set(nucbs_keys) - set(tfbs_keys)
        print(f"[WARN] Key mismatch: {len(tfbs_only)} TFBS-only, "
              f"{len(nucbs_only)} NucBS-only", flush=True)

    # Use sorted intersection for deterministic order
    region_strs = sorted(set(tfbs_keys) & set(nucbs_keys))
    n_regions = len(region_strs)
    print(f"[LOAD] {n_regions:,} regions ({time.time()-t0:.1f}s)", flush=True)

    # ── Preallocate ──────────────────────────────────────────────────────
    TFBS_prob = np.empty((n_regions, N_TILES), dtype=np.float32)
    NucBS_raw = np.empty((n_regions, N_TILES), dtype=np.float32)

    # ── Extract ──────────────────────────────────────────────────────────
    t0 = time.time()
    for i, rstr in enumerate(region_strs):
        # TFBS: shape (1, 180) → squeeze to (180,)
        tfbs_arr = f_tfbs["obsm"][rstr][:].squeeze()
        if tfbs_arr.shape != (N_TILES,):
            print(f"[WARN] {rstr} TFBS shape {tfbs_arr.shape}, expected ({N_TILES},)",
                  flush=True)
        TFBS_prob[i] = tfbs_arr

        # NucBS: shape (1, 180) → squeeze to (180,)
        nucbs_arr = f_nucbs["obsm"][rstr][:].squeeze()
        if nucbs_arr.shape != (N_TILES,):
            print(f"[WARN] {rstr} NucBS shape {nucbs_arr.shape}, expected ({N_TILES},)",
                  flush=True)
        NucBS_raw[i] = nucbs_arr

        if (i + 1) % 5000 == 0:
            print(f"    [{cond}] {i+1:,}/{n_regions:,} regions...", flush=True)

    f_tfbs.close()
    f_nucbs.close()
    print(f"[EXTRACT] Done in {time.time()-t0:.1f}s", flush=True)

    # ── Apply sigmoid to NucBS ───────────────────────────────────────────
    NucBS_prob = sigmoid(NucBS_raw).astype(np.float32)

    # ── Classify (percentile-based thresholds) ────────────────────────────
    tfbs_bound_thresh = np.percentile(TFBS_prob, args.tfbs_bound_percentile)
    tfbs_unbound_thresh = np.percentile(TFBS_prob, args.tfbs_unbound_percentile)
    tf_bound = TFBS_prob > tfbs_bound_thresh
    tf_unbound = TFBS_prob < tfbs_unbound_thresh

    nucbs_occ_thresh = np.percentile(NucBS_prob, args.nucbs_occ_percentile)
    nucbs_free_thresh = np.percentile(NucBS_prob, args.nucbs_free_percentile)
    nuc_occupied = NucBS_prob > nucbs_occ_thresh
    nuc_free = NucBS_prob < nucbs_free_thresh

    # ── Summary statistics ───────────────────────────────────────────────
    n_total = n_regions * N_TILES
    print(f"\n--- Summary ({cond}) ---")
    print(f"  Regions: {n_regions:,}")
    print(f"  Total tiles: {n_total:,}")

    print(f"\n  TFBS (probability [0,1]):")
    print(f"    Range: [{TFBS_prob.min():.4f}, {TFBS_prob.max():.4f}]")
    print(f"    Mean: {TFBS_prob.mean():.4f}, Median: {np.median(TFBS_prob):.4f}")
    print(f"    Bound (P{args.tfbs_bound_percentile} > {tfbs_bound_thresh:.4f}): "
          f"{tf_bound.sum():,} ({tf_bound.mean()*100:.2f}%)")
    print(f"    Unbound (P{args.tfbs_unbound_percentile} < {tfbs_unbound_thresh:.4f}): "
          f"{tf_unbound.sum():,} ({tf_unbound.mean()*100:.2f}%)")
    print(f"    Intermediate: "
          f"{n_total - tf_bound.sum() - tf_unbound.sum():,} "
          f"({(1 - tf_bound.mean() - tf_unbound.mean())*100:.2f}%)")
    n_regions_with_bound = (tf_bound.any(axis=1)).sum()
    print(f"    Regions with >=1 bound tile: "
          f"{n_regions_with_bound:,} ({n_regions_with_bound/n_regions*100:.1f}%)")

    print(f"\n  NucBS raw (unbounded score):")
    print(f"    Range: [{NucBS_raw.min():.4f}, {NucBS_raw.max():.4f}]")
    print(f"    Mean: {NucBS_raw.mean():.4f}, Median: {np.median(NucBS_raw):.4f}")

    print(f"\n  NucBS sigmoid (probability [0,1]):")
    print(f"    Range: [{NucBS_prob.min():.4f}, {NucBS_prob.max():.4f}]")
    print(f"    Mean: {NucBS_prob.mean():.4f}, Median: {np.median(NucBS_prob):.4f}")
    print(f"    Occupied (P{args.nucbs_occ_percentile} > {nucbs_occ_thresh:.4f}): "
          f"{nuc_occupied.sum():,} ({nuc_occupied.mean()*100:.2f}%)")
    print(f"    Free (P{args.nucbs_free_percentile} < {nucbs_free_thresh:.4f}): "
          f"{nuc_free.sum():,} ({nuc_free.mean()*100:.2f}%)")
    print(f"    Intermediate: "
          f"{n_total - nuc_occupied.sum() - nuc_free.sum():,} "
          f"({(1 - nuc_occupied.mean() - nuc_free.mean())*100:.2f}%)")
    n_regions_with_occ = (nuc_occupied.any(axis=1)).sum()
    print(f"    Regions with >=1 occupied tile: "
          f"{n_regions_with_occ:,} ({n_regions_with_occ/n_regions*100:.1f}%)")

    # Percentiles
    for label, arr in [("TFBS_prob", TFBS_prob), ("NucBS_prob", NucBS_prob)]:
        print(f"\n  {label} percentiles:")
        for p in [1, 5, 25, 50, 75, 95, 99, 99.9]:
            print(f"    P{p}: {np.percentile(arr, p):.4f}")

    # ── Save NPZ ────────────────────────────────────────────────────────
    out_path = os.path.join(args.outdir, f"_bs_{cond}.npz")
    np.savez_compressed(
        out_path,
        region_strs=np.array(region_strs, dtype=object),
        TFBS_prob=TFBS_prob,
        NucBS_raw=NucBS_raw,
        NucBS_prob=NucBS_prob,
        tile_bp=TILE_BP.astype(np.int32),
        tf_bound=tf_bound,
        tf_unbound=tf_unbound,
        nuc_occupied=nuc_occupied,
        nuc_free=nuc_free,
        # Store thresholds for reproducibility
        tfbs_bound_threshold=np.array([tfbs_bound_thresh]),
        tfbs_unbound_threshold=np.array([tfbs_unbound_thresh]),
        tfbs_bound_percentile=np.array([args.tfbs_bound_percentile]),
        tfbs_unbound_percentile=np.array([args.tfbs_unbound_percentile]),
        nucbs_occ_threshold=np.array([nucbs_occ_thresh]),
        nucbs_free_threshold=np.array([nucbs_free_thresh]),
        nucbs_occ_percentile=np.array([args.nucbs_occ_percentile]),
        nucbs_free_percentile=np.array([args.nucbs_free_percentile]),
    )
    print(f"\n[SAVE] {out_path}")
    print(f"  Arrays: TFBS_prob {TFBS_prob.shape}, NucBS_raw {NucBS_raw.shape}, "
          f"NucBS_prob {NucBS_prob.shape}")
    print(f"  tile_bp: {TILE_BP.shape}")
    print(f"  Boolean masks: tf_bound, tf_unbound, nuc_occupied, nuc_free")
    print(f"  Thresholds: TFBS bound=P{args.tfbs_bound_percentile} "
          f"({tfbs_bound_thresh:.4f}), unbound=P{args.tfbs_unbound_percentile} "
          f"({tfbs_unbound_thresh:.4f})")
    print(f"  Thresholds: NucBS occupied=P{args.nucbs_occ_percentile} "
          f"({nucbs_occ_thresh:.4f}), free=P{args.nucbs_free_percentile} "
          f"({nucbs_free_thresh:.4f})")

    print(f"\n[DONE] {cond} extraction complete.", flush=True)


if __name__ == "__main__":
    main()
