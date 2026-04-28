#!/usr/bin/env python3
"""
v4_03c_acr_tile_table — Per-ACR tile classification matching Venn diagram groups.

For each ACR, counts how many of its 180 tiles fall into each overlap category
(shared, leaf_only, proto_only) at each threshold, for both TFBS and NucBS.

Output: one TSV per threshold combo, rows = (acr_class, overlap_group, acr_id, n_tiles).
Only rows where n_tiles > 0 are included.

Usage:
  /opt/anaconda3/bin/python3 -u v4/v4_03c_acr_tile_table.py
"""

import argparse
import os

import numpy as np
import pandas as pd


ACR_CLASSES = ["proto_gain", "stable", "leaf_gain"]
OVERLAP_GROUPS = ["shared", "leaf_only", "proto_only"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bs-dir", default="results/v4_03a_binding_scores")
    p.add_argument("--metadata", default="v4/data/acr_metadata.tsv.gz")
    p.add_argument("--mapping", default="data/acr_native_to_resized.tsv")
    p.add_argument("--outdir", default="results/v4_03c_binding_overlap")
    p.add_argument("--native-only", action="store_true",
                   help="Restrict to tiles inside native ACR boundaries")
    return p.parse_args()


def build_acr_info(metadata_path, mapping_path, region_strs):
    """Map region strings -> (acr_class, native_coord, chr, start, end, logFC, fdr, genomic_context)."""
    meta = pd.read_csv(metadata_path, sep="\t")
    meta["native_str"] = (meta["chr"].str.lower() + ":" +
                          meta["start"].astype(str) + "-" +
                          meta["end"].astype(str))
    mapping = pd.read_csv(mapping_path, sep="\t")
    nat_to_resized = dict(zip(mapping["native_str"], mapping["resized_str"]))
    meta["resized_str"] = meta["native_str"].map(nat_to_resized)

    # Build lookup: resized_str -> row of metadata
    meta_cols = ["acr_class", "native_str", "chr", "start", "end"]
    # v4 metadata uses edgeR_ prefix and 'width' instead of 'acr_width'
    col_map = {
        "edgeR_logFC": "logFC", "edgeR_fdr": "fdr", "edgeR_logCPM": "logCPM",
        "genomic_context": "genomic_context", "width": "acr_width",
        "nearest_gene": "nearest_gene", "distance_to_tss": "distance_to_tss",
    }
    for src, dst in col_map.items():
        if src in meta.columns:
            if src != dst:
                meta = meta.rename(columns={src: dst})
            meta_cols.append(dst)
    meta_lookup = meta.set_index("resized_str")[meta_cols].to_dict("index")

    records = []
    for r in region_strs:
        info = meta_lookup.get(r, {})
        records.append({
            "resized_str": r,
            "acr_class": info.get("acr_class", "unknown"),
            "native_coord": info.get("native_str", ""),
            "chr": info.get("chr", ""),
            "start": info.get("start", ""),
            "end": info.get("end", ""),
            "logFC": info.get("logFC", np.nan),
            "logCPM": info.get("logCPM", np.nan),
            "fdr": info.get("fdr", np.nan),
            "genomic_context": info.get("genomic_context", ""),
            "acr_width": info.get("acr_width", ""),
            "nearest_gene": info.get("nearest_gene", ""),
            "distance_to_tss": info.get("distance_to_tss", np.nan),
        })
    return pd.DataFrame(records)


def classify_tiles(mask_leaf, mask_proto):
    """Per-tile overlap category: 0=neither, 1=shared, 2=leaf_only, 3=proto_only."""
    out = np.zeros(mask_leaf.shape, dtype=np.int8)
    out[mask_leaf & mask_proto] = 1    # shared
    out[mask_leaf & ~mask_proto] = 2   # leaf_only
    out[~mask_leaf & mask_proto] = 3   # proto_only
    return out


def build_table(tile_classes, acr_info, score_type, top_pct,
                tfbs_leaf, tfbs_proto, nucbs_leaf, nucbs_proto,
                valid_mask=None):
    """Build per-ACR rows: one row per (acr, overlap_group) with n_tiles > 0.

    Adds per-group probability summary stats and whole-ACR max-tile deltas.
    valid_mask: optional (n_regions, 180) bool — tiles to consider for
                whole-ACR stats (native mask when --native-only).
    """
    group_map = {1: "shared", 2: "leaf_only", 3: "proto_only"}
    rows = []
    for i in range(tile_classes.shape[0]):
        acr = acr_info.iloc[i]
        if acr["acr_class"] == "unknown":
            continue

        # Whole-ACR max-tile stats (across all valid tiles, not just classified)
        if valid_mask is not None:
            vm = valid_mask[i]
        else:
            vm = np.ones(tile_classes.shape[1], dtype=bool)
        if vm.sum() > 0:
            tf_all_l = tfbs_leaf[i, vm]
            tf_all_p = tfbs_proto[i, vm]
            nuc_all_l = nucbs_leaf[i, vm]
            nuc_all_p = nucbs_proto[i, vm]
            acr_stats = {
                "acr_tfbs_max_leaf": float(tf_all_l.max()),
                "acr_tfbs_max_proto": float(tf_all_p.max()),
                "acr_tfbs_max_delta": float(tf_all_l.max() - tf_all_p.max()),
                "acr_nucbs_max_leaf": float(nuc_all_l.max()),
                "acr_nucbs_max_proto": float(nuc_all_p.max()),
                "acr_nucbs_max_delta": float(nuc_all_l.max() - nuc_all_p.max()),
                "acr_n_valid_tiles": int(vm.sum()),
            }
        else:
            acr_stats = {
                "acr_tfbs_max_leaf": np.nan, "acr_tfbs_max_proto": np.nan,
                "acr_tfbs_max_delta": np.nan, "acr_nucbs_max_leaf": np.nan,
                "acr_nucbs_max_proto": np.nan, "acr_nucbs_max_delta": np.nan,
                "acr_n_valid_tiles": 0,
            }

        tiles = tile_classes[i]  # shape (180,)
        for code, group_name in group_map.items():
            mask = tiles == code
            n = int(mask.sum())
            if n == 0:
                continue
            # Probability stats for the classified tiles
            tf_l = tfbs_leaf[i, mask]
            tf_p = tfbs_proto[i, mask]
            nuc_l = nucbs_leaf[i, mask]
            nuc_p = nucbs_proto[i, mask]
            row = {
                "score_type": score_type,
                "top_pct": top_pct,
                "acr_class": acr["acr_class"],
                "overlap_group": group_name,
                "acr_id": acr["resized_str"],
                "native_coord": acr["native_coord"],
                "chr": acr["chr"],
                "start": acr["start"],
                "end": acr["end"],
                "n_tiles": n,
                "logFC": acr["logFC"],
                "logCPM": acr["logCPM"],
                "fdr": acr["fdr"],
                "genomic_context": acr["genomic_context"],
                "acr_width": acr["acr_width"],
                "nearest_gene": acr["nearest_gene"],
                "distance_to_tss": acr["distance_to_tss"],
                # TFBS prob stats at the classified tiles
                "tfbs_prob_leaf_mean": float(tf_l.mean()),
                "tfbs_prob_leaf_max": float(tf_l.max()),
                "tfbs_prob_proto_mean": float(tf_p.mean()),
                "tfbs_prob_proto_max": float(tf_p.max()),
                "tfbs_prob_delta_mean": float((tf_l - tf_p).mean()),
                # NucBS prob stats at the classified tiles
                "nucbs_prob_leaf_mean": float(nuc_l.mean()),
                "nucbs_prob_leaf_max": float(nuc_l.max()),
                "nucbs_prob_proto_mean": float(nuc_p.mean()),
                "nucbs_prob_proto_max": float(nuc_p.max()),
                "nucbs_prob_delta_mean": float((nuc_l - nuc_p).mean()),
            }
            row.update(acr_stats)
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    print("=== v4_03c: ACR tile classification table ===\n")

    # Load binding scores
    leaf = np.load(os.path.join(args.bs_dir, "_bs_leaf.npz"), allow_pickle=True)
    proto = np.load(os.path.join(args.bs_dir, "_bs_proto.npz"), allow_pickle=True)
    region_strs = leaf["region_strs"]
    assert np.array_equal(region_strs, proto["region_strs"])
    n_regions, n_tiles = leaf["TFBS_prob"].shape
    print(f"  {n_regions:,} regions x {n_tiles} tiles\n")

    # ACR metadata
    print("[MAP] Building ACR info...")
    acr_info = build_acr_info(args.metadata, args.mapping, region_strs)
    for cls in ACR_CLASSES:
        print(f"  {cls}: {(acr_info['acr_class'] == cls).sum():,}")
    print()

    # ── Native-only masking ───────────────────────────────────────────
    native_mask = None
    if args.native_only:
        import sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib"))
        from _tile_utils import build_native_tile_mask
        print("[NATIVE] Building native ACR tile mask...")
        native_mask, _ = build_native_tile_mask(
            region_strs, args.metadata, args.mapping)
        print()

    # Thresholds matching the figure panels
    threshold_combos = [
        (2, 2),    # Panel A/B
        (5, 5),    # Panel C/D
        (10, 10),  # Panel E/F
        (5, 2),    # mixed (default in v4_03c)
    ]

    for tf_pct, nuc_pct in threshold_combos:
        tf_cutoff = 100 - tf_pct
        nuc_cutoff = 100 - nuc_pct

        tf_t_l = np.percentile(leaf["TFBS_prob"], tf_cutoff)
        tf_t_p = np.percentile(proto["TFBS_prob"], tf_cutoff)
        nuc_t_l = np.percentile(leaf["NucBS_prob"], nuc_cutoff)
        nuc_t_p = np.percentile(proto["NucBS_prob"], nuc_cutoff)

        # Probability arrays (float, for summary stats)
        tfbs_l = leaf["TFBS_prob"]
        tfbs_p = proto["TFBS_prob"]
        nucbs_l = leaf["NucBS_prob"]
        nucbs_p = proto["NucBS_prob"]

        # TFBS tile classification
        tf_leaf = tfbs_l > tf_t_l
        tf_proto = tfbs_p > tf_t_p
        if native_mask is not None:
            tf_leaf = tf_leaf & native_mask
            tf_proto = tf_proto & native_mask
        tf_tiles = classify_tiles(tf_leaf, tf_proto)
        df_tf = build_table(tf_tiles, acr_info, "TFBS", tf_pct,
                            tfbs_l, tfbs_p, nucbs_l, nucbs_p,
                            valid_mask=native_mask)

        # NucBS tile classification
        nuc_leaf = nucbs_l > nuc_t_l
        nuc_proto = nucbs_p > nuc_t_p
        if native_mask is not None:
            nuc_leaf = nuc_leaf & native_mask
            nuc_proto = nuc_proto & native_mask
        nuc_tiles = classify_tiles(nuc_leaf, nuc_proto)
        df_nuc = build_table(nuc_tiles, acr_info, "NucBS", nuc_pct,
                             tfbs_l, tfbs_p, nucbs_l, nucbs_p,
                             valid_mask=native_mask)

        # Combine
        df = pd.concat([df_tf, df_nuc], ignore_index=True)
        df = df.sort_values(["score_type", "acr_class", "overlap_group", "n_tiles"],
                            ascending=[True, True, True, False])

        tag = f"tf{tf_pct}_nuc{nuc_pct}"
        if args.native_only:
            tag += "_native"
        out_path = os.path.join(args.outdir, f"acr_tile_table_{tag}.tsv")
        df.to_csv(out_path, sep="\t", index=False, float_format="%.4f")

        # Summary
        n_tf = len(df_tf)
        n_nuc = len(df_nuc)
        print(f"[{tag}] TFBS top {tf_pct}% (thresh: L>{tf_t_l:.4f}, P>{tf_t_p:.4f}): "
              f"{n_tf:,} ACR-group rows")
        print(f"[{tag}] NucBS top {nuc_pct}% (thresh: L>{nuc_t_l:.4f}, P>{nuc_t_p:.4f}): "
              f"{n_nuc:,} ACR-group rows")
        print(f"[SAVE] {out_path}  ({len(df):,} rows)")

        # Quick sanity: totals should match Venn counts
        for st, pct in [("TFBS", tf_pct), ("NucBS", nuc_pct)]:
            sub = df[df["score_type"] == st]
            for cls in ACR_CLASSES:
                cls_sub = sub[sub["acr_class"] == cls]
                for grp in OVERLAP_GROUPS:
                    total = cls_sub.loc[cls_sub["overlap_group"] == grp, "n_tiles"].sum()
                    if total > 0:
                        print(f"  {st} {cls:12s} {grp:11s}: {total:>7,} tiles "
                              f"across {(cls_sub['overlap_group'] == grp).sum():,} ACRs")
        print()

    print("[DONE]")


if __name__ == "__main__":
    main()
