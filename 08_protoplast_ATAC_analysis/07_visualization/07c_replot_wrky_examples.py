#!/usr/bin/env python3
"""Quick replot of WRKY example regions with Blues cmap and 10/10% thresholds."""

import os
import sys
import numpy as np

import importlib.util
_spec = importlib.util.spec_from_file_location(
    "plot_regions_families",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "07b_plot_regions_families.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
load_extracted = _mod.load_extracted
plot_one_region_with_families = _mod.plot_one_region_with_families

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

# ── Config ────────────────────────────────────────────────────────────────
CATEGORY = "NucBS_leaf_gain_proto_only"
BASE_DIR = "results/v4_03f_region_families"
BS_DIR = "results/v4_03a_binding_scores"
TFBS_PCT = 10
NUCBS_PCT = 10
ZOOM_NATIVE = True
ZOOM_PAD = 200

# The 23 example regions (from the WRKY/examples directory)
EXAMPLE_REGIONS_RESIZED = [
    "chr1:10274824-10276824",
    "chr1:11219230-11221230",
    "chr1:17535114-17537114",
    "chr1:1860977-1862977",
    "chr1:19256160-19258160",
    "chr1:20733661-20735661",
    "chr1:20937443-20939443",
    "chr1:23554136-23556136",
    "chr1:25029089-25031089",
    "chr1:26337869-26339869",
    "chr1:2681150-2683150",
    "chr2:14291079-14293079",
    "chr2:19533368-19535368",
    "chr2:8704673-8706673",
    "chr3:1173637-1175637",
    "chr3:1993605-1995605",
    "chr3:4408149-4410149",
    "chr3:9306705-9308705",
    "chr5:17448239-17450239",
    "chr5:1833614-1835614",
    "chr5:26023698-26025698",
    "chr5:5205872-5207872",
    "chr5:7520297-7522297",
]

def main():
    cat_dir = os.path.join(BASE_DIR, CATEGORY)
    npz_file = os.path.join(cat_dir, f"{CATEGORY}_extracted.npz")

    if not os.path.exists(npz_file):
        print(f"ERROR: NPZ not found: {npz_file}")
        sys.exit(1)

    # Load extracted data
    print(f"[LOAD] {npz_file}")
    (regions, scales, _, _,
     _, _,
     enriched_families, fam_colors, enriched_or,
     data) = load_extracted(npz_file)

    # Recompute thresholds at 10/10% from full BS NPZs
    print(f"[THRESH] Recomputing at TFBS={TFBS_PCT}% / NucBS={NUCBS_PCT}%...")
    bs_leaf = np.load(os.path.join(BS_DIR, "_bs_leaf.npz"), allow_pickle=True)
    bs_proto = np.load(os.path.join(BS_DIR, "_bs_proto.npz"), allow_pickle=True)

    tfbs_thresh_l = float(np.percentile(bs_leaf["TFBS_prob"], 100 - TFBS_PCT))
    tfbs_thresh_p = float(np.percentile(bs_proto["TFBS_prob"], 100 - TFBS_PCT))
    nucbs_thresh_l = float(np.percentile(bs_leaf["NucBS_prob"], 100 - NUCBS_PCT))
    nucbs_thresh_p = float(np.percentile(bs_proto["NucBS_prob"], 100 - NUCBS_PCT))
    print(f"  TFBS: L>{tfbs_thresh_l:.4f}, P>{tfbs_thresh_p:.4f}")
    print(f"  NucBS: L>{nucbs_thresh_l:.4f}, P>{nucbs_thresh_p:.4f}")
    del bs_leaf, bs_proto

    # Rebuild color map
    family_color_map = {}
    for i, fam in enumerate(enriched_families):
        c = fam_colors[i] if i < len(fam_colors) else "#888888"
        or_val = enriched_or[i] if i < len(enriched_or) else 0
        family_color_map[fam] = (c, or_val)

    # Output dir
    outdir = os.path.join(cat_dir, "WRKY", "examples_blues_10pct")
    os.makedirs(outdir, exist_ok=True)

    # Filter to example regions
    example_set = set(EXAMPLE_REGIONS_RESIZED)
    to_plot = [r for r in regions if r in example_set]
    missing = example_set - set(regions)
    if missing:
        print(f"  WARNING: {len(missing)} regions not in NPZ: {missing}")
    print(f"\n[PLOT] {len(to_plot)} example regions → {outdir}/")

    for i, region_id in enumerate(to_plot):
        rec = data[region_id]

        meta_row = {}
        for key, val in rec.items():
            if key.startswith("meta_"):
                meta_row[key[5:]] = val

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
            outdir, CATEGORY, None,
            zoom_native=ZOOM_NATIVE, zoom_pad=ZOOM_PAD,
        )
        print(f"  [{i+1}/{len(to_plot)}] {fname}")

    print(f"\n[DONE] {len(to_plot)} plots saved to {outdir}/")


if __name__ == "__main__":
    main()
