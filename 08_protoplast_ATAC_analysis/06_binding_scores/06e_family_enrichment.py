#!/usr/bin/env python3
"""
v4_03e — TF family enrichment at significant-delta tile positions.

For each of 18 boxes (3 ACR classes x 3 overlap groups x 2 directions),
tests whether specific TF families are enriched among tiles with significant
FP deltas compared to the genome-wide background of all active tiles.

Uses v3 motif signature hits mapped to tile positions (10bp windows).

Usage (local):
  /opt/anaconda3/bin/python3 -u v4/v4_03e_family_enrichment.py
"""

import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

# ── Constants ─────────────────────────────────────────────────────────────────
def pct_tag(tfbs_pct, nucbs_pct):
    def _fmt(v):
        return str(int(v)) if v == int(v) else f"{v:.1f}"
    return f"_tf{_fmt(tfbs_pct)}_nuc{_fmt(nucbs_pct)}"


ACR_CLASSES = ["proto_gain", "stable", "leaf_gain"]
OVERLAP_GROUPS = ["shared", "leaf_only", "proto_only"]
DIRECTIONS = ["leaf_enriched", "proto_enriched"]
CLASS_COLORS = {"proto_gain": "#E64B35", "stable": "#808080", "leaf_gain": "#4DBBD5"}
DIR_COLORS = {"leaf_enriched": "#D62728", "proto_enriched": "#1F77B4"}

TILE_SIZE = 10
TILE_HALF = TILE_SIZE // 2  # ±5bp overlap window
N_TILES = 180
TILE_BP = np.arange(N_TILES) * TILE_SIZE + 100 + TILE_SIZE // 2  # [105, 115, ..., 1895]

Z_THRESH = 1.0


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bs-dir", default="results/v4_03a_binding_scores")
    p.add_argument("--delta-dir", default="results/v4_03d_binding_deltas")
    p.add_argument("--v3-chunks", default="data/v3_chunks")
    p.add_argument("--sig-metadata", default="data/motif_signatures/signature_metadata.tsv")
    p.add_argument("--mapping", default="data/acr_native_to_resized.tsv")
    p.add_argument("--metadata", default="v4/data/acr_metadata.tsv.gz")
    p.add_argument("--outdir", default="results/v4_03e_family_enrichment")
    p.add_argument("--tfbs-pct", type=float, default=5)
    p.add_argument("--nucbs-pct", type=float, default=2)
    p.add_argument("--native-only", action="store_true",
                   help="Restrict to tiles inside native ACR boundaries")
    return p.parse_args()


# ── Helpers ───────────────────────────────────────────────────────────────────

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
    return np.array([resized_to_class.get(r, "unknown") for r in region_strs])


def load_motif_hits(v3_chunks_dir, mapping_path, sig_metadata_path):
    """Load all v3 signature hits, map to resized tile indices.

    Returns DataFrame with columns:
      resized_str, tile_idx, motif_id, family
    """
    # Native -> resized mapping
    mapping = pd.read_csv(mapping_path, sep="\t")
    nat_to_resized = dict(zip(mapping["native_str"], mapping["resized_str"]))
    resized_starts = dict(zip(mapping["resized_str"],
                              mapping["resized_str"].str.split(":").str[1]
                              .str.split("-").str[0].astype(int)))

    # Signature -> family
    sig_meta = pd.read_csv(sig_metadata_path, sep="\t",
                           usecols=["signature_id", "primary_family"])
    sig_to_family = dict(zip(sig_meta["signature_id"], sig_meta["primary_family"]))

    chunk_files = sorted(glob.glob(os.path.join(v3_chunks_dir, "chunk_*/motif_hits.tsv.gz")))
    print(f"  Loading {len(chunk_files)} chunk files...")

    frames = []
    for i, f in enumerate(chunk_files):
        df = pd.read_csv(f, sep="\t", usecols=["region_str", "motif_id", "hit_center"])
        # Map region to resized
        df["resized_str"] = df["region_str"].map(nat_to_resized)
        df = df.dropna(subset=["resized_str"])
        # Map hit_center to resized bp position
        df["resized_start"] = df["resized_str"].map(resized_starts)
        df["hit_bp_resized"] = df["hit_center"] - df["resized_start"]
        # Map to nearest tile index
        df["tile_idx"] = ((df["hit_bp_resized"] - TILE_BP[0] + TILE_HALF) // TILE_SIZE).astype(int)
        # Clip to valid tile range
        df = df[(df["tile_idx"] >= 0) & (df["tile_idx"] < N_TILES)]
        # Map to family
        df["family"] = df["motif_id"].map(sig_to_family)
        df = df.dropna(subset=["family"])

        frames.append(df[["resized_str", "tile_idx", "motif_id", "family"]])

        if (i + 1) % 10 == 0:
            print(f"    {i+1}/{len(chunk_files)} chunks...", flush=True)

    hits = pd.concat(frames, ignore_index=True)
    # Deduplicate: same region + tile + family counted once
    hits_dedup = hits.drop_duplicates(subset=["resized_str", "tile_idx", "family"])
    print(f"  Total hits: {len(hits):,}, dedup (region×tile×family): {len(hits_dedup):,}")
    print(f"  Families: {hits_dedup['family'].nunique()}")
    return hits_dedup


def build_tile_keys(region_strs, mask):
    """Convert a boolean mask (n_regions, n_tiles) to a set of (resized_str, tile_idx) tuples."""
    rows, cols = np.where(mask)
    return set(zip(region_strs[rows], cols))


def compute_enrichment(fg_keys, bg_keys, hits_df, families):
    """Fisher's exact test per family: fg vs bg tile sets.

    Returns list of dicts with OR, p, counts per family.
    """
    # Index hits by (resized_str, tile_idx)
    hit_keys = set(zip(hits_df["resized_str"], hits_df["tile_idx"]))

    # For each family, get its tile keys
    family_tile_keys = {}
    for fam, grp in hits_df.groupby("family"):
        family_tile_keys[fam] = set(zip(grp["resized_str"], grp["tile_idx"]))

    n_fg = len(fg_keys)
    n_bg = len(bg_keys)

    results = []
    for fam in families:
        fam_keys = family_tile_keys.get(fam, set())

        # Foreground: tiles in fg that have this family
        a = len(fg_keys & fam_keys)
        b = n_fg - a
        # Background: tiles in bg that have this family
        c = len(bg_keys & fam_keys)
        d = n_bg - c

        if a + c == 0:
            results.append(dict(family=fam, a=a, b=b, c=c, d=d,
                                OR=np.nan, pvalue=1.0))
            continue

        OR, p = fisher_exact([[a, b], [c, d]], alternative="two-sided")
        results.append(dict(family=fam, a=a, b=b, c=c, d=d, OR=OR, pvalue=p))

    return results


def bh_fdr(pvals):
    """Benjamini-Hochberg FDR correction."""
    n = len(pvals)
    if n == 0:
        return np.array([])
    pvals = np.array(pvals)
    order = np.argsort(pvals)
    fdr = np.empty(n)
    fdr[order] = pvals[order] * n / np.arange(1, n + 1)
    fdr[order] = np.minimum.accumulate(fdr[order][::-1])[::-1]
    return np.clip(fdr, 0, 1)


def _fix_family_name(name):
    """Prefix bZIP subgroup names for clarity."""
    if name.startswith("Group "):
        return f"bZIP {name}"
    return name


def plot_enrichment_grid(all_results, score_type, outdir, tag, suffix=""):
    """3x3 volcano-style grid: x=log2(OR), y=-log10(FDR).
    All families shown as gray; significant ones colored + labeled.
    Dot size proportional to |log2(OR)|. Shared y-axis per score type."""
    y_clip = {"TFBS": 4, "NucBS": 10}.get(score_type, 5)
    fig, axes = plt.subplots(3, 3, figsize=(14, 12), sharey=True)
    FDR_LINE = -np.log10(0.05)

    for row, cls in enumerate(ACR_CLASSES):
        for col, grp in enumerate(OVERLAP_GROUPS):
            ax = axes[row, col]
            has_legend = False

            for direction in DIRECTIONS:
                key = (score_type, cls, grp, direction)
                df = all_results.get(key, pd.DataFrame())
                if df.empty:
                    continue

                df = df.copy()
                df["log2OR"] = np.log2(df["OR"].clip(0.01, 100))
                df["neg_log10_fdr"] = np.clip(
                    -np.log10(df["fdr"].clip(1e-50, 1)), 0, y_clip)
                df["family_label"] = df["family"].apply(_fix_family_name)
                df["dot_size"] = np.clip(np.abs(df["log2OR"]) * 20, 10, 120)

                is_sig = df["fdr"] < 0.05
                color = DIR_COLORS[direction]

                # All families as gray
                ax.scatter(df.loc[~is_sig, "log2OR"],
                           df.loc[~is_sig, "neg_log10_fdr"],
                           s=df.loc[~is_sig, "dot_size"],
                           c="#D9D9D9", alpha=0.4, edgecolors="none")

                # Significant families colored
                if is_sig.any():
                    lbl = direction.replace("_", " ") if not has_legend else ""
                    ax.scatter(df.loc[is_sig, "log2OR"],
                               df.loc[is_sig, "neg_log10_fdr"],
                               s=df.loc[is_sig, "dot_size"],
                               c=color, alpha=0.8, edgecolors="k",
                               linewidths=0.4, label=lbl, zorder=5)
                    has_legend = True

                    # Label significant families
                    for _, r in df[is_sig].iterrows():
                        ax.text(r["log2OR"] + 0.08, r["neg_log10_fdr"],
                                r["family_label"], fontsize=5, va="center",
                                color=color, fontweight="bold")

            ax.axvline(0, color="gray", ls="--", lw=0.5)
            ax.axhline(FDR_LINE, color="gray", ls=":", lw=0.5, alpha=0.7)
            ax.set_ylim(-0.2, y_clip)
            ax.set_xlabel("log2(OR)", fontsize=8)
            if col == 0:
                ax.set_ylabel(f"{cls}\n-log10(FDR)", fontsize=9,
                              color=CLASS_COLORS[cls], fontweight="bold")

            n_sig_total = 0
            for d in DIRECTIONS:
                _df = all_results.get((score_type, cls, grp, d), pd.DataFrame())
                if "fdr" in _df.columns:
                    n_sig_total += int((_df["fdr"] < 0.05).sum())
            ax.set_title(f"{grp} ({n_sig_total} sig)",
                         fontsize=8, fontweight="bold",
                         color={'shared': '#7B2D8E', 'leaf_only': '#D62728',
                                'proto_only': '#1F77B4'}[grp])

    # Manual legend at figure level
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#D62728",
               markersize=8, markeredgecolor="k", markeredgewidth=0.4,
               label="leaf enriched (z >= 1)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#1F77B4",
               markersize=8, markeredgecolor="k", markeredgewidth=0.4,
               label="proto enriched (z <= -1)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#D9D9D9",
               markersize=6, markeredgecolor="none",
               label="not significant"),
    ]
    # Size legend
    for or_val, sz_label in [(1, "|log2 OR|=1"), (3, "|log2 OR|=3")]:
        legend_elements.append(
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#999",
                   markersize=np.sqrt(or_val * 20), markeredgecolor="k",
                   markeredgewidth=0.3, label=sz_label))

    fig.legend(handles=legend_elements, loc="lower center", ncol=5,
               fontsize=8, framealpha=0.9, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(f"{score_type}: TF family enrichment at significant-delta tiles\n"
                 f"(|z| >= {Z_THRESH}, volcano: x=log2 OR, y=-log10 FDR)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    path = os.path.join(outdir, f"fig_{tag}_family_enrichment{suffix}")
    for fmt in ("pdf", "png"):
        fig.savefig(f"{path}.{fmt}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVE] {path}.pdf/.png")


def plot_summary_heatmap(all_results, families, outdir, suffix=""):
    """Two summary heatmaps (TFBS + NucBS), one per score type."""
    # Collect significant results
    rows = []
    for key, df in all_results.items():
        if df.empty:
            continue
        score_type, cls, grp, direction = key
        sig = df[df["fdr"] < 0.05]
        for _, r in sig.iterrows():
            rows.append(dict(score_type=score_type, acr_class=cls, group=grp,
                             direction=direction, family=r["family"],
                             log2OR=np.log2(max(r["OR"], 0.01)),
                             fdr=r["fdr"]))

    if not rows:
        print("[WARN] No significant enrichments for summary heatmap")
        return

    sig_df = pd.DataFrame(rows)

    for score_type, stag in [("TFBS", "C_tfbs"), ("NucBS", "D_nucbs")]:
        st_df = sig_df[sig_df["score_type"] == score_type]
        if st_df.empty:
            continue

        # Top families by frequency
        top_fams = st_df["family"].value_counts().head(25).index.tolist()
        top_fam_labels = [_fix_family_name(f) for f in top_fams]

        # Columns: ACR class × overlap × direction
        box_specs = [(cls, grp, d)
                     for cls in ACR_CLASSES for grp in OVERLAP_GROUPS
                     for d in DIRECTIONS]
        box_labels = [f"{cls}\n{grp}\n{d[:5]}" for cls, grp, d in box_specs]

        mat = np.full((len(top_fams), len(box_specs)), np.nan)
        for i, fam in enumerate(top_fams):
            for j, (cls, grp, d) in enumerate(box_specs):
                key = (score_type, cls, grp, d)
                df = all_results.get(key, pd.DataFrame())
                if df.empty:
                    continue
                match = df[df["family"] == fam]
                if not match.empty and match.iloc[0]["fdr"] < 0.05:
                    mat[i, j] = np.log2(max(match.iloc[0]["OR"], 0.01))

        fig, ax = plt.subplots(figsize=(14, max(5, len(top_fams) * 0.35)))
        vmax = np.nanmax(np.abs(mat[np.isfinite(mat)])) if np.any(np.isfinite(mat)) else 1
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")

        ax.set_xticks(range(len(box_labels)))
        ax.set_xticklabels(box_labels, fontsize=6, rotation=90)
        ax.set_yticks(range(len(top_fams)))
        ax.set_yticklabels(top_fam_labels, fontsize=8)
        plt.colorbar(im, ax=ax, shrink=0.6, label="log2(OR)")
        ax.set_title(f"{score_type}: enriched TF families (FDR < 0.05)",
                     fontsize=12, fontweight="bold")
        fig.tight_layout()

        path = os.path.join(outdir, f"fig_{stag}_summary_heatmap{suffix}")
        for fmt in ("pdf", "png"):
            fig.savefig(f"{path}.{fmt}", dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"[SAVE] {path}.pdf/.png")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    print("=== v4_03e: TF Family Enrichment at Significant-Delta Tiles ===\n")

    # ── Load binding scores & build masks ─────────────────────────────────
    print("[LOAD] Binding scores...")
    leaf_bs = np.load(os.path.join(args.bs_dir, "_bs_leaf.npz"), allow_pickle=True)
    proto_bs = np.load(os.path.join(args.bs_dir, "_bs_proto.npz"), allow_pickle=True)
    region_strs = leaf_bs["region_strs"]
    n_regions = len(region_strs)

    tf_cutoff = 100 - args.tfbs_pct
    nuc_cutoff = 100 - args.nucbs_pct

    tf_leaf = leaf_bs["TFBS_prob"] > np.percentile(leaf_bs["TFBS_prob"], tf_cutoff)
    tf_proto = proto_bs["TFBS_prob"] > np.percentile(proto_bs["TFBS_prob"], tf_cutoff)
    nuc_leaf = leaf_bs["NucBS_prob"] > np.percentile(leaf_bs["NucBS_prob"], nuc_cutoff)
    nuc_proto = proto_bs["NucBS_prob"] > np.percentile(proto_bs["NucBS_prob"], nuc_cutoff)

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

    # Active masks: bound/occupied in at least one condition
    tf_active = tf_leaf | tf_proto
    nuc_active = nuc_leaf | nuc_proto

    print(f"  TFBS active tiles: {tf_active.sum():,}")
    print(f"  NucBS active tiles: {nuc_active.sum():,}")

    # ── ACR class ─────────────────────────────────────────────────────────
    print("\n[MAP] ACR classes...")
    rcls = build_region_to_class(args.metadata, args.mapping, region_strs)
    for c in ACR_CLASSES:
        print(f"  {c}: {(rcls == c).sum():,} regions")

    # ── Load deltas & z-scores from v4_03d ────────────────────────────────
    _tag = pct_tag(args.tfbs_pct, args.nucbs_pct)
    if args.native_only:
        _tag += "_native"
    delta_file = os.path.join(args.delta_dir, f"tile_deltas{_tag}.npz")
    if not os.path.exists(delta_file):
        # Fall back to untagged file for legacy runs
        delta_file = os.path.join(args.delta_dir, "tile_deltas.npz")
    print(f"\n[LOAD] Tile deltas from {delta_file}")
    npz = np.load(delta_file, allow_pickle=True)

    # ── Load motif hits ───────────────────────────────────────────────────
    print("\n[LOAD] v3 motif signature hits...")
    hits = load_motif_hits(args.v3_chunks, args.mapping, args.sig_metadata)
    families = sorted(hits["family"].unique())
    print(f"  {len(families)} families")

    # ── Build tile key sets ───────────────────────────────────────────────
    print("\n[BUILD] Tile key sets...")

    # Genome-wide background: all active tiles
    tf_bg_keys = build_tile_keys(region_strs, tf_active)
    nuc_bg_keys = build_tile_keys(region_strs, nuc_active)
    print(f"  TFBS background: {len(tf_bg_keys):,} tile positions")
    print(f"  NucBS background: {len(nuc_bg_keys):,} tile positions")

    # Category masks (same logic as v4_03d)
    print("\n[BUILD] Category foreground sets...")
    all_results = {}
    enrich_rows = []

    for score_type, active_mask, cond_leaf, cond_proto, bg_keys in [
        ("TFBS", tf_active, tf_leaf, tf_proto, tf_bg_keys),
        ("NucBS", nuc_active, nuc_leaf, nuc_proto, nuc_bg_keys),
    ]:
        print(f"\n{'='*60}")
        print(f"  {score_type}")
        print(f"{'='*60}")

        for cls in ACR_CLASSES:
            cls_idx = rcls == cls
            cls_expand = np.zeros_like(active_mask)
            cls_expand[cls_idx] = True

            ml = cond_leaf & cls_expand
            mp = cond_proto & cls_expand

            cat_masks = {
                "shared": ml & mp,
                "leaf_only": ml & ~mp,
                "proto_only": ~ml & mp,
            }

            for grp in OVERLAP_GROUPS:
                mask = cat_masks[grp]
                n_cat = mask.sum()

                # Get z-scores for this category
                z_key = f"{score_type}_{cls}_{grp}_z"
                z = npz.get(z_key, np.array([]))

                if len(z) == 0 or n_cat == 0:
                    print(f"  {cls}/{grp}: no data")
                    continue

                # Reconstruct which tiles are in this category
                # (same order as v4_03d extraction — row-major through the mask)
                cat_rows, cat_cols = np.where(mask)

                for direction, z_cond in [
                    ("leaf_enriched", z >= Z_THRESH),
                    ("proto_enriched", z <= -Z_THRESH),
                ]:
                    sig_idx = np.where(z_cond)[0]
                    if len(sig_idx) == 0:
                        continue

                    # Map to tile keys
                    fg_keys = set()
                    for si in sig_idx:
                        if si < len(cat_rows):
                            fg_keys.add((region_strs[cat_rows[si]], cat_cols[si]))

                    n_fg = len(fg_keys)
                    print(f"  {cls}/{grp}/{direction}: {n_fg:,} sig tiles")

                    if n_fg < 3:
                        continue

                    # Fisher enrichment per family
                    res = compute_enrichment(fg_keys, bg_keys, hits, families)
                    res_df = pd.DataFrame(res)
                    res_df["fdr"] = bh_fdr(res_df["pvalue"].values)

                    n_sig_fam = (res_df["fdr"] < 0.05).sum()
                    print(f"    -> {n_sig_fam} significant families (FDR<0.05)")

                    all_results[(score_type, cls, grp, direction)] = res_df

                    for _, r in res_df.iterrows():
                        enrich_rows.append(dict(
                            score_type=score_type, acr_class=cls, group=grp,
                            direction=direction, **r.to_dict()))

    # ── Save TSV ──────────────────────────────────────────────────────────
    _tag = pct_tag(args.tfbs_pct, args.nucbs_pct)
    if args.native_only:
        _tag += "_native"
    if enrich_rows:
        tsv_path = os.path.join(args.outdir, f"enrichment_results{_tag}.tsv")
        pd.DataFrame(enrich_rows).to_csv(tsv_path, sep="\t", index=False,
                                          float_format="%.6f")
        print(f"\n[SAVE] {tsv_path}")

    # ── Plots ─────────────────────────────────────────────────────────────
    print("\n[PLOT]")
    plot_enrichment_grid(all_results, "TFBS", args.outdir, "A_tfbs", suffix=_tag)
    plot_enrichment_grid(all_results, "NucBS", args.outdir, "B_nucbs", suffix=_tag)
    plot_summary_heatmap(all_results, families, args.outdir, suffix=_tag)

    print("\n[DONE]")


if __name__ == "__main__":
    main()
