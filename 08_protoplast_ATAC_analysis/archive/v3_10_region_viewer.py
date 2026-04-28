#!/usr/bin/env python3
"""
v3 Step 10: Motif-annotated multiscale footprint region viewer.

For each requested region, produces a 3-panel page:
  1) Leaf FP heatmap (mean of active reps)
  2) Proto FP heatmap (mean of active reps)
  3) Delta heatmap (leaf − proto) with signature-hit annotations

Uses v3 signature metadata for family filtering and display names.
Motif logos parsed from MEME file.

Output: results/v3_10_region_viewer/
"""

from __future__ import annotations

import argparse
import gc
import os
import pickle
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

try:
    import logomaker
    _HAS_LOGOMAKER = True
except Exception:
    _HAS_LOGOMAKER = False

try:
    from adjustText import adjust_text
    _HAS_ADJUSTTEXT = True
except Exception:
    _HAS_ADJUSTTEXT = False

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from _utils import nature_figure_defaults, nature_savefig

nature_figure_defaults()

EXCLUDE_REPS = {3}
ACTIVE_REPS = sorted({1, 2, 3} - EXCLUDE_REPS)
LEAF_IDS = [f"leaf_rep{r}" for r in ACTIVE_REPS]
PROTO_IDS = [f"proto_rep{r}" for r in ACTIVE_REPS]


# ── MEME motif logo loader ──────────────────────────────────────────────────

def load_meme_logos(meme_path: str) -> Dict[str, np.ndarray]:
    """Parse MEME file → {sig_id: probability matrix (4, width)}."""
    logos: Dict[str, np.ndarray] = {}
    sig_counter = 0
    with open(meme_path) as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("MOTIF "):
            sig_counter += 1
            sig_id = f"sig_{sig_counter:03d}"
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("letter-probability"):
                i += 1
            if i >= len(lines):
                break
            header = lines[i].strip()
            m = re.search(r"w=\s*(\d+)", header)
            if not m:
                i += 1
                continue
            width = int(m.group(1))
            rows = []
            i += 1
            for _ in range(width):
                if i >= len(lines):
                    break
                vals = lines[i].strip().split()
                if len(vals) >= 4:
                    rows.append([float(v) for v in vals[:4]])
                i += 1
            if len(rows) == width:
                logos[sig_id] = np.array(rows, dtype=float).T  # (4, width)
        else:
            i += 1
    return logos


# ── FP tensor helpers ────────────────────────────────────────────────────────

def get_fp_tensor(fp_adata, region: str) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(fp_adata.obsm[region])
    if arr.ndim == 3:
        arr = arr[0]
    scales = np.asarray(fp_adata.uns["scales"], dtype=float)
    return arr, scales


def load_printers_and_average(
    print_dir: str, genome, region: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    import scprinter as scp
    scales_out: Optional[np.ndarray] = None

    def _load_tensor(sample_id: str) -> np.ndarray:
        nonlocal scales_out
        h5ad_path = os.path.join(print_dir, f"printer_{sample_id}_bulk.h5ad")
        printer = scp.load_printer(h5ad_path, genome)
        fp_key = f"FP_{sample_id}_ALL".replace("-", "_").replace(".", "_")
        tensor, scales = get_fp_tensor(printer.footprintsadata[fp_key], region)
        if scales_out is None:
            scales_out = scales
        printer.close()
        del printer
        gc.collect()
        return tensor

    leaf_tensors = []
    for sid in LEAF_IDS:
        try:
            leaf_tensors.append(_load_tensor(sid))
        except Exception as e:
            print(f"  [WARN] Could not load {sid}: {e}", flush=True)
    if not leaf_tensors:
        raise ValueError(f"No leaf tensors loaded for {region}")

    proto_tensors = []
    for sid in PROTO_IDS:
        try:
            proto_tensors.append(_load_tensor(sid))
        except Exception as e:
            print(f"  [WARN] Could not load {sid}: {e}", flush=True)
    if not proto_tensors:
        raise ValueError(f"No proto tensors loaded for {region}")

    leaf_mean = np.mean(leaf_tensors, axis=0)
    proto_mean = np.mean(proto_tensors, axis=0)
    if scales_out is None:
        scales_out = np.arange(leaf_mean.shape[0], dtype=float)
    return leaf_mean, proto_mean, scales_out


# ── Region parsing ───────────────────────────────────────────────────────────

def parse_region(region: str) -> Tuple[str, int, int]:
    chrom, rest = region.split(":")
    start_s, end_s = rest.split("-")
    return chrom, int(start_s), int(end_s)


def regions_from_string(regions_str: str) -> List[str]:
    parts = [p.strip() for p in regions_str.split(",") if p.strip()]
    out = []
    for p in parts:
        out.extend([x for x in p.split() if x])
    seen = set()
    return [r for r in out if r not in seen and not seen.add(r)]


def regions_from_bed(path: str) -> List[str]:
    df = pd.read_csv(path, sep="\t", header=None, comment="#").iloc[:, :3]
    df.columns = ["Chromosome", "Start", "End"]
    return [f"{r.Chromosome}:{int(r.Start)}-{int(r.End)}"
            for r in df.itertuples(index=False)]


# ── Load and filter signature hits ───────────────────────────────────────────

def load_family_hits(
    merged_path: str,
    sig_meta_path: str,
    family: str,
    summary_path: Optional[str] = None,
    q_cutoff: Optional[float] = None,
    min_abs_delta: float = 0.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load hits filtered to a TF family using v3 signature metadata."""
    sig_meta = pd.read_csv(sig_meta_path, sep="\t")
    family_sigs = set(
        sig_meta.loc[sig_meta["primary_family"] == family, "signature_id"].unique()
    )
    if not family_sigs:
        raise ValueError(f"No signatures found for family '{family}' in {sig_meta_path}")
    print(f"[INFO] Family '{family}': {len(family_sigs)} signatures", flush=True)

    dn_map = dict(zip(sig_meta["signature_id"], sig_meta["display_name"]))

    use_cols = ["motif_id", "region_str", "hit_center", "hit_start", "hit_end",
                "strand", "score"]
    chunks = []
    for chunk in pd.read_csv(merged_path, sep="\t", chunksize=500_000):
        available = [c for c in use_cols if c in chunk.columns]
        delta_cols = [c for c in chunk.columns if c.startswith("delta_rep")]
        sub = chunk[chunk["motif_id"].isin(family_sigs)][available + delta_cols].copy()
        if len(sub) > 0:
            chunks.append(sub)

    if not chunks:
        raise ValueError(f"No hits found for family '{family}'")

    hits = pd.concat(chunks, ignore_index=True)
    hits["display_name"] = hits["motif_id"].map(dn_map).fillna("")

    # Compute dominant delta
    delta_b1_cols = [c for c in hits.columns if re.match(r"delta_rep\d+_band1$", c)]
    if delta_b1_cols:
        hits["delta_dominant"] = hits[delta_b1_cols].mean(axis=1)
    else:
        delta_any = [c for c in hits.columns if c.startswith("delta_rep")]
        hits["delta_dominant"] = hits[delta_any].mean(axis=1) if delta_any else np.nan

    print(f"[INFO] Loaded {len(hits):,} hits for family '{family}'", flush=True)

    # Join q-values from v3_08 or step 05 summary
    hits["q_value"] = np.nan
    if summary_path and os.path.exists(summary_path):
        summary = pd.read_csv(summary_path, sep="\t")
        for c in ["dominant_q_value", "perm_q", "lmm_q", "q_value"]:
            if c in summary.columns:
                q_map = (summary[summary["motif_id"].isin(family_sigs)]
                         [["motif_id", c]].drop_duplicates("motif_id")
                         .rename(columns={c: "q_value_summary"}))
                hits = hits.merge(q_map, on="motif_id", how="left")
                hits["q_value"] = hits["q_value_summary"]
                hits.drop(columns=["q_value_summary"], inplace=True)
                break

    hits["region_str_lower"] = hits["region_str"].str.replace(r"^Chr", "chr", regex=True)
    hits_all = hits.copy()

    mask = pd.Series(True, index=hits.index)
    if q_cutoff is not None and hits["q_value"].notna().any():
        mask &= hits["q_value"] <= q_cutoff
    if min_abs_delta > 0 and hits["delta_dominant"].notna().any():
        mask &= hits["delta_dominant"].abs() >= min_abs_delta

    hits_filtered = hits[mask].copy()
    print(f"[INFO] After filtering: {len(hits_filtered):,} hits", flush=True)
    return hits_all, hits_filtered


# ── Plotting ─────────────────────────────────────────────────────────────────

def merge_close_positions(xs: List[int], max_gap: int) -> List[List[int]]:
    if not xs:
        return []
    xs = sorted(int(x) for x in xs)
    groups = [[xs[0]]]
    for x in xs[1:]:
        if x - groups[-1][-1] <= max_gap:
            groups[-1].append(x)
        else:
            groups.append([x])
    return groups


def plot_region(
    region: str,
    leaf_tensor: np.ndarray,
    proto_tensor: np.ndarray,
    hits_filtered: pd.DataFrame,
    hits_all: pd.DataFrame,
    logos: Optional[Dict[str, np.ndarray]] = None,
    family: str = "",
    vmin: float = 0.5,
    vmax: float = 2.0,
    delta_vmax: float = 0.5,
    max_motifs_to_draw: int = 15,
    label_merge_bp: int = 10,
    max_logo_cols: int = 5,
    scale_values: Optional[np.ndarray] = None,
) -> plt.Figure:
    delta = leaf_tensor - proto_tensor
    chrom, rstart, rend = parse_region(region)
    n_pos = leaf_tensor.shape[1]
    region_len = rend - rstart

    def _compute_x(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if region_len == n_pos:
            df["x"] = df["hit_center"].astype(int) - rstart
        else:
            frac = (df["hit_center"].astype(int) - rstart) / max(1, region_len)
            df["x"] = np.clip(np.round(frac * (n_pos - 1)), 0, n_pos - 1).astype(int)
        return df

    hits_draw = _compute_x(hits_filtered)
    hits_all_draw = _compute_x(hits_all)

    mot_stats = (
        hits_draw.groupby("motif_id", as_index=False)
        .agg(
            abs_delta_max=("delta_dominant", lambda x: np.nanmax(np.abs(x))),
            q_min=("q_value", "min"),
            display_name=("display_name", "first"),
            n_hits=("hit_center", "count"),
        )
    ).sort_values(["abs_delta_max", "q_min"], ascending=[False, True])

    draw_sigs = mot_stats["motif_id"].tolist()[:max_motifs_to_draw]
    hits_draw = hits_draw[hits_draw["motif_id"].isin(draw_sigs)].copy()

    fig, axs = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    plt.subplots_adjust(hspace=0.15, bottom=0.28)

    im0 = axs[0].imshow(leaf_tensor, aspect="auto", cmap="Blues",
                         vmin=vmin, vmax=vmax, origin="lower")
    axs[0].set_title("Leaf (mean rep1+rep2)", fontsize=10)

    im1 = axs[1].imshow(proto_tensor, aspect="auto", cmap="Blues",
                         vmin=vmin, vmax=vmax, origin="lower")
    axs[1].set_title("Protoplast (mean rep1+rep2)", fontsize=10)

    norm = mcolors.TwoSlopeNorm(vcenter=0, vmin=-delta_vmax, vmax=delta_vmax)
    im2 = axs[2].imshow(delta, aspect="auto", cmap="coolwarm",
                         norm=norm, origin="lower")
    axs[2].set_title(f"Delta (leaf - proto) | {family} signatures annotated",
                     fontsize=10)

    if scale_values is not None:
        n_ticks = min(6, len(scale_values))
        tick_idx = np.round(np.linspace(0, len(scale_values) - 1, n_ticks)).astype(int)
        for ax in axs:
            ax.set_yticks(tick_idx)
            ax.set_yticklabels([f"{scale_values[i]:.0f}" for i in tick_idx], fontsize=7)
        axs[1].set_ylabel("Scale (bp)", fontsize=8)
    else:
        for ax in axs:
            ax.set_ylabel("Scale index", fontsize=8)
    axs[2].set_xlabel("Position within region (bp)", fontsize=9, labelpad=18)

    tick_positions = np.linspace(0, n_pos - 1, 5)
    tick_labels = [f"{int(rstart + t * region_len / n_pos)}" for t in tick_positions]
    for ax in axs:
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, fontsize=7)

    fig.suptitle(f"Multiscale footprints — {region}\nFamily: {family}",
                 y=0.995, fontsize=11)

    # Motif annotations
    delta_ax = axs[2]
    n_scales = leaf_tensor.shape[0]
    bands = [int(np.clip(b, 0, n_scales - 1)) for b in [45, 55, 65, 75]]

    line_kws_context = dict(color="gray", ls="--", lw=0.4, alpha=0.3)
    line_kws_sig = dict(color="black", ls="--", lw=0.6, alpha=0.7)
    label_kws = dict(
        rotation=90, ha="center", va="center", fontsize=3, color="black",
        bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.65,
                  linewidth=0),
        clip_on=True,
    )

    for x in sorted(hits_all_draw["x"].unique()):
        axs[0].axvline(x, **line_kws_context)
        axs[1].axvline(x, **line_kws_context)

    text_objects = []
    band_i = 0
    mot_stats_r = mot_stats.set_index("motif_id")

    for mid in draw_sigs:
        xs = hits_draw.loc[hits_draw["motif_id"] == mid, "x"].astype(int).tolist()
        if not xs:
            continue
        for x in xs:
            delta_ax.axvline(x, **line_kws_sig)
        clusters = merge_close_positions(xs, label_merge_bp)

        dname = ""
        abs_d = np.nan
        qmin = np.nan
        if mid in mot_stats_r.index:
            dname = str(mot_stats_r.loc[mid, "display_name"])
            abs_d = float(mot_stats_r.loc[mid, "abs_delta_max"])
            qmin = float(mot_stats_r.loc[mid, "q_min"])

        label = dname if dname and dname != "nan" else mid
        if np.isfinite(abs_d):
            label = f"{label}\n|D|={abs_d:.3f}"
        if np.isfinite(qmin):
            label = f"{label}\nq={qmin:.2g}"

        for cl in clusters:
            x_center = int(round(np.mean(cl)))
            y_use = bands[band_i % len(bands)]
            band_i += 1
            txt = delta_ax.text(x_center, y_use, label, **label_kws)
            text_objects.append(txt)

    if text_objects and _HAS_ADJUSTTEXT:
        adjust_text(
            text_objects, ax=delta_ax,
            only_move={"text": "xy"},
            arrowprops=dict(arrowstyle="-", color="gray", lw=0.4, alpha=0.5),
            expand=(1.5, 1.8),
            force_text=(0.8, 1.0),
        )

    # Colorbars
    cax0 = fig.add_axes([0.92, 0.62, 0.015, 0.22])
    fig.colorbar(im0, cax=cax0, label="FP score")
    cax2 = fig.add_axes([0.92, 0.12, 0.015, 0.22])
    fig.colorbar(im2, cax=cax2, label="Delta FP")

    # Motif logos from MEME
    if logos and _HAS_LOGOMAKER and draw_sigs:
        n = len(draw_sigs)
        ncols = min(max_logo_cols, n)
        nrows = int(np.ceil(n / ncols))
        bottom_margin = min(0.22 + nrows * 0.10, 0.55)
        plt.subplots_adjust(bottom=bottom_margin)

        left0, right0 = 0.15, 0.90
        total_w = right0 - left0
        w = total_w / ncols
        logo_width = w * 0.55
        h = 0.05
        base_y = 0.01
        row_spacing = 0.07

        for i, mid in enumerate(draw_sigs):
            if mid not in logos:
                continue
            pfm = logos[mid]
            if pfm.ndim != 2 or pfm.shape[0] != 4:
                continue
            row = i // ncols
            col = i % ncols
            x0 = left0 + col * w
            y0 = base_y + row * row_spacing
            ax_logo = fig.add_axes([x0, y0, logo_width, h])
            ax_logo.set_xticks([])
            ax_logo.set_yticks([])
            for sp in ax_logo.spines.values():
                sp.set_visible(False)

            logo_df = pd.DataFrame(pfm.T, columns=list("ACGT"))
            logo_df = logo_df.div(
                logo_df.sum(axis=1).replace(0, np.nan), axis=0
            ).fillna(0.0)
            logomaker.Logo(logo_df, ax=ax_logo, stack_order="small_on_top")
            ax_logo.set_ylim(0, 1.0)

            dname = mot_stats_r.loc[mid, "display_name"] if mid in mot_stats_r.index else mid
            ax_logo.set_title(str(dname), fontsize=4)

    return fig


# ── Select example regions ───────────────────────────────────────────────────

def select_example_regions(
    broadscale_path: str,
    nuc_summary_path: str,
    top_n: int = 10,
) -> pd.DataFrame:
    """Select top example regions per ACR class (adapted from v2 14b)."""
    bs = pd.read_csv(broadscale_path, sep="\t")
    ns = pd.read_csv(nuc_summary_path, sep="\t")

    df = bs.merge(
        ns[["region_str", "occ_delta", "profile_l2_dissimilarity"]],
        left_on="resized_str", right_on="region_str", how="inner",
    )

    leaf_thresh = df["mean_leaf"].quantile(0.5)
    df = df[df["mean_leaf"] >= leaf_thresh].copy()
    df["abs_delta"] = df["mean_delta"].abs()
    df["score"] = df["abs_delta"] * df["profile_l2_dissimilarity"]

    def _overlaps(r1, r2, min_frac=0.5):
        chr1, rest1 = r1.split(":")
        s1, e1 = map(int, rest1.split("-"))
        chr2, rest2 = r2.split(":")
        s2, e2 = map(int, rest2.split("-"))
        if chr1 != chr2:
            return False
        overlap = max(0, min(e1, e2) - max(s1, s2))
        return overlap / max(min(e1 - s1, e2 - s2), 1) >= min_frac

    all_selected = []
    for cls in ["proto_gain", "stable", "leaf_gain"]:
        candidates = df[df["acr_class"] == cls].sort_values("score", ascending=False)
        picked = []
        for _, row in candidates.iterrows():
            rs = row["resized_str"]
            if any(_overlaps(rs, prev) for prev in picked):
                continue
            picked.append(rs)
            all_selected.append(row)
            if len(picked) >= top_n:
                break

    return pd.DataFrame(all_selected)


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="v3 Step 10: Motif-annotated multiscale FP viewer"
    )
    p.add_argument("--genome-pkl", default="3_PRINT_bulk/At_genome_OBJ")
    p.add_argument("--print-dir", default="3_PRINT_per_rep")
    p.add_argument("--merged",
                   default="data/v3_merged_motif_hits_fpband_per_rep.tsv.gz")
    p.add_argument("--sig-metadata",
                   default="data/motif_signatures/signature_metadata.tsv")
    p.add_argument("--summary", default=None,
                   help="Step 05 or v3_08 summary for q-values (optional)")
    p.add_argument("--meme",
                   default="data/motif_signatures/At_Motif_SignatureDB.meme")
    p.add_argument("--acr-coord-mapping",
                   default="data/acr_native_to_resized.tsv")

    p.add_argument("--family", required=True,
                   help="TF family to annotate (e.g., WRKY, bZIP)")

    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--regions", default=None,
                   help="Comma-separated region strings (resized coords)")
    g.add_argument("--regions-bed", default=None,
                   help="BED file of regions (resized 2000bp coords)")
    g.add_argument("--select-examples", action="store_true",
                   help="Auto-select example regions from broadscale + nuc data")

    p.add_argument("--broadscale",
                   default="results/13a_nuc_genome_view/broadscale_summary.tsv.gz")
    p.add_argument("--nuc-summary",
                   default="results/09_nucleosome_shift/nucleosome_shift_summary.tsv.gz")
    p.add_argument("--max-regions", type=int, default=50)
    p.add_argument("--top-n-examples", type=int, default=10)

    p.add_argument("--q-cutoff", type=float, default=None)
    p.add_argument("--min-abs-delta", type=float, default=0.0)

    p.add_argument("--scale-min", type=float, default=4.0)
    p.add_argument("--scale-max", type=float, default=90.0)

    p.add_argument("--vmin", type=float, default=0.5)
    p.add_argument("--vmax", type=float, default=2.0)
    p.add_argument("--delta-vmax", type=float, default=0.5)
    p.add_argument("--max-motifs", type=int, default=15)
    p.add_argument("--label-merge-bp", type=int, default=10)
    p.add_argument("--max-logo-cols", type=int, default=5)

    p.add_argument("--outdir", default="results/v3_10_region_viewer")
    return p.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load genome
    print("[INFO] Loading genome...", flush=True)
    with open(args.genome_pkl, "rb") as f:
        genome = pickle.load(f)

    # Load logos from MEME
    logos = {}
    if args.meme and os.path.exists(args.meme):
        print("[INFO] Loading motif logos from MEME...", flush=True)
        logos = load_meme_logos(args.meme)
        print(f"[INFO] Loaded {len(logos)} logos", flush=True)

    # Load and filter hits
    print(f"\n{'=' * 60}", flush=True)
    print(f"Loading {args.family} signature hits", flush=True)
    print("=" * 60, flush=True)
    hits_all, hits_filtered = load_family_hits(
        merged_path=args.merged,
        sig_meta_path=args.sig_metadata,
        family=args.family,
        summary_path=args.summary,
        q_cutoff=args.q_cutoff,
        min_abs_delta=args.min_abs_delta,
    )

    # Coordinate mapping
    coord_map = pd.read_csv(args.acr_coord_mapping, sep="\t")
    coord_map["native_lower"] = coord_map["native_str"].str.lower()
    coord_map["resized_lower"] = coord_map["resized_str"].str.lower()
    native_to_resized = dict(zip(coord_map["native_lower"],
                                 coord_map["resized_lower"]))
    resized_to_native = dict(zip(coord_map["resized_lower"],
                                 coord_map["native_lower"]))

    hits_all["resized_str"] = hits_all["region_str_lower"].map(native_to_resized)
    hits_filtered["resized_str"] = hits_filtered["region_str_lower"].map(native_to_resized)

    # Determine regions
    if args.select_examples:
        print("[INFO] Auto-selecting example regions...", flush=True)
        sel = select_example_regions(
            args.broadscale, args.nuc_summary, args.top_n_examples)
        regions = sel["resized_str"].tolist()
        sel_path = outdir / "selected_regions.tsv"
        sel.to_csv(str(sel_path), sep="\t", index=False)
        print(f"[INFO] Selected {len(regions)} example regions", flush=True)
    elif args.regions:
        regions = regions_from_string(args.regions)
    else:
        regions = regions_from_bed(args.regions_bed)

    # Match regions to hits
    region_set_native = set(hits_filtered["region_str_lower"].dropna())
    region_set_resized = set(hits_filtered["resized_str"].dropna())

    regions_matched = []
    for r in regions:
        r_lower = r.lower()
        if r_lower in region_set_resized:
            native = resized_to_native.get(r_lower)
            if native and native in region_set_native:
                regions_matched.append((r_lower, native))
        elif r_lower in region_set_native:
            resized = native_to_resized.get(r_lower)
            if resized:
                regions_matched.append((resized, r_lower))

    # Prioritize by max |delta|
    if len(regions_matched) > args.max_regions:
        native_set = {native for _, native in regions_matched}
        region_delta = (
            hits_filtered[hits_filtered["region_str_lower"].isin(native_set)]
            .groupby("region_str_lower")["delta_dominant"]
            .apply(lambda x: np.nanmax(np.abs(x)))
            .sort_values(ascending=False)
        )
        top_natives = set(region_delta.index[:args.max_regions])
        regions_matched = [(res, nat) for res, nat in regions_matched
                           if nat in top_natives]

    print(f"\n[INFO] Regions to plot: {len(regions_matched)}", flush=True)
    if not regions_matched:
        print("[WARN] No regions with filtered hits found.", flush=True)
        return

    # Generate figures
    safe_family = args.family.replace(" ", "_")
    n_ok = 0

    for i, (resized_region, native_region) in enumerate(regions_matched):
        print(f"\n[{i+1}/{len(regions_matched)}] {resized_region}", flush=True)

        try:
            leaf_t, proto_t, scales = load_printers_and_average(
                args.print_dir, genome, resized_region)
        except Exception as e:
            print(f"  [WARN] Skipping: {e}", flush=True)
            continue

        scale_mask = (scales >= args.scale_min) & (scales <= args.scale_max)
        scale_idx = np.where(scale_mask)[0]
        if scale_idx.size == 0:
            print(f"  [WARN] No scales in range, skipping", flush=True)
            continue
        leaf_t = leaf_t[scale_idx, :]
        proto_t = proto_t[scale_idx, :]
        scale_values = scales[scale_idx]

        r_hits_all = hits_all[hits_all["region_str_lower"] == native_region].copy()
        r_hits_filt = hits_filtered[hits_filtered["region_str_lower"] == native_region].copy()

        if r_hits_filt.empty:
            print("  [WARN] No filtered hits, skipping", flush=True)
            plt.close("all")
            continue

        fig = plot_region(
            region=resized_region,
            leaf_tensor=leaf_t,
            proto_tensor=proto_t,
            hits_filtered=r_hits_filt,
            hits_all=r_hits_all,
            logos=logos,
            family=args.family,
            vmin=args.vmin,
            vmax=args.vmax,
            delta_vmax=args.delta_vmax,
            max_motifs_to_draw=args.max_motifs,
            label_merge_bp=args.label_merge_bp,
            max_logo_cols=args.max_logo_cols,
            scale_values=scale_values,
        )

        safe_r = resized_region.replace(":", "_").replace("-", "_")
        pdf_path = outdir / f"fp_{safe_family}_{safe_r}.pdf"
        fig.savefig(str(pdf_path), bbox_inches="tight")
        png_path = outdir / f"fp_{safe_family}_{safe_r}.png"
        fig.savefig(str(png_path), dpi=200, bbox_inches="tight")

        plt.close(fig)
        n_ok += 1
        print(f"  [OK] Saved: {pdf_path.name}", flush=True)

        del leaf_t, proto_t
        gc.collect()

    if n_ok == 0:
        print("[ERROR] No regions produced plots.", flush=True)
        return

    print(f"\n[DONE] {n_ok} regions saved to {outdir}", flush=True)


if __name__ == "__main__":
    main()
