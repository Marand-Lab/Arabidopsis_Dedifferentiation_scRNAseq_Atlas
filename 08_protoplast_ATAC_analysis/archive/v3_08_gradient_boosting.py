#!/usr/bin/env python3
"""
v3 Step 08: Gradient Boosting + SHAP on per-scale FP features.

Three tiers:
  T1 — family × scale (~34 families × ~99 scales ≈ 3,400 features)
  T2 — signature × scale (~150 sigs × ~99 scales ≈ 15,000 features)
  T3 — ElasticNet on SHAP-selected informative signatures at best scale

Includes classification (multinomial) integrated — no separate 12b needed.
Nucleosome interpretation: SHAP analysis at large scales (>80bp).

Per-feature OLS residualization on {log_width, logCPM, C(genomic_context)}.

Adapts v2 15b_gradient_boosting_model.py for v3 signatures.

Output: results/v3_08_gradient_boosting/
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from matplotlib.colors import TwoSlopeNorm
from scipy.cluster.hierarchy import leaves_list, linkage, optimal_leaf_ordering
from scipy.spatial.distance import pdist
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix,
                             f1_score, r2_score)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler

from _utils import (
    ACR_CLASS_COLORS,
    PALETTE,
    load_acr_metadata,
    nature_figure_defaults,
    nature_savefig,
    residualize_features,
    residualize_response,
)

warnings.filterwarnings("ignore", category=FutureWarning)

try:
    import logomaker
    _HAS_LOGOMAKER = True
except ImportError:
    _HAS_LOGOMAKER = False

EXCLUDE_REPS = {3}
BAND_EDGES = [20, 50]  # bp — reference lines on scale heatmaps
ACTIVE_REPS = sorted({1, 2, 3} - EXCLUDE_REPS)

BASE = os.path.dirname(os.path.abspath(__file__))
CLASS_ORDER = ["proto_gain", "stable", "leaf_gain"]
MEME_PATH = os.path.join(BASE, "data", "motif_signatures",
                          "At_Motif_SignatureDB.meme")

# Display-time renaming for ambiguous family subgroup names
FAMILY_RENAME = {
    "Group A": "bZIP Group A", "Group B": "bZIP Group B",
    "Group D": "bZIP Group D", "Group G": "bZIP Group G",
    "Group H": "bZIP Group H", "Group I": "bZIP Group I",
    "Group K": "bZIP Group K", "Group S": "bZIP Group S",
    "Type II": "MADS Type II",
}


def rename_family(name):
    """Prefix ambiguous subgroup names with their parent class."""
    return FAMILY_RENAME.get(name, name)


def load_meme_logos(meme_path: str) -> dict:
    """Parse MEME file → {sig_id: probability matrix (4, width)}."""
    import re as _re
    logos: dict = {}
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
            m = _re.search(r"w=\s*(\d+)", header)
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


def _ic_profile(pfm):
    """Information content per position. pfm shape (4, width)."""
    ic = np.zeros(pfm.shape[1])
    for j in range(pfm.shape[1]):
        col = pfm[:, j]
        col = col / col.sum()
        col = np.clip(col, 1e-10, 1.0)
        ic[j] = 2.0 + np.sum(col * np.log2(col))
    return ic


def _revcomp_pfm(pfm):
    """Reverse complement a PFM (4, width) → (4, width).
    Row order: A=0, C=1, G=2, T=3."""
    return pfm[[3, 2, 1, 0], ::-1]


def _column_pcc(ref_pfm, query_pfm, offset):
    """Mean Pearson r between overlapping frequency columns at *offset*."""
    q_start = max(0, -offset)
    q_end = min(query_pfm.shape[1], ref_pfm.shape[1] - offset)
    r_start = max(0, offset)
    overlap = q_end - q_start
    if overlap < 1:
        return -1.0, overlap
    rs = []
    for k in range(overlap):
        qcol = query_pfm[:, q_start + k]
        rcol = ref_pfm[:, r_start + k]
        r = np.corrcoef(qcol, rcol)[0, 1]
        if np.isnan(r):
            r = 0.0
        rs.append(r)
    return float(np.mean(rs)), overlap


def _align_pwms(pwm_dict):
    """Align PWMs by column-level Pearson correlation with revcomp trial.

    Uses the highest-IC motif as reference. For each other motif, tries
    all offsets in both forward and reverse-complement orientations,
    scoring each by mean Pearson r between overlapping 4-letter frequency
    columns. Picks the (orientation, offset) with the highest score.
    Pads with zero columns (render as blank in logo).

    Parameters
    ----------
    pwm_dict : dict
        {sig_id: np.array shape (4, width)} — frequency matrices

    Returns
    -------
    dict : {sig_id: np.array shape (4, aligned_width)}
    """
    if len(pwm_dict) <= 1:
        return {k: v.copy() for k, v in pwm_dict.items()}

    ids = list(pwm_dict.keys())
    ics = {sid: _ic_profile(pwm_dict[sid]) for sid in ids}

    # Reference = motif with highest total IC
    ref_id = max(ids, key=lambda s: ics[s].sum())
    ref_pfm = pwm_dict[ref_id]

    # Best (offset, orientation) for each motif vs reference
    offsets = {}
    oriented_pfms = {}  # store the chosen orientation
    for sid in ids:
        if sid == ref_id:
            offsets[sid] = 0
            oriented_pfms[sid] = pwm_dict[sid]
            continue

        pfm_fwd = pwm_dict[sid]
        pfm_rc = _revcomp_pfm(pfm_fwd)
        w_q = pfm_fwd.shape[1]
        w_r = ref_pfm.shape[1]
        min_overlap = min(3, min(w_q, w_r))

        # Collect all (pcc, overlap, offset, pfm) candidates
        candidates = []
        for trial_pfm in (pfm_fwd, pfm_rc):
            for offset in range(-w_q + 1, w_r):
                q_start = max(0, -offset)
                q_end = min(w_q, w_r - offset)
                overlap = q_end - q_start
                if overlap < min_overlap:
                    continue
                score, _ = _column_pcc(ref_pfm, trial_pfm, offset)
                candidates.append((score, overlap, offset, trial_pfm))

        # Among candidates within 0.01 of best PCC, pick longest overlap
        if candidates:
            best_pcc = max(c[0] for c in candidates)
            near_best = [c for c in candidates
                         if c[0] >= best_pcc - 0.01]
            winner = max(near_best, key=lambda c: c[1])
            offsets[sid] = winner[2]
            oriented_pfms[sid] = winner[3]
        else:
            offsets[sid] = 0
            oriented_pfms[sid] = pfm_fwd

    # Compute aligned width
    min_start = min(offsets[sid] for sid in ids)
    max_end = max(offsets[sid] + oriented_pfms[sid].shape[1] for sid in ids)
    aligned_width = max_end - min_start
    shift = -min_start

    # Build aligned PWMs (pad with zeros → blank columns in logo)
    aligned = {}
    for sid in ids:
        pfm = oriented_pfms[sid]
        w = pfm.shape[1]
        start = offsets[sid] + shift
        result = np.zeros((4, aligned_width))
        result[:, start:start + w] = pfm
        aligned[sid] = result
    return aligned


# ── Data loading ─────────────────────────────────────────────────────────────

def load_signature_metadata(path):
    return pd.read_csv(path, sep="\t")


def load_data(args):
    """Load all required data: NPZs, ACR metadata, signature metadata."""
    print("[1] Loading data...", flush=True)

    # Family-level delta
    fam_npz = np.load(os.path.join(BASE, args.perscale_dir,
                                    "delta_acr_family_scale.npz"),
                       allow_pickle=True)
    family_delta = fam_npz["delta"]
    family_ids = fam_npz["family_ids"]
    acr_ids_fam = fam_npz["acr_ids"]
    scales = fam_npz["scales"]
    print(f"  Family delta: {family_delta.shape}", flush=True)

    # Signature-level delta
    sig_npz_path = os.path.join(BASE, args.perscale_dir,
                                 "delta_acr_signature_scale.npz")
    sig_delta = sig_ids = acr_ids_sig = None
    if not args.skip_tier2 and os.path.exists(sig_npz_path):
        sig_npz = np.load(sig_npz_path, allow_pickle=True)
        sig_delta = sig_npz["delta"]
        sig_ids = sig_npz["signature_ids"]
        acr_ids_sig = sig_npz["acr_ids"]
        print(f"  Signature delta: {sig_delta.shape}", flush=True)

    # ACR metadata
    acr_meta = load_acr_metadata(os.path.join(BASE, args.acr_metadata))

    # Coordinate mapping (native → resized)
    coord_map = pd.read_csv(os.path.join(BASE, args.acr_coord_mapping), sep="\t")
    coord_map["native_str"] = coord_map["native_str"].str.lower()
    coord_map["resized_str"] = coord_map["resized_str"].str.lower()
    acr_meta["region_str_lower"] = acr_meta["region_str"].str.lower()
    acr_meta = acr_meta.merge(
        coord_map[["native_str", "resized_str"]],
        left_on="region_str_lower", right_on="native_str", how="left")
    acr_meta = acr_meta.set_index("resized_str")

    # Signature metadata
    sig_meta = load_signature_metadata(
        os.path.join(BASE, args.sig_metadata))
    print(f"  Signatures: {len(sig_meta)}, "
          f"Families: {sig_meta['primary_family'].nunique()}", flush=True)

    return {
        "family_delta": family_delta, "family_ids": family_ids,
        "acr_ids": acr_ids_fam, "scales": scales,
        "sig_delta": sig_delta, "sig_ids": sig_ids,
        "acr_ids_sig": acr_ids_sig,
        "acr_meta": acr_meta, "sig_meta": sig_meta,
    }


# ── Feature construction ─────────────────────────────────────────────────────

def build_family_scale_features(family_delta, family_ids, acr_ids, scales):
    """Flatten (n_acrs, n_fam, n_scales) → DataFrame columns: {family}_s{idx}."""
    n_acrs, n_fam, n_scales = family_delta.shape
    cols = [f"{family_ids[fi]}_s{si}" for fi in range(n_fam) for si in range(n_scales)]
    flat = family_delta.reshape(n_acrs, -1)
    df = pd.DataFrame(np.nan_to_num(flat, nan=0.0), index=acr_ids, columns=cols)
    return df


def build_sig_scale_features(sig_delta, sig_ids, acr_ids, scales):
    """Flatten (n_acrs, n_sigs, n_scales) → DataFrame columns: {sig_id}_s{idx}."""
    n_acrs, n_sigs, n_scales = sig_delta.shape
    cols = [f"{sig_ids[si]}_s{sci}" for si in range(n_sigs) for sci in range(n_scales)]
    flat = sig_delta.reshape(n_acrs, -1).astype(np.float32)
    df = pd.DataFrame(np.nan_to_num(flat, nan=0.0), index=acr_ids, columns=cols)
    return df


def build_confounders(acr_meta):
    """Build confounder DataFrame from ACR metadata."""
    cols = []
    if "acr_width" in acr_meta.columns:
        acr_meta = acr_meta.copy()
        acr_meta["log_width"] = np.log1p(acr_meta["acr_width"])
        cols.append("log_width")
    if "logCPM" in acr_meta.columns:
        cols.append("logCPM")
    if "genomic_context" in acr_meta.columns:
        gc_map = {"Promoter": 0, "Gene body": 1, "Intergenic": 2}
        acr_meta = acr_meta.copy()
        acr_meta["gc_code"] = acr_meta["genomic_context"].map(gc_map).fillna(2).astype(int)
        cols.append("gc_code")
    return acr_meta[cols].copy()


# ── Per-feature OLS residualization (now in _utils.py) ────────────────────────
# residualize_features() and residualize_response() imported from _utils


def assemble_xy(tf_features, acr_meta, acr_subset=None):
    """Align features + response, residualize, return X, y, acr_class."""
    y_df = acr_meta[["edgeR_logFC", "acr_class"]].copy()
    common = tf_features.index.intersection(y_df.index)
    if acr_subset:
        mask = y_df.loc[common, "acr_class"].isin(acr_subset)
        common = common[mask]
    notna = y_df.loc[common, "edgeR_logFC"].notna()
    common = common[notna]

    X_resid = residualize_features(tf_features.loc[common], acr_meta)
    y_resid, r2_conf = residualize_response(
        y_df.loc[common, "edgeR_logFC"], acr_meta)

    # Re-align
    common = X_resid.index.intersection(y_resid.index)
    return (X_resid.loc[common], y_resid.loc[common],
            y_df.loc[common, "acr_class"], r2_conf)


# ── Model fitting ────────────────────────────────────────────────────────────

def fit_gb_regressor(X, y, acr_class, seed=42, test_size=0.2):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(sss.split(X, acr_class))

    model = HistGradientBoostingRegressor(
        max_depth=5, learning_rate=0.05, max_iter=500,
        min_samples_leaf=20, validation_fraction=0.1,
        n_iter_no_change=10, random_state=seed)
    model.fit(X.values[train_idx], y.values[train_idx])

    pred = model.predict(X.values[test_idx])
    r2 = r2_score(y.values[test_idx], pred)

    return {"model": model, "r2": r2, "train_idx": train_idx,
            "test_idx": test_idx, "X_test": X.values[test_idx],
            "y_test": y.values[test_idx], "feature_names": list(X.columns),
            "acr_class_test": acr_class.values[test_idx]}


def fit_gb_classifier(X, acr_class, seed=42, test_size=0.2):
    le = LabelEncoder()
    y_enc = le.fit_transform(acr_class)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(sss.split(X, y_enc))

    # Balanced weights
    from collections import Counter
    counts = Counter(y_enc[train_idx])
    n_classes = len(counts)
    weights = {c: len(train_idx) / (n_classes * cnt) for c, cnt in counts.items()}
    sw_train = np.array([weights[c] for c in y_enc[train_idx]])

    model = HistGradientBoostingClassifier(
        max_depth=5, learning_rate=0.05, max_iter=500,
        min_samples_leaf=20, validation_fraction=0.1,
        n_iter_no_change=10, random_state=seed)
    model.fit(X.values[train_idx], y_enc[train_idx], sample_weight=sw_train)

    pred = model.predict(X.values[test_idx])
    ba = balanced_accuracy_score(y_enc[test_idx], pred)
    f1 = f1_score(y_enc[test_idx], pred, average="macro")
    cm = confusion_matrix(y_enc[test_idx], pred, normalize="true")

    return {"model": model, "ba": ba, "f1": f1, "cm": cm,
            "le": le, "test_idx": test_idx, "X_test": X.values[test_idx],
            "feature_names": list(X.columns),
            "acr_class_test": acr_class.values[test_idx]}


# ── SHAP ─────────────────────────────────────────────────────────────────────

def compute_shap(model, X_test, feature_names):
    explainer = shap.TreeExplainer(model)
    X_df = pd.DataFrame(X_test, columns=feature_names)
    return explainer(X_df)


def build_shap_table(shap_values, feature_names, id_field="family"):
    """Build mean |SHAP| per (id, scale) from flat feature names like {id}_s{idx}."""
    abs_mean = np.abs(shap_values.values).mean(axis=0)
    rows = []
    for fi, fname in enumerate(feature_names):
        parts = fname.rsplit("_s", 1)
        if len(parts) != 2:
            continue
        fid = parts[0]
        try:
            sidx = int(parts[1])
        except ValueError:
            continue
        rows.append({id_field: fid, "scale_idx": sidx,
                     "mean_abs_shap": abs_mean[fi]})
    return pd.DataFrame(rows)


def select_informative_signatures(shap_table, sig_meta, threshold=0.80):
    """Select informative signatures per family via cumulative SHAP cutoff."""
    # Total |SHAP| per signature
    sig_total = shap_table.groupby("signature_id")["mean_abs_shap"].sum()
    sig_total = sig_total.reset_index()
    sig_total.columns = ["signature_id", "total_shap"]

    # Add family and best scale
    best_scale = (shap_table.sort_values("mean_abs_shap", ascending=False)
                  .groupby("signature_id").first()
                  .reset_index()[["signature_id", "scale_idx"]])
    sig_total = sig_total.merge(best_scale, on="signature_id", how="left")

    fam_map = dict(zip(sig_meta["signature_id"], sig_meta["primary_family"]))
    sig_total["family"] = sig_total["signature_id"].map(fam_map)

    results = []
    for fam, grp in sig_total.groupby("family"):
        grp = grp.sort_values("total_shap", ascending=False).copy()
        fam_total = grp["total_shap"].sum()
        if fam_total <= 0:
            continue
        grp["cumfrac"] = grp["total_shap"].cumsum() / fam_total
        grp["rank"] = range(1, len(grp) + 1)
        n_needed = (grp["cumfrac"] < threshold).sum() + 1
        grp["is_informative"] = grp["rank"] <= n_needed
        results.append(grp)

    if not results:
        return pd.DataFrame()
    return pd.concat(results, ignore_index=True)


# ── Per-family permutation importance ─────────────────────────────────────────

def family_permutation_importance(model, X, feature_names, test_idx, y_test,
                                   family_feature_map, n_perm=100, seed=42,
                                   scoring="r2"):
    rng = np.random.RandomState(seed)
    X_test = X[test_idx]
    score_fn = r2_score if scoring == "r2" else balanced_accuracy_score
    base_score = score_fn(y_test, model.predict(X_test))

    rows = []
    for fam, col_indices in family_feature_map.items():
        if not col_indices:
            continue
        drops = []
        for _ in range(n_perm):
            X_perm = X_test.copy()
            for ci in col_indices:
                X_perm[:, ci] = rng.permutation(X_perm[:, ci])
            perm_score = score_fn(y_test, model.predict(X_perm))
            drops.append(base_score - perm_score)
        rows.append({"family": fam, "n_features": len(col_indices),
                     "perm_importance_mean": np.mean(drops),
                     "perm_importance_std": np.std(drops)})
    return pd.DataFrame(rows)


def build_family_feature_map(feature_names, family_ids=None, sig_meta=None,
                              tier="T1"):
    """Map family → list of column indices."""
    fam_map = {}
    if tier == "T1":
        for fi, fname in enumerate(feature_names):
            parts = fname.rsplit("_s", 1)
            fam = parts[0]
            fam_map.setdefault(fam, []).append(fi)
    else:
        # T2: signature → family
        sig_to_fam = dict(zip(sig_meta["signature_id"],
                               sig_meta["primary_family"]))
        for fi, fname in enumerate(feature_names):
            parts = fname.rsplit("_s", 1)
            sig_id = parts[0]
            fam = sig_to_fam.get(sig_id, "Unknown")
            fam_map.setdefault(fam, []).append(fi)
    return fam_map


# ── Tier runners ─────────────────────────────────────────────────────────────

def run_tier1(D, tf_features, acr_subset, pass_label, pass_outdir, args,
              all_summary):
    """Run Tier 1: family × scale."""
    print(f"\n  [T1] family × scale: {tf_features.shape[1]} features", flush=True)

    # Check sentinel
    sentinel_files = ["regression_results.tsv", "scale_family_shap.tsv",
                      "family_permutation_importance.tsv"]
    if not args.force and all(os.path.exists(os.path.join(pass_outdir, f))
                              for f in sentinel_files):
        print("  [T1] Sentinel found — loading cached results", flush=True)
        shap_table = pd.read_csv(os.path.join(pass_outdir, "scale_family_shap.tsv"),
                                  sep="\t")
        return shap_table

    X, y, acr_class, r2_conf = assemble_xy(tf_features, D["acr_meta"], acr_subset)
    print(f"  ACRs: {len(X)}, r2_conf: {r2_conf:.4f}", flush=True)

    # Regression
    result = fit_gb_regressor(X, y, acr_class, seed=args.seed,
                               test_size=args.test_size)
    print(f"  T1 R²: {result['r2']:.4f}", flush=True)

    # SHAP
    shap_values = compute_shap(result["model"], result["X_test"],
                                result["feature_names"])
    shap_table = build_shap_table(shap_values, result["feature_names"],
                                   id_field="family")
    shap_table.to_csv(os.path.join(pass_outdir, "scale_family_shap.tsv"),
                       sep="\t", index=False)

    # Permutation importance
    fam_feat_map = build_family_feature_map(result["feature_names"], tier="T1")
    perm_df = family_permutation_importance(
        result["model"], X.values, result["feature_names"],
        result["test_idx"], result["y_test"],
        fam_feat_map, n_perm=args.n_permutations, seed=args.seed)
    perm_df.to_csv(os.path.join(pass_outdir,
                                 "family_permutation_importance.tsv"),
                    sep="\t", index=False)

    # Classification
    clf_result = fit_gb_classifier(X, acr_class, seed=args.seed,
                                    test_size=args.test_size)
    print(f"  T1 BA: {clf_result['ba']:.4f}, F1: {clf_result['f1']:.4f}",
          flush=True)

    # Save results
    pd.DataFrame([{
        "pass": pass_label, "tier": "T1", "r2": result["r2"],
        "r2_conf": r2_conf, "n_acrs": len(X),
        "n_features": X.shape[1], "ba": clf_result["ba"],
        "f1": clf_result["f1"],
    }]).to_csv(os.path.join(pass_outdir, "regression_results.tsv"),
               sep="\t", index=False)

    # Save confusion matrix for plotting
    np.savez_compressed(
        os.path.join(pass_outdir, "confusion_matrix_tier1.npz"),
        cm=clf_result["cm"],
        classes=np.array(clf_result["le"].classes_))

    # Save raw T1 SHAP for beeswarm (includes X_test for color scale)
    np.savez_compressed(
        os.path.join(pass_outdir, "raw_shap_tier1.npz"),
        shap_values=shap_values.values.astype(np.float32),
        X_test=result["X_test"].astype(np.float32),
        feature_names=np.array(result["feature_names"]),
        acr_class_test=result["acr_class_test"],
        acr_ids_test=np.array(X.index[result["test_idx"]], dtype=object))

    # Classification SHAP → family × scale table
    print("  Computing classification SHAP (T1)...", flush=True)
    clf_shap = compute_shap(clf_result["model"], clf_result["X_test"],
                             clf_result["feature_names"])
    clf_abs = np.abs(clf_shap.values)
    if clf_abs.ndim == 3:
        # Multiclass: average |SHAP| across classes
        clf_abs = clf_abs.mean(axis=2)
    clf_abs_mean = clf_abs.mean(axis=0)
    clf_rows = []
    for fi, fname in enumerate(clf_result["feature_names"]):
        parts = fname.rsplit("_s", 1)
        if len(parts) != 2:
            continue
        try:
            sidx = int(parts[1])
        except ValueError:
            continue
        clf_rows.append({"family": parts[0], "scale_idx": sidx,
                         "mean_abs_shap": clf_abs_mean[fi]})
    clf_shap_table = pd.DataFrame(clf_rows)
    clf_shap_table.to_csv(
        os.path.join(pass_outdir, "scale_family_shap_clf.tsv"),
        sep="\t", index=False)
    print(f"  Saved scale_family_shap_clf.tsv ({len(clf_shap_table)} rows)",
          flush=True)

    # Save raw T1 classification SHAP NPZ (for by-class heatmap)
    np.savez_compressed(
        os.path.join(pass_outdir, "raw_shap_tier1_clf.npz"),
        shap_values=clf_shap.values.astype(np.float32),
        feature_names=np.array(clf_result["feature_names"]),
        acr_class_test=clf_result["acr_class_test"],
        acr_ids_test=np.array(X.index[clf_result["test_idx"]], dtype=object),
        class_names=np.array(clf_result["le"].classes_))

    # Classifier permutation importance (BA drop)
    clf_perm_df = family_permutation_importance(
        clf_result["model"], X.values, result["feature_names"],
        clf_result["test_idx"],
        clf_result["le"].transform(
            clf_result["acr_class_test"]),
        fam_feat_map, n_perm=args.n_permutations, seed=args.seed,
        scoring="ba")
    clf_perm_df.to_csv(
        os.path.join(pass_outdir, "family_permutation_importance_clf.tsv"),
        sep="\t", index=False)

    all_summary.append({"pass": pass_label, "tier": "T1",
                        "r2": result["r2"], "ba": clf_result["ba"]})
    return shap_table


def run_tier2(D, acr_subset, pass_label, pass_outdir, args,
              all_summary, family_shap_t1):
    """Run Tier 2: signature × scale."""
    if D["sig_delta"] is None:
        print("  [T2] No signature NPZ — skipping", flush=True)
        return None

    # Check sentinel
    sentinel_files = ["scale_signature_shap.tsv",
                      "family_permutation_importance_tier2.tsv",
                      "scale_signature_shap_clf.tsv"]
    if not args.force and all(os.path.exists(os.path.join(pass_outdir, f))
                              for f in sentinel_files):
        print("  [T2] Sentinel found — loading cached", flush=True)
        shap_table = pd.read_csv(
            os.path.join(pass_outdir, "scale_signature_shap.tsv"), sep="\t")
        info_df = select_informative_signatures(
            shap_table, D["sig_meta"],
            threshold=args.cumulative_threshold)
        info_df.to_csv(os.path.join(pass_outdir, "informative_signatures.tsv"),
                        sep="\t", index=False)
        return info_df

    tf_features = build_sig_scale_features(
        D["sig_delta"], D["sig_ids"], D["acr_ids_sig"], D["scales"])
    print(f"\n  [T2] signature × scale: {tf_features.shape[1]} features",
          flush=True)

    X, y, acr_class, r2_conf = assemble_xy(tf_features, D["acr_meta"], acr_subset)
    print(f"  ACRs: {len(X)}, r2_conf: {r2_conf:.4f}", flush=True)

    result = fit_gb_regressor(X, y, acr_class, seed=args.seed,
                               test_size=args.test_size)
    print(f"  T2 R²: {result['r2']:.4f}", flush=True)

    # Save R² early (crash protection before SHAP)
    pd.DataFrame([{
        "pass": pass_label, "tier": "T2", "r2": result["r2"],
        "r2_conf": r2_conf, "n_acrs": len(X), "n_features": X.shape[1],
    }]).to_csv(os.path.join(pass_outdir, "regression_results_tier2.tsv"),
               sep="\t", index=False)

    # SHAP
    t0 = time.time()
    shap_values = compute_shap(result["model"], result["X_test"],
                                result["feature_names"])
    dt = time.time() - t0
    print(f"  T2 SHAP computed [{dt:.1f}s]", flush=True)

    shap_table = build_shap_table(shap_values, result["feature_names"],
                                   id_field="signature_id")
    shap_table.to_csv(os.path.join(pass_outdir, "scale_signature_shap.tsv"),
                       sep="\t", index=False)

    # Save raw SHAP for later use (includes X_test for color scale)
    np.savez_compressed(
        os.path.join(pass_outdir, "raw_shap_tier2.npz"),
        shap_values=shap_values.values.astype(np.float32),
        X_test=result["X_test"].astype(np.float32),
        feature_names=np.array(result["feature_names"]),
        acr_class_test=result["acr_class_test"],
        acr_ids_test=np.array(X.index[result["test_idx"]], dtype=object))

    # Permutation importance
    fam_feat_map = build_family_feature_map(
        result["feature_names"], sig_meta=D["sig_meta"], tier="T2")
    perm_df = family_permutation_importance(
        result["model"], X.values, result["feature_names"],
        result["test_idx"], result["y_test"],
        fam_feat_map, n_perm=args.n_permutations, seed=args.seed)
    perm_df.to_csv(os.path.join(pass_outdir,
                                 "family_permutation_importance_tier2.tsv"),
                    sep="\t", index=False)

    # Classification
    clf_result = fit_gb_classifier(X, acr_class, seed=args.seed,
                                    test_size=args.test_size)
    print(f"  T2 BA: {clf_result['ba']:.4f}", flush=True)
    pd.DataFrame([{
        "pass": pass_label, "tier": "T2",
        "ba": clf_result["ba"], "f1": clf_result["f1"],
    }]).to_csv(os.path.join(pass_outdir, "classification_results_tier2.tsv"),
               sep="\t", index=False)

    # Save confusion matrix for plotting
    np.savez_compressed(
        os.path.join(pass_outdir, "confusion_matrix_tier2.npz"),
        cm=clf_result["cm"],
        classes=np.array(clf_result["le"].classes_))

    # T2 Classification SHAP
    print("  Computing classification SHAP (T2)...", flush=True)
    t0_clf = time.time()
    clf_shap = compute_shap(clf_result["model"], clf_result["X_test"],
                             clf_result["feature_names"])
    dt_clf = time.time() - t0_clf
    print(f"  T2 classification SHAP computed [{dt_clf:.1f}s]", flush=True)

    clf_shap_arr = clf_shap.values  # (n_test, n_feat) or (n_test, n_feat, n_classes)
    class_names = list(clf_result["le"].classes_)

    # Class-averaged table for heatmap (same format as regression)
    clf_abs = np.abs(clf_shap_arr)
    if clf_abs.ndim == 3:
        clf_abs_avg = clf_abs.mean(axis=2)  # average across classes
    else:
        clf_abs_avg = clf_abs
    clf_abs_mean = clf_abs_avg.mean(axis=0)

    clf_rows = []
    for fi, fname in enumerate(clf_result["feature_names"]):
        parts = fname.rsplit("_s", 1)
        if len(parts) != 2:
            continue
        try:
            sidx = int(parts[1])
        except ValueError:
            continue
        clf_rows.append({"signature_id": parts[0], "scale_idx": sidx,
                         "mean_abs_shap": clf_abs_mean[fi]})
    clf_shap_table = pd.DataFrame(clf_rows)
    clf_shap_table.to_csv(
        os.path.join(pass_outdir, "scale_signature_shap_clf.tsv"),
        sep="\t", index=False)

    # Save raw classification SHAP tensor (per-class SHAP for downstream)
    save_dict = {
        "feature_names": np.array(clf_result["feature_names"]),
        "acr_class_test": clf_result["acr_class_test"],
        "acr_ids_test": np.array(X.index[clf_result["test_idx"]], dtype=object),
        "class_names": np.array(class_names),
    }
    if clf_shap_arr.ndim == 3:
        # (n_test, n_features, n_classes) — keep per-class resolution
        save_dict["shap_values"] = clf_shap_arr.astype(np.float32)
    else:
        save_dict["shap_values"] = clf_shap_arr.astype(np.float32)
    np.savez_compressed(
        os.path.join(pass_outdir, "raw_shap_tier2_clf.npz"), **save_dict)
    print(f"  Saved raw_shap_tier2_clf.npz + scale_signature_shap_clf.tsv",
          flush=True)

    all_summary.append({"pass": pass_label, "tier": "T2",
                        "r2": result["r2"], "ba": clf_result["ba"]})

    # Informative signatures
    info_df = select_informative_signatures(
        shap_table, D["sig_meta"], threshold=args.cumulative_threshold)
    info_df.to_csv(os.path.join(pass_outdir, "informative_signatures.tsv"),
                    sep="\t", index=False)
    n_info = info_df["is_informative"].sum() if len(info_df) > 0 else 0
    print(f"  Informative signatures ({args.cumulative_threshold:.0%} cutoff): "
          f"{n_info}/{len(info_df)}", flush=True)

    # Nucleosome-scale SHAP interpretation
    _nucleosome_shap_analysis(shap_table, D["scales"], D["sig_meta"],
                               pass_outdir)

    return info_df


def _nucleosome_shap_analysis(shap_table, scales, sig_meta, outdir):
    """Analyze SHAP at large scales (>80bp) for nucleosome interpretation."""
    nuc_mask = shap_table["scale_idx"].apply(
        lambda si: si < len(scales) and scales[si] > 80)
    nuc_shap = shap_table[nuc_mask].copy()
    if nuc_shap.empty:
        return

    # Top signatures at nucleosome scales
    nuc_top = (nuc_shap.groupby("signature_id")["mean_abs_shap"]
               .sum().sort_values(ascending=False).head(20))
    dn = dict(zip(sig_meta["signature_id"], sig_meta["display_name"]))

    nuc_df = pd.DataFrame({
        "signature_id": nuc_top.index,
        "display_name": [dn.get(s, s) for s in nuc_top.index],
        "total_nuc_shap": nuc_top.values,
    })
    nuc_df.to_csv(os.path.join(outdir, "nucleosome_scale_shap.tsv"),
                   sep="\t", index=False)
    print(f"  Nucleosome-scale top signatures saved", flush=True)


def run_tier3(D, info_df, acr_subset, pass_label, pass_outdir, args,
              all_summary):
    """Run Tier 3: ElasticNet on informative signatures at best scale."""
    if info_df is None or info_df.empty:
        print("  [T3] No informative signatures — skipping", flush=True)
        return

    informative = info_df[info_df["is_informative"]].copy()
    if informative.empty:
        return

    # Build feature matrix: one column per informative sig at best scale
    sig_delta = D["sig_delta"]
    sig_ids = D["sig_ids"]
    acr_ids = D["acr_ids_sig"]
    scales = D["scales"]

    sig_idx_map = {s: i for i, s in enumerate(sig_ids)}
    cols = []
    col_names = []
    for _, row in informative.iterrows():
        si = sig_idx_map.get(row["signature_id"])
        sci = int(row["scale_idx"]) if pd.notna(row["scale_idx"]) else 0
        if si is not None and sci < sig_delta.shape[2]:
            cols.append(sig_delta[:, si, sci])
            col_names.append(f"{row['signature_id']}_s{sci}")

    if not cols:
        return

    X_diag = pd.DataFrame(
        np.column_stack(cols), index=acr_ids, columns=col_names)
    X_diag = X_diag.fillna(0)

    # Assemble y
    y_df = D["acr_meta"][["edgeR_logFC", "acr_class"]].copy()
    common = X_diag.index.intersection(y_df.index)
    if acr_subset:
        mask = y_df.loc[common, "acr_class"].isin(acr_subset)
        common = common[mask]
    notna = y_df.loc[common, "edgeR_logFC"].notna()
    common = common[notna]

    if len(common) < 50:
        return

    X_en = X_diag.loc[common]
    y_en = y_df.loc[common, "edgeR_logFC"]

    # Residualize
    X_resid = residualize_features(X_en, D["acr_meta"])
    y_resid, r2_conf = residualize_response(y_en, D["acr_meta"])
    common = X_resid.index.intersection(y_resid.index)
    X_resid = X_resid.loc[common]
    y_resid = y_resid.loc[common]

    # Fit ElasticNet
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_resid)

    en = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 1.0],
                       n_alphas=100, cv=5, max_iter=10000, random_state=args.seed)
    en.fit(X_scaled, y_resid.values)
    r2_en = en.score(X_scaled, y_resid.values)
    n_nz = np.sum(en.coef_ != 0)

    print(f"  [T3] EN R²={r2_en:.4f}, nonzero={n_nz}/{X_resid.shape[1]}",
          flush=True)

    pd.DataFrame([{
        "pass": pass_label, "tier": "T3", "r2": r2_en,
        "r2_conf": r2_conf, "n_features": X_resid.shape[1],
        "n_nonzero": n_nz, "n_acrs": len(common),
    }]).to_csv(os.path.join(pass_outdir, "diagnostic_en_results.tsv"),
               sep="\t", index=False)

    # Save feature matrix for v3_09
    np.savez_compressed(
        os.path.join(pass_outdir, "informative_features_tier3.npz"),
        features=X_resid.values.astype(np.float32),
        acr_ids=np.array(common, dtype=object),
        feature_names=np.array(list(X_resid.columns), dtype=object))

    all_summary.append({"pass": pass_label, "tier": "T3", "r2": r2_en})


# ── Post-run figures (read from existing TSVs/NPZs — safe to re-run) ─────────

def plot_permutation_importance(pass_outdir, pass_label):
    """Bar chart of family permutation importance (T2, fallback T1)."""
    nature_figure_defaults()
    t2_path = os.path.join(pass_outdir, "family_permutation_importance_tier2.tsv")
    t1_path = os.path.join(pass_outdir, "family_permutation_importance.tsv")

    if os.path.exists(t2_path):
        perm_df = pd.read_csv(t2_path, sep="\t")
        tier_label = "T2"
    elif os.path.exists(t1_path):
        perm_df = pd.read_csv(t1_path, sep="\t")
        tier_label = "T1"
    else:
        return

    perm_df = perm_df.sort_values("perm_importance_mean", ascending=True)
    n = len(perm_df)
    colors = ["#e34a33" if v > 0 else "#bdbdbd"
              for v in perm_df["perm_importance_mean"]]

    has_std = "perm_importance_std" in perm_df.columns
    xerr = perm_df["perm_importance_std"].values if has_std else None

    fig, ax = plt.subplots(figsize=(7, max(4, n * 0.3)))
    ax.barh(range(n), perm_df["perm_importance_mean"],
            xerr=xerr, color=colors, edgecolor="white", linewidth=0.5,
            capsize=2, ecolor="black", error_kw={"lw": 0.6})
    ax.set_yticks(range(n))
    ax.set_yticklabels(
        [f"{rename_family(r['family'])} (n={r['n_features']})"
         for _, r in perm_df.iterrows()],
        fontsize=7)
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_xlabel("Permutation importance (R² drop ± SD)")
    ax.set_title(f"Family Permutation Importance ({tier_label}, pass={pass_label})")
    plt.tight_layout()
    nature_savefig(fig, f"fig_perm_importance_{pass_label}", pass_outdir)
    plt.close(fig)
    print(f"  [Fig] permutation importance ({pass_label})", flush=True)


def plot_shap_scale_heatmap(pass_outdir, pass_label, scales,
                             tsv_name="scale_family_shap.tsv",
                             fig_suffix="", title_extra=""):
    """Heatmap: family × scale, color = mean |SHAP|.

    Parameters
    ----------
    tsv_name : str
        TSV file to read (default: regression SHAP).
    fig_suffix : str
        Appended to figure name before pass_label (e.g. "_clf").
    title_extra : str
        Appended to title (e.g. " (classification)").
    """
    nature_figure_defaults()
    shap_path = os.path.join(pass_outdir, tsv_name)
    if not os.path.exists(shap_path):
        return

    shap_table = pd.read_csv(shap_path, sep="\t")
    shap_table["family"] = shap_table["family"].map(rename_family)
    pivot = shap_table.pivot_table(
        index="family", columns="scale_idx",
        values="mean_abs_shap", aggfunc="mean")

    # Label columns with actual bp values
    col_labels = [f"{scales[i]:.0f}" if i < len(scales) else str(i)
                  for i in pivot.columns]
    pivot.columns = col_labels

    # Sort families by total SHAP (most important on top)
    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]

    n_xticks = max(1, len(pivot.columns) // 10)
    # Clip at 98th percentile to reveal structure beyond dominant families
    vals = pivot.values[pivot.values > 0]
    vmax = float(np.percentile(vals, 98)) if len(vals) else 1.0
    fig, ax = plt.subplots(
        figsize=(max(10, len(pivot.columns) * 0.12), max(5, len(pivot) * 0.35)))
    sns.heatmap(pivot, cmap="YlOrRd", ax=ax, linewidths=0, vmax=vmax,
                cbar_kws={"label": f"Mean |SHAP| (clipped @{vmax:.4f})"})
    xticks = ax.get_xticks()[::n_xticks]
    ax.set_xticks(xticks)
    ax.tick_params(axis="x", rotation=45, labelsize=7)
    ax.tick_params(axis="y", labelsize=7)
    ax.set_xlabel("Scale (bp)")
    ax.set_ylabel("TF Family")
    ax.set_title(f"Mean |SHAP| by Family \u00d7 Scale "
                 f"(T1, pass={pass_label}{title_extra})")
    plt.tight_layout()
    nature_savefig(fig,
                   f"fig_shap_scale_heatmap{fig_suffix}_{pass_label}",
                   pass_outdir)
    plt.close(fig)
    label = "SHAP \u00d7 scale heatmap" + (title_extra or "")
    print(f"  [Fig] {label} ({pass_label})", flush=True)


def _load_raw_shap_and_parse(npz_path, id_field="family"):
    """Load raw SHAP NPZ, parse feature names → (id, scale_idx).

    Returns (shap_vals, parsed_features, acr_class, acr_ids) where
    parsed_features is a list of (id_value, scale_idx) tuples aligned
    with the feature axis.  For regression NPZ, shap_vals is 2D; for
    classification NPZ it may be 3D (n_test, n_feat, n_classes).
    """
    if not os.path.exists(npz_path):
        return None
    npz = np.load(npz_path, allow_pickle=True)
    shap_vals = npz["shap_values"]
    feat_names = list(npz["feature_names"])
    acr_class = npz["acr_class_test"]

    parsed = []
    for fname in feat_names:
        parts = fname.rsplit("_s", 1)
        if len(parts) == 2:
            try:
                parsed.append((parts[0], int(parts[1])))
            except ValueError:
                parsed.append((None, None))
        else:
            parsed.append((None, None))

    return {
        "shap_values": shap_vals,
        "parsed_features": parsed,
        "acr_class": acr_class,
        "feature_names": feat_names,
    }


def plot_shap_scale_heatmap_by_class(pass_outdir, pass_label, scales,
                                      sig_meta, npz_name="raw_shap_tier1.npz",
                                      fig_suffix="", title_extra=""):
    """Family × scale heatmap stratified by ACR class (3 panels).

    Reads raw SHAP NPZ, subsets test ACRs by class, computes **signed**
    mean SHAP per family × scale within each class.  Diverging colormap
    (RdBu_r) centered at 0: red = pushes toward proto-enriched logFC,
    blue = pushes toward leaf-enriched logFC.
    """
    nature_figure_defaults()
    data = _load_raw_shap_and_parse(
        os.path.join(pass_outdir, npz_name), id_field="family")
    if data is None:
        return

    shap_vals = data["shap_values"]
    parsed = data["parsed_features"]
    acr_class = data["acr_class"]

    # For 3D (classification), average signed SHAP across classes
    if shap_vals.ndim == 3:
        shap_signed = shap_vals.mean(axis=2)
    else:
        shap_signed = shap_vals

    # Map features to families (T1 features are already family-named)
    fam_map = dict(zip(sig_meta["signature_id"], sig_meta["primary_family"]))
    fam_scale_map = {}  # (family, scale_idx) → list of column indices
    for fi, (fid, sidx) in enumerate(parsed):
        if fid is None:
            continue
        fam = fam_map.get(fid, fid)  # T1: fid is already family; T2: map sig→fam
        fam = rename_family(fam)
        fam_scale_map.setdefault((fam, sidx), []).append(fi)

    class_order = ["proto_gain", "stable", "leaf_gain"]
    class_labels = {"proto_gain": "Proto-gain", "stable": "Stable",
                    "leaf_gain": "Leaf-gain"}

    # Build pivot per class
    all_fams = sorted(set(k[0] for k in fam_scale_map))
    all_scales_idx = sorted(set(k[1] for k in fam_scale_map))
    pivots = {}

    for cls in class_order:
        cls_mask = acr_class == cls
        if cls_mask.sum() == 0:
            pivots[cls] = pd.DataFrame(
                np.nan, index=all_fams,
                columns=[f"{scales[si]:.0f}" if si < len(scales) else str(si)
                         for si in all_scales_idx])
            continue

        mat = np.zeros((len(all_fams), len(all_scales_idx)))
        for fi_fam, fam in enumerate(all_fams):
            for fi_sc, sidx in enumerate(all_scales_idx):
                cols = fam_scale_map.get((fam, sidx), [])
                if cols:
                    mat[fi_fam, fi_sc] = shap_signed[cls_mask][:, cols].mean()

        col_labels = [f"{scales[si]:.0f}" if si < len(scales) else str(si)
                      for si in all_scales_idx]
        pivots[cls] = pd.DataFrame(mat, index=all_fams, columns=col_labels)

    # Sort families by total |SHAP| across all classes (avoid sign cancellation)
    total = sum(p.abs().sum(axis=1) for p in pivots.values())
    fam_order = total.sort_values(ascending=False).index

    # Shared symmetric vmax (98th percentile of |values| across all classes)
    all_vals = np.concatenate([p.values.ravel() for p in pivots.values()])
    all_vals = all_vals[np.isfinite(all_vals)]
    vmax = float(np.percentile(np.abs(all_vals), 98)) if len(all_vals) else 1.0

    n_xticks = max(1, len(all_scales_idx) // 10)
    fig, axes = plt.subplots(
        1, 3, figsize=(max(10, len(all_scales_idx) * 0.12) * 1.1,
                       max(5, len(all_fams) * 0.35)),
        sharey=True)

    for ci, cls in enumerate(class_order):
        ax = axes[ci]
        pivot = pivots[cls].reindex(fam_order)
        sns.heatmap(pivot, cmap="RdBu_r", ax=ax, linewidths=0,
                    vmin=-vmax, vmax=vmax, center=0,
                    cbar=(ci == 2),
                    cbar_kws={"label": f"Mean SHAP (clipped \u00b1{vmax:.4f})",
                              "shrink": 0.6} if ci == 2 else {})
        xticks = ax.get_xticks()[::n_xticks]
        ax.set_xticks(xticks)
        ax.tick_params(axis="x", rotation=45, labelsize=6)
        ax.tick_params(axis="y", labelsize=6 if ci == 0 else 0)
        ax.set_xlabel("Scale (bp)", fontsize=7)
        ax.set_title(f"{class_labels[cls]} "
                     f"(n={int((acr_class == cls).sum()):,})",
                     fontsize=9, fontweight="bold")
        if ci > 0:
            ax.set_ylabel("")

    fig.suptitle(f"Mean signed SHAP by Family \u00d7 Scale \u00d7 ACR class"
                 f"{title_extra} ({pass_label})", fontsize=10)
    plt.tight_layout()
    nature_savefig(fig,
                   f"fig_shap_scale_heatmap_by_class{fig_suffix}_{pass_label}",
                   pass_outdir)
    plt.close(fig)
    print(f"  [Fig] SHAP \u00d7 scale heatmap by ACR class{title_extra} "
          f"({pass_label})", flush=True)


def plot_signature_importance_per_family_by_class(
        pass_outdir, pass_label, sig_meta,
        npz_name="raw_shap_tier2.npz",
        fig_suffix="", title_extra="",
        top_n_families=15, logos=None):
    """Per-family signature bar chart stratified by ACR class.

    Layout per family (one PDF each):
      Col 0: Overall total |SHAP| (all ACRs) — blue gradient bars
      Col 1-3: Per-class total |SHAP| (proto_gain / stable / leaf_gain)
      Col 4 (if logos available): IC-aligned motif logos

    Unmasks signatures whose importance is diluted by the large stable-
    ACR majority in the overall average.
    """
    nature_figure_defaults()
    data = _load_raw_shap_and_parse(
        os.path.join(pass_outdir, npz_name), id_field="signature_id")
    if data is None:
        return

    shap_vals = data["shap_values"]
    parsed = data["parsed_features"]
    acr_class = data["acr_class"]

    # For 3D (classification), average |SHAP| across classes
    if shap_vals.ndim == 3:
        shap_abs = np.abs(shap_vals).mean(axis=2)
    else:
        shap_abs = np.abs(shap_vals)

    fam_map = dict(zip(sig_meta["signature_id"], sig_meta["primary_family"]))
    dn_map = dict(zip(sig_meta["signature_id"], sig_meta["display_name"]))

    # Map features → signature → column indices (sum across scales)
    sig_col_map = {}
    for fi, (sid, sidx) in enumerate(parsed):
        if sid is None:
            continue
        sig_col_map.setdefault(sid, []).append(fi)

    class_order = ["proto_gain", "stable", "leaf_gain"]
    class_labels = {"proto_gain": "Proto-gain", "stable": "Stable",
                    "leaf_gain": "Leaf-gain"}

    # Compute per-sig total |SHAP| per class + overall
    rows = []
    for sig_id, col_idxs in sig_col_map.items():
        fam = rename_family(fam_map.get(sig_id, "Unknown"))
        display = dn_map.get(sig_id, sig_id)
        # Overall (all ACRs)
        total_all = shap_abs[:, col_idxs].sum(axis=1).mean()
        rows.append({"signature_id": sig_id, "family": fam,
                     "display_name": display, "acr_class": "overall",
                     "total_shap": total_all})
        # Per class
        for cls in class_order:
            cls_mask = acr_class == cls
            if cls_mask.sum() == 0:
                continue
            total = shap_abs[cls_mask][:, col_idxs].sum(axis=1).mean()
            rows.append({"signature_id": sig_id, "family": fam,
                         "display_name": display, "acr_class": cls,
                         "total_shap": total})
    df = pd.DataFrame(rows)
    if df.empty:
        return

    # Rank families by overall total |SHAP|
    fam_rank = (df[df["acr_class"] == "overall"]
                .groupby("family")["total_shap"]
                .sum().sort_values(ascending=False))
    top_fams = fam_rank.head(top_n_families).index.tolist()

    has_logos = logos and _HAS_LOGOMAKER
    all_cols = ["overall"] + class_order
    all_col_labels = {"overall": "Overall", **class_labels}
    n_bar_cols = len(all_cols)  # 4: overall + 3 classes
    n_cols = n_bar_cols + (1 if has_logos else 0)

    for fam in top_fams:
        fam_df = df[df["family"] == fam].copy()
        # Pivot: display_name × (overall + class_order)
        pivot = fam_df.pivot(index="display_name", columns="acr_class",
                              values="total_shap").reindex(columns=all_cols)

        # Sort by overall total
        overall = pivot["overall"].fillna(0).sort_values(ascending=True)
        pivot = pivot.reindex(overall.index)
        n_sigs = len(pivot)

        # Collect sig_ids in display order (for logos)
        sig_id_map = dict(zip(
            fam_df[fam_df["acr_class"] == "overall"]["display_name"],
            fam_df[fam_df["acr_class"] == "overall"]["signature_id"]))
        sig_ids_ordered = [sig_id_map.get(dn) for dn in pivot.index]

        width_ratios = [2] + [1.5] * 3 + ([1.2] if has_logos else [])
        fig, axes = plt.subplots(
            1, n_cols,
            figsize=(4 * n_bar_cols + (3 if has_logos else 0),
                     max(2, n_sigs * 0.4)),
            gridspec_kw={"width_ratios": width_ratios},
            sharey=True)

        for ci, col_key in enumerate(all_cols):
            ax = axes[ci]
            vals = pivot[col_key].fillna(0).values

            if col_key == "overall":
                # Blue gradient like the original per-family figure
                max_val = vals.max() if vals.max() > 0 else 1
                colors = plt.cm.Blues(0.3 + 0.7 * vals / max_val)
                ax.barh(range(n_sigs), vals, color=colors,
                        edgecolor="black", linewidth=0.3)
            else:
                color = ACR_CLASS_COLORS.get(col_key, "#888888")
                ax.barh(range(n_sigs), vals, color=color,
                        edgecolor="white", linewidth=0.3)

            ax.set_yticks(range(n_sigs))
            if ci == 0:
                ax.set_yticklabels(pivot.index, fontsize=6)
            ax.set_xlabel("Total |SHAP|", fontsize=7)

            if col_key == "overall":
                n_total = len(acr_class)
                ax.set_title(f"Overall (n={n_total:,})",
                             fontsize=8, fontweight="bold")
            else:
                n_cls = int((acr_class == col_key).sum())
                ax.set_title(f"{all_col_labels[col_key]} (n={n_cls:,})",
                             fontsize=8, fontweight="bold")

        # Hide logo placeholder axes (logos placed after tight_layout)
        if has_logos:
            axes[-1].set_axis_off()

        fig.suptitle(f"{fam} — signature importance by ACR class"
                     f"{title_extra} ({pass_label})", fontsize=10)
        plt.tight_layout()

        # --- Place IC-aligned logos after tight_layout ---
        if has_logos:
            ax_bar = axes[0]  # reference bar axis for y-positions
            ax_logo = axes[-1]

            fam_pwms = {}
            for sid in sig_ids_ordered:
                if sid is None:
                    continue
                pfm = logos.get(sid)
                if pfm is not None and pfm.ndim == 2 and pfm.shape[0] == 4:
                    fam_pwms[sid] = pfm
            if fam_pwms:
                aligned = _align_pwms(fam_pwms)
                logo_bbox = ax_logo.get_position()
                trans = ax_bar.transData + fig.transFigure.inverted()

                for j, sid in enumerate(sig_ids_ordered):
                    if sid is None:
                        continue
                    apfm = aligned.get(sid)
                    if apfm is None:
                        continue
                    bar_center_fig = trans.transform((0, j))
                    bar_top_fig = trans.transform((0, j + 0.4))
                    bar_bot_fig = trans.transform((0, j - 0.4))
                    h_fig = bar_top_fig[1] - bar_bot_fig[1]

                    x0 = logo_bbox.x0 + 0.005
                    y0 = bar_center_fig[1] - h_fig / 2
                    w = logo_bbox.width * 0.92
                    h = h_fig

                    ax_ins = fig.add_axes([x0, y0, w, h])
                    logo_df = pd.DataFrame(apfm.T, columns=list("ACGT"))
                    logo_df = logo_df.div(
                        logo_df.sum(axis=1).replace(0, np.nan), axis=0)
                    logo_df = logo_df.fillna(0)
                    logomaker.Logo(logo_df, ax=ax_ins,
                                   stack_order="small_on_top")
                    ax_ins.set_ylim(0, 1.0)
                    ax_ins.set_xticks([])
                    ax_ins.set_yticks([])
                    for sp in ax_ins.spines.values():
                        sp.set_visible(False)

        safe_fam = fam.replace("/", "_").replace(" ", "_")
        panels_dir = os.path.join(pass_outdir, "sig_shap_panels")
        os.makedirs(panels_dir, exist_ok=True)
        nature_savefig(
            fig,
            f"fig_sig_importance_by_class{fig_suffix}_{safe_fam}_{pass_label}",
            panels_dir)
        plt.close(fig)

    print(f"  [Fig] signature importance by ACR class{title_extra}: "
          f"{len(top_fams)} families ({pass_label})", flush=True)


def plot_family_acr_class_heatmap(D, pass_outdir, pass_label, acr_subset):
    """Heatmap: family × ACR class, color = mean z-delta (family-scale mean delta)."""
    nature_figure_defaults()
    family_delta = D["family_delta"]   # (n_acrs, n_fam, n_scales)
    family_ids = [rename_family(f) for f in D["family_ids"]]
    acr_ids = list(D["acr_ids"])
    acr_meta = D["acr_meta"]

    # Mean across scales → (n_acrs, n_fam)
    acr_mean = np.nanmean(family_delta, axis=2)
    df_wide = pd.DataFrame(acr_mean, index=acr_ids, columns=family_ids)

    # Z-score per family across ACRs
    mu = df_wide.mean(axis=0)
    sd = df_wide.std(axis=0).replace(0, 1)
    df_z = (df_wide - mu) / sd

    # Attach acr_class
    df_z["acr_class"] = acr_meta["acr_class"].reindex(df_z.index)
    df_z = df_z.dropna(subset=["acr_class"])
    if acr_subset:
        df_z = df_z[df_z["acr_class"].isin(acr_subset)]

    matrix = df_z.groupby("acr_class")[family_ids].mean()
    class_order = [c for c in CLASS_ORDER if c in matrix.index]
    matrix = matrix.loc[class_order]
    if matrix.empty:
        return

    # Cluster families by Ward
    fam_data = matrix.T.values
    if fam_data.shape[0] > 2:
        dist = pdist(np.nan_to_num(fam_data), metric="euclidean")
        Z = linkage(dist, method="ward")
        order = leaves_list(Z)
        matrix = matrix.iloc[:, order]

    fig, ax = plt.subplots(
        figsize=(max(8, len(family_ids) * 0.4), max(3, len(class_order) * 0.4)))
    sns.heatmap(matrix.T, cmap="RdBu_r", center=0, vmin=-1.5, vmax=1.5,
                ax=ax, linewidths=0.3, linecolor="white",
                cbar_kws={"label": "Mean z-delta"},
                annot=True, fmt=".2f", annot_kws={"fontsize": 6})
    ax.set_xlabel("ACR class")
    ax.set_ylabel("TF Family")
    ax.tick_params(axis="x", rotation=0, labelsize=8)
    ax.tick_params(axis="y", labelsize=7)
    ax.set_title(f"Family × ACR Class: Mean z-delta (pass={pass_label})")
    plt.tight_layout()
    nature_savefig(fig, f"fig_family_acr_class_{pass_label}", pass_outdir)
    plt.close(fig)
    print(f"  [Fig] family × ACR class heatmap ({pass_label})", flush=True)


def plot_shap_asymmetry(pass_outdir, pass_label, sig_meta,
                        npz_name="raw_shap_tier1.npz"):
    """Systematic comparison of signed SHAP at proto-gain vs leaf-gain ACRs.

    For each family (T1) or signature (T2), computes mean signed SHAP
    summed across scales, separately for proto-gain and leaf-gain ACRs.
    Positive SHAP = pushes logFC toward proto-enriched; negative = toward
    leaf-enriched.

    Produces:
      - TSV: per-family signed SHAP at each ACR class + asymmetry metrics
      - Figure A: scatter of mean signed SHAP at proto-gain vs leaf-gain
        per family — diagonal = symmetric; off-diagonal = asymmetric
      - Figure B: bar chart of |SHAP_proto| − |SHAP_leaf| per family,
        ranked — positive = stronger effect at gaining ACRs
    """
    nature_figure_defaults()
    data = _load_raw_shap_and_parse(
        os.path.join(pass_outdir, npz_name), id_field="family")
    if data is None:
        return

    shap_vals = data["shap_values"]
    parsed = data["parsed_features"]
    acr_class = data["acr_class"]

    # For 3D (classification), use mean signed SHAP across target classes
    if shap_vals.ndim == 3:
        shap_signed = shap_vals.mean(axis=2)
    else:
        shap_signed = shap_vals

    # Aggregate feature columns by family (sum SHAP across scales)
    fam_map = dict(zip(sig_meta["signature_id"], sig_meta["primary_family"]))
    fam_col_map = {}
    for fi, (fid, sidx) in enumerate(parsed):
        if fid is None:
            continue
        fam = fam_map.get(fid, fid)
        fam = rename_family(fam)
        fam_col_map.setdefault(fam, []).append(fi)

    # Compute per-family mean signed SHAP per ACR class
    class_order = ["proto_gain", "stable", "leaf_gain"]
    class_labels = {"proto_gain": "Proto-gain", "stable": "Stable",
                    "leaf_gain": "Leaf-gain"}
    rows = []
    for fam, col_idxs in sorted(fam_col_map.items()):
        row = {"family": fam}
        for cls in class_order:
            cls_mask = acr_class == cls
            if cls_mask.sum() == 0:
                row[f"shap_{cls}"] = np.nan
                row[f"n_{cls}"] = 0
                continue
            # Sum SHAP across all scales for this family, then mean across ACRs
            row[f"shap_{cls}"] = float(
                shap_signed[cls_mask][:, col_idxs].sum(axis=1).mean())
            row[f"n_{cls}"] = int(cls_mask.sum())
        # Asymmetry: |SHAP at proto-gain| − |SHAP at leaf-gain|
        sp = row.get("shap_proto_gain", 0) or 0
        sl = row.get("shap_leaf_gain", 0) or 0
        row["abs_shap_proto_gain"] = abs(sp)
        row["abs_shap_leaf_gain"] = abs(sl)
        row["asymmetry"] = abs(sp) - abs(sl)
        row["ratio"] = (abs(sp) / abs(sl)) if abs(sl) > 1e-8 else np.nan
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return

    # Save TSV
    tsv_path = os.path.join(pass_outdir, "shap_asymmetry_by_class.tsv")
    df.to_csv(tsv_path, sep="\t", index=False, float_format="%.6f")
    print(f"  [Table] shap_asymmetry_by_class.tsv: {len(df)} families",
          flush=True)

    # ── Figure A: Scatter proto-gain vs leaf-gain ──────────────────────────
    fig, ax = plt.subplots(figsize=(6, 6))
    x = df["shap_proto_gain"].values
    y = df["shap_leaf_gain"].values

    ax.scatter(x, y, s=40, alpha=0.7, c="#555555",
               edgecolors="black", linewidth=0.4, zorder=3)

    # Diagonal reference and zero lines
    lim = max(np.abs(x).max(), np.abs(y).max()) * 1.2
    ax.plot([-lim, lim], [lim, -lim], "k--", lw=0.5, alpha=0.4,
            label="Perfect symmetry")
    ax.axhline(0, color="gray", lw=0.3, alpha=0.5)
    ax.axvline(0, color="gray", lw=0.3, alpha=0.5)

    # Label families
    try:
        from adjustText import adjust_text
        texts = []
        for i, fam in enumerate(df["family"]):
            texts.append(ax.text(x[i], y[i], fam, fontsize=5.5, alpha=0.85))
        adjust_text(texts, ax=ax, force_text=(0.3, 0.3),
                    force_points=(0.2, 0.2), arrowprops=dict(
                        arrowstyle="-", color="gray", lw=0.3, alpha=0.4))
    except ImportError:
        for i, fam in enumerate(df["family"]):
            ax.annotate(fam, (x[i], y[i]), fontsize=4.5, alpha=0.7)

    ax.set_xlabel("Mean signed SHAP at Proto-gain ACRs", fontsize=8)
    ax.set_ylabel("Mean signed SHAP at Leaf-gain ACRs", fontsize=8)
    ax.set_title(f"Family SHAP asymmetry: proto-gain vs leaf-gain "
                 f"({pass_label})", fontsize=9)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.legend(fontsize=7, loc="upper right")
    plt.tight_layout()
    nature_savefig(fig, f"fig_shap_asymmetry_scatter_{pass_label}",
                   pass_outdir)
    plt.close(fig)

    # ── Figure B: Asymmetry bar chart ──────────────────────────────────────
    df_sorted = df.sort_values("asymmetry", ascending=True)
    n_fam = len(df_sorted)

    fig, ax = plt.subplots(figsize=(7, max(4, n_fam * 0.3)))
    colors = ["#D64045" if v > 0 else "#3A7D44"
              for v in df_sorted["asymmetry"]]
    ax.barh(range(n_fam), df_sorted["asymmetry"].values,
            color=colors, edgecolor="black", linewidth=0.3)
    ax.set_yticks(range(n_fam))
    ax.set_yticklabels(df_sorted["family"].values, fontsize=6)
    ax.axvline(0, color="black", lw=0.5)
    ax.set_xlabel("|SHAP proto-gain| − |SHAP leaf-gain|", fontsize=8)
    ax.set_title(f"SHAP magnitude asymmetry by family ({pass_label})\n"
                 f"Red = stronger at gaining ACRs, "
                 f"Green = stronger at losing ACRs", fontsize=9)
    plt.tight_layout()
    nature_savefig(fig, f"fig_shap_asymmetry_bar_{pass_label}",
                   pass_outdir)
    plt.close(fig)

    # ── Figure C: Grouped bar — signed SHAP per class ─────────────────────
    # Top 20 families by max |SHAP| across classes
    df["max_abs"] = df[["abs_shap_proto_gain", "abs_shap_leaf_gain"]].max(
        axis=1)
    df_top = df.nlargest(20, "max_abs").sort_values(
        "shap_proto_gain", ascending=True)
    n_top = len(df_top)

    fig, ax = plt.subplots(figsize=(8, max(4, n_top * 0.35)))
    bar_h = 0.25
    y_pos = np.arange(n_top)

    for ci, cls in enumerate(class_order):
        col = f"shap_{cls}"
        offset = (ci - 1) * bar_h
        color = ACR_CLASS_COLORS.get(cls, "#888888")
        ax.barh(y_pos + offset, df_top[col].values, height=bar_h,
                color=color, edgecolor="white", linewidth=0.3,
                label=class_labels[cls])

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_top["family"].values, fontsize=6)
    ax.axvline(0, color="black", lw=0.5)
    ax.set_xlabel("Mean signed SHAP (sum across scales)", fontsize=8)
    ax.set_title(f"Per-family signed SHAP by ACR class ({pass_label})",
                 fontsize=9)
    ax.legend(fontsize=7, loc="lower right")
    plt.tight_layout()
    nature_savefig(fig, f"fig_shap_signed_by_class_{pass_label}",
                   pass_outdir)
    plt.close(fig)

    print(f"  [Fig] SHAP asymmetry analysis ({pass_label})", flush=True)


def plot_shap_beeswarm(pass_outdir, pass_label, scales=None, sig_meta=None,
                       top_n_scales=2):
    """SHAP beeswarm: loads raw_shap_tier2.npz (fallback tier1).

    Renames feature labels from ``sig_XXX_sYY`` to
    ``display_name [family] @ ZZbp`` for readability.
    Filters to top_n_scales per signature to avoid DOF domination.
    """
    import re as _re
    from collections import defaultdict
    nature_figure_defaults()

    # Build lookup maps for display names / families
    dn_map, fam_map = {}, {}
    if sig_meta is not None:
        dn_map = dict(zip(sig_meta["signature_id"], sig_meta["display_name"]))
        fam_map = dict(zip(sig_meta["signature_id"], sig_meta["primary_family"]))

    def _pretty(feat):
        """sig_042_s15 → 'WRKY_WRKY33 [WRKY] @ 16 bp'.
        Also handles T1 names: 'Group S_s15' → 'bZIP Group S @ 16 bp'."""
        # T2: sig_XXX_sYY
        m = _re.match(r"(sig_\d+)_s(\d+)", feat)
        if m:
            sid, si = m.group(1), int(m.group(2))
            dname = dn_map.get(sid, sid)
            fam = rename_family(fam_map.get(sid, ""))
            bp = (f"{scales[si]:.0f}"
                  if scales is not None and si < len(scales) else str(si))
            return f"{dname} [{fam}] @ {bp} bp"
        # T1: {family}_s{idx}
        m = _re.match(r"(.+)_s(\d+)$", feat)
        if m:
            fam_raw, si = m.group(1), int(m.group(2))
            fam = rename_family(fam_raw)
            bp = (f"{scales[si]:.0f}"
                  if scales is not None and si < len(scales) else str(si))
            return f"{fam} @ {bp} bp"
        return feat

    def _filter_top_scales(shap_vals, feat_names, n):
        """Keep only top-n scales per signature/family by mean |SHAP|."""
        mean_abs = np.mean(np.abs(shap_vals), axis=0)
        # Group feature indices by signature/family prefix
        groups = defaultdict(list)
        for i, name in enumerate(feat_names):
            m = _re.match(r"(sig_\d+)_s\d+", name)
            if m:
                groups[m.group(1)].append(i)
            else:
                m2 = _re.match(r"(.+)_s\d+$", name)
                groups[m2.group(1) if m2 else name].append(i)
        # Select top-n indices per group
        keep = []
        for indices in groups.values():
            ranked = sorted(indices, key=lambda i: mean_abs[i], reverse=True)
            keep.extend(ranked[:n])
        keep.sort()
        return np.array(keep)

    for tier in ["tier2", "tier1"]:
        npz_path = os.path.join(pass_outdir, f"raw_shap_{tier}.npz")
        if os.path.exists(npz_path):
            d = np.load(npz_path, allow_pickle=True)
            shap_vals = d["shap_values"].astype(np.float64)
            feat_names = list(d["feature_names"])
            X_data = (d["X_test"].astype(np.float64)
                      if "X_test" in d else None)

            # Filter to top-n scales per signature
            if top_n_scales and top_n_scales > 0:
                mask = _filter_top_scales(shap_vals, feat_names, top_n_scales)
                shap_vals = shap_vals[:, mask]
                feat_names = [feat_names[i] for i in mask]
                if X_data is not None:
                    X_data = X_data[:, mask]

            pretty_names = [_pretty(f) for f in feat_names]
            exp = shap.Explanation(
                values=shap_vals,
                data=X_data,
                feature_names=pretty_names)
            fig, ax = plt.subplots(figsize=(7, 9))
            shap.plots.beeswarm(exp, max_display=30, show=False)
            nature_savefig(plt.gcf(), f"fig_shap_beeswarm_{pass_label}",
                           pass_outdir)
            plt.close("all")
            n_kept = len(feat_names)
            print(f"  [Fig] SHAP beeswarm ({pass_label}, {tier}, "
                  f"{n_kept} features after top-{top_n_scales} filter)",
                  flush=True)
            return
    print(f"  [Fig] SHAP beeswarm skipped — no raw SHAP NPZ", flush=True)


def plot_clf_results(pass_outdir, pass_label):
    """Confusion matrix + BA/F1 bar; loads confusion_matrix_tier2/1.npz."""
    nature_figure_defaults()
    cm_data = None
    for tier in ["tier2", "tier1"]:
        p = os.path.join(pass_outdir, f"confusion_matrix_{tier}.npz")
        if os.path.exists(p):
            cm_data = np.load(p, allow_pickle=True)
            tier_used = tier
            break
    if cm_data is None:
        return

    # Load BA/F1 from TSV
    ba, f1 = None, None
    for tsv in ["classification_results_tier2.tsv", "regression_results.tsv"]:
        p = os.path.join(pass_outdir, tsv)
        if os.path.exists(p):
            df = pd.read_csv(p, sep="\t")
            if "ba" in df.columns:
                ba = float(df["ba"].iloc[0])
            if "f1" in df.columns:
                f1 = float(df["f1"].iloc[0])
            break
    if ba is None:
        return

    cm = cm_data["cm"]
    classes = list(cm_data["classes"])
    n_classes = len(classes)

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))

    # Confusion matrix
    ax = axes[0]
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels(classes, fontsize=7, rotation=30)
    ax.set_yticklabels(classes, fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix ({tier_used.upper()})")
    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                    fontsize=8,
                    color="white" if cm[i, j] > 0.5 else "black")
    plt.colorbar(im, ax=ax, shrink=0.8)

    # BA / F1 bars
    ax = axes[1]
    chance = 1.0 / n_classes
    bars = ["BA (chance)", "BA (model)", "F1 (macro)"]
    vals = [chance, ba, f1]
    colors = ["#999999", PALETTE.get("proto_gain", "#2166ac"), "#4daf4a"]
    ax.barh(bars, vals, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Score")
    ax.set_title(f"Classification metrics (pass={pass_label})")

    plt.tight_layout()
    nature_savefig(fig, f"fig_clf_results_{pass_label}", pass_outdir)
    plt.close(fig)
    print(f"  [Fig] clf results ({pass_label})", flush=True)


def plot_family_perm_importance_clf(pass_outdir, pass_label):
    """BA-drop permutation importance; loads family_permutation_importance_clf.tsv."""
    nature_figure_defaults()
    p = os.path.join(pass_outdir, "family_permutation_importance_clf.tsv")
    if not os.path.exists(p):
        return

    perm_df = pd.read_csv(p, sep="\t")
    perm_df = perm_df.sort_values("perm_importance_mean", ascending=True)
    n = len(perm_df)

    colors = []
    for _, row in perm_df.iterrows():
        if (row["perm_importance_mean"] > 0
                and row["perm_importance_mean"] > 2 * row["perm_importance_std"]):
            colors.append(PALETTE.get("leaf_gain", "#2ca02c"))
        else:
            colors.append("#cccccc")

    has_std = "perm_importance_std" in perm_df.columns
    xerr = perm_df["perm_importance_std"].values if has_std else None

    fig, ax = plt.subplots(figsize=(7, max(4, n * 0.3)))
    ax.barh(range(n), perm_df["perm_importance_mean"],
            xerr=xerr, color=colors, edgecolor="none", linewidth=0,
            capsize=2, ecolor="black", error_kw={"lw": 0.6})
    ax.set_yticks(range(n))
    ax.set_yticklabels(
        [f"{rename_family(r['family'])} (n={r['n_features']})"
         for _, r in perm_df.iterrows()],
        fontsize=7)
    ax.axvline(0, color="k", lw=0.5, ls="--")
    ax.set_xlabel("Permutation importance (BA drop ± SD)")
    ax.set_title(f"Family clf permutation importance (pass={pass_label})")
    plt.tight_layout()
    nature_savefig(fig, f"fig_perm_importance_clf_{pass_label}", pass_outdir)
    plt.close(fig)
    print(f"  [Fig] clf perm importance ({pass_label})", flush=True)


def plot_scale_signature_heatmap(pass_outdir, pass_label, scales, sig_meta):
    """T2 heatmap: top signatures × scale with family sidebar; Ward-clustered."""
    nature_figure_defaults()
    shap_path = os.path.join(pass_outdir, "scale_signature_shap.tsv")
    if not os.path.exists(shap_path):
        return

    shap_table = pd.read_csv(shap_path, sep="\t")

    dn_map = dict(zip(sig_meta["signature_id"], sig_meta["display_name"]))
    fam_map = dict(zip(sig_meta["signature_id"], sig_meta["primary_family"]))

    # Top 50 signatures by total |SHAP|
    sig_total = (shap_table.groupby("signature_id")["mean_abs_shap"]
                 .sum().sort_values(ascending=False))
    top_sigs = sig_total.head(50).index.tolist()

    n_scales = len(scales)
    n_sigs = len(top_sigs)
    mat = np.zeros((n_sigs, n_scales))
    sig_to_row = {s: i for i, s in enumerate(top_sigs)}
    sub = shap_table[shap_table["signature_id"].isin(top_sigs)]
    for _, row in sub.iterrows():
        ri = sig_to_row.get(row["signature_id"])
        if ri is not None and int(row["scale_idx"]) < n_scales:
            mat[ri, int(row["scale_idx"])] = row["mean_abs_shap"]

    labels = [f"{s} {dn_map.get(s, s)} [{rename_family(fam_map.get(s, '?'))}]"
              for s in top_sigs]
    families = [rename_family(fam_map.get(s, "?")) for s in top_sigs]

    # Ward clustering on signatures
    if n_sigs > 2:
        dist = pdist(mat, metric="euclidean")
        Z = linkage(dist, method="ward")
        try:
            Z = optimal_leaf_ordering(Z, dist)
        except Exception:
            pass
        order = leaves_list(Z)
    else:
        order = np.arange(n_sigs)

    mat_ord = mat[order]
    labels_ord = [labels[i] for i in order]
    fams_ord = [families[i] for i in order]

    # Family color sidebar
    unique_fams = sorted(set(fams_ord))
    cmap_fam = plt.cm.tab20(np.linspace(0, 1, max(len(unique_fams), 1)))
    fam_color = {f: cmap_fam[i] for i, f in enumerate(unique_fams)}

    nonzero = mat[mat > 0]
    vmax = float(np.percentile(nonzero, 98)) if len(nonzero) else 1.0

    fig, (ax_side, ax_main) = plt.subplots(
        1, 2, figsize=(10, max(4, n_sigs * 0.25)),
        gridspec_kw={"width_ratios": [0.03, 1], "wspace": 0.02})

    for i, fam in enumerate(fams_ord):
        ax_side.barh(i, 1, color=fam_color[fam], edgecolor="none")
    ax_side.set_ylim(-0.5, n_sigs - 0.5)
    ax_side.invert_yaxis()
    ax_side.set_xticks([])
    ax_side.set_yticks([])

    im = ax_main.imshow(mat_ord, aspect="auto", interpolation="nearest",
                        cmap="viridis", vmin=0, vmax=vmax)
    tick_pos = []
    for bp in [5, 10, 20, 50, 100]:
        idx = int(np.argmin(np.abs(scales - bp)))
        if idx not in tick_pos:
            tick_pos.append(idx)
    ax_main.set_xticks(tick_pos)
    ax_main.set_xticklabels([f"{scales[t]:.0f}" for t in tick_pos])
    ax_main.set_xlabel("Scale (bp)")
    ax_main.set_yticks(range(n_sigs))
    ax_main.set_yticklabels(labels_ord, fontsize=5)
    for edge in BAND_EDGES:
        idx = int(np.argmin(np.abs(scales - edge)))
        ax_main.axvline(idx, color="red", ls="--", lw=0.5, alpha=0.7)
    plt.colorbar(im, ax=ax_main,
                 label=f"Mean |SHAP| (clipped @{vmax:.4f})", shrink=0.6)
    ax_main.set_title(f"Signature × Scale SHAP (T2, pass={pass_label})")
    plt.tight_layout()
    nature_savefig(fig, f"fig_scale_signature_heatmap_{pass_label}", pass_outdir)
    plt.close(fig)
    print(f"  [Fig] signature × scale heatmap ({pass_label})", flush=True)


def plot_signature_importance_per_family(pass_outdir, pass_label, sig_meta,
                                         top_n_families=20, logos=None,
                                         tsv_name="scale_signature_shap.tsv",
                                         fig_suffix="", title_extra=""):
    """Per-family bar charts with IC-aligned motif-logo companion panel.

    Left column: horizontal bar chart of total |SHAP| per signature.
    Right column: sequence logos aligned by information content cross-
    correlation (conserved cores line up across logos within each family).
    Logos are placed after tight_layout() for accurate vertical alignment.

    Parameters
    ----------
    tsv_name : str
        TSV file to read (default: regression SHAP).
    fig_suffix : str
        Appended to figure name before pass_label (e.g. "_clf").
    title_extra : str
        Appended to title (e.g. " (classification)").
    """
    nature_figure_defaults()
    shap_path = os.path.join(pass_outdir, tsv_name)
    if not os.path.exists(shap_path):
        return

    shap_table = pd.read_csv(shap_path, sep="\t")
    dn_map = dict(zip(sig_meta["signature_id"], sig_meta["display_name"]))
    fam_map = dict(zip(sig_meta["signature_id"], sig_meta["primary_family"]))

    df = shap_table.copy()
    df["family"] = df["signature_id"].map(fam_map).map(rename_family)
    df = df.dropna(subset=["family"])

    sig_total = (df.groupby(["family", "signature_id"])["mean_abs_shap"]
                 .sum().reset_index()
                 .rename(columns={"mean_abs_shap": "total_shap"}))
    fam_rank = (sig_total.groupby("family")["total_shap"]
                .sum().sort_values(ascending=False))
    top_families = fam_rank.head(top_n_families).index.tolist()
    if not top_families:
        return

    has_logos = logos and _HAS_LOGOMAKER
    n_cols = 2 if has_logos else 1
    # Compute row heights: proportional to number of sigs per family
    fam_n_sigs = []
    fam_sig_ids = {}  # store sorted sig_ids per family for logo placement
    for fam in top_families:
        fam_sigs = (sig_total[sig_total["family"] == fam]
                    .sort_values("total_shap", ascending=True))
        sids = fam_sigs["signature_id"].tolist()
        fam_sig_ids[fam] = sids
        fam_n_sigs.append(max(len(sids), 1))
    row_height = 0.45  # inches per signature bar
    heights = [n * row_height for n in fam_n_sigs]

    fig, axes = plt.subplots(
        len(top_families), n_cols,
        figsize=(12 if has_logos else 8, sum(heights) + 1.5),
        gridspec_kw={"width_ratios": [3, 2] if has_logos else [1],
                     "height_ratios": heights,
                     "hspace": 0.6},
        squeeze=False)

    for row_i, fam in enumerate(top_families):
        ax_bar = axes[row_i, 0]
        sig_ids = fam_sig_ids[fam]
        fam_sigs = (sig_total[sig_total["family"] == fam]
                    .sort_values("total_shap", ascending=True))
        names = [f"{s} {dn_map.get(s, s)}" for s in sig_ids]
        vals = fam_sigs["total_shap"].values
        max_val = vals.max() if vals.max() > 0 else 1
        colors = plt.cm.Blues(0.3 + 0.7 * vals / max_val)
        ax_bar.barh(range(len(names)), vals, color=colors,
                    edgecolor="black", linewidth=0.3)
        ax_bar.set_yticks(range(len(names)))
        ax_bar.set_yticklabels(names, fontsize=6)
        ax_bar.set_xlabel("Total |SHAP|", fontsize=7)
        ax_bar.set_title(f"{fam} ({len(names)} sigs, "
                         f"total={fam_rank[fam]:.4f})", fontsize=9)

        # Hide logo placeholder axes (logos placed after tight_layout)
        if has_logos:
            axes[row_i, 1].set_axis_off()

    # Finalize layout before placing logos
    plt.tight_layout()

    # --- Place IC-aligned logos after tight_layout for accurate positioning ---
    if has_logos:
        for row_i, fam in enumerate(top_families):
            ax_bar = axes[row_i, 0]
            ax_logo = axes[row_i, 1]
            sig_ids = fam_sig_ids[fam]
            n_sigs = len(sig_ids)
            if n_sigs == 0:
                continue

            # Collect available PWMs for this family, then align them
            fam_pwms = {}
            for sid in sig_ids:
                pfm = logos.get(sid)
                if pfm is not None and pfm.ndim == 2 and pfm.shape[0] == 4:
                    fam_pwms[sid] = pfm
            if not fam_pwms:
                continue

            aligned = _align_pwms(fam_pwms)

            # Map bar y-positions to figure coordinates
            logo_bbox = ax_logo.get_position()
            trans = ax_bar.transData + fig.transFigure.inverted()

            for j, sid in enumerate(sig_ids):
                apfm = aligned.get(sid)
                if apfm is None:
                    continue

                # Bar center in data coords → figure coords
                bar_center_fig = trans.transform((0, j))
                bar_top_fig = trans.transform((0, j + 0.4))
                bar_bot_fig = trans.transform((0, j - 0.4))
                h_fig = bar_top_fig[1] - bar_bot_fig[1]

                x0 = logo_bbox.x0 + 0.005
                y0 = bar_center_fig[1] - h_fig / 2
                w = logo_bbox.width * 0.92
                h = h_fig

                ax_ins = fig.add_axes([x0, y0, w, h])
                logo_df = pd.DataFrame(apfm.T, columns=list("ACGT"))
                logo_df = logo_df.div(
                    logo_df.sum(axis=1).replace(0, np.nan), axis=0)
                logo_df = logo_df.fillna(0)  # zero-padded cols → blank
                logomaker.Logo(logo_df, ax=ax_ins,
                               stack_order="small_on_top")
                ax_ins.set_ylim(0, 1.0)
                ax_ins.set_xticks([])
                ax_ins.set_yticks([])
                for sp in ax_ins.spines.values():
                    sp.set_visible(False)

    nature_savefig(fig,
                   f"fig_signature_importance_per_family{fig_suffix}_{pass_label}",
                   pass_outdir)
    plt.close(fig)
    label = "signature importance per family" + (title_extra or "")
    print(f"  [Fig] {label} ({pass_label})", flush=True)


def _tier_concordance_scatter(ax, merged, x_col, y_col,
                              clip_family="DOF", xlabel="", ylabel="",
                              title="", top_n_labels=8):
    """Helper: one tier-concordance scatter panel with DOF clipping."""
    from adjustText import adjust_text
    x_raw = merged[x_col].values.copy()
    y_raw = merged[y_col].values.copy()
    families = merged["family"].values

    # Clip outlier family (e.g. DOF) to 95th percentile of remaining data
    clip_mask = families == clip_family
    if clip_mask.any():
        other_max = max(x_raw[~clip_mask].max(),
                        y_raw[~clip_mask].max()) if (~clip_mask).any() else 1
        clip_val = other_max * 1.15  # slight margin above the rest
        x_plot = np.where(clip_mask, np.minimum(x_raw, clip_val), x_raw)
        y_plot = np.where(clip_mask, np.minimum(y_raw, clip_val), y_raw)
        clipped = clip_mask & ((x_raw > clip_val) | (y_raw > clip_val))
    else:
        x_plot, y_plot = x_raw, y_raw
        clipped = np.zeros(len(x_raw), dtype=bool)

    ax.scatter(x_plot, y_plot, s=25, alpha=0.7, c="#555555",
               edgecolors="black", linewidth=0.3, zorder=3)

    lim = max(x_plot.max(), y_plot.max()) * 1.1
    ax.plot([0, lim], [0, lim], "k--", lw=0.5, alpha=0.4)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect("equal")
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(title, fontsize=9)

    # Label top-N families by magnitude + any clipped families
    magnitude = np.maximum(x_plot, y_plot)
    top_idx = set(np.argsort(magnitude)[-top_n_labels:])
    texts = []
    for i, fam in enumerate(families):
        if i in top_idx or clipped[i]:
            label = fam
            if clipped[i]:
                label += f" ({x_raw[i]:.3f}, {y_raw[i]:.3f})"
            texts.append(
                ax.text(x_plot[i], y_plot[i], label,
                        fontsize=5.5, alpha=0.85))
    if texts:
        try:
            adjust_text(texts, ax=ax,
                        force_text=(0.4, 0.4), force_points=(0.3, 0.3),
                        expand_text=(1.15, 1.25))
        except Exception:
            pass


def plot_tier_concordance(pass_outdir, pass_label, sig_meta):
    """Tier concordance: two separate figures.

    Figure 1 (regression): T1 vs T2 family-level total |SHAP| concordance.
    Figure 2 (classification): T1 vs T2 perm importance (R² vs BA drop).
    Clips DOF to avoid axis compression. Labels significant families.
    """
    from scipy.stats import spearmanr
    nature_figure_defaults()

    # ── Figure 1: Regression — T1 vs T2 SHAP concordance ──
    t1_path = os.path.join(pass_outdir, "scale_family_shap.tsv")
    t2_path = os.path.join(pass_outdir, "scale_signature_shap.tsv")
    if os.path.exists(t1_path) and os.path.exists(t2_path):
        t1 = pd.read_csv(t1_path, sep="\t")
        t1["family"] = t1["family"].map(rename_family)
        t1_total = (t1.groupby("family")["mean_abs_shap"]
                    .sum().reset_index()
                    .rename(columns={"mean_abs_shap": "total_shap_t1"}))

        fam_map = dict(zip(sig_meta["signature_id"],
                           sig_meta["primary_family"]))
        t2 = pd.read_csv(t2_path, sep="\t")
        t2["family"] = t2["signature_id"].map(fam_map).map(rename_family)
        t2_total = (t2.dropna(subset=["family"])
                    .groupby("family")["mean_abs_shap"]
                    .sum().reset_index()
                    .rename(columns={"mean_abs_shap": "total_shap_t2"}))

        merged = t1_total.merge(t2_total, on="family", how="inner")
        if len(merged) >= 3:
            rho, _ = spearmanr(merged["total_shap_t1"],
                               merged["total_shap_t2"])
            fig, ax = plt.subplots(figsize=(6, 5.5))
            _tier_concordance_scatter(
                ax, merged, "total_shap_t1", "total_shap_t2",
                xlabel="T1: Family-level total |SHAP|",
                ylabel="T2: Signature-aggregated total |SHAP|",
                title=f"Regression T1 vs T2 SHAP (ρ={rho:.3f})")
            plt.tight_layout()
            nature_savefig(fig,
                           f"fig_tier_concordance_reg_{pass_label}",
                           pass_outdir)
            plt.close(fig)
            print(f"  [Fig] tier concordance regression ({pass_label})",
                  flush=True)

    # ── Figure 2: Classification — T1 reg vs clf perm importance ──
    perm_reg_path = (
        os.path.join(pass_outdir, "family_permutation_importance_tier2.tsv")
        if os.path.exists(os.path.join(
            pass_outdir, "family_permutation_importance_tier2.tsv"))
        else os.path.join(pass_outdir, "family_permutation_importance.tsv"))
    perm_clf_path = os.path.join(
        pass_outdir, "family_permutation_importance_clf.tsv")

    if os.path.exists(perm_reg_path) and os.path.exists(perm_clf_path):
        pr = pd.read_csv(perm_reg_path, sep="\t")[
            ["family", "perm_importance_mean"]].rename(
            columns={"perm_importance_mean": "perm_r2_drop"})
        pr["family"] = pr["family"].map(rename_family)
        pc = pd.read_csv(perm_clf_path, sep="\t")[
            ["family", "perm_importance_mean"]].rename(
            columns={"perm_importance_mean": "perm_ba_drop"})
        pc["family"] = pc["family"].map(rename_family)
        perm_merged = pr.merge(pc, on="family", how="inner")

        if len(perm_merged) >= 3:
            rho_p, _ = spearmanr(perm_merged["perm_r2_drop"],
                                  perm_merged["perm_ba_drop"])
            fig, ax = plt.subplots(figsize=(6, 5.5))
            _tier_concordance_scatter(
                ax, perm_merged, "perm_r2_drop", "perm_ba_drop",
                xlabel="Regression perm. importance (R² drop)",
                ylabel="Classification perm. importance (BA drop)",
                title=f"Reg. vs Clf. perm importance (ρ={rho_p:.3f})")
            plt.tight_layout()
            nature_savefig(fig,
                           f"fig_tier_concordance_clf_{pass_label}",
                           pass_outdir)
            plt.close(fig)
            print(f"  [Fig] tier concordance classification ({pass_label})",
                  flush=True)


def plot_sig_shap_per_family(pass_outdir, pass_label, scales, sig_meta,
                              top_n_families=20):
    """Per-family per-signature SHAP panels (one file per family).

    Loads raw_shap_tier2.npz + informative_signatures.tsv.
    Layout per signature row: 4 columns
      Col 0: mean |SHAP| profile (Y = scale bp, X = mean |SHAP|)
      Col 1–3: signed SHAP violin at top-1, top-2, top-3 scales
    Saves to sig_shap_panels/ subdirectory.
    """
    import re as _re
    nature_figure_defaults()
    npz_path = os.path.join(pass_outdir, "raw_shap_tier2.npz")
    info_path = os.path.join(pass_outdir, "informative_signatures.tsv")
    if not (os.path.exists(npz_path) and os.path.exists(info_path)):
        return

    d = np.load(npz_path, allow_pickle=True)
    raw_shap = d["shap_values"].astype(np.float32)
    feature_names = list(d["feature_names"])
    acr_class_test = d["acr_class_test"]

    info_df = pd.read_csv(info_path, sep="\t")
    info_df["family"] = info_df["family"].map(rename_family)
    info_only = info_df[info_df["is_informative"]].copy()
    if info_only.empty:
        return

    dn_map = dict(zip(sig_meta["signature_id"], sig_meta["display_name"]))

    fam_rank = (info_only.groupby("family")["total_shap"]
                .sum().sort_values(ascending=False))
    top_families = fam_rank.head(top_n_families).index.tolist()

    feat_to_idx = {f: i for i, f in enumerate(feature_names)}
    n_scales = len(scales)
    panel_dir = os.path.join(pass_outdir, "sig_shap_panels")
    os.makedirs(panel_dir, exist_ok=True)

    n_top_scales = 3  # number of top scales to show as violins

    for fam in top_families:
        fam_sigs = info_only[info_only["family"] == fam].sort_values(
            "total_shap", ascending=False)
        if fam_sigs.empty:
            continue

        n_sigs = len(fam_sigs)
        n_cols = 1 + n_top_scales  # profile + 3 violin panels
        fig, axes = plt.subplots(
            n_sigs, n_cols,
            figsize=(3.5 * n_cols, max(3, 2.5 * n_sigs)),
            gridspec_kw={"width_ratios": [2] + [1] * n_top_scales},
            squeeze=False)

        for row_i, (_, srow) in enumerate(fam_sigs.iterrows()):
            sid = srow["signature_id"]
            sname = dn_map.get(sid, sid)

            # Compute mean |SHAP| at each scale
            scale_shap, scale_bps, scale_indices = [], [], []
            for si in range(n_scales):
                ci = feat_to_idx.get(f"{sid}_s{si}")
                if ci is not None:
                    scale_shap.append(np.abs(raw_shap[:, ci]).mean())
                    scale_bps.append(float(scales[si]))
                    scale_indices.append(si)

            # Panel A: mean |SHAP| profile — Y = scale (bp), X = mean |SHAP|
            ax = axes[row_i, 0]
            if scale_bps:
                ax.plot(scale_shap, scale_bps, color="black", linewidth=1)
                ax.fill_betweenx(scale_bps, 0, scale_shap,
                                 alpha=0.2, color="steelblue")
                # Mark top-3 scales with horizontal lines
                top_k = min(n_top_scales, len(scale_shap))
                top_si_local = np.argsort(scale_shap)[-top_k:][::-1]
                rank_colors = ["#e34a33", "#fc8d59", "#fdbb84"]
                for rank, li in enumerate(top_si_local):
                    bp_val = scale_bps[li]
                    ax.axhline(bp_val, color=rank_colors[rank],
                               ls="--", lw=0.8, alpha=0.7)
                    ax.annotate(f"#{rank+1}", (scale_shap[li], bp_val),
                                fontsize=6, color=rank_colors[rank],
                                xytext=(3, 2), textcoords="offset points")
            for edge in BAND_EDGES:
                ax.axhline(edge, color="grey", ls=":", lw=0.5, alpha=0.5)
            ax.set_xlabel("Mean |SHAP|", fontsize=7)
            ax.set_ylabel("Scale (bp)", fontsize=7)
            ax.set_title(f"{sid} {sname} (rank {int(srow['rank'])})",
                         fontsize=8)

            # Panels B–D: signed SHAP violins at top-1, top-2, top-3 scales
            # Shared y-axis across the 3 violin panels for this signature
            if scale_shap:
                top_k = min(n_top_scales, len(scale_shap))
                top_si_local = np.argsort(scale_shap)[-top_k:][::-1]

                # Pre-compute shared y range across top-k scales
                y_lo, y_hi = 0.0, 0.0
                for vi in range(top_k):
                    si = scale_indices[top_si_local[vi]]
                    ci = feat_to_idx.get(f"{sid}_s{si}")
                    if ci is not None:
                        sv = raw_shap[:, ci]
                        y_lo = min(y_lo, float(np.nanmin(sv)))
                        y_hi = max(y_hi, float(np.nanmax(sv)))
                y_margin = max(0.05, (y_hi - y_lo) * 0.08)
                shared_ylim = (y_lo - y_margin, y_hi + y_margin)

                for vi in range(n_top_scales):
                    ax_v = axes[row_i, 1 + vi]
                    if vi >= top_k:
                        ax_v.set_visible(False)
                        continue
                    si = scale_indices[top_si_local[vi]]
                    bp_val = float(scales[si])
                    ci = feat_to_idx.get(f"{sid}_s{si}")
                    if ci is None:
                        ax_v.set_visible(False)
                        continue
                    shap_vals = raw_shap[:, ci]
                    for cls_i, cls in enumerate(CLASS_ORDER):
                        mask = acr_class_test == cls
                        if mask.sum() > 0:
                            color = PALETTE.get(cls, "#888888")
                            parts = ax_v.violinplot(
                                [shap_vals[mask]], positions=[cls_i],
                                showmedians=True, widths=0.7)
                            for pc in parts["bodies"]:
                                pc.set_facecolor(color)
                                pc.set_alpha(0.5)
                            for key in ["cmins", "cmaxes", "cbars",
                                        "cmedians"]:
                                if key in parts:
                                    parts[key].set_color(color)
                    ax_v.set_ylim(shared_ylim)
                    ax_v.set_xticks(range(len(CLASS_ORDER)))
                    ax_v.set_xticklabels(CLASS_ORDER, fontsize=6,
                                         rotation=25)
                    ax_v.axhline(0, color="grey", ls="--", lw=0.5)
                    rank_label = f"Top-{vi+1}"
                    ax_v.set_title(f"{rank_label}: {bp_val:.0f} bp",
                                   fontsize=7)
                    if vi == 0:
                        ax_v.set_ylabel("Signed SHAP", fontsize=7)
            else:
                for vi in range(n_top_scales):
                    axes[row_i, 1 + vi].set_visible(False)

        fig.suptitle(f"{fam} — informative signatures ({n_sigs})",
                     fontsize=10, y=1.01)
        plt.tight_layout()
        safe = _re.sub(r"[/\\: ]", "_", fam)
        nature_savefig(fig, f"sig_shap_{safe}", panel_dir)
        plt.close(fig)

    print(f"  [Fig] per-family sig SHAP panels: {len(top_families)} families "
          f"→ {panel_dir}", flush=True)


# ── Per-ACR family SHAP analysis ─────────────────────────────────────────────

def analyze_per_acr_family_shap(D, pass_outdir, pass_label, acr_subset,
                                 top_n=30):
    """Per-ACR family SHAP: rank ACRs by family influence, find opposing pairs.

    Loads raw_shap_tier1.npz (family × scale features), sums SHAP across all
    scales per family per ACR, and produces:
      - per_acr_family_shap.tsv.gz (reusable data table)
      - Fig A: top ACRs by DOF |SHAP|
      - Fig B: top ACRs by most negative WRKY SHAP
      - Fig C: DOF vs WRKY opposing scatter
      - Fig D: top-10 families boxplot by genomic context
    """
    import re as _re

    npz_path = os.path.join(pass_outdir, "raw_shap_tier1.npz")
    if not os.path.exists(npz_path):
        print("  [per-ACR SHAP] No T1 NPZ — skipping", flush=True)
        return

    npz = np.load(npz_path, allow_pickle=True)
    if "acr_ids_test" not in npz.files:
        print("  [per-ACR SHAP] NPZ missing acr_ids_test — re-run with "
              "--force to regenerate", flush=True)
        return

    shap_vals = npz["shap_values"]          # (n_test, n_features)
    feat_names = list(npz["feature_names"])
    acr_class = npz["acr_class_test"]
    acr_ids = npz["acr_ids_test"]

    # Build family→column index map (reuse existing function)
    fam_feat_map = build_family_feature_map(feat_names, tier="T1")

    # Signed sum of SHAP across all scales per family per ACR
    fam_shap = {}
    for fam, col_idxs in fam_feat_map.items():
        fam_shap[f"shap_{fam}"] = shap_vals[:, col_idxs].sum(axis=1)

    shap_df = pd.DataFrame(fam_shap, index=acr_ids)
    shap_df.index.name = "acr_id"
    shap_df["acr_class"] = acr_class

    # Merge genomic annotations from acr_meta
    meta_cols = ["genomic_context", "edgeR_logFC", "logCPM"]
    available = [c for c in meta_cols if c in D["acr_meta"].columns]
    if available:
        shap_df = shap_df.join(D["acr_meta"][available], how="left")

    # Save table
    shap_df.to_csv(os.path.join(pass_outdir, "per_acr_family_shap.tsv.gz"),
                    sep="\t", compression="gzip")
    print(f"  [per-ACR SHAP] Table saved: {len(shap_df)} ACRs × "
          f"{len(fam_feat_map)} families", flush=True)

    # -- Identify top families for display names --
    fam_names = list(fam_feat_map.keys())
    fam_abs_total = {f: np.abs(shap_df[f"shap_{f}"]).sum() for f in fam_names}
    top10_fams = sorted(fam_abs_total, key=fam_abs_total.get, reverse=True)[:10]

    # Consistent ACR class order + colors
    class_order = ["proto_gain", "stable", "leaf_gain"]
    class_colors = ACR_CLASS_COLORS

    # ── Figure A: Top ACRs by DOF |SHAP| ────────────────────────────────────
    _plot_top_acrs_by_family(
        shap_df, family="DOF", top_n=top_n, class_order=class_order,
        class_colors=class_colors, pass_outdir=pass_outdir,
        pass_label=pass_label, fig_name="fig_acr_dof_influence")

    # ── Figure B: Top ACRs with strongest proto-enriched WRKY SHAP ──────────
    _plot_top_acrs_by_family(
        shap_df, family="WRKY", top_n=top_n, class_order=class_order,
        class_colors=class_colors, pass_outdir=pass_outdir,
        pass_label=pass_label, fig_name="fig_acr_wrky_proto",
        direction="negative")

    # ── Figure C: DOF vs WRKY scatter ────────────────────────────────────────
    _plot_opposing_scatter(shap_df, fam_x="DOF", fam_y="WRKY",
                           class_order=class_order, class_colors=class_colors,
                           pass_outdir=pass_outdir, pass_label=pass_label)

    # ── Figure D: Top-10 families boxplot by genomic context ─────────────────
    if "genomic_context" in shap_df.columns:
        _plot_shap_by_context(shap_df, top10_fams, class_colors=class_colors,
                               pass_outdir=pass_outdir, pass_label=pass_label)

    print(f"  [per-ACR SHAP] 4 figures saved", flush=True)


def _plot_top_acrs_by_family(shap_df, family, top_n, class_order,
                              class_colors, pass_outdir, pass_label,
                              fig_name, direction="absolute"):
    """Bar chart of top ACRs ranked by family SHAP influence."""
    nature_figure_defaults()
    col = f"shap_{family}"
    if col not in shap_df.columns:
        print(f"  [per-ACR SHAP] {family} not in features — skipping "
              f"{fig_name}", flush=True)
        return

    df = shap_df[[col, "acr_class"]].copy()
    if "genomic_context" in shap_df.columns:
        df["genomic_context"] = shap_df["genomic_context"]
    if "edgeR_logFC" in shap_df.columns:
        df["edgeR_logFC"] = shap_df["edgeR_logFC"]

    if direction == "negative":
        # Most negative (proto-enriched) — sort ascending
        df = df.sort_values(col, ascending=True).head(top_n)
        title = f"Top {top_n} ACRs — strongest proto-enriched {family} SHAP"
        xlabel = f"{family} signed SHAP sum (most negative = strongest proto)"
    else:
        # Largest |SHAP|
        df["abs_shap"] = df[col].abs()
        df = df.sort_values("abs_shap", ascending=False).head(top_n)
        df = df.sort_values(col, ascending=True)  # visual order
        title = f"Top {top_n} ACRs by {family} |SHAP| influence"
        xlabel = f"{family} signed SHAP sum"

    fig, ax = plt.subplots(figsize=(6, 0.25 * top_n + 1))
    colors = [class_colors.get(c, "#888888") for c in df["acr_class"]]
    bars = ax.barh(range(len(df)), df[col].values, color=colors, edgecolor="none")

    # Y-axis: ACR IDs (shortened)
    labels = []
    for acr_id, row in df.iterrows():
        lbl = str(acr_id).replace("chr", "Chr")
        if "genomic_context" in row.index and pd.notna(row.get("genomic_context")):
            lbl += f" [{row['genomic_context'][:4]}]"
        labels.append(lbl)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.axvline(0, color="black", linewidth=0.5, linestyle="-")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=class_colors[c], label=c)
                       for c in class_order if c in set(df["acr_class"])]
    ax.legend(handles=legend_elements, fontsize=6, loc="lower right")

    plt.tight_layout()
    nature_savefig(fig, f"{fig_name}_{pass_label}", pass_outdir)
    plt.close(fig)


def _plot_opposing_scatter(shap_df, fam_x, fam_y, class_order, class_colors,
                            pass_outdir, pass_label):
    """Scatter: fam_x SHAP vs fam_y SHAP per ACR, colored by ACR class."""
    nature_figure_defaults()
    col_x = f"shap_{fam_x}"
    col_y = f"shap_{fam_y}"
    if col_x not in shap_df.columns or col_y not in shap_df.columns:
        return

    fig, ax = plt.subplots(figsize=(5, 5))

    for cls in class_order:
        mask = shap_df["acr_class"] == cls
        if mask.sum() == 0:
            continue
        ax.scatter(shap_df.loc[mask, col_x], shap_df.loc[mask, col_y],
                   c=class_colors.get(cls, "#888888"), label=cls, s=6,
                   alpha=0.4, edgecolors="none", rasterized=True)

    ax.axhline(0, color="black", linewidth=0.4, linestyle="--", alpha=0.5)
    ax.axvline(0, color="black", linewidth=0.4, linestyle="--", alpha=0.5)

    # Highlight opposing quadrants
    ax.text(0.95, 0.05, f"{fam_x}+ / {fam_y}−\n(opposing)",
            transform=ax.transAxes, fontsize=6, ha="right", va="bottom",
            color="#555555", style="italic")
    ax.text(0.05, 0.95, f"{fam_x}− / {fam_y}+\n(opposing)",
            transform=ax.transAxes, fontsize=6, ha="left", va="top",
            color="#555555", style="italic")

    # Count opposing ACRs
    opposing = ((shap_df[col_x] > 0) & (shap_df[col_y] < 0)) | \
               ((shap_df[col_x] < 0) & (shap_df[col_y] > 0))
    n_opp = opposing.sum()
    pct_opp = 100 * n_opp / len(shap_df)

    # Spearman correlation
    from scipy.stats import spearmanr
    rho, pval = spearmanr(shap_df[col_x], shap_df[col_y])

    ax.set_xlabel(f"{rename_family(fam_x)} signed SHAP sum", fontsize=8)
    ax.set_ylabel(f"{rename_family(fam_y)} signed SHAP sum", fontsize=8)
    ax.set_title(f"{rename_family(fam_x)} vs {rename_family(fam_y)} — "
                 f"per-ACR SHAP\n"
                 f"ρ={rho:.3f} (p={pval:.1e}), "
                 f"{n_opp:,} opposing ({pct_opp:.1f}%)",
                 fontsize=8)
    ax.legend(fontsize=7, markerscale=2)

    plt.tight_layout()
    nature_savefig(fig, f"fig_acr_opposing_families_{pass_label}", pass_outdir)
    plt.close(fig)


def _plot_shap_by_context(shap_df, top_families, class_colors,
                           pass_outdir, pass_label):
    """Boxplot: signed SHAP per family, faceted by genomic context."""
    nature_figure_defaults()
    contexts = ["Promoter", "Gene body", "Intergenic"]
    n_fam = len(top_families)

    fig, axes = plt.subplots(1, 3, figsize=(4 * 3, 0.3 * n_fam + 1.5),
                              sharey=True)

    for ci, ctx in enumerate(contexts):
        ax = axes[ci]
        ctx_mask = shap_df["genomic_context"] == ctx
        sub = shap_df.loc[ctx_mask]
        if sub.empty:
            ax.set_title(f"{ctx} (n=0)", fontsize=8)
            continue

        data = []
        labels = []
        for fam in reversed(top_families):
            col = f"shap_{fam}"
            if col in sub.columns:
                vals = sub[col].dropna().values
                data.append(vals)
                labels.append(rename_family(fam))

        bp = ax.boxplot(data, vert=False, widths=0.6,
                        patch_artist=True, showfliers=False,
                        medianprops=dict(color="black", linewidth=0.8))
        for patch in bp["boxes"]:
            patch.set_facecolor("#B0C4DE")
            patch.set_edgecolor("#666666")
            patch.set_linewidth(0.5)

        ax.set_yticks(range(1, len(labels) + 1))
        ax.set_yticklabels(labels, fontsize=7)
        ax.axvline(0, color="black", linewidth=0.4, linestyle="--", alpha=0.5)
        ax.set_title(f"{ctx} (n={ctx_mask.sum():,})", fontsize=8)
        if ci == 0:
            ax.set_ylabel("")
        ax.set_xlabel("Signed SHAP sum", fontsize=7)

    fig.suptitle(f"Per-ACR family SHAP by genomic context — {pass_label}",
                 fontsize=10, y=1.02)
    plt.tight_layout()
    nature_savefig(fig, f"fig_acr_shap_by_context_{pass_label}", pass_outdir)
    plt.close(fig)


# ── Per-class classification SHAP for individual signatures ──────────────────

def plot_clf_shap_per_class(pass_outdir, pass_label, sig_meta,
                             top_n_families=10):
    """Show how individual signatures within each family push toward each class.

    Loads raw_shap_tier2_clf.npz (n_test, n_features, n_classes).
    For each top family, plots a grouped bar: one bar per signature,
    colored by target class, showing mean signed SHAP per class.

    This reveals e.g. whether individual DOF signatures push toward
    proto_gain vs leaf_gain classification — confirming or refuting the
    opposing pattern seen in fig_acr_opposing_families.
    """
    nature_figure_defaults()

    npz_path = os.path.join(pass_outdir, "raw_shap_tier2_clf.npz")
    if not os.path.exists(npz_path):
        return

    npz = np.load(npz_path, allow_pickle=True)
    shap_vals = npz["shap_values"]      # (n_test, n_feat, n_classes) or 2D
    feat_names = list(npz["feature_names"])
    class_names = list(npz["class_names"])

    if shap_vals.ndim != 3:
        print("  [Fig] clf SHAP per class: need 3D tensor — skipping",
              flush=True)
        return

    n_test, n_feat, n_classes = shap_vals.shape
    dn_map = dict(zip(sig_meta["signature_id"], sig_meta["display_name"]))
    fam_map = dict(zip(sig_meta["signature_id"], sig_meta["primary_family"]))

    # Parse feature names → (signature_id, scale_idx)
    # Sum SHAP across all scales per signature per class per ACR
    sig_col_map = {}  # sig_id → list of column indices
    for fi, fname in enumerate(feat_names):
        parts = fname.rsplit("_s", 1)
        if len(parts) != 2:
            continue
        sig_id = parts[0]
        sig_col_map.setdefault(sig_id, []).append(fi)

    # Mean signed SHAP per signature per class (averaged across test ACRs)
    rows = []
    for sig_id, col_idxs in sig_col_map.items():
        fam = fam_map.get(sig_id, "Unknown")
        # Sum across scales, then mean across test samples
        sig_shap = shap_vals[:, col_idxs, :].sum(axis=1)  # (n_test, n_classes)
        for ci, cls in enumerate(class_names):
            rows.append({
                "signature_id": sig_id,
                "family": rename_family(fam),
                "display_name": dn_map.get(sig_id, sig_id),
                "class": cls,
                "mean_signed_shap": float(sig_shap[:, ci].mean()),
                "mean_abs_shap": float(np.abs(sig_shap[:, ci]).mean()),
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return

    # Save table for reuse
    df.to_csv(os.path.join(pass_outdir, "signature_clf_shap_per_class.tsv"),
              sep="\t", index=False)

    # Rank families by total |SHAP| across classes
    fam_rank = (df.groupby("family")["mean_abs_shap"]
                .sum().sort_values(ascending=False))
    top_fams = fam_rank.head(top_n_families).index.tolist()

    class_colors = {}
    for cls in class_names:
        class_colors[cls] = ACR_CLASS_COLORS.get(cls, "#888888")

    n_fams = len(top_fams)
    fig, axes = plt.subplots(n_fams, 1,
                              figsize=(10, max(2.5 * n_fams, 6)),
                              squeeze=False)

    for fi, fam in enumerate(top_fams):
        ax = axes[fi, 0]
        fam_df = df[df["family"] == fam].copy()
        # Pivot: signatures × classes
        pivot = fam_df.pivot(index="display_name", columns="class",
                              values="mean_signed_shap")
        pivot = pivot.reindex(columns=class_names)
        # Sort by total absolute SHAP
        abs_total = fam_df.groupby("display_name")["mean_abs_shap"].sum()
        sig_order = abs_total.sort_values(ascending=True).index
        pivot = pivot.reindex(sig_order)

        x = np.arange(len(pivot))
        width = 0.25
        for ci, cls in enumerate(class_names):
            offset = (ci - (n_classes - 1) / 2) * width
            vals = pivot[cls].values
            ax.barh(x + offset, vals, height=width * 0.9,
                    color=class_colors.get(cls, "#888888"),
                    label=cls if fi == 0 else None,
                    edgecolor="white", linewidth=0.3)

        ax.set_yticks(x)
        ax.set_yticklabels(pivot.index, fontsize=6)
        ax.axvline(0, color="grey", lw=0.8)
        ax.set_title(f"{fam}", fontsize=9, fontweight="bold")
        ax.set_xlabel("Mean signed SHAP (classification)", fontsize=7)

    axes[0, 0].legend(fontsize=7, loc="upper right", ncol=n_classes)
    fig.suptitle(f"Per-signature classification SHAP by target class "
                 f"(T2, {pass_label})", fontsize=11)
    plt.tight_layout()
    nature_savefig(fig, f"fig_clf_shap_per_class_{pass_label}", pass_outdir)
    plt.close(fig)
    print(f"  [Fig] classification SHAP per class ({pass_label})", flush=True)


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="v3 Step 08: Gradient Boosting + SHAP")
    p.add_argument("--perscale-dir", default="results/v3_06_perscale_fp")
    p.add_argument("--sig-metadata",
                   default="data/motif_signatures/signature_metadata.tsv")
    p.add_argument("--acr-metadata", default="data/acr_metadata.tsv.gz")
    p.add_argument("--acr-coord-mapping",
                   default="data/acr_native_to_resized.tsv")
    p.add_argument("--outdir", default="results/v3_08_gradient_boosting")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--n-permutations", type=int, default=100)
    p.add_argument("--cumulative-threshold", type=float, default=0.80)
    p.add_argument("--skip-tier2", action="store_true")
    p.add_argument("--skip-elastic-net", action="store_true")
    p.add_argument("--force", action="store_true",
                   help="Force re-run, ignore sentinels (regenerates NPZs)")
    p.add_argument("--top-n-families", type=int, default=20)
    return p.parse_args()


def main():
    args = parse_args()
    outdir = os.path.join(BASE, args.outdir)
    os.makedirs(outdir, exist_ok=True)

    print("=" * 60, flush=True)
    print("v3_08 — Gradient Boosting + SHAP", flush=True)
    print("=" * 60, flush=True)

    nature_figure_defaults()
    D = load_data(args)

    # Build T1 features
    tf_features = build_family_scale_features(
        D["family_delta"], D["family_ids"], D["acr_ids"], D["scales"])
    print(f"  T1 features: {tf_features.shape}", flush=True)

    # Load motif logos once (for signature importance figure)
    meme_logos = {}
    if os.path.exists(MEME_PATH):
        try:
            meme_logos = load_meme_logos(MEME_PATH)
            print(f"  Loaded {len(meme_logos)} motif logos from MEME",
                  flush=True)
        except Exception as e:
            print(f"  [WARN] Could not load MEME logos: {e}", flush=True)

    # Passes
    pass_configs = [
        ("all", None),
        ("changing", ["proto_gain", "leaf_gain"]),
    ]
    all_summary = []

    for pass_label, acr_subset in pass_configs:
        pass_outdir = os.path.join(outdir, pass_label)
        os.makedirs(pass_outdir, exist_ok=True)

        print(f"\n{'='*60}", flush=True)
        print(f"Pass: {pass_label}", flush=True)
        print("=" * 60, flush=True)

        # T1
        shap_t1 = run_tier1(D, tf_features, acr_subset, pass_label,
                             pass_outdir, args, all_summary)

        # T2
        info_df = None
        if not args.skip_tier2:
            info_df = run_tier2(D, acr_subset, pass_label, pass_outdir,
                                args, all_summary, shap_t1)

        # T3
        if not args.skip_elastic_net:
            run_tier3(D, info_df, acr_subset, pass_label, pass_outdir,
                      args, all_summary)

        # Figures (always run — read from cached TSVs/NPZs, skip if missing)
        print(f"\n  [Figures] pass={pass_label}", flush=True)
        plot_permutation_importance(pass_outdir, pass_label)
        plot_shap_scale_heatmap(pass_outdir, pass_label, D["scales"])
        plot_shap_scale_heatmap(pass_outdir, pass_label, D["scales"],
                                 tsv_name="scale_family_shap_clf.tsv",
                                 fig_suffix="_clf",
                                 title_extra=", classification")
        # ACR-class-stratified scale heatmaps (regression + classification)
        plot_shap_scale_heatmap_by_class(
            pass_outdir, pass_label, D["scales"], D["sig_meta"],
            npz_name="raw_shap_tier1.npz",
            title_extra=" (regression)")
        plot_shap_scale_heatmap_by_class(
            pass_outdir, pass_label, D["scales"], D["sig_meta"],
            npz_name="raw_shap_tier1_clf.npz",
            fig_suffix="_clf",
            title_extra=" (classification)")
        plot_family_acr_class_heatmap(D, pass_outdir, pass_label, acr_subset)
        plot_shap_asymmetry(pass_outdir, pass_label, D["sig_meta"])
        plot_shap_beeswarm(pass_outdir, pass_label,
                           D["scales"], D["sig_meta"])
        plot_clf_results(pass_outdir, pass_label)
        plot_family_perm_importance_clf(pass_outdir, pass_label)
        plot_scale_signature_heatmap(pass_outdir, pass_label,
                                     D["scales"], D["sig_meta"])
        plot_signature_importance_per_family(pass_outdir, pass_label,
                                             D["sig_meta"],
                                             logos=meme_logos)
        plot_signature_importance_per_family(pass_outdir, pass_label,
                                             D["sig_meta"],
                                             logos=meme_logos,
                                             tsv_name="scale_signature_shap_clf.tsv",
                                             fig_suffix="_clf",
                                             title_extra=", classification")
        # ACR-class-stratified per-family signature bars (reg + clf)
        plot_signature_importance_per_family_by_class(
            pass_outdir, pass_label, D["sig_meta"],
            npz_name="raw_shap_tier2.npz",
            title_extra=" (regression)",
            logos=meme_logos)
        plot_signature_importance_per_family_by_class(
            pass_outdir, pass_label, D["sig_meta"],
            npz_name="raw_shap_tier2_clf.npz",
            fig_suffix="_clf",
            title_extra=" (classification)",
            logos=meme_logos)
        plot_tier_concordance(pass_outdir, pass_label, D["sig_meta"])
        plot_sig_shap_per_family(pass_outdir, pass_label,
                                 D["scales"], D["sig_meta"])
        analyze_per_acr_family_shap(D, pass_outdir, pass_label, acr_subset)
        plot_clf_shap_per_class(pass_outdir, pass_label, D["sig_meta"])

    # Summary
    summary_df = pd.DataFrame(all_summary)
    summary_df.to_csv(os.path.join(outdir, "model_summary.tsv"),
                       sep="\t", index=False)
    print(f"\n[DONE] model_summary.tsv: {len(summary_df)} rows", flush=True)


if __name__ == "__main__":
    main()
