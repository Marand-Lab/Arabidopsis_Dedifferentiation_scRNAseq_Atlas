#!/usr/bin/env python3
"""
v3 Step 09: SHAP interaction analysis.

Builds a feature matrix with ALL 122 signatures at their best scale (from
v3_08 T2 SHAP table), fits regression + classification models, and computes
SHAP interaction values.

Two-phase design with independent sentinels:
  Phase 1 (regression):  SHAP interaction values → raw tensor + mean matrices
  Phase 2 (classifier):  BA/F1/confusion matrix for performance comparison

Output: results/v3_09_shap_interactions/
"""
from __future__ import annotations

import argparse
import gc
import os
import time
import warnings

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, r2_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder

from _utils import (
    load_acr_metadata,
    nature_figure_defaults,
    nature_savefig,
)

warnings.filterwarnings("ignore", category=FutureWarning)

BASE = os.path.dirname(os.path.abspath(__file__))


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_signature_metadata(path):
    return pd.read_csv(path, sep="\t")


def _build_design_matrix(acr_meta, index):
    cols = []
    if "acr_width" in acr_meta.columns:
        acr_meta = acr_meta.copy()
        acr_meta["log_width"] = np.log1p(acr_meta["acr_width"])
        cols.append("log_width")
    if "logCPM" in acr_meta.columns:
        cols.append("logCPM")
    if "genomic_context" in acr_meta.columns:
        dummies = pd.get_dummies(acr_meta["genomic_context"], prefix="gc", drop_first=True)
        acr_meta = pd.concat([acr_meta, dummies], axis=1)
        cols.extend(dummies.columns.tolist())
    C = acr_meta.loc[index, cols].astype(float)
    valid = C.notna().all(axis=1)
    return C.loc[valid], valid


def residualize_features(tf_features, acr_meta):
    common = tf_features.index.intersection(acr_meta.index)
    C, _ = _build_design_matrix(acr_meta, common)
    common = C.index
    X = tf_features.loc[common]
    resid = X.copy()
    from numpy.linalg import lstsq
    C_arr = np.column_stack([np.ones(len(C)), C.values])
    for col in X.columns:
        y = X[col].values
        finite = np.isfinite(y)
        if finite.sum() < 20:
            continue
        beta, _, _, _ = lstsq(C_arr[finite], y[finite], rcond=None)
        resid[col] = y - C_arr @ beta
    return resid.astype(np.float32)


def residualize_response(y, acr_meta):
    C, _ = _build_design_matrix(acr_meta, y.index)
    common = C.index.intersection(y.index)
    C = C.loc[common]; y = y.loc[common]
    from numpy.linalg import lstsq
    C_arr = np.column_stack([np.ones(len(C)), C.values])
    beta, _, _, _ = lstsq(C_arr, y.values, rcond=None)
    return pd.Series(y.values - C_arr @ beta, index=common)


def _build_feature_matrix(perscale_dir, best_scale_map, acr_meta, acr_subset, args):
    """Build residualized feature matrix deterministically. Returns data dict."""
    sig_npz = np.load(os.path.join(BASE, perscale_dir,
                                    "delta_acr_signature_scale.npz"),
                       allow_pickle=True)
    sig_delta = sig_npz["delta"]
    sig_ids = sig_npz["signature_ids"]
    acr_ids = sig_npz["acr_ids"]
    scales = sig_npz["scales"]

    sig_idx_map = {s: i for i, s in enumerate(sig_ids)}
    cols, col_names = [], []
    for sig_id in sig_ids:
        si = sig_idx_map.get(sig_id)
        sci = best_scale_map.get(sig_id, 0)
        if si is not None and sci < sig_delta.shape[2]:
            cols.append(sig_delta[:, si, sci])
            col_names.append(sig_id)

    X_df = pd.DataFrame(np.column_stack(cols), index=acr_ids,
                         columns=col_names).fillna(0)

    y_df = acr_meta[["edgeR_logFC", "acr_class"]].copy()
    common = X_df.index.intersection(y_df.index)
    if acr_subset:
        mask = y_df.loc[common, "acr_class"].isin(acr_subset)
        common = common[mask]
    common = common[y_df.loc[common, "edgeR_logFC"].notna()]

    X_resid = residualize_features(X_df.loc[common], acr_meta)
    y_resid = residualize_response(y_df.loc[common, "edgeR_logFC"], acr_meta)
    common = X_resid.index.intersection(y_resid.index)
    X_resid = X_resid.loc[common]
    y_resid = y_resid.loc[common]
    acr_class = y_df.loc[common, "acr_class"]

    sss = StratifiedShuffleSplit(n_splits=1, test_size=args.test_size,
                                  random_state=args.seed)
    train_idx, test_idx = next(sss.split(X_resid, acr_class))

    return {
        "X_resid": X_resid, "y_resid": y_resid,
        "acr_class": acr_class,
        "train_idx": train_idx, "test_idx": test_idx,
        "sig_ids": sig_ids, "col_names": col_names, "scales": scales,
    }


# ── Main pass runner ──────────────────────────────────────────────────────────

def run_pass(D, sig_meta, acr_meta, pass_label, acr_subset, outdir, args):
    """Run SHAP interaction analysis (regression + classification) for one pass."""
    pass_outdir = os.path.join(outdir, pass_label)
    os.makedirs(pass_outdir, exist_ok=True)

    # Load best scale for ALL signatures from v3_08 SHAP table
    shap_path = os.path.join(BASE, args.gb_dir, pass_label, "scale_signature_shap.tsv")
    if not os.path.exists(shap_path):
        print(f"  [{pass_label}] No scale_signature_shap.tsv — skipping", flush=True)
        return
    shap_table = pd.read_csv(shap_path, sep="\t")
    best_scale_map = (shap_table.sort_values("mean_abs_shap", ascending=False)
                      .groupby("signature_id")["scale_idx"].first()
                      .astype(int).to_dict())

    # Bookkeeping: count informative signatures
    info_path = os.path.join(BASE, args.gb_dir, pass_label, "informative_signatures.tsv")
    n_informative = 0
    if os.path.exists(info_path):
        info_df = pd.read_csv(info_path, sep="\t")
        n_informative = int(info_df["is_informative"].sum())

    # ── Phase 1: Regression SHAP ──────────────────────────────────────────────
    reg_sentinel = os.path.join(pass_outdir, "shap_interaction_matrix.tsv")
    run_phase1 = not os.path.exists(reg_sentinel) or args.force
    if run_phase1 and os.path.exists(reg_sentinel):
        os.remove(reg_sentinel)
        print(f"  [{pass_label}] --force: rerunning Phase 1 (regression)", flush=True)

    print(f"  [{pass_label}] Building feature matrix (ALL sigs, informative subset was"
          f" {n_informative})...", flush=True)
    data = _build_feature_matrix(args.perscale_dir, best_scale_map,
                                   acr_meta, acr_subset, args)
    n_feat = data["X_resid"].shape[1]
    print(f"  [{pass_label}] ACRs: {len(data['X_resid'])}, features: {n_feat}", flush=True)

    if run_phase1:
        # Fit regression model
        model = HistGradientBoostingRegressor(
            max_depth=5, learning_rate=0.05, max_iter=500,
            min_samples_leaf=20, validation_fraction=0.1,
            n_iter_no_change=10, random_state=args.seed)
        model.fit(data["X_resid"].values[data["train_idx"]],
                   data["y_resid"].values[data["train_idx"]])
        r2 = r2_score(data["y_resid"].values[data["test_idx"]],
                       model.predict(data["X_resid"].values[data["test_idx"]]))
        print(f"  [{pass_label}] Regression R²: {r2:.4f}", flush=True)

        # SHAP interaction values
        print(f"  [{pass_label}] Computing regression SHAP interactions...", flush=True)
        t0 = time.time()
        explainer = shap.TreeExplainer(model)
        X_test_df = pd.DataFrame(data["X_resid"].values[data["test_idx"]],
                                  columns=data["X_resid"].columns)
        interaction_values = explainer.shap_interaction_values(X_test_df)
        dt = time.time() - t0
        print(f"  [{pass_label}] SHAP interactions computed [{dt:.1f}s]", flush=True)

        # Save raw interaction tensor (for bootstrap CI + class-stratified viz)
        acr_class_test = data["acr_class"].values[data["test_idx"]].astype(str)
        acr_ids_test = data["X_resid"].index[data["test_idx"]].values.astype(str)
        np.savez_compressed(
            os.path.join(pass_outdir, "raw_interaction_tensor.npz"),
            interaction_values=interaction_values.astype(np.float32),
            acr_class=acr_class_test,
            acr_ids_test=acr_ids_test,
            feature_names=np.array(list(data["X_resid"].columns)),
        )
        print(f"  [{pass_label}] Saved raw_interaction_tensor.npz "
              f"({interaction_values.nbytes / 1e6:.0f} MB)", flush=True)

        # Save best scale in bp for all signatures (for viz scale hexbin)
        scales = data["scales"]
        bp_rows = [{"signature_id": sid,
                     "scale_idx": best_scale_map.get(sid, 0),
                     "scale_bp": float(scales[best_scale_map.get(sid, 0)])}
                    for sid in data["col_names"]]
        pd.DataFrame(bp_rows).to_csv(
            os.path.join(pass_outdir, "best_scale_bp.tsv"), sep="\t", index=False)

        # Mean interaction matrices (existing outputs)
        abs_matrix = np.mean(np.abs(interaction_values), axis=0)
        signed_matrix = np.mean(interaction_values, axis=0)

        feat_names = list(data["X_resid"].columns)
        dn = dict(zip(sig_meta["signature_id"], sig_meta["display_name"]))
        display_names = [dn.get(f, f) for f in feat_names]

        fam_idx = sig_meta.set_index("signature_id")["primary_family"]

        pd.DataFrame(abs_matrix, index=display_names,
                      columns=display_names).to_csv(
            os.path.join(pass_outdir, "shap_interaction_matrix.tsv"), sep="\t")
        pd.DataFrame(signed_matrix, index=display_names,
                      columns=display_names).to_csv(
            os.path.join(pass_outdir, "shap_interaction_matrix_signed.tsv"), sep="\t")

        # Top interactions table
        rows = []
        for i in range(n_feat):
            for j in range(i + 1, n_feat):
                fam_i = fam_idx.get(feat_names[i], "")
                fam_j = fam_idx.get(feat_names[j], "")
                rows.append({
                    "sig_i": display_names[i], "sig_j": display_names[j],
                    "sig_id_i": feat_names[i], "sig_id_j": feat_names[j],
                    "mean_abs_interaction": abs_matrix[i, j],
                    "mean_signed_interaction": signed_matrix[i, j],
                    "same_family": fam_i == fam_j,
                    "family_i": fam_i, "family_j": fam_j,
                })
        (pd.DataFrame(rows)
         .sort_values("mean_abs_interaction", ascending=False)
         .to_csv(os.path.join(pass_outdir, "top_interactions.tsv"),
                  sep="\t", index=False))

        # Regression results
        pd.DataFrame([{"pass": pass_label, "r2": r2, "n_features": n_feat,
                        "n_signatures_total": len(data["sig_ids"]),
                        "n_signatures_informative": n_informative,
                        "n_acrs": len(data["X_resid"]),
                        "interaction_time_s": dt}]).to_csv(
            os.path.join(pass_outdir, "regression_results.tsv"),
            sep="\t", index=False)

        del interaction_values, explainer, model
        gc.collect()

    else:
        print(f"  [{pass_label}] Phase 1 sentinel found — skipping regression SHAP",
              flush=True)

    # ── Phase 2: Classification model ─────────────────────────────────────────
    clf_sentinel = os.path.join(pass_outdir, "clf_results.tsv")
    if os.path.exists(clf_sentinel) and not args.force:
        print(f"  [{pass_label}] Phase 2 sentinel found — skipping classification",
              flush=True)
        print(f"  [{pass_label}] Done", flush=True)
        return

    print(f"  [{pass_label}] Phase 2: Fitting classification model...", flush=True)

    le = LabelEncoder()
    acr_class = data["acr_class"]
    y_enc = le.fit_transform(acr_class)
    n_classes = len(le.classes_)
    train_idx, test_idx = data["train_idx"], data["test_idx"]

    # Balanced class weights
    counts = np.bincount(y_enc[train_idx])
    sample_weight = np.array([len(train_idx) / (n_classes * counts[c])
                               for c in y_enc[train_idx]])

    clf = HistGradientBoostingClassifier(
        max_depth=5, learning_rate=0.05, max_iter=500,
        min_samples_leaf=20, validation_fraction=0.1,
        n_iter_no_change=10, random_state=args.seed)
    clf.fit(data["X_resid"].values[train_idx], y_enc[train_idx],
             sample_weight=sample_weight)

    pred = clf.predict(data["X_resid"].values[test_idx])
    ba = balanced_accuracy_score(y_enc[test_idx], pred)
    f1 = f1_score(y_enc[test_idx], pred, average="macro")
    cm = confusion_matrix(y_enc[test_idx], pred, normalize="true")
    print(f"  [{pass_label}] Classification BA: {ba:.4f}, F1: {f1:.4f}", flush=True)

    np.savez_compressed(
        os.path.join(pass_outdir, "confusion_matrix_clf.npz"),
        cm=cm, classes=le.classes_)

    pd.DataFrame([{"pass": pass_label, "ba": ba, "f1": f1,
                    "n_features": n_feat, "n_acrs": len(acr_class)}]).to_csv(
        clf_sentinel, sep="\t", index=False)

    print(f"  [{pass_label}] Done", flush=True)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="v3 Step 09: SHAP interactions")
    p.add_argument("--perscale-dir", default="results/v3_06_perscale_fp")
    p.add_argument("--gb-dir", default="results/v3_08_gradient_boosting")
    p.add_argument("--sig-metadata",
                   default="data/motif_signatures/signature_metadata.tsv")
    p.add_argument("--acr-metadata", default="data/acr_metadata.tsv.gz")
    p.add_argument("--acr-coord-mapping",
                   default="data/acr_native_to_resized.tsv")
    p.add_argument("--outdir", default="results/v3_09_shap_interactions")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--force", action="store_true",
                   help="Remove existing sentinels and rerun all phases")
    return p.parse_args()


def main():
    args = parse_args()
    outdir = os.path.join(BASE, args.outdir)
    os.makedirs(outdir, exist_ok=True)

    print("=" * 60, flush=True)
    print("v3_09 — SHAP Interaction Analysis", flush=True)
    print("=" * 60, flush=True)

    sig_meta = load_signature_metadata(os.path.join(BASE, args.sig_metadata))
    acr_meta = load_acr_metadata(os.path.join(BASE, args.acr_metadata))

    # Coordinate mapping: native → resized coords for h5ad lookup
    coord_map = pd.read_csv(os.path.join(BASE, args.acr_coord_mapping), sep="\t")
    coord_map["native_str"] = coord_map["native_str"].str.lower()
    coord_map["resized_str"] = coord_map["resized_str"].str.lower()
    acr_meta["region_str_lower"] = acr_meta["region_str"].str.lower()
    acr_meta = acr_meta.merge(
        coord_map[["native_str", "resized_str"]],
        left_on="region_str_lower", right_on="native_str", how="left")
    acr_meta = acr_meta.set_index("resized_str")

    D = {"sig_meta": sig_meta, "acr_meta": acr_meta}

    for pass_label, acr_subset in [("all", None),
                                    ("changing", ["proto_gain", "leaf_gain"])]:
        print(f"\n{'─'*40}", flush=True)
        print(f"Pass: {pass_label}", flush=True)
        run_pass(D, sig_meta, acr_meta, pass_label, acr_subset, outdir, args)

    print("\n[DONE]", flush=True)


if __name__ == "__main__":
    main()
