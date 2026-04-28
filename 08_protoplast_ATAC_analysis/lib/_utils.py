#!/usr/bin/env python3
"""
Shared utilities for TF footprinting analysis pipeline (steps 06-11).

Provides:
  - Chunked loading of per-rep hit data with dominant-band computation
  - Delta prefix detection (absolute vs fractional)
  - Per-motif z-scoring
  - JASPAR family metadata loading (with bZIP remapping)
  - ACR metadata loading (with region_str construction)
  - Step 05 summary loading (all/stable/changing)
  - LMM fitting helpers
  - Color palette and Nature-style figure defaults
  - Statistical helpers (partial Spearman, bootstrap CI, effect sizes)
"""

from __future__ import annotations

import gzip
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from statsmodels.regression.mixed_linear_model import MixedLM


# ── Color palette (colorblind-safe, Nature-style) ────────────────────────────

PALETTE = dict(
    # Conditions
    leaf="#3A7D44",          # forest green
    proto="#D64045",         # vermilion red
    # ACR classes
    proto_gain="#D64045",
    stable="#8C8C8C",
    leaf_gain="#3A7D44",
    # Feature highlights
    wrky="#2166AC",          # strong blue
    lec2="#E8A838",          # warm amber
    other_fam="#BFBFBF",     # light gray (non-highlighted families)
    sig_other="#4A4A4A",     # dark gray (significant non-highlighted)
    # Nucleosome landmarks
    nfr="#FDB863",           # warm orange
    nuc_m1="#B2ABD2",        # light purple
    nuc_p1="#5E4FA2",        # deep purple
)

ACR_CLASS_COLORS = {
    "proto_gain": PALETTE["proto_gain"],
    "stable": PALETTE["stable"],
    "leaf_gain": PALETTE["leaf_gain"],
}

LANDMARK_COLORS = {
    "nfr": PALETTE["nfr"],
    "nuc_m1": PALETTE["nuc_m1"],
    "nuc_p1": PALETTE["nuc_p1"],
}


def nature_figure_defaults():
    """Set matplotlib rcParams for Nature-style figures."""
    import matplotlib.pyplot as plt
    try:
        import seaborn as sns
        sns.set_context("paper", font_scale=1.1)
    except ImportError:
        pass

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,      # editable text in PDF
        "ps.fonttype": 42,
        "svg.fonttype": "none",  # use system fonts in SVG (editable in Illustrator)
    })


def nature_savefig(fig, name, outdir, formats=("pdf", "png", "svg")):
    """Save figure in multiple formats at 300 DPI."""
    from pathlib import Path
    outdir = Path(outdir)
    for fmt in formats:
        path = outdir / f"{name}.{fmt}"
        fig.savefig(str(path), dpi=300, bbox_inches="tight", facecolor="white")
    import matplotlib.pyplot as plt
    plt.close(fig)


# ── Delta prefix detection ──────────────────────────────────────────────────

def detect_delta_prefix(hit_file: str | Path) -> str:
    """Read header to determine 'delta_frac_rep' vs 'delta_rep' prefix."""
    with gzip.open(hit_file, "rt") as f:
        header = f.readline().strip().split("\t")
    if "delta_frac_rep1_band1" in header:
        return "delta_frac_rep"
    return "delta_rep"


# ── Chunked loading ─────────────────────────────────────────────────────────

def load_hits_chunked(
    hit_file: str | Path,
    acr_regions: set[str] | None,
    motif_band_map: pd.DataFrame,
    delta_prefix: str,
    keep_per_rep: bool = False,
    extra_usecols: list[str] | None = None,
    chunksize: int = 500_000,
    motif_filter: set[str] | None = None,
    active_reps: list[int] | tuple[int, ...] = (1, 2, 3),
) -> pd.DataFrame:
    """Chunked load of merged per-rep hits with dominant-band computation.

    Parameters
    ----------
    hit_file : path to merged_motif_hits_fpband_per_rep.tsv.gz
    acr_regions : set of region_str values to keep (None = keep all)
    motif_band_map : DataFrame with columns [motif_id, dominant_band_idx]
    delta_prefix : 'delta_rep' or 'delta_frac_rep'
    keep_per_rep : if True, retain delta_rep{r}_dom columns for active reps
    extra_usecols : additional columns to load (e.g. ['hit_center', 'Chromosome'])
    chunksize : rows per chunk
    motif_filter : optional set of motif_ids to keep
    active_reps : which replicates to use (default all 3)

    Returns
    -------
    DataFrame with columns:
      region_str, motif_id, motif_name, delta_dominant
      + extra_usecols if provided
      + delta_rep{r}_dom for each r in active_reps if keep_per_rep=True
    """
    delta_rep_cols = [
        f"{delta_prefix}{r}_band{b}" for r in active_reps for b in [1, 2, 3]
    ]
    base_cols = ["region_str", "motif_id", "motif_name"]
    usecols = list(set(base_cols + delta_rep_cols + (extra_usecols or [])))

    chunks = []
    n_chunks = 0
    for chunk in pd.read_csv(hit_file, sep="\t", chunksize=chunksize, usecols=usecols):
        n_chunks += 1

        if acr_regions is not None:
            chunk = chunk[chunk["region_str"].isin(acr_regions)]
        if motif_filter is not None:
            chunk = chunk[chunk["motif_id"].isin(motif_filter)]
        if len(chunk) == 0:
            continue

        # Merge dominant band index
        chunk = chunk.merge(
            motif_band_map[["motif_id", "dominant_band_idx"]],
            on="motif_id", how="left",
        )

        dom_bi = chunk["dominant_band_idx"].values

        # Compute per-rep dominant-band deltas
        for bi in [1, 2, 3]:
            mask = dom_bi == bi
            if not mask.any():
                continue
            for r in active_reps:
                col = f"{delta_prefix}{r}_band{bi}"
                if keep_per_rep:
                    # Initialize rep column if not yet present
                    if f"delta_rep{r}_dom" not in chunk.columns:
                        chunk[f"delta_rep{r}_dom"] = np.nan
                    chunk.loc[mask, f"delta_rep{r}_dom"] = chunk.loc[mask, col].values

        # Compute replicate-mean delta at dominant band
        if keep_per_rep:
            rep_dom_cols = [f"delta_rep{r}_dom" for r in active_reps]
            chunk["delta_dominant"] = chunk[rep_dom_cols].mean(axis=1)
        else:
            mean_deltas = np.full(len(chunk), np.nan)
            for bi in [1, 2, 3]:
                mask = dom_bi == bi
                if mask.any():
                    cols = [f"{delta_prefix}{r}_band{bi}" for r in active_reps]
                    vals = chunk.loc[mask, cols].values
                    mean_deltas[mask] = np.nanmean(vals, axis=1)
            chunk["delta_dominant"] = mean_deltas

        # Select output columns
        keep_cols = base_cols + (extra_usecols or []) + ["delta_dominant"]
        if keep_per_rep:
            keep_cols += [f"delta_rep{r}_dom" for r in active_reps]
        chunk = chunk[[c for c in keep_cols if c in chunk.columns]]
        chunk = chunk.dropna(subset=["delta_dominant"])
        chunks.append(chunk)

        if n_chunks % 5 == 0:
            print(f"  Processed {n_chunks} chunks...", flush=True)

    if not chunks:
        return pd.DataFrame()

    hits = pd.concat(chunks, ignore_index=True)
    print(f"  Total hits loaded: {len(hits):,}", flush=True)
    return hits


# ── Z-scoring ───────────────────────────────────────────────────────────────

def zscore_delta(hits: pd.DataFrame) -> pd.DataFrame:
    """Add z_delta column: per-motif z-scored delta_dominant."""
    motif_stats = hits.groupby("motif_id")["delta_dominant"].agg(["mean", "std"])
    motif_stats.columns = ["motif_mean", "motif_std"]
    motif_stats["motif_std"] = motif_stats["motif_std"].replace(0, 1.0)
    hits = hits.merge(motif_stats, on="motif_id", how="left")
    hits["z_delta"] = (hits["delta_dominant"] - hits["motif_mean"]) / hits["motif_std"]
    hits.drop(columns=["motif_mean", "motif_std"], inplace=True)
    return hits


# ── Family metadata ─────────────────────────────────────────────────────────

def load_family_metadata(fam_file: str | Path) -> pd.DataFrame:
    """Load + remap JASPAR family metadata (bZIP groups, class fallbacks)."""
    fam_df = pd.read_csv(fam_file, sep="\t")
    fam_df = fam_df[["motif_id", "motif_name", "tf_class", "tf_family"]].copy()
    fam_df["tf_family"] = fam_df["tf_family"].fillna("").astype(str).str.strip()
    fam_df["tf_class"] = fam_df["tf_class"].fillna("").astype(str).str.strip()

    # bZIP remapping
    bzip_remap = {
        "S": "Group S", "D": "Group D", "group A": "Group A",
        "B": "Group B", "C": "Group C", "G": "Group G",
        "H": "Group H", "I": "Group I", "K": "Group K",
    }
    fam_df["tf_family"] = fam_df["tf_family"].replace(bzip_remap)
    _grp_mask = fam_df["tf_family"].str.startswith("Group ")
    fam_df.loc[_grp_mask, "tf_family"] = "bZIP " + fam_df.loc[_grp_mask, "tf_family"]

    # Class-to-family fallbacks
    class_to_family = {"TCP": "TCP", "CPP": "CPP", "EIL": "EIL", "RWP-RK": "RWP-RK"}
    for idx, row in fam_df.iterrows():
        if row["tf_family"] == "":
            cls = row["tf_class"]
            if cls in class_to_family:
                fam_df.loc[idx, "tf_family"] = class_to_family[cls]
            elif "AP2" in cls:
                fam_df.loc[idx, "tf_family"] = "ERF/DREB"
            elif "bHLH" in cls:
                fam_df.loc[idx, "tf_family"] = "bHLH"
            elif "bZIP" in cls:
                fam_df.loc[idx, "tf_family"] = "bZIP"
            elif "C2H2" in cls:
                fam_df.loc[idx, "tf_family"] = "C2H2"
            elif "HC3" in cls:
                fam_df.loc[idx, "tf_family"] = "HC3"
    fam_df["tf_family"] = fam_df["tf_family"].replace("", "Unknown")
    return fam_df


# ── ACR metadata ────────────────────────────────────────────────────────────

def load_acr_metadata(acr_meta_file: str | Path) -> pd.DataFrame:
    """Load ACR metadata with region_str construction."""
    with gzip.open(acr_meta_file, "rt") as f:
        acr_meta = pd.read_csv(f, sep="\t")
    acr_meta["region_str"] = (
        acr_meta["chr"].astype(str).str.lower() + ":"
        + acr_meta["start"].astype(str) + "-"
        + acr_meta["end"].astype(str)
    )
    return acr_meta


# ── Step 05 summaries ───────────────────────────────────────────────────────

def load_step05_summaries(
    results_dir: str | Path,
) -> dict[str, pd.DataFrame]:
    """Load all/stable/changing motif summaries from step 05.

    Returns dict with keys 'all', 'stable', 'changing'.
    Missing files are silently skipped.
    """
    results_dir = Path(results_dir)
    files = {
        "all": results_dir / "motif_diff_summary_replicates.tsv.gz",
        "stable": results_dir / "motif_diff_summary_replicates_stable.tsv.gz",
        "changing": results_dir / "motif_diff_summary_replicates_changing.tsv.gz",
    }
    summaries = {}
    for key, path in files.items():
        if path.exists():
            with gzip.open(path, "rt") as f:
                df = pd.read_csv(f, sep="\t")
            df["abs_mean_delta"] = df["dominant_mean_delta"].abs()
            df["significant"] = df["dominant_perm_q"] < 0.1
            summaries[key] = df
            n_sig = df["significant"].sum()
            print(f"  Loaded {key} summary: {len(df)} motifs, {n_sig} significant", flush=True)
        else:
            print(f"  [WARN] {key} summary not found: {path}", flush=True)
    return summaries


def load_step05_byband(
    results_dir: str | Path,
) -> dict[str, pd.DataFrame]:
    """Load all/stable/changing by-band results from step 05."""
    results_dir = Path(results_dir)
    files = {
        "all": results_dir / "motif_diff_replicates.tsv.gz",
        "stable": results_dir / "motif_diff_replicates_stable.tsv.gz",
        "changing": results_dir / "motif_diff_replicates_changing.tsv.gz",
    }
    bybands = {}
    for key, path in files.items():
        if path.exists():
            with gzip.open(path, "rt") as f:
                bybands[key] = pd.read_csv(f, sep="\t")
    return bybands


# ── LMM helpers ─────────────────────────────────────────────────────────────

def fit_lmm_intercept(values: np.ndarray, groups: np.ndarray):
    """Fit intercept-only LMM with random intercept for group.

    Returns (t_stat, p_value) or (NaN, NaN) on failure.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            exog = np.ones((len(values), 1))
            model = MixedLM(values, exog, groups=groups)
            result = model.fit(reml=True, maxiter=200)
            return float(result.tvalues[0]), float(result.pvalues[0])
    except Exception:
        return np.nan, np.nan


def fit_lmm_one_covariate(
    values: np.ndarray,
    covariate: np.ndarray,
    groups: np.ndarray,
):
    """Fit LMM: values ~ 1 + covariate + (1|group).

    Returns (coef, t_stat, p_value) or (NaN, NaN, NaN) on failure.
    The covariate effect is index 1 in the fixed-effects.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            exog = np.column_stack([np.ones(len(values)), covariate])
            model = MixedLM(values, exog, groups=groups)
            result = model.fit(reml=True, maxiter=200)
            coef = float(result.fe_params[1])
            t_stat = float(result.tvalues[1])
            p_val = float(result.pvalues[1])
            return coef, t_stat, p_val
    except Exception:
        return np.nan, np.nan, np.nan


# ── BH-FDR ──────────────────────────────────────────────────────────────────

def fdr_bh(pvalues):
    """Benjamini-Hochberg FDR correction."""
    pvals = np.asarray(pvalues, dtype=float)
    n = len(pvals)
    if n == 0:
        return pvals.copy()
    order = np.argsort(pvals)
    ranked = np.empty(n)
    ranked[order] = np.arange(1, n + 1)
    padj = pvals * n / ranked
    padj_sorted = padj[np.argsort(ranked)[::-1]]
    np.minimum.accumulate(padj_sorted, out=padj_sorted)
    padj[np.argsort(ranked)[::-1]] = padj_sorted
    return np.clip(padj, 0, 1)


# ── Motif band map construction ─────────────────────────────────────────────

BAND_COL_MAP = {"<20": 1, "20-50": 2, ">50": 3}


def build_motif_band_map(summary: pd.DataFrame) -> pd.DataFrame:
    """Build motif_id → dominant_band_idx mapping from step 05 summary."""
    motif_band = summary[["motif_id", "dominant_band_label"]].copy()
    motif_band["dominant_band_idx"] = motif_band["dominant_band_label"].map(BAND_COL_MAP)
    return motif_band


# ── Statistical helpers ──────────────────────────────────────────────────────

def partial_spearman(x, y, covariates):
    """Partial Spearman correlation via rank residuals.

    Regress rank(x) and rank(y) on rank(covariates), then correlate residuals.

    Parameters
    ----------
    x, y : 1D arrays (same length)
    covariates : 2D array (n_obs, n_covariates) or 1D array

    Returns
    -------
    (rho_partial, p_value)
    """
    from scipy.stats import rankdata, spearmanr

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    covariates = np.asarray(covariates, dtype=float)
    if covariates.ndim == 1:
        covariates = covariates.reshape(-1, 1)

    # Drop rows with NaN in any input
    mask = np.isfinite(x) & np.isfinite(y) & np.all(np.isfinite(covariates), axis=1)
    if mask.sum() < 10:
        return np.nan, np.nan
    x, y, covariates = x[mask], y[mask], covariates[mask]

    # Rank everything
    rx = rankdata(x)
    ry = rankdata(y)
    rc = np.column_stack([rankdata(covariates[:, j]) for j in range(covariates.shape[1])])

    # OLS residuals: rank(x) ~ rank(covariates), same for y
    rc_aug = np.column_stack([np.ones(len(rx)), rc])
    try:
        beta_x = np.linalg.lstsq(rc_aug, rx, rcond=None)[0]
        beta_y = np.linalg.lstsq(rc_aug, ry, rcond=None)[0]
    except np.linalg.LinAlgError:
        return np.nan, np.nan

    resid_x = rx - rc_aug @ beta_x
    resid_y = ry - rc_aug @ beta_y

    rho, p = spearmanr(resid_x, resid_y)
    return float(rho), float(p)


def bootstrap_ci_spearman(x, y, n_boot=10000, ci=0.95, seed=42):
    """BCa bootstrap confidence interval on Spearman rho.

    Returns (rho, ci_low, ci_high).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 10:
        return np.nan, np.nan, np.nan
    x, y = x[mask], y[mask]
    n = len(x)

    rho_obs = float(sp_stats.spearmanr(x, y).statistic)

    rng = np.random.RandomState(seed)
    boot_rhos = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.randint(0, n, size=n)
        boot_rhos[b] = sp_stats.spearmanr(x[idx], y[idx]).statistic

    alpha = 1.0 - ci
    lo = np.percentile(boot_rhos, 100 * alpha / 2)
    hi = np.percentile(boot_rhos, 100 * (1 - alpha / 2))
    return rho_obs, float(lo), float(hi)


def steiger_test(r12, r13, r23, n):
    """Steiger's test for equality of two dependent correlations.

    Tests H0: rho(X,Y) = rho(X,Z) where Y and Z are measured on the same
    sample of size n, and r23 = cor(Y,Z).

    Returns (z_statistic, p_value_two_sided).
    """
    if n < 4 or any(not np.isfinite(v) for v in [r12, r13, r23]):
        return np.nan, np.nan

    # Steiger (1980) formula
    r_bar = (r12 + r13) / 2.0
    det = 1 - r12**2 - r13**2 - r23**2 + 2 * r12 * r13 * r23
    det = max(det, 1e-12)
    denom = (1 - r23) * (1 + r23)
    if denom < 1e-12:
        return np.nan, np.nan

    # Williams' modification
    f = (1 - r23) / (2 * det / (n - 1))
    f = max(f, 1e-12)

    z = (r12 - r13) * np.sqrt((n - 1) * (1 + r23)) / np.sqrt(
        2 * det * (n - 1) / (n - 3) + (r_bar**2) * (1 - r23)**3 / 4
    )
    # Fallback: simpler Steiger formula if numerical issues
    if not np.isfinite(z):
        z = (r12 - r13) * np.sqrt((n - 3) * (1 + r23) / (
            2 * (1 - r23**2 - r12**2 - r13**2 + 2 * r12 * r13 * r23) + 1e-12
        ))

    if not np.isfinite(z):
        return np.nan, np.nan

    p = 2 * sp_stats.norm.sf(abs(z))
    return float(z), float(p)


def rank_biserial_r(group1, group2):
    """Rank-biserial correlation (effect size for Mann-Whitney U).

    r = 1 - 2U/(n1*n2), where U is from mannwhitneyu.
    Returns (r, U, p_two_sided).
    """
    g1 = np.asarray(group1, dtype=float)
    g2 = np.asarray(group2, dtype=float)
    g1 = g1[np.isfinite(g1)]
    g2 = g2[np.isfinite(g2)]
    if len(g1) < 3 or len(g2) < 3:
        return np.nan, np.nan, np.nan
    U, p = sp_stats.mannwhitneyu(g1, g2, alternative="two-sided")
    n1, n2 = len(g1), len(g2)
    r = 1.0 - 2.0 * U / (n1 * n2)
    return float(r), float(U), float(p)


def cohens_d_one_sample(values, mu0=0.0):
    """One-sample Cohen's d: (mean - mu0) / std."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return np.nan
    s = np.std(values, ddof=1)
    if s < 1e-12:
        return np.nan
    return float((np.mean(values) - mu0) / s)


# ── OLS residualization (shared by v3_07, v3_08) ────────────────────────────

def _build_design_matrix(acr_meta, index):
    """Build confounder design matrix C for OLS.

    Confounders: log_width (or log1p(acr_width)), logCPM, genomic_context
    (one-hot encoded, drop_first=True).
    """
    cols = []
    if "log_width" in acr_meta.columns:
        cols.append("log_width")
    elif "acr_width" in acr_meta.columns:
        acr_meta = acr_meta.copy()
        acr_meta["log_width"] = np.log1p(acr_meta["acr_width"])
        cols.append("log_width")
    if "logCPM" in acr_meta.columns:
        cols.append("logCPM")
    if "genomic_context" in acr_meta.columns:
        dummies = pd.get_dummies(acr_meta["genomic_context"], prefix="gc",
                                 drop_first=True)
        acr_meta = pd.concat([acr_meta, dummies], axis=1)
        cols.extend(dummies.columns.tolist())

    C = acr_meta.loc[index, cols].astype(float)
    valid = C.notna().all(axis=1)
    return C.loc[valid], valid


def residualize_features(tf_features, acr_meta):
    """OLS-residualize each column of tf_features on confounders.

    Parameters
    ----------
    tf_features : DataFrame
        Columns = features, index = ACR IDs (must overlap acr_meta index).
    acr_meta : DataFrame
        Must contain log_width/acr_width, logCPM, genomic_context columns.

    Returns
    -------
    DataFrame (float32) with same shape, confounder effects removed per column.
    """
    from numpy.linalg import lstsq

    common = tf_features.index.intersection(acr_meta.index)
    C, valid = _build_design_matrix(acr_meta, common)
    common = C.index

    X = tf_features.loc[common]
    resid = X.copy()
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
    """OLS-residualize a response variable (e.g., logFC) on confounders.

    Returns
    -------
    (y_resid: Series, r2_conf: float)
        Residualized response and confounder R².
    """
    from numpy.linalg import lstsq

    C, valid = _build_design_matrix(acr_meta, y.index)
    common = C.index.intersection(y.index)
    C = C.loc[common]
    y = y.loc[common]

    C_arr = np.column_stack([np.ones(len(C)), C.values])
    beta, _, _, _ = lstsq(C_arr, y.values, rcond=None)
    pred = C_arr @ beta
    ss_res = np.sum((y.values - pred) ** 2)
    ss_tot = np.sum((y.values - y.mean()) ** 2)
    r2_conf = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    y_resid = pd.Series(y.values - pred, index=common)
    return y_resid, r2_conf
