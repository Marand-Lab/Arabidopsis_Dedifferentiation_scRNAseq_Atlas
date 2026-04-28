#!/usr/bin/env python3
"""
v3 Step 04: MOODS motif scanning with collapsed Arabidopsis signatures.

Scans native ACR regions for motif matches using 115 non-redundant
Arabidopsis motif signatures (collapsed from 465 JASPAR2024 PFMs via
motifStack clustering; see data/motif_signatures/reference_motif_cleaning.R).

Input:
  - Genome FASTA
  - BED file with ACR regions (native coordinates)
  - MEME-format signature PFMs (data/motif_signatures/At_Motif_SignatureDB.meme)

Output per chunk:
  - motif_hits.tsv.gz  (region_str, motif_id, hit_center, strand, score, ...)

Adapted from v2 04a_motif_scan_only.py — same MOODS scanning logic,
new MEME parser replaces JASPAR ZIP parser.
"""

from __future__ import annotations

import os
import warnings

warnings.filterwarnings("ignore", message=r".*pkg_resources is deprecated.*",
                        category=UserWarning)
os.environ.setdefault("PYTHONWARNINGS",
                      "ignore:pkg_resources is deprecated as an API:UserWarning")

import argparse
import gzip
import re
import time
from multiprocessing import get_context
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import MOODS.scan
import MOODS.tools


# ── Helpers ──────────────────────────────────────────────────────────────────
_RC = str.maketrans(
    {"A": "T", "C": "G", "G": "C", "T": "A", "N": "N",
     "a": "t", "c": "g", "g": "c", "t": "a", "n": "n"}
)


def revcomp_dna(seq: str) -> str:
    return seq.translate(_RC)[::-1]


def write_tsv_gz(df: pd.DataFrame, path: str | Path) -> None:
    with gzip.open(str(path), "wt") as f:
        df.to_csv(f, sep="\t", index=False)


def read_regions_bed(bed_path: str) -> pd.DataFrame:
    df = pd.read_csv(bed_path, sep="\t", header=None, comment="#")
    cols = ["Chromosome", "Start", "End"]
    if df.shape[1] >= 4:
        cols.append("Label")
        df = df.iloc[:, :4]
    else:
        df = df.iloc[:, :3]
        df["Label"] = "NA"
        cols.append("Label")
    df.columns = cols
    df["Chromosome"] = (df["Chromosome"].astype(str)
                        .str.replace(r"^Chr", "chr", regex=True))
    df["Start"] = df["Start"].astype(np.int64, copy=False)
    df["End"] = df["End"].astype(np.int64, copy=False)
    df["region_str"] = df.apply(
        lambda r: f"{r['Chromosome']}:{int(r['Start'])}-{int(r['End'])}",
        axis=1,
    )
    return df


# ── MEME motif parser ───────────────────────────────────────────────────────
class Motif:
    __slots__ = ("motif_id", "name", "matrix")

    def __init__(self, motif_id: str, name: str, matrix: np.ndarray):
        self.motif_id = motif_id
        self.name = name
        self.matrix = matrix  # shape (4, width) — rows = A, C, G, T


def _load_display_names(metadata_path: str) -> Dict[str, str]:
    """Load signature_id -> display_name mapping from signature_metadata.tsv."""
    meta = pd.read_csv(metadata_path, sep="\t", usecols=["signature_id", "display_name"])
    return dict(zip(meta["signature_id"], meta["display_name"]))


def load_meme_pfms(meme_path: str,
                   metadata_path: str | None = None) -> Dict[str, Motif]:
    """Parse a MEME-format file into {motif_id: Motif} dict.

    The MEME MOTIF line format produced by universalmotif / motifStack is:
        MOTIF <name>
        letter-probability matrix: alength= 4 w= <width> nsites= ...
        <probability rows, one per position (transposed relative to JASPAR)>

    If metadata_path is provided, uses display_name from signature_metadata.tsv
    (e.g., "WRKY_WRKY15") instead of parsing names from the MEME file.
    """
    # Load display names if metadata available
    display_names: Dict[str, str] = {}
    if metadata_path and Path(metadata_path).exists():
        display_names = _load_display_names(metadata_path)

    motifs: Dict[str, Motif] = {}
    sig_counter = 0

    with open(meme_path) as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if line.startswith("MOTIF "):
            full_name = line[6:].strip()
            sig_counter += 1
            sig_id = f"sig_{sig_counter:03d}"

            # Look for letter-probability matrix header
            i += 1
            while i < len(lines) and not lines[i].strip().startswith(
                    "letter-probability"):
                i += 1
            if i >= len(lines):
                break

            # Parse width from header
            header = lines[i].strip()
            m = re.search(r"w=\s*(\d+)", header)
            if not m:
                i += 1
                continue
            width = int(m.group(1))

            # Read probability rows (width rows, 4 columns each)
            rows: List[List[float]] = []
            i += 1
            for _ in range(width):
                if i >= len(lines):
                    break
                vals = lines[i].strip().split()
                if len(vals) >= 4:
                    rows.append([float(v) for v in vals[:4]])
                i += 1

            if len(rows) != width:
                continue

            # rows is (width x 4), transpose to (4 x width) for MOODS
            mat = np.array(rows, dtype=float).T

            # Convert probabilities to pseudo-counts (MOODS expects count-like)
            # Multiply by nsites or use 100 as default
            nsites_match = re.search(r"nsites=\s*(\d+)", header)
            nsites = int(nsites_match.group(1)) if nsites_match else 100
            mat_counts = mat * nsites

            # Use display_name from metadata if available, else parse from MEME
            if sig_id in display_names:
                short_name = display_names[sig_id]
            else:
                members = full_name.split(";")
                first_parts = members[0].split("_", 2)
                short_name = first_parts[2] if len(first_parts) >= 3 else members[0]

            motifs[sig_id] = Motif(
                motif_id=sig_id,
                name=short_name,
                matrix=mat_counts,
            )
        else:
            i += 1

    if not motifs:
        raise ValueError(f"No motifs parsed from MEME file: {meme_path}")

    print(f"[INFO] Loaded {len(motifs)} motif signatures from {meme_path}",
          flush=True)
    return motifs


# ── MOODS scanners (one per motif) ───────────────────────────────────────────
def _match_pos_score(m):
    if hasattr(m, "pos") and hasattr(m, "score"):
        return int(m.pos), float(m.score)
    if hasattr(m, "position") and hasattr(m, "score"):
        return int(m.position), float(m.score)
    try:
        return int(m[0]), float(m[1])
    except Exception as e:
        raise TypeError(f"Unknown MOODS match object: {type(m)}") from e


def build_moods_scanners(motifs, pseudocount, pvalue):
    """Build one MOODS scanner per motif to avoid cross-motif max_len
    contamination that silently inflates thresholds for shorter motifs."""
    t0 = time.time()
    bg = [0.25, 0.25, 0.25, 0.25]
    scanners = []
    motif_id_batches = []
    for mid in motifs:
        pfm = motifs[mid].matrix
        pwm = MOODS.tools.log_odds(pfm, bg, pseudocount)
        thr = MOODS.tools.threshold_from_p(pwm, bg, pvalue)
        scanner = MOODS.scan.Scanner(7)
        scanner.set_motifs([pwm], bg, [thr])
        scanners.append(scanner)
        motif_id_batches.append([mid])
    print(f"[INFO] scanners built | motifs={len(motifs)} "
          f"dt={time.time() - t0:.1f}s", flush=True)
    return scanners, motif_id_batches


# ── Worker globals ───────────────────────────────────────────────────────────
_G_FA = None
_G_MOTIFS = None
_G_SCANNERS = None
_G_MOTIF_ID_BATCHES = None


def _init_scanners(meme_path, metadata_path, pseudocount, pvalue):
    """Load MEME + build MOODS scanners in the PARENT process (called once).

    Forked workers inherit these globals via copy-on-write, so scanners are
    built only once regardless of n_jobs.  This avoids the OOM death-loop
    where n_jobs concurrent spawn-workers each independently rebuild all 122
    scanners and exceed the cgroup memory limit.
    """
    global _G_MOTIFS, _G_SCANNERS, _G_MOTIF_ID_BATCHES
    _G_MOTIFS = load_meme_pfms(meme_path, metadata_path=metadata_path)
    _G_SCANNERS, _G_MOTIF_ID_BATCHES = build_moods_scanners(
        _G_MOTIFS, pseudocount, pvalue)


def _init_worker_fasta(fasta_path):
    """Open a per-worker Fasta handle after fork.

    File handles must not be shared across forked processes, so each worker
    opens its own.  Called as the pool initializer with fork context.
    """
    from pyfaidx import Fasta
    global _G_FA
    _G_FA = Fasta(fasta_path, as_raw=True, sequence_always_upper=True)


def _scan_chunk_worker(chunk_df):
    global _G_FA, _G_MOTIFS, _G_SCANNERS, _G_MOTIF_ID_BATCHES
    rows = []
    for r in chunk_df.itertuples(index=False):
        chrom, start0, end0, label, region_str = (
            r.Chromosome, int(r.Start), int(r.End), r.Label, r.region_str)
        seq = str(_G_FA[chrom][start0:end0]).upper()
        seq_rc = revcomp_dna(seq)
        L = len(seq)
        for scanner, batch_ids in zip(_G_SCANNERS, _G_MOTIF_ID_BATCHES):
            hits_fwd = scanner.scan(seq)
            hits_rev = scanner.scan(seq_rc)
            for mi, mhits in enumerate(hits_fwd):
                mid = batch_ids[mi]
                mlen = _G_MOTIFS[mid].matrix.shape[1]
                mname = _G_MOTIFS[mid].name
                for m in mhits:
                    pos, score = _match_pos_score(m)
                    hstart = start0 + int(pos)
                    hend = hstart + mlen
                    hcenter = hstart + (mlen // 2)
                    rows.append((
                        region_str, label, chrom, start0, end0,
                        mid, mname, hstart, hend, hcenter, "+",
                        float(score),
                    ))
            for mi, mhits in enumerate(hits_rev):
                mid = batch_ids[mi]
                mlen = _G_MOTIFS[mid].matrix.shape[1]
                mname = _G_MOTIFS[mid].name
                for m in mhits:
                    pos, score = _match_pos_score(m)
                    fpos = L - int(pos) - mlen
                    hstart = start0 + fpos
                    hend = hstart + mlen
                    hcenter = hstart + (mlen // 2)
                    rows.append((
                        region_str, label, chrom, start0, end0,
                        mid, mname, hstart, hend, hcenter, "-",
                        float(score),
                    ))
    return pd.DataFrame(rows, columns=[
        "region_str", "Label", "Chromosome", "Start", "End",
        "motif_id", "motif_name", "hit_start", "hit_end", "hit_center",
        "strand", "score",
    ])


def scan_regions_parallel(regions, fasta_path, meme_path, metadata_path,
                          pseudocount, pvalue, n_jobs=8, chunk_size=1000):
    empty_cols = [
        "region_str", "Label", "Chromosome", "Start", "End",
        "motif_id", "motif_name", "hit_start", "hit_end",
        "hit_center", "strand", "score",
    ]
    if regions.empty:
        return pd.DataFrame(columns=empty_cols)

    # Build MOODS scanners ONCE in the parent process.
    # With fork context, workers inherit these globals via copy-on-write,
    # avoiding the OOM caused by n_jobs concurrent spawn-workers each
    # independently rebuilding all 122 scanners simultaneously.
    _init_scanners(meme_path, metadata_path, pseudocount, pvalue)

    if n_jobs <= 1:
        _init_worker_fasta(fasta_path)
        return _scan_chunk_worker(regions.copy()).reset_index(drop=True)

    chunks = [regions.iloc[i : i + chunk_size].copy()
              for i in range(0, len(regions), chunk_size)]
    ctx = get_context("fork")   # inherit pre-built scanner globals via CoW
    dfs = []
    t0 = time.time()
    with ctx.Pool(
        processes=n_jobs,
        initializer=_init_worker_fasta,   # only opens a fresh Fasta per worker
        initargs=(fasta_path,),
    ) as pool:
        for i, df_chunk in enumerate(
            pool.imap(_scan_chunk_worker, chunks, chunksize=1), 1
        ):
            dfs.append(df_chunk)
            if i % 10 == 0 or i == len(chunks):
                print(f"[INFO] chunk {i}/{len(chunks)} "
                      f"dt={time.time() - t0:.1f}s", flush=True)
    return (pd.concat(dfs, ignore_index=True) if dfs
            else pd.DataFrame(columns=empty_cols))


# ── CLI ──────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="v3 Step 04: MOODS scanning with Arabidopsis motif signatures",
    )
    p.add_argument("--fasta", required=True,
                   help="Genome FASTA (indexed)")
    p.add_argument("--bed-in", required=True,
                   help="BED file with ACR regions (native coordinates)")
    p.add_argument("--outdir", required=True,
                   help="Output directory for motif_hits.tsv.gz")
    p.add_argument("--meme",
                   default="data/motif_signatures/At_Motif_SignatureDB.meme",
                   help="MEME-format signature PFMs")
    p.add_argument("--metadata",
                   default="data/motif_signatures/signature_metadata.tsv",
                   help="Signature metadata TSV with display_name column")
    p.add_argument("--pseudocount", type=float, default=1e-4)
    p.add_argument("--pvalue", type=float, default=5e-5)
    p.add_argument("--n-jobs", type=int,
                   default=int(os.environ.get("SLURM_CPUS_PER_TASK", "2")))
    p.add_argument("--chunk-size", type=int, default=1000,
                   help="Regions per parallel worker chunk")
    return p.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    regions = read_regions_bed(args.bed_in)
    print(f"[INFO] regions={len(regions)} from {args.bed_in}", flush=True)

    hits = scan_regions_parallel(
        regions=regions,
        fasta_path=args.fasta,
        meme_path=args.meme,
        metadata_path=args.metadata,
        pseudocount=args.pseudocount,
        pvalue=args.pvalue,
        n_jobs=args.n_jobs,
        chunk_size=args.chunk_size,
    )
    hits = hits.reset_index(drop=True)
    hits["hit_id"] = np.arange(hits.shape[0], dtype=np.int64)

    print(f"[INFO] total hits: {hits.shape[0]:,}", flush=True)
    hits_path = outdir / "motif_hits.tsv.gz"
    write_tsv_gz(hits, hits_path)
    print(f"[DONE] {hits_path}", flush=True)


if __name__ == "__main__":
    main()
