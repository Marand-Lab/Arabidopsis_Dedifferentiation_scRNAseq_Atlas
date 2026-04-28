#!/usr/bin/env python3
"""
v4_02c — Run scPrinter bulk footprinting on merged condition BAMs.

Input:  v4/fragments_1based/{condition}_merged.bulk.1based.tsv.gz
Output: v4/3_PRINT/printer_{condition}_merged_bulk.h5ad

One pooled BAM per condition (no replicates). Same FP scoring as v2 step 02:
  - 99 scales (2-100 bp), region_width=None (pre-resized 2000 bp ACRs)
  - No coverage filters (min_num_fragments=0, min_tsse=0)

Usage:
  python v4_02c_run_print.py leaf
  python v4_02c_run_print.py proto
"""

import os
import sys
import re
import gzip
import time
import shutil
import pickle
import gc

import numpy as np
import pandas as pd
import scprinter as scp

# ── HPC hygiene ──────────────────────────────────────────────────────────────
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

# ── Args ─────────────────────────────────────────────────────────────────────
CONDITION = sys.argv[1]  # "leaf" or "proto"
SAMPLE_ID = f"{CONDITION}_merged"

# ── Paths ────────────────────────────────────────────────────────────────────
FRAG_1BASED_DIR = "v4/fragments_1based"
OUT = "v4/3_PRINT"
WL_DIR = os.path.join(OUT, "_wl")
REGIONS_BED = "v4/data/acr_resized_2000bp.bed"
N_JOBS = int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))

os.makedirs(OUT, exist_ok=True)
os.makedirs(WL_DIR, exist_ok=True)

# ── Load genome object ──────────────────────────────────────────────────────
genome_obj_path = "3_PRINT_bulk/At_genome_OBJ"
if not os.path.exists(genome_obj_path):
    raise FileNotFoundError(f"Missing genome object: {genome_obj_path}")

with open(genome_obj_path, "rb") as f:
    genome = pickle.load(f)

genome_keys = list(genome.chrom_sizes.keys())
genome_key_set = set(genome_keys)
print(f"[INFO] genome chrom keys: {genome_keys[:10]}")


# ── Chrom mapping ────────────────────────────────────────────────────────────
def chrom_to_genome_key(ch: str) -> str:
    ch = str(ch)
    if ch in genome_key_set:
        return ch
    base = re.sub(r"^(chr|Chr)", "", ch)
    if base in genome_key_set:
        return base
    uses_lower = any(k.startswith("chr") for k in genome_keys)
    uses_upper = any(k.startswith("Chr") for k in genome_keys)
    if base.isdigit():
        if uses_lower:
            cand = f"chr{base}"
        elif uses_upper:
            cand = f"Chr{base}"
        else:
            cand = base
        if cand in genome_key_set:
            return cand
    org_map = {"M": ["M", "Mt", "MT"], "C": ["C", "Pt", "PT"]}
    for target, aliases in org_map.items():
        if base in aliases:
            if uses_lower:
                cand = f"chr{target}"
            elif uses_upper:
                cand = f"Chr{target}"
            else:
                cand = target
            if cand in genome_key_set:
                return cand
    for prefix in ("chr", "Chr"):
        cand = prefix + base
        if cand in genome_key_set:
            return cand
    raise KeyError(f"Chrom '{ch}' cannot be mapped. Genome keys: {genome_keys[:10]}")


def safe_key(s: str) -> str:
    return s.replace("-", "_").replace(".", "_").replace("/", "_")


def supp_dir_of(printer_path: str) -> str:
    base_dir = os.path.dirname(printer_path)
    base_stem = os.path.splitext(os.path.basename(printer_path))[0]
    return os.path.join(base_dir, f"{base_stem}_supp")


def wait_for(paths, tries=90, sleep=1):
    for _ in range(tries):
        for p in paths:
            if os.path.exists(p) and os.path.getsize(p) > 0:
                return p
        time.sleep(sleep)
    return None


# ── Resolve input fragments ─────────────────────────────────────────────────
bulk_frag = os.path.join(FRAG_1BASED_DIR, f"{SAMPLE_ID}.bulk.1based.tsv.gz")
if not os.path.exists(bulk_frag):
    raise FileNotFoundError(f"Missing 1-based fragments: {bulk_frag}")

print(f"[INFO] condition={CONDITION}")
print(f"[INFO] sample_id={SAMPLE_ID}")
print(f"[INFO] fragments={bulk_frag}")

# Verify chrom keys in bulk_frag match genome (sample check)
with gzip.open(bulk_frag, "rt") as ih:
    for j in range(1000):
        line = ih.readline()
        if not line:
            break
        chrom = line.split("\t", 1)[0]
        if chrom not in genome_key_set:
            raise RuntimeError(
                f"[ERR] bulk_frag chrom '{chrom}' not in genome keys "
                f"(example keys: {genome_keys[:10]})"
            )

# ── Build printer object ────────────────────────────────────────────────────
wl_file = os.path.join(WL_DIR, f"{SAMPLE_ID}.barcodes.txt")
with open(wl_file, "w") as oh:
    oh.write("bulk\n")

out_h5 = os.path.join(OUT, f"printer_{SAMPLE_ID}_bulk.h5ad")
if os.path.exists(out_h5):
    os.remove(out_h5)

print(f"[IMPORT] scPrinter import_fragments for {SAMPLE_ID}")
pr = None
gc.collect()
try:
    pr = scp.pp.import_fragments(
        path_to_frags=[bulk_frag],
        barcodes=[wl_file],
        sample_names=[SAMPLE_ID],
        savename=out_h5,
        genome=genome,
        auto_detect_shift=False,
        plus_shift=4,
        minus_shift=-5,
        min_num_fragments=0,
        min_tsse=0,
        sorted_by_barcode=False,
        n_jobs=1,
    )
finally:
    if pr is not None:
        pr.close()
    gc.collect()

# ── TFBS + FP scoring on ACR regions ────────────────────────────────────────
printer = scp.load_printer(out_h5, genome)
printer.load_disp_model()

cell_ids = list(printer.obs_names)
if "bulk" in cell_ids:
    cell_grouping = [["bulk"]]
else:
    assert len(cell_ids) == 1, f"Expected 1 cell, found {len(cell_ids)}: {cell_ids[:5]}"
    cell_grouping = [[cell_ids[0]]]

group_names = [SAMPLE_ID]

# Load regions
regions = pd.read_csv(
    REGIONS_BED, sep="\t", header=None, usecols=[0, 1, 2],
    names=["Chromosome", "Start", "End"]
)
regions["Chromosome"] = regions["Chromosome"].astype(str).map(chrom_to_genome_key)
regions["Start"] = regions["Start"].astype(np.int64, copy=False)
regions["End"] = regions["End"].astype(np.int64, copy=False)
regions.drop_duplicates(subset=["Chromosome", "Start", "End"], inplace=True)
regions.sort_values(["Chromosome", "Start", "End"], inplace=True, ignore_index=True)
print(f"[INFO] regions={len(regions)} from {REGIONS_BED}")

# TFBS
printer.load_bindingscore_model("TF", scp.datasets.pretrained_TFBS_model)

supp_dir = supp_dir_of(out_h5)
os.makedirs(supp_dir, exist_ok=True)

tf_key = safe_key(f"TFBS_{SAMPLE_ID}_ALL")
print(f"[RUN] TFBS key={tf_key} n_jobs={N_JOBS}")
scp.tl.get_binding_score(
    printer,
    cell_grouping=cell_grouping,
    group_names=group_names,
    regions=regions,
    model_key="TF",
    n_jobs=N_JOBS,
    contextRadius=100,
    region_width=None,
    downsample=5,
    save_key=tf_key,
    backed=True,
    overwrite=True,
)

tf_src = wait_for([
    os.path.join(supp_dir, f"{tf_key}.h5ad"),
    os.path.join(supp_dir, f"{safe_key(tf_key)}.h5ad"),
])
if tf_src is None:
    raise FileNotFoundError(f"TFBS backed output not found for key {tf_key} in {supp_dir}")

# NucBS (nucleosome binding score)
printer.load_bindingscore_model("Nuc", scp.datasets.pretrained_NucBS_model)

nuc_key = safe_key(f"NucBS_{SAMPLE_ID}_ALL")
print(f"[RUN] NucBS key={nuc_key} n_jobs={N_JOBS}")
scp.tl.get_binding_score(
    printer,
    cell_grouping=cell_grouping,
    group_names=group_names,
    regions=regions,
    model_key="Nuc",
    n_jobs=N_JOBS,
    contextRadius=100,
    region_width=None,
    downsample=5,
    save_key=nuc_key,
    backed=True,
    overwrite=True,
)

nuc_src = wait_for([
    os.path.join(supp_dir, f"{nuc_key}.h5ad"),
    os.path.join(supp_dir, f"{safe_key(nuc_key)}.h5ad"),
])
if nuc_src is None:
    raise FileNotFoundError(f"NucBS backed output not found for key {nuc_key} in {supp_dir}")

# FP
fp_key = safe_key(f"FP_{SAMPLE_ID}_ALL")
print(f"[RUN] FP key={fp_key} n_jobs={N_JOBS}")
scp.tl.get_footprint_score(
    printer,
    cell_grouping=cell_grouping,
    group_names=group_names,
    regions=regions,
    modes=np.arange(2, 101),
    n_jobs=N_JOBS,
    region_width=None,
    save_key=fp_key,
    backed=True,
    overwrite=True,
)

fp_src = wait_for([
    os.path.join(supp_dir, f"{fp_key}.h5ad"),
    os.path.join(supp_dir, f"{safe_key(fp_key)}.h5ad"),
])
if fp_src is None:
    raise FileNotFoundError(f"FP backed output not found for key {fp_key} in {supp_dir}")

printer.close()

# ── Copy backed outputs to canonical locations ───────────────────────────────
os.makedirs(os.path.join(OUT, "TFBS"), exist_ok=True)
os.makedirs(os.path.join(OUT, "NucBS"), exist_ok=True)
os.makedirs(os.path.join(OUT, "FP"), exist_ok=True)

tf_dst = os.path.join(OUT, "TFBS", f"{SAMPLE_ID}__ALL.h5ad")
nuc_dst = os.path.join(OUT, "NucBS", f"{SAMPLE_ID}__ALL.h5ad")
fp_dst = os.path.join(OUT, "FP", f"{SAMPLE_ID}__ALL.h5ad")

for src, dst in [(tf_src, tf_dst), (nuc_src, nuc_dst), (fp_src, fp_dst)]:
    tmp = dst + ".tmp"
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)

print(f"[OK] {SAMPLE_ID}: TFBS -> {tf_dst}; NucBS -> {nuc_dst}; FP -> {fp_dst}")
