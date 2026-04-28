# Final Pipeline: Differential TF Footprinting in *Arabidopsis thaliana* Leaf vs Protoplast

Self-contained script collection for the complete analysis. All scripts are
**copies** — originals remain in their source directories (`v4/`, `v3/`, `data/motif_signatures/`).

## How to run

All scripts assume **cwd = project root** (`5_TF_FP/`).

```bash
# SLURM steps: sbatch from project root
cd /nfs/turbo/lsa-YOURNAME/fabio_home/Projects/13_Arbidopsis_protoplast/5_TF_FP
sbatch final/02_library_equalization/02a_subsample_bams.sh

# Local steps: run from project root
python -u final/00_metadata/00a_build_acr_metadata.py
```

Data paths (`v4/`, `data/`, `1_ACRs/`, `3_PRINT_per_rep/`, etc.) reference their
original locations relative to the project root.

---

## Pipeline execution order

### Step 00 — Metadata (local)

| Script | Description |
|--------|-------------|
| `00_metadata/00a_build_acr_metadata.py` | ACR metadata: edgeR + genomic context. Output: `v4/data/acr_metadata.tsv.gz` |
| `00_metadata/00b_extract_library_sizes.R` | Per-replicate library sizes from edgeR counts. Output: `data/library_sizes.tsv` |
| `00_metadata/00c_validate_edgeR_2rep.R` | Validates 2-rep vs 3-rep edgeR concordance |

### Step 01 — Motif signatures (SLURM)

| Script | Description |
|--------|-------------|
| `01_motif_signatures/01a_motif_clustering.R` (+`.sh`) | Per-family JASPAR motif clustering. Output: `data/motif_signatures/At_Motif_SignatureDB.meme` |
| `01_motif_signatures/01b_build_signature_metadata.py` | Signature-to-family mapping. Output: `data/motif_signatures/signature_metadata.tsv` |

### Step 02 — Library equalization (SLURM)

| Script | Description |
|--------|-------------|
| `02_library_equalization/02a_subsample_bams.sh` | Subsample 4 BAMs (reps 1+2) to equal depth |
| `02_library_equalization/02b_merge_conditions.sh` | Merge subsampled reps per condition |
| `02_library_equalization/02c_prep_fragments.sh` | BAM to 1-based fragment files for scPrinter |

Chain: `02a` -> `02b` -> `02c`

### Step 03 — Footprinting (SLURM)

| Script | Description |
|--------|-------------|
| `03_footprinting/03a_run_print.py` (+`.sh`) | scPrinter: TFBS + NucBS + FP scoring (array 0-1) |
| `03_footprinting/03b_qc_scale_bias.py` (+`.sh`) | Scale-resolved condition bias QC at null loci |

### Step 04 — Motif scanning (SLURM)

| Script | Description |
|--------|-------------|
| `04_motif_scanning/04a_motif_scan.py` (+`.sh`) | MOODS scan with 122 signatures. Array x50 chunks |

Output: `data/v3_chunks/chunk_NN/motif_hits.tsv.gz`

### Step 05 — Per-scale FP extraction (SLURM)

| Script | Description |
|--------|-------------|
| `05_perscale_fp/05a_extract_fp.py` (+`.sh`, `_merge.sh`) | Per-hit FP band extraction. 4-phase chain via `05a_submit_all.sh` |
| `05_perscale_fp/05b_perscale_fp.py` (+`.sh`, `_merge.sh`) | Per-scale FP (99 scales). Array x50 via `05b_submit_all.sh` |

Output: `results/v3_06_perscale_fp/delta_acr_signature_scale.npz`

### Step 06 — Binding score analysis (SLURM + local)

| Script | Description |
|--------|-------------|
| `06_binding_scores/06a_extract_binding_scores.py` (+`.sh`) | Extract TFBS/NucBS from h5ad (per condition) |
| `06_binding_scores/06b_bs_fp_correlation.py` (+`.sh`) | Correlate binding scores with FP scales |
| `06_binding_scores/06c_binding_overlap.py` (+`.sh`) | Venn overlap: bound/occupied tiles leaf vs proto |
| `06_binding_scores/06c_acr_tile_table.py` | Per-ACR tile classification table |
| `06_binding_scores/06d_binding_deltas.py` (+`.sh`) | FP deltas at bound/occupied positions |
| `06_binding_scores/06d_replot.py` | Local replot from cached NPZ |
| `06_binding_scores/06e_family_enrichment.py` | Fisher's exact test: TF family enrichment at significant-delta tiles |

Chain: `06a` -> `06b`, `06c` -> `06d` -> `06e`

### Step 07 — Visualization (SLURM extract + local replot)

| Script | Description |
|--------|-------------|
| `07_visualization/07a_plot_regions.py` (+`.sh`, `_batch.sh`) | Per-ACR multiscale FP viewer (2-phase) |
| `07_visualization/07b_plot_regions_families.py` (+`.sh`) | Per-ACR family-annotated viewer (2-phase) |
| `07_visualization/07c_replot_wrky_examples.py` | WRKY-specific replotting utility |

---

## Dependency chain

```
Step 00 (metadata)
Step 01 (signatures)
  |
  v
Step 02 (subsample -> merge -> fragments) -> Step 03 (scPrinter -> QC)
  |                                              |
  v                                              v
Step 04 (motif scan) -----> Step 05 (FP extraction)
                                |
                                v
                     Step 06 (binding scores -> overlap -> deltas -> enrichment)
                                |
                                v
                     Step 07 (region viewers)
```

---

## Shared utilities (`lib/`)

| File | Source | Used by |
|------|--------|---------|
| `_utils.py` | `_utils.py` (root) | Archive scripts (v3_07-09) |
| `_tile_utils.py` | `v4/_tile_utils.py` | Steps 06b-06e, 07b (native ACR tile masking) |
| `bam_to_fragment.py` | `v1/2_00_bam_to_fragment.py` | Step 02c (fragment extraction) |

---

## Archive (`archive/`)

V3 gradient boosting + SHAP pipeline — kept for reference in case reviewers request.
These scripts have **unmodified paths** and would need adjustment to run from `final/`.

| Script | Description |
|--------|-------------|
| `v3_07_top_signatures.py` (+`.sh`) | Top-N signature intro figures (6 figures A-F) |
| `v3_08_gradient_boosting.py` (+`.sh`) | 3-tier GB + SHAP (R^2=0.60 all, 0.81 changing) |
| `v3_08b_wrky_summary.py` (+`.sh`) | 6-page WRKY detective figure |
| `v3_08c_wrky_nucleosome_extract.py` (+`.sh`) | Spatial nucleosome profiles around WRKY hits |
| `v3_08c_plot.py` | Nucleosome profile plots |
| `v3_09_shap_interactions.py` (+`.sh`) | 122x122 SHAP pairwise interactions |
| `v3_09_viz.py` (+`.sh`) | Interaction visualization (6 figure types) |
| `v3_10_region_viewer.py` (+`.sh`) | v3 multiscale FP locus viewer |

---

## Sign conventions

| Metric | Positive | Negative |
|--------|----------|----------|
| TF FP delta (leaf - proto) | Leaf-enriched footprint | Proto-enriched footprint |
| edgeR logFC (log2 proto/leaf) | More accessible in protoplast | More accessible in leaf |
| SHAP value | Pushes toward proto-enriched | Pushes toward leaf-enriched |

## ACR classification

- **proto_gain**: fdr < 0.05 and logFC > 0
- **leaf_gain**: fdr < 0.05 and logFC < 0
- **stable**: fdr >= 0.05

## Replicate 3

Excluded from all analyses due to confirmed label swap. See `REPs_README.md`.
