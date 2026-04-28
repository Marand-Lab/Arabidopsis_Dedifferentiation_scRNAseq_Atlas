# Final Pipeline — Differential TF Footprinting in *Arabidopsis thaliana*

## Big picture

Differential TF footprinting between intact leaf tissue and protoplasts using bulk ATAC-seq. The pipeline uses scPrinter for multi-scale footprint (FP) scoring across 99 protection scales (2--100 bp), plus pretrained TF binding score (TFBS) and nucleosome binding score (NucBS) models. TF binding sites are identified by MOODS scanning with 122 non-redundant Arabidopsis motif signatures clustered from JASPAR 2026 CORE plants. The v4 design subsamples BAMs to equal library depth and merges replicates per condition, producing two pooled samples (leaf, proto) for a direct two-condition comparison. Replicate 3 is excluded due to a confirmed label swap (see `REPs_README.md`).

---

## Pipeline steps

All scripts run from the project root (`5_TF_FP/`).

| Step | Directory | Description | Where |
|------|-----------|-------------|-------|
| 00 | `00_metadata/` | ACR metadata (edgeR + genomic context), library sizes, 2-rep edgeR validation | Local |
| 01 | `01_motif_signatures/` | Per-family JASPAR motif clustering (762 -> 122 signatures) + metadata | SLURM |
| 02 | `02_library_equalization/` | Subsample 4 BAMs to equal depth, merge per condition, extract fragments | SLURM |
| 03 | `03_footprinting/` | scPrinter: TFBS + NucBS + multi-scale FP scoring; scale-resolved QC | SLURM |
| 04 | `04_motif_scanning/` | MOODS scan with 122 signatures across 50 ACR chunks | SLURM |
| 05 | `05_perscale_fp/` | Per-hit FP band extraction + per-scale FP (99 scales) across all hits | SLURM |
| 06 | `06_binding_scores/` | TFBS/NucBS extraction, BS-FP correlation, binding overlap, FP deltas, family enrichment | SLURM + Local |
| 07 | `07_visualization/` | Per-ACR multiscale FP region viewers with family motif annotations | SLURM + Local |

**Dependency chain**: 00, 01 (independent) -> 02 -> 03 -> 04 -> 05 -> 06 -> 07

---

## Sign conventions

| Metric | Positive | Negative |
|--------|----------|----------|
| TF FP delta (leaf - proto) | Leaf-enriched footprint | Proto-enriched footprint |
| edgeR logFC (log2 proto/leaf) | More accessible in protoplast | More accessible in leaf |
| ACR class | proto_gain (fdr < 0.05, logFC > 0) | leaf_gain (fdr < 0.05, logFC < 0) |

---

## scPrinter: methodological notes and limitations

### Models overview

scPrinter produces three types of scores per genomic region:

| Score | Type | Resolution | Output | Organism-specific? |
|-------|------|-----------|--------|-------------------|
| **Multi-scale FP** | Statistical test (-log10 p) | 2000 bp (per-position) x 99 scales | Shape (1, 99, 2000) | No (uses custom At Tn5 bias model) |
| **TFBS** | Neural network (2 hidden, sigmoid) | 180 tiles (10 bp spacing) | Shape (1, 180), probability [0, 1] | Training data is human |
| **NucBS** | Neural network (2 hidden, no activation) | 180 tiles (10 bp spacing) | Shape (1, 180), raw unbounded | Training data is human |

### 1. Human-trained binding models

TFBS was trained on human ChIP-seq (HepG2, GM12878, K562 cell lines, bone marrow hematopoiesis). NucBS was trained on human nucleosome chemical mapping. Both models detect footprint *shape* patterns from multi-scale FP inputs, which are largely organism-agnostic -- the Tn5 insertion dip created by a bound TF or nucleosome has similar geometry across species.

**However**: absolute probability values are not calibrated for Arabidopsis. Treat TFBS/NucBS outputs as **relative rankings** (within and between conditions), not as calibrated binding probabilities. Ideally, future work should retrain these models with plant-specific ChIP-seq and nucleosome mapping data for properly calibrated scores.

### 2. NucBS missing output activation

The NucBS pretrained model lacks an output sigmoid activation. Raw scores are unbounded (observed range approximately [-10, +6]). A numerically stable sigmoid must be applied post-hoc to convert to [0, 1] probability scale. This is handled in step 06a (`06a_extract_binding_scores.py`). TFBS has sigmoid built into the model and does not require this correction.

### 3. Nucleosome repeat length mismatch

Arabidopsis nucleosome repeat length (NRL) is ~165 bp vs human ~195 bp. The NucBS model was trained on human nucleosome spacing, so flanking linker DNA signals may appear at slightly different positions in Arabidopsis. This primarily affects interpretation of NucBS at region edges and inter-nucleosome boundaries.

### 4. No native library-size normalization

scPrinter FP scores are raw -log10(p-values) with no built-in normalization for sequencing depth differences between samples. Two strategies are used in this project:

- **v4 (this pipeline)**: BAMs are subsampled to equal read count *before* scPrinter scoring, ensuring equal input depth. This is the cleanest solution.
- **v2/v3 (archived)**: Post-hoc correction via DESeq2-style size factors applied to FP band/flank values, plus fractional delta (depth/flank ratio) which is coverage-invariant by construction.

### 5. No native replicate handling

scPrinter processes each sample independently -- there is no built-in framework for replicate-aware testing, reproducibility assessment, or variance estimation across biological replicates. All replicate-aware statistics in this project (linear mixed models, permutation tests, replicate concordance checks) are implemented externally in the pipeline scripts.

### 6. Multi-scale FP is organism-agnostic

Unlike TFBS/NucBS, the multi-scale FP score is a statistical test (not a trained neural network). It uses a custom Arabidopsis-specific Tn5 bias model (`3_PRINT_bulk/bias.h5`) fit to the actual ATAC-seq data. The FP scores are fully valid for any organism with a properly trained bias model.

---

## Future direction: tile-based motif scoring

The current motif-level analysis extracts FP at exact hit center positions (single-bp resolution). Single-bp FP measurements are inherently noisy -- a more robust approach would convert motif hit positions into tiles (e.g., 10 bp windows centered on each hit) and aggregate FP across the tile before computing condition deltas.

This mirrors the TFBS/NucBS tile architecture (180 tiles x 10 bp per 2000 bp region), but applied to motif hits rather than a fixed grid. The tiling strategy is independent of the pretrained models -- it is purely a spatial aggregation that improves signal-to-noise for per-hit cross-condition comparisons. A tile width matching the motif width or a fixed 10 bp window would both be reasonable choices.

---

## Key data paths (relative to project root)

| Data | Path |
|------|------|
| Native ACRs (22,582) | `1_ACRs/Athaliana_leaf_protoplast.mergedACRs.bed` |
| edgeR results | `1_ACRs/differential_ACRs_tests.unfiltered.txt` |
| ACR metadata (v4, FDR-only) | `v4/data/acr_metadata.tsv.gz` |
| Signature MEME file (122 sigs) | `data/motif_signatures/At_Motif_SignatureDB.meme` |
| Signature metadata | `data/motif_signatures/signature_metadata.tsv` |
| Motif hits (per chunk) | `data/v3_chunks/chunk_NN/motif_hits.tsv.gz` |
| Merged BAMs (equal depth) | `v4/merged_bams/{leaf,proto}_merged.bam` |
| scPrinter FP h5ad (~33 GB each) | `v4/3_PRINT/FP/{cond}_merged__ALL.h5ad` |
| scPrinter TFBS h5ad (~29 MB each) | `v4/3_PRINT/TFBS/{cond}_merged__ALL.h5ad` |
| scPrinter NucBS h5ad (~29 MB each) | `v4/3_PRINT/NucBS/{cond}_merged__ALL.h5ad` |
| Tn5 bias model (Arabidopsis) | `3_PRINT_bulk/bias.h5` |
| Genome object (pickled) | `3_PRINT_bulk/At_genome_OBJ` |
| Resized ACRs (2000 bp) | `data/acr_resized_2000bp.bed` |
| Native-to-resized mapping | `data/acr_native_to_resized.tsv` |

## Python environment

- **Cluster (SLURM)**: `conda activate scprinter-cpu` (under `~/home_turbo/fabio_home/LocalInstall/miniconda3/`)
- **Local**: `~/Local_installs/miniconda3/bin/python3` or `conda run -n scprinter-local python -u`
- Key packages: scPrinter, MOODS, anndata, h5py, numpy, pandas, scipy, statsmodels, scikit-learn, matplotlib, seaborn
