# Methods: Library-Equalized TF and Nucleosome Footprinting Between Leaf and Protoplast in *Arabidopsis thaliana* (v4)

## Overview

To address systematic library-size imbalance and limited replicate leverage in the per-replicate pipeline (v2/v3), we developed a library-equalized, replicate-merged analysis (v4). Rather than relying on post-hoc normalization (TMM, size factors) to correct a nearly two-fold depth imbalance between leaf replicates (28M vs 15M reads), v4 subsamples all BAM files to equal sequencing depth before merging replicates within each condition. This produces two pooled BAMs (leaf, proto) with identical library sizes, eliminating coverage-driven bias at the source.

Additionally, v4 introduces two methodological refinements: (1) a relaxed ACR classification using FDR < 0.05 without an effect-size threshold, capturing 4,095 changing ACRs (vs 3,242 in v2/v3); and (2) nucleosome binding score (NucBS) prediction via scPrinter's pre-trained neural network model alongside the standard TF binding score (TFBS), enabling joint TF--nucleosome occupancy analysis at single-tile resolution.

---

## 1. Validation of edgeR Results Across Replicate Configurations

### 1.1 Two-replicate edgeR reanalysis

Before committing to a two-replicate design, we re-ran the edgeR differential accessibility analysis using only replicates 1 and 2 (excluding replicate 3, which carries a confirmed label swap; see v2 methods). The count matrix from the original three-replicate edgeR analysis was subset to four samples (leaf_rep1, leaf_rep2, proto_rep1, proto_rep2). TMM normalization was re-computed on the reduced matrix, and a quasi-likelihood F-test (`glmQLFit` + `glmQLFTest`) was applied with the contrast proto vs leaf. A pre-filter of >= 10 total reads across 4 samples was applied.

### 1.2 Concordance with three-replicate results

The two-replicate logFC values were highly correlated with the original three-replicate logFC values (Pearson r = 0.938, Spearman rho = 0.897). Classification concordance (using |logFC| > 1 and FDR < 0.05 on both) was 93.5%, with zero direction flips among ACRs classified as significant in both analyses. This validates that the three-replicate logFC estimates are reliable and can be used as the reference for ACR classification in v4, despite the replicate-3 exclusion.

---

## 2. ACR Preparation and Metadata Construction

### 2.1 Accessible chromatin regions

The analysis starts from the same 22,582 ACRs used in v2/v3, obtained by merging ATAC-seq peaks across all conditions and replicates. Each ACR is defined by its native genomic coordinates (variable width, median ~400 bp, range 150--6,952 bp).

### 2.2 Differential accessibility classification (FDR-only)

ACR classification in v4 uses FDR < 0.05 only, with direction determined by the sign of the original three-replicate edgeR logFC:

- **proto_gain**: FDR < 0.05 and logFC > 0 (more accessible in protoplast; 1,222 ACRs, 5.4%)
- **leaf_gain**: FDR < 0.05 and logFC < 0 (more accessible in leaf; 2,873 ACRs, 12.7%)
- **stable**: FDR >= 0.05 (18,487 ACRs, 81.9%)

This yields 4,095 changing ACRs, a 26% increase over the 3,242 obtained with the v2/v3 threshold (|logFC| > 1 and FDR < 0.05). The relaxed threshold captures ACRs with statistically significant but modest-magnitude accessibility differences that were previously excluded.

### 2.3 Genomic context annotation

Each ACR was annotated with its genomic context relative to the TAIR10 v60 gene annotation (GFF3), using the same classification as v2/v3:

- **Promoter**: ACR midpoint within +/-1,000 bp of a transcription start site
- **Gene body**: ACR midpoint overlaps a gene model but outside the promoter window
- **Intergenic**: all remaining ACRs

Priority: Promoter > Gene body > Intergenic. Chromosome naming between GFF3 (lowercase `chr1`) and ACR BED files (uppercase `Chr1`) was automatically harmonized.

### 2.4 Region resizing

All ACRs were center-extended to a uniform 2,000 bp width for scPrinter batch processing (see v2/v3 methods, Section 1.5). One ACR near the end of Chr2 was dropped, yielding 22,581 resized regions. A native-to-resized coordinate mapping table was generated for downstream motif-to-region lookups.

---

## 3. Library Equalization and BAM Merging

### 3.1 Subsampling to equal depth

Four BAM files (leaf_rep1, leaf_rep2, proto_rep1, proto_rep2; all MAPQ >= 30, PCR duplicates removed) were subsampled to match the library with the fewest aligned reads. For each BAM, `samtools view -c` determined the total read count, and the minimum across all four was used as the target depth. Each BAM was then subsampled using `samtools view -bs` with a fixed random seed (seed = 42) and the appropriate subsampling fraction (target / original). The BAM already at minimum depth was copied unchanged. Output BAMs were indexed and read counts verified.

Per-ACR read counts were generated from the subsampled BAMs using `bedtools multicov` on the native ACR BED file, providing a verification of uniform coverage across conditions.

### 3.2 Replicate merging

Within each condition, the two subsampled replicate BAMs were merged using `samtools merge`, producing two final BAMs: `leaf_merged.bam` and `proto_merged.bam`. Each merged BAM contains exactly twice the minimum per-replicate read count, ensuring perfect library-size balance between conditions. Merged BAMs were indexed and read counts verified.

### 3.3 Fragment preparation

Merged BAMs were converted to scPrinter-compatible 1-based fragment files through a three-step process:

1. **Name-sorting**: `samtools sort -n` to group mate pairs
2. **Fragment extraction**: Custom `bam_to_fragment.py` script (bulk mode, minimum MAPQ = 20, 8 threads) to extract paired-end fragment coordinates
3. **Coordinate conversion**: AWK pipeline converting 0-based to 1-based start coordinates (`start_1based = start_0based + 1`), remapping chromosome names to scPrinter conventions, and filtering invalid fragments (start >= end or start < 1). Mitochondrial and chloroplast reads were excluded.

Output: gzipped 4-column TSV (chromosome, start_1based, end, "bulk") per condition.

---

## 4. scPrinter Footprint and Binding Score Computation

### 4.1 Experimental design

Unlike the per-replicate design in v2/v3 (six h5ad files), v4 processes two merged BAMs per condition, producing two independent scPrinter output sets. This simplification is appropriate because statistical inference on replicate variability is handled by the original three-replicate edgeR results; the v4 footprinting analysis characterizes the magnitude and spatial pattern of binding differences rather than estimating per-replicate uncertainty.

### 4.2 Three-model scoring

For each condition (leaf, proto), scPrinter was run with standard Tn5 transposase strand offsets (+4 bp plus strand, -5 bp minus strand) on the 22,581 resized ACR regions. Three scoring models were applied:

1. **TF binding score (TFBS)**: A pre-trained three-layer neural network (`scp.datasets.pretrained_TFBS_model`) that consumes footprint signals at 6 scales (10, 20, 30, 50, 80, 100 bp) within a +/-100 bp context window, producing a per-tile sigmoid-activated probability in [0, 1]. The model uses `contextRadius=100` and `tileSize=10` on 2,000 bp regions, yielding **180 tiles** per region with tile centers at bp positions 105, 115, 125, ..., 1895 (formula: `bp = tile_index * 10 + 105`).

2. **Nucleosome binding score (NucBS)**: A pre-trained three-layer neural network (`scp.datasets.pretrained_NucBS_model`) using 5 FP scales (10, 20, 30, 50, 80 bp) within the same tiling scheme (180 tiles per region). Critically, **the NucBS model lacks an output activation function**, producing unbounded raw scores (observed range approximately [-10, +6]). A sigmoid transformation must be applied post-hoc to convert to [0, 1] probability (see Section 7.1).

3. **Multi-scale footprint score (FP)**: Standard scPrinter footprint scoring at 99 spatial scales (2 to 100 bp) via `get_footprint_score()` with `modes=np.arange(2, 101)` and `region_width=None`. This produces a 3D tensor per region of shape (1, 99, 2000), representing -log10(p-value) footprint significance at each scale and genomic position.

### 4.3 Output structure

Each condition produces three h5ad files:

| Model | Shape per region | Size | Content |
|-------|-----------------|------|---------|
| TFBS | (1, 180) | ~29 MB | TF binding probability [0, 1] |
| NucBS | (1, 180) | ~29 MB | Raw nucleosome score (unbounded) |
| FP | (1, 99, 2000) | ~33 GB | -log10(p-value) at 99 scales x 2000 positions |

Region keys in h5ad `obsm` are stored as sanitized coordinate strings (e.g., `chr1:2054_4054`), accessible via `anndata.read_h5ad(path, backed="r").obsm[key]`.

### 4.4 Cross-species applicability of pretrained models

Both the TFBS and NucBS neural network models were trained on human data: TFBS on ChIP-seq from HepG2, GM12878, and K562 cell lines plus bone marrow hematopoiesis, and NucBS on human nucleosome chemical mapping. The models detect footprint *shape* patterns from multi-scale FP inputs -- the Tn5 insertion protection dip created by a bound TF or nucleosome has similar geometry across species -- which makes them largely organism-agnostic in principle. However, three caveats apply to Arabidopsis usage:

1. **Probability calibration**: The sigmoid output was trained on human ChIP-seq "bound" labels. Absolute probability values in [0, 1] are not calibrated for Arabidopsis TF binding or nucleosome occupancy. We therefore treat TFBS and NucBS outputs as relative rankings (within and between conditions) rather than calibrated binding probabilities.

2. **Nucleosome repeat length mismatch**: Arabidopsis nucleosome repeat length (NRL) is approximately 165 bp compared to ~195 bp in human. The NucBS model was trained on human nucleosome spacing, so flanking linker DNA signals may appear at slightly different positions in Arabidopsis, particularly affecting interpretation at region edges and inter-nucleosome boundaries.

3. **Training label bias**: The "bound" labels reflect the human TF repertoire and may be better at detecting footprint shapes typical of human TF families (e.g., zinc fingers) than plant-specific families. Ideally, future work should retrain these models with plant-specific ChIP-seq and nucleosome mapping data.

In contrast, the multi-scale FP score is a statistical test (-log10 p-value), not a trained model, and uses a custom Arabidopsis-specific Tn5 bias model fit to the actual ATAC-seq data. FP scores are fully valid for any organism with a properly trained bias model.

---

## 5. Non-Redundant Motif Signature Construction and Scanning

### 5.1 Motivation

The JASPAR 2026 CORE plants non-redundant collection contains 762 motifs, many of which are highly similar within TF families (e.g., multiple DOF or WRKY motifs with near-identical position weight matrices). This within-family redundancy inflates feature counts in downstream analyses without proportional information gain. To address this, we clustered motifs within each family to produce non-redundant signatures that preserve between-family diversity while collapsing within-family redundancy.

### 5.2 Per-family motif clustering

Starting from 465 Arabidopsis-specific motifs extracted from the JASPAR 2026 CORE plants collection (identified via the JASPAR2024 R package), motifs were clustered independently within each TF family using motifStack (R). The per-family design prevents biologically distinct motifs from different families from merging based on superficial PWM similarity (e.g., bHLH E-box vs bZIP E-box).

The clustering procedure was:

1. **Singletons** (1 motif in family): passed through as a single signature.
2. **Pairs** (2 motifs in family): PWM distance computed via `clusterMotifs()`. If the hierarchical clustering height was < 0.5, the two PFMs were averaged position-by-position (shorter matrix padded with uniform 0.25 columns) to produce a consensus signature; otherwise both were retained as separate signatures.
3. **Families with >= 3 motifs**: Full motifStack clustering via `clusterMotifs()`, conversion to phylogenetic tree via `hclust2phylog()`, and signature extraction via `motifSignature()` with `cutoffPval = 0.0001` and `min.freq = 1`. Each resulting signature is a consensus PFM representing a cluster of similar motifs.

For families where clustering failed (e.g., numerical issues with highly similar motifs), all motifs were retained as individual signatures (fallback). The output was a single MEME-format file containing all signature PFMs and a cluster membership table mapping each signature to its constituent motifs.

### 5.3 Results

The procedure reduced 465 Arabidopsis motifs to **122 non-redundant signatures** across approximately 42 TF families. A companion metadata table maps each signature to its primary family, constituent motif IDs, display name, and multi-family flags.

### 5.4 MOODS motif scanning

TF binding motifs were scanned on native (non-resized) ACR sequences using MOODS (Motif Occurrence Detection Suite) with the 122 signature PFMs. Position frequency matrices were converted to position weight matrices via log-odds transformation against a uniform nucleotide background (0.25 each) with a pseudocount of 1 x 10^-4. Both forward and reverse complement strands were scanned with a per-motif p-value threshold of 5 x 10^-5.

Each hit was recorded with its genomic coordinates, signature identity, strand, MOODS score, and hit center position (start + motif_length / 2), which anchors downstream footprint extraction windows.

A critical technical fix was applied to the MOODS scanner configuration: each signature receives its own `Scanner(7)` instance rather than batching multiple motifs per scanner. The MOODS `Scanner(n)` function precomputes a 4^n k-mer prefilter table for efficient scanning. Using `n = motif_width` for wide motifs (w = 17--30 bp in our MEME file) caused prohibitive memory allocations (4^17 = 17 billion entries, ~137 GB). Setting `n = 7` uses the standard MOODS default (4^7 = 16,384 entries, ~128 KB). The window parameter affects only the k-mer prefilter speed, not scanning accuracy -- results are identical, just slightly slower for very wide motifs. Scanners were built once in the parent process; fork workers inherited them via copy-on-write memory sharing, avoiding out-of-memory crashes from redundant scanner construction.

The scan was parallelized as a 50-chunk SLURM array job, splitting the native ACR BED file into 50 equal-sized blocks. Total output: **3,785,216 hits** across 22,581 ACRs -- a 31% reduction from the v2 full-motif count of 5,518,010 (762 motifs), reflecting the removal of within-family redundancy.

---

## 6. Scale-Resolved Bias Quality Control

### 6.1 Rationale

Library equalization eliminates global depth differences but cannot correct for scale-dependent Tn5 insertion biases or subtle condition-specific chromatin artifacts. Before interpreting FP deltas (leaf - proto), we tested whether any systematic bias exists at each of the 99 FP scales.

### 6.2 Null locus construction

We identified null loci -- positions within ACRs that are far from any known TF motif -- to estimate the expected FP signal in the absence of specific TF binding. From the v3 motif hit database, all hit positions were buffered by +/-50 bp. Within each ACR, candidate positions at least 15 bp from ACR edges and outside all motif exclusion zones were enumerated. A random sample of 5,000 null loci was drawn (seed = 0) from 4,937 qualifying loci across the genome.

### 6.3 Statistical testing

For each FP scale (2--100 bp), the delta (leaf - proto) was computed at (a) all 22,338 ACR center positions (position 1000 within the 2,000 bp window) and (b) all null loci. One-sample t-tests against the null hypothesis delta = 0 were performed at each scale, with Bonferroni correction for 99 tests (alpha_Bonferroni = 0.05 / 99 = 5.05 x 10^-4).

### 6.4 Results and interpretation

- **Sub-nucleosomal scales (2--20 bp)**: No Bonferroni-significant bias was detected at null loci, confirming that TF-scale FP signals are clean and interpretable without correction.
- **Mid-range scales (43--69 bp)**: 27 of 99 scales showed significant bias at null loci, but with small effect sizes (maximum |delta| = 0.047, compared to typical biological signal of ~0.09).
- **Oscillatory pattern**: A 2--4 bp leaf-favored to 7--8 bp proto-favored oscillation was visible at both ACR centers and null loci but remained below the significance threshold at null loci, indicating high per-locus variance at small scales.

The full null delta matrix (4,937 loci x 99 scales) was saved for downstream per-scale empirical null correction: `z(s) = (delta_motif(s) - mean_null(s)) / sd_null(s)`, which automatically corrects for scale-dependent bias, Tn5 artifacts, and global accessibility shifts.

---

## 7. Binding Score Extraction and Classification

### 7.1 Extraction

TFBS and NucBS binding score arrays were extracted from the h5ad tile arrays for each condition independently. The NucBS raw scores were transformed via a numerically stable sigmoid function: `sigmoid(x) = 1 / (1 + exp(-x))`, handling extreme values separately to avoid floating-point overflow.

### 7.2 Tile classification

Tiles were classified using per-condition percentile thresholds computed across all regions and tiles:

| Score | Classification | Threshold |
|-------|---------------|-----------|
| TFBS | Bound | > 95th percentile |
| TFBS | Unbound | < 5th percentile |
| NucBS (sigmoid) | Occupied | > 95th percentile |
| NucBS (sigmoid) | Free | < 5th percentile |

The 95th/5th percentile thresholds were chosen rather than absolute cutoffs (e.g., 0.8/0.2) because the score distributions are condition- and model-dependent. Percentile-based thresholds ensure equal representation of extreme tiles across conditions, facilitating balanced comparisons.

### 7.3 Output

Per-condition NPZ archives contain: TFBS probabilities (22,581 x 180), NucBS raw and sigmoid-transformed scores (22,581 x 180 each), tile center positions (180 values), and boolean classification masks.

---

## 8. Binding Score--Footprint Correlation

### 8.1 Rationale

The TFBS and NucBS models were trained on FP features at specific scale subsets (6 and 5 scales, respectively), but their predictions integrate information non-linearly. To understand which FP scales carry the strongest binding signal, we correlated binding scores with FP across all 99 scales.

### 8.2 Method

For each condition, tiles were stratified into bound/unbound (TFBS, top/bottom 5%) and occupied/free (NucBS, top/bottom 2%) groups. Within each group, Spearman rank correlation was computed between the binding score and FP signal at each of the 99 scales across all qualifying tiles.

### 8.3 Key finding

TFBS binding probability correlates most strongly with FP at **4--10 bp** scales (the TF protection zone, where protein-DNA contacts shield against Tn5 insertion). NucBS occupancy peaks at **20--30 bp** scales but extends through **40--60 bp** (the nucleosome zone, where ~147 bp of wrapped DNA creates broad protection). The crossover between TF- and nucleosome-dominated signal occurs at approximately 50--60 bp. This confirms that the two binding score models capture biologically distinct protection signatures, validating their use as complementary readouts.

---

## 9. Condition-Specific Binding Overlap

### 9.1 Method

Tile-level overlap of bound (TFBS top 5%) and occupied (NucBS top 2%) positions between leaf and proto was quantified using Venn-style set analysis, stratified by ACR class (proto_gain, stable, leaf_gain). For each ACR class, tiles were categorized as:

- **Shared**: bound/occupied in both conditions
- **Leaf-only**: bound/occupied in leaf but unbound/free in proto
- **Proto-only**: bound/occupied in proto but unbound/free in leaf

Enrichment or depletion of overlap relative to expectation was assessed using a hypergeometric test on the universe of active tiles (all tiles classified as bound or occupied in at least one condition).

### 9.2 Key finding

All overlaps were significantly less than expected by chance (fold enrichment 0.1--0.5x, all hypergeometric tests significant). This indicates that predicted binding/occupancy positions are **highly condition-specific** -- the same genomic positions rarely carry high binding probability in both leaf and proto. This condition-specificity is consistent with condition-dependent TF occupancy and nucleosome positioning, validating the approach for differential analysis.

---

## 10. FP Delta Distributions at Bound and Occupied Positions

### 10.1 Rationale

Having established that binding positions are condition-specific (Section 8), we asked whether these positions also show differential footprint depth. The FP delta (leaf - proto) at bound/occupied tiles should reflect the direction of the underlying accessibility change.

### 10.2 Scale selection

Two biologically motivated scale ranges were analyzed:

- **TF scale**: 4--10 bp (7 scales) -- captures direct protein-DNA contact protection
- **Nucleosome scale**: 40--60 bp (21 scales) -- captures broad nucleosome wrapping protection

For each tile, FP was extracted at all scales within the range and averaged to produce a single TF-scale and nucleosome-scale value per tile per condition.

### 10.3 Genome-wide null distribution

A null distribution was constructed from tiles classified as unbound/free in both conditions (bottom percentile tiles in both leaf and proto). These represent genomic positions with minimal predicted binding in either condition, serving as a baseline for FP delta in the absence of differential occupancy. A maximum of 50,000 null tiles was sampled to keep computation tractable.

### 10.4 Category-level testing

For each of 9 categories (3 ACR classes x 3 overlap groups: leaf-only, shared, proto-only), the Mann-Whitney U test was used to compare FP deltas against the genome-wide null. Benjamini-Hochberg FDR correction was applied across the 9 tests. All categories showed highly significant shifts (all FDR < 0.001).

### 10.5 Tile-level classification

Each tile was z-scored against the null distribution: `z = (delta_tile - mean_null) / sd_null`. Tiles with |z| >= 1 were classified as significantly shifted:

- **Leaf-enriched**: z >= 1 (FP deeper in leaf than expected under the null)
- **Proto-enriched**: z <= -1 (FP deeper in proto)
- **Non-significant**: |z| < 1

BH-FDR correction was applied within each category.

### 10.6 Key findings

- **TFBS categories**: All categories show the expected direction of FP shift -- proto-enriched deltas at proto_gain ACRs, leaf-enriched deltas at leaf_gain ACRs. The magnitude of per-tile effects is moderate.
- **NucBS categories**: Nucleosome-scale deltas show much larger per-tile effects, with 60--80% of tiles reaching |z| >= 1. This indicates that nucleosome repositioning is more spatially pervasive than TF rebinding.
- **Stable ACRs**: Notably, even stable ACRs (no significant accessibility change) show substantial nucleosome-scale reorganization in both directions, suggesting that nucleosome dynamics can occur independently of net accessibility changes.

---

## 11. TF Family Enrichment at Significant-Delta Tiles

### 11.1 Rationale

The preceding analyses established that bound/occupied tiles carry condition-specific FP deltas. The final question is whether specific TF families are enriched at tiles with the strongest differential signal, which would implicate those families in the accessibility transition.

### 11.2 Motif-to-tile mapping

TF motif hits from the v3 non-redundant motif signature database were mapped to the 180-tile coordinate system. Each motif hit center (in native ACR coordinates) was translated to resized coordinates via the coordinate mapping table, then assigned to the overlapping tile (tile center +/- 5 bp). Hits were deduplicated by (region, tile, family) to avoid double-counting multiple motifs from the same family at the same tile.

### 11.3 Statistical testing

For each of 18 boxes (9 categories x 2 directions: leaf-enriched z >= 1, proto-enriched z <= -1), Fisher's exact test (two-sided) was performed per TF family. The 2 x 2 contingency table compared:

- **Foreground**: tiles with significant delta in the specified direction within the category
- **Background**: all active tiles genome-wide (bound or occupied in at least one condition)
- **With family**: tile contains at least one motif hit from the focal family
- **Without family**: tile has no hit from the focal family

Benjamini-Hochberg FDR correction was applied per box across all tested families.

### 11.4 Key finding -- WRKY compaction model

The family enrichment analysis revealed a striking asymmetry for WRKY TFs in leaf_gain ACRs:

- **At the TF binding level (TFBS)**: WRKY is enriched at leaf-only tiles with positive FP delta (leaf-enriched TF signal). WRKY binding sites coincide with strong leaf TF footprints, consistent with active WRKY occupancy in leaf tissue.

- **At the nucleosome level (NucBS)**: WRKY is depleted at leaf-only tiles with leaf-enriched nucleosome signal (WRKY avoids nucleosome-occupied positions in leaf), but enriched at proto-only tiles with proto-enriched nucleosome signal (nucleosomes invade former WRKY sites in protoplast).

This pattern supports a model in which WRKY TFs -- known stress-responsive factors that interact with chromatin remodelers (HDA19, TPL/TPR) -- may actively recruit nucleosome remodeling machinery upon protoplast induction. Rather than WRKY being passively displaced by nucleosome encroachment ("victim"), WRKY overexpression in protoplasts may trigger local chromatin compaction at its own target sites ("agent of compaction"). This interpretation is consistent with the WRKY compaction model identified independently in the v3 multi-scale analysis (v3_08b, page 5).

---

## Sign Conventions

| Quantity | Positive direction | Negative direction |
|----------|-------------------|-------------------|
| FP delta (leaf - proto) | Leaf-enriched (deeper FP in leaf) | Proto-enriched (deeper FP in proto) |
| TFBS delta (leaf - proto) | Higher TF binding probability in leaf | Higher in proto |
| NucBS delta (leaf - proto) | Higher nucleosome occupancy in leaf | Higher in proto |
| edgeR logFC | Proto-enriched (more accessible in proto) | Leaf-enriched |
| ACR class | proto_gain (logFC > 0, FDR < 0.05) | leaf_gain (logFC < 0, FDR < 0.05) |

Note: FP delta and edgeR logFC have **opposite sign conventions** for the same biological direction. A proto_gain ACR (positive logFC) is expected to show negative FP delta (deeper FP in proto) at TF sites bound in proto.

---

## Software and Parameters

| Tool | Version | Key parameters |
|------|---------|----------------|
| samtools | >= 1.17 | `view -bs {seed}.{frac}` (subsampling); `merge -f` (merging) |
| bedtools | >= 2.31 | `multicov` (ACR read counts) |
| edgeR | >= 3.40 | TMM normalization, glmQLFit + glmQLFTest, pre-filter >= 10 reads |
| scPrinter | 1.2.1 | Tn5 shifts +4/-5; `tileSize=10`, `contextRadius=100`; 99 FP scales (2--100 bp) |
| MOODS | >= 1.9.4 | Per-motif `Scanner(7)`; p-value threshold 5 x 10^-5; both strands |
| motifStack | >= 1.38 | `clusterMotifs()`, `motifSignature(cutoffPval=0.0001, min.freq=1)` |
| Python | >= 3.10 | NumPy, SciPy (t-test, Mann-Whitney, Fisher, BH-FDR), anndata |
| R | >= 4.3 | edgeR, limma, motifStack, JASPAR2024, universalmotif |

---

## Pipeline Summary

| Step | Script (in `final/`) | Environment | Description |
|------|---------------------|-------------|-------------|
| 00a | `00_metadata/00a_build_acr_metadata.py` | Local | ACR metadata with FDR-only classification |
| 00b | `00_metadata/00b_extract_library_sizes.R` | Local | Per-replicate library sizes from edgeR counts |
| 00c | `00_metadata/00c_validate_edgeR_2rep.R` | Local (R) | 2-rep edgeR reanalysis + 3-rep validation |
| 01a | `01_motif_signatures/01a_motif_clustering.R` | SLURM | Per-family JASPAR motif clustering (762 -> 122 sigs) |
| 01b | `01_motif_signatures/01b_build_signature_metadata.py` | Local | Signature-to-family metadata table |
| 02a | `02_library_equalization/02a_subsample_bams.sh` | SLURM | Subsample 4 BAMs to equal depth |
| 02b | `02_library_equalization/02b_merge_conditions.sh` | SLURM | Merge subsampled reps per condition |
| 02c | `02_library_equalization/02c_prep_fragments.sh` | SLURM | BAM to 1-based fragment conversion |
| 03a | `03_footprinting/03a_run_print.py` (+`.sh`) | SLURM | scPrinter: TFBS + NucBS + FP scoring |
| 03b | `03_footprinting/03b_qc_scale_bias.py` (+`.sh`) | SLURM | Scale-resolved bias QC at null loci |
| 04a | `04_motif_scanning/04a_motif_scan.py` (+`.sh`) | SLURM | MOODS scan with 122 signatures (50-chunk array) |
| 05a | `05_perscale_fp/05a_extract_fp.py` (+`.sh`) | SLURM | Per-hit FP band extraction (4-phase chain) |
| 05b | `05_perscale_fp/05b_perscale_fp.py` (+`.sh`) | SLURM | Per-scale FP at 99 scales (50-chunk array + merge) |
| 06a | `06_binding_scores/06a_extract_binding_scores.py` (+`.sh`) | SLURM | TFBS/NucBS extraction + sigmoid + classification |
| 06b | `06_binding_scores/06b_bs_fp_correlation.py` (+`.sh`) | SLURM | Binding score--FP correlation across scales |
| 06c | `06_binding_scores/06c_binding_overlap.py` (+`.sh`) | Local/SLURM | Condition-specific binding overlap + hypergeometric test |
| 06d | `06_binding_scores/06d_binding_deltas.py` (+`.sh`) | SLURM | FP delta at bound/occupied tiles + z-score classification |
| 06e | `06_binding_scores/06e_family_enrichment.py` | Local | TF family enrichment at significant-delta tiles |
| 07a | `07_visualization/07a_plot_regions.py` (+`.sh`) | SLURM + Local | Per-ACR multiscale FP region viewer |
| 07b | `07_visualization/07b_plot_regions_families.py` (+`.sh`) | SLURM + Local | Per-ACR family-annotated viewer |
