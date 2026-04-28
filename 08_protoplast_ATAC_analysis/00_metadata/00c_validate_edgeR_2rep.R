#!/usr/bin/env Rscript
# v4_01_differential_acrs_edgeR.R
# ────────────────────────────────────────────────────────────────────
# edgeR differential accessibility: protoplast vs leaf — 2 replicates only
#
# Motivation: Rep3 has a confirmed label swap at BAM source level.
# The original edgeR results (3 reps) are contaminated.
# This script re-runs edgeR with reps 1+2 only for clean logFC values.
#
# Input:  1_ACRs/Athaliana_leaf_protoplast.mergedACRs.counts.rds
# Output: v4/differential_ACRs_2rep.tsv
#         v4/comparison_2rep_vs_3rep.pdf
#
# Contrast: proto vs leaf → positive logFC = proto-enriched
#
# Run: Rscript v4/v4_01_differential_acrs_edgeR.R
# ────────────────────────────────────────────────────────────────────

suppressPackageStartupMessages({
  library(edgeR)
})

# Set working directory to project root (parent of v4/)
args <- commandArgs(trailingOnly = FALSE)
script_path <- sub("--file=", "", args[grep("--file=", args)])
if (length(script_path) > 0) {
  BASE <- dirname(dirname(normalizePath(script_path)))
} else {
  BASE <- getwd()
}
setwd(BASE)
cat("[INFO] Working directory:", BASE, "\n")

# ── 1. Read counts matrix ────────────────────────────────────────────
counts_path <- "1_ACRs/Athaliana_leaf_protoplast.mergedACRs.counts.rds"
if (!file.exists(counts_path)) {
  stop("[ERR] Counts file not found: ", counts_path)
}

cat("[INFO] Reading counts:", counts_path, "\n")
counts <- readRDS(counts_path)
cat("[INFO] Dimensions:", paste(dim(counts), collapse = " x "), "\n")
cat("[INFO] Column names:", paste(colnames(counts), collapse = ", "), "\n")

# Normalize column names: protoplast_* → proto_*
colnames(counts) <- gsub("^protoplast_", "proto_", colnames(counts))

# ── 2. Subset to rep1 + rep2 only ───────────────────────────────────
keep_cols <- c("leaf_rep1", "leaf_rep2", "proto_rep1", "proto_rep2")
missing <- setdiff(keep_cols, colnames(counts))
if (length(missing) > 0) {
  stop("[ERR] Missing columns: ", paste(missing, collapse = ", "),
       "\n  Available: ", paste(colnames(counts), collapse = ", "))
}

counts_2rep <- counts[, keep_cols]
cat(sprintf("[INFO] Subset to 2 reps: %d ACRs x %d samples\n",
            nrow(counts_2rep), ncol(counts_2rep)))

# ── 3. Sample metadata ──────────────────────────────────────────────
condition <- factor(
  c("leaf", "leaf", "proto", "proto"),
  levels = c("leaf", "proto")  # leaf is reference
)
cat("[INFO] Design: ~ condition (reference = leaf)\n")
cat("[INFO] Positive logFC = proto-enriched\n")

# ── 4. edgeR analysis ───────────────────────────────────────────────
# Create DGEList
y <- DGEList(counts = counts_2rep, group = condition)

# Pre-filter: ≥ 10 total reads across 4 samples
keep <- rowSums(y$counts) >= 10
cat(sprintf("[INFO] Pre-filter: keeping %d / %d ACRs (>= 10 total reads)\n",
            sum(keep), length(keep)))
y <- y[keep, , keep.lib.sizes = FALSE]

# TMM normalization
y <- calcNormFactors(y, method = "TMM")
cat("[INFO] TMM normalization factors:\n")
print(y$samples$norm.factors)

# Design matrix
design <- model.matrix(~ condition)
colnames(design) <- c("Intercept", "proto_vs_leaf")

# Estimate dispersions
y <- estimateDisp(y, design, robust = TRUE)
cat(sprintf("[INFO] Common dispersion: %.4f (BCV = %.3f)\n",
            y$common.dispersion, sqrt(y$common.dispersion)))

# GLM QL F-test
fit <- glmQLFit(y, design, robust = TRUE)
qlf <- glmQLFTest(fit, coef = "proto_vs_leaf")

# Extract results
res <- topTags(qlf, n = Inf, sort.by = "none")$table
cat(sprintf("[INFO] Results: %d ACRs\n", nrow(res)))

# ── 5. Add lec2 column from original results ────────────────────────
orig_path <- "1_ACRs/differential_ACRs_tests.unfiltered.txt"
if (file.exists(orig_path)) {
  orig <- read.delim(orig_path, row.names = 1)
  # Match lec2 column by row name
  res$lec2 <- orig[rownames(res), "lec2"]
  res$lec2[is.na(res$lec2)] <- 0
} else {
  cat("[WARN] Original results not found — skipping lec2 column\n")
  res$lec2 <- 0
}

# Add size column (ACR width from row names)
parts <- strsplit(rownames(res), "_")
res$size <- as.integer(sapply(parts, function(x) as.integer(x[3]) - as.integer(x[2])))

# ── 6. Rename columns to match original format ──────────────────────
# Original columns: logFC, logCPM, F, PValue, fdr, lec2, size
colnames(res)[colnames(res) == "FDR"] <- "fdr"
# topTags output has: logFC, logCPM, F, PValue, FDR → rename FDR to fdr

# Reorder to match original
res <- res[, c("logFC", "logCPM", "F", "PValue", "fdr", "lec2", "size")]

# ── 7. Write output ────────────────────────────────────────────────
out_path <- "v4/differential_ACRs_2rep.tsv"
write.table(res, out_path, sep = "\t", row.names = TRUE, quote = FALSE)
cat(sprintf("[INFO] Written: %s\n", out_path))

# ── 8. Summary ──────────────────────────────────────────────────────
LOGFC_THRESH <- 1
FDR_THRESH <- 0.05

n_tested <- nrow(res)
n_sig <- sum(res$fdr < FDR_THRESH & abs(res$logFC) > LOGFC_THRESH, na.rm = TRUE)
n_proto <- sum(res$fdr < FDR_THRESH & res$logFC > LOGFC_THRESH, na.rm = TRUE)
n_leaf <- sum(res$fdr < FDR_THRESH & res$logFC < -LOGFC_THRESH, na.rm = TRUE)

cat("\n══════════════════════════════════════════════\n")
cat(sprintf("  ACRs tested:              %d\n", n_tested))
cat(sprintf("  Significant (|logFC|>1, fdr<0.05): %d\n", n_sig))
cat(sprintf("    Proto-gain (logFC > 1):  %d\n", n_proto))
cat(sprintf("    Leaf-gain  (logFC < -1): %d\n", n_leaf))
cat(sprintf("    Stable:                  %d\n", n_tested - n_sig))
cat("══════════════════════════════════════════════\n")

# ── 9. Comparison with 3-rep results ────────────────────────────────
if (file.exists(orig_path)) {
  cat("\n[INFO] Comparing 2-rep vs 3-rep results...\n")
  orig <- read.delim(orig_path, row.names = 1)

  # Match by row name
  shared <- intersect(rownames(res), rownames(orig))
  cat(sprintf("[INFO] Shared ACRs: %d\n", length(shared)))

  lfc_2rep <- res[shared, "logFC"]
  lfc_3rep <- orig[shared, "logFC"]

  r_pearson <- cor(lfc_2rep, lfc_3rep, use = "complete.obs")
  r_spearman <- cor(lfc_2rep, lfc_3rep, use = "complete.obs", method = "spearman")
  cat(sprintf("[INFO] logFC correlation: Pearson = %.4f, Spearman = %.4f\n",
              r_pearson, r_spearman))

  # Classification concordance
  classify <- function(logfc, fdr_val) {
    cls <- rep("stable", length(logfc))
    cls[!is.na(fdr_val) & fdr_val < FDR_THRESH & logfc > LOGFC_THRESH] <- "proto_gain"
    cls[!is.na(fdr_val) & fdr_val < FDR_THRESH & logfc < -LOGFC_THRESH] <- "leaf_gain"
    factor(cls, levels = c("proto_gain", "stable", "leaf_gain"))
  }

  cls_2rep <- classify(res[shared, "logFC"], res[shared, "fdr"])
  cls_3rep <- classify(orig[shared, "logFC"], orig[shared, "fdr"])
  concordance <- table(`2rep` = cls_2rep, `3rep` = cls_3rep)
  cat("\n[INFO] Classification concordance (2-rep vs 3-rep):\n")
  print(concordance)

  n_agree <- sum(diag(concordance))
  cat(sprintf("[INFO] Agreement: %d / %d (%.1f%%)\n",
              n_agree, length(shared), 100 * n_agree / length(shared)))

  # ── Comparison plot ──────────────────────────────────────────────
  pdf("v4/comparison_2rep_vs_3rep.pdf", width = 10, height = 5)
  par(mfrow = c(1, 2), mar = c(4.5, 4.5, 3, 1))

  # Panel A: logFC scatter
  cls_color <- rep("gray70", length(shared))
  # Color by 2-rep classification
  cls_color[cls_2rep == "proto_gain"] <- "#D64045"
  cls_color[cls_2rep == "leaf_gain"] <- "#3A7D44"

  plot(lfc_3rep, lfc_2rep,
       pch = 16, cex = 0.3, col = adjustcolor(cls_color, alpha.f = 0.4),
       xlab = "logFC (3 replicates)", ylab = "logFC (2 replicates)",
       main = "A. edgeR logFC: 2-rep vs 3-rep",
       xlim = range(c(lfc_2rep, lfc_3rep), na.rm = TRUE),
       ylim = range(c(lfc_2rep, lfc_3rep), na.rm = TRUE))
  abline(0, 1, col = "black", lwd = 1.5, lty = 2)
  abline(h = c(-1, 1), col = "gray50", lty = 3)
  abline(v = c(-1, 1), col = "gray50", lty = 3)
  legend("topleft", bty = "n", cex = 0.8,
         legend = c(sprintf("Pearson r = %.3f", r_pearson),
                    sprintf("Spearman ρ = %.3f", r_spearman),
                    sprintf("n = %d ACRs", length(shared))))

  # Panel B: concordance mosaic
  barplot_mat <- concordance
  barplot(barplot_mat, beside = TRUE,
          col = c("#D64045", "gray70", "#3A7D44"),
          main = "B. Classification concordance",
          xlab = "3-rep class", ylab = "Count (2-rep class)",
          legend.text = rownames(barplot_mat),
          args.legend = list(title = "2-rep class", cex = 0.7,
                             x = "topright", bty = "n"))
  mtext(sprintf("Agreement: %.1f%%", 100 * n_agree / length(shared)),
        side = 3, line = -1.5, cex = 0.8)

  dev.off()
  cat("[INFO] Saved: v4/comparison_2rep_vs_3rep.pdf\n")
} else {
  cat("[WARN] Skipping comparison — original results file not found\n")
}

cat("\n[DONE]\n")
