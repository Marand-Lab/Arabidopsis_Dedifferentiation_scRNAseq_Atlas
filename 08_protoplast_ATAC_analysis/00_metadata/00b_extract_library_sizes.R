#!/usr/bin/env Rscript
# Extract per-replicate library sizes from edgeR counts matrix.
# Input:  1_ACRs/Athaliana_leaf_protoplast.mergedACRs.counts.rds
# Output: data/library_sizes.tsv (bam_name, total_counts)
#
# Run: Rscript 00c_extract_library_sizes.R

counts_path <- "1_ACRs/Athaliana_leaf_protoplast.mergedACRs.counts.rds"
out_path    <- "data/library_sizes.tsv"

if (!file.exists(counts_path)) {
  stop("Counts file not found: ", counts_path)
}

cat("Reading counts matrix:", counts_path, "\n")
counts <- readRDS(counts_path)

# Print structure for diagnostic
cat("Class:", class(counts), "\n")
cat("Dimensions:", paste(dim(counts), collapse=" x "), "\n")
cat("Column names:\n")
print(colnames(counts))

# Compute per-replicate total counts (library sizes)
lib_sizes <- colSums(counts)
cat("\nLibrary sizes:\n")
print(lib_sizes)

# Compute size factors (DESeq2-style: divide by geometric mean)
geom_mean <- exp(mean(log(lib_sizes)))
size_factors <- lib_sizes / geom_mean
cat("\nGeometric mean:", geom_mean, "\n")
cat("Size factors:\n")
print(size_factors)

# Normalize sample names to match pipeline convention (protoplast → proto)
sample_ids <- gsub("^protoplast_", "proto_", names(lib_sizes))
cat("\nSample IDs (normalized):\n")
print(sample_ids)

# Output
df <- data.frame(
  sample_id     = sample_ids,
  total_counts  = as.numeric(lib_sizes),
  size_factor   = as.numeric(size_factors),
  stringsAsFactors = FALSE
)
write.table(df, out_path, sep = "\t", row.names = FALSE, quote = FALSE)
cat("\nWritten:", out_path, "\n")
