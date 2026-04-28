####################################################################################
## v3 Per-Family Motif Clustering
##
## Clusters 465 Arabidopsis JASPAR2024 motifs WITHIN each TF family using
## motifStack, preventing cross-family merges (e.g., bHLH E-box vs bZIP E-box).
## Outputs a single MEME file + cluster membership table for the v3 pipeline.
##
## Input:
##   - 1_At_Motifs_names: 465 Arabidopsis motif IDs (MA0001.2_AGL3 format)
##   - JASPAR_redundant/: individual MEME files per motif
##   - at_motif_family_assignments.tsv: motif-to-family mapping (465 rows)
##
## Output:
##   - At_Motif_SignatureDB.meme: per-family signature PFMs (MEME format)
##   - At_MotifClusters.txt: cluster membership (MOTIFN\t<semicolon-joined members>)
##   - v3_clustering_summary.tsv: per-family clustering summary
##
## Usage:
##   module load GCC/11.2.0 OpenMPI/4.1.1 R/4.3.1
##   Rscript v3_motif_clustering.R
####################################################################################

suppressMessages(library(tidyverse))
suppressMessages(library(motifStack))
suppressMessages(library(data.table))
library(igraph)
library(Rgraphviz)
library(ade4)
library(universalmotif)
library(JASPAR2024)
library(TFBSTools)

set.seed(42)

## ---- Helper functions ----

chop <- function(myStr, mySep, myField) {
  choppedString <- sapply(strsplit(myStr, mySep), "[", myField)
  if (length(myField) > 1) {
    choppedString <- apply(choppedString, 2, function(x) {
      paste0(x[!is.na(x)], collapse = mySep)
    })
  }
  return(choppedString)
}

ReadFirstMotif <- function(x) {
  tem <- importMatrix(x, format = "meme", to = "pfm")
  return(tem)
}

## ---- Read input data ----

# All paths relative to project root (5_TF_FP/).
# Input/output data stays in data/motif_signatures/.
SIG_DIR <- "data/motif_signatures"

JASPAR_DIR <- "/nfs/turbo/lsa-YOURNAME/fabio_home/Projects/7_Zea282/0_Data_CHIP_DAP/4_Motif/JASPAR_redundant"

# Read 465 At motif names
filesAthMotif <- chop(read.table(file.path(SIG_DIR, "1_At_Motifs_names"), h = FALSE)$V1, '[_]', 1)

# Filter to those available in JASPAR_redundant
filescontrol <- gsub('.meme', '', list.files(JASPAR_DIR, pattern = "*meme"))
filesAthMotif <- filesAthMotif[filesAthMotif %in% filescontrol]
cat(sprintf("[INFO] %d motifs found in JASPAR_redundant\n", length(filesAthMotif)))

# Read all motif PFMs
At_Motif <- lapply(paste0(JASPAR_DIR, '/', filesAthMotif, '.meme'), ReadFirstMotif)
At_Motif <- unlist(At_Motif)
cat(sprintf("[INFO] %d PFMs loaded\n", length(At_Motif)))

# Normalize names: replace dots with underscores for consistency
names(At_Motif) <- gsub('[.]', '_', names(At_Motif))

# Read family assignments
fam_df <- read.table(file.path(SIG_DIR, "at_motif_family_assignments.tsv"), header = TRUE, sep = "\t",
                     stringsAsFactors = FALSE, quote = "")

# Build lookup: MAxxxx_version -> family
# Use motif_version (e.g. MA0001.2) rather than full at_name (e.g. MA0001.2_AGL3)
# because TF names in MEME files may differ from the TSV (e.g. ERF036 vs AT3G16280)
fam_df$motif_key <- gsub('[.]', '_', fam_df$motif_version)
family_lookup <- setNames(fam_df$tf_family, fam_df$motif_key)

# Map each loaded motif to its family using only the MAxxxx_version prefix
# e.g. names(At_Motif) = "MA1253_1_ERF036" -> extract "MA1253_1"
motif_names <- names(At_Motif)
motif_version_keys <- sub("^(MA[0-9]+_[0-9]+)_.*", "\\1", motif_names)
# Handle motifs whose full name IS the version (no TF suffix)
motif_version_keys[motif_version_keys == motif_names] <- motif_names[motif_version_keys == motif_names]
motif_families <- family_lookup[motif_version_keys]

# Check for unmapped
n_unmapped <- sum(is.na(motif_families))
if (n_unmapped > 0) {
  cat(sprintf("[WARN] %d motifs have no family assignment\n", n_unmapped))
  # Assign to "Unknown" family
  motif_families[is.na(motif_families)] <- "Unknown"
}

families <- unique(motif_families)
cat(sprintf("[INFO] %d unique families\n", length(families)))

## ---- Per-family clustering ----

all_signatures <- list()
cluster_table <- list()  # MOTIFN -> semicolon-joined members
summary_rows <- list()

sig_counter <- 0

for (fam in sort(families)) {
  fam_idx <- which(motif_families == fam)
  fam_motifs <- At_Motif[fam_idx]
  n_motifs <- length(fam_motifs)

  if (n_motifs == 0) next

  if (n_motifs == 1) {
    # Singleton: pass through as its own signature
    sig_counter <- sig_counter + 1
    sig_name <- paste0("MOTIF", sig_counter)
    member_name <- names(fam_motifs)[1]

    # Create pfm object for motifStack compatibility
    pfm_obj <- new("pfm", mat = fam_motifs[[1]]@mat, name = member_name)

    all_signatures[[sig_name]] <- pfm_obj
    cluster_table[[sig_name]] <- member_name

    summary_rows[[length(summary_rows) + 1]] <- data.frame(
      family = fam, n_input = 1, n_signatures = 1,
      stringsAsFactors = FALSE
    )
    cat(sprintf("[INFO] %s: 1 motif -> 1 signature (singleton)\n", fam))
    next
  }

  if (n_motifs == 2) {
    # Two motifs: cluster but motifSignature needs >= 3 leaves for phylog
    # Compare them; if similar merge, otherwise keep both
    # Use motifStack's clusterMotifs to get distance
    pfms_fam <- mapply(fam_motifs, names(fam_motifs),
                       FUN = function(.pfm, .name) {
                         new("pfm", mat = fam_motifs[[.name]]@mat, name = .name)
                       })

    tryCatch({
      hc <- clusterMotifs(pfms_fam)
      # If height < 0.5, merge into one signature; otherwise keep separate
      if (max(hc$height) < 0.5) {
        sig_counter <- sig_counter + 1
        sig_name <- paste0("MOTIF", sig_counter)
        # Average the two PFMs (simple consensus)
        # Use first motif's matrix dimensions
        m1 <- pfms_fam[[1]]@mat
        m2 <- pfms_fam[[2]]@mat
        # Pad shorter to match longer
        maxw <- max(ncol(m1), ncol(m2))
        if (ncol(m1) < maxw) m1 <- cbind(m1, matrix(0.25, 4, maxw - ncol(m1)))
        if (ncol(m2) < maxw) m2 <- cbind(m2, matrix(0.25, 4, maxw - ncol(m2)))
        avg_mat <- (m1 + m2) / 2
        merged_name <- paste0(names(fam_motifs), collapse = ";")
        pfm_merged <- new("pfm", mat = avg_mat, name = merged_name)
        all_signatures[[sig_name]] <- pfm_merged
        cluster_table[[sig_name]] <- merged_name
        cat(sprintf("[INFO] %s: 2 motifs -> 1 signature (merged, height=%.3f)\n",
                    fam, max(hc$height)))
        summary_rows[[length(summary_rows) + 1]] <- data.frame(
          family = fam, n_input = 2, n_signatures = 1,
          stringsAsFactors = FALSE
        )
      } else {
        # Keep both as separate signatures
        for (j in 1:2) {
          sig_counter <- sig_counter + 1
          sig_name <- paste0("MOTIF", sig_counter)
          all_signatures[[sig_name]] <- pfms_fam[[j]]
          cluster_table[[sig_name]] <- names(fam_motifs)[j]
        }
        cat(sprintf("[INFO] %s: 2 motifs -> 2 signatures (distinct, height=%.3f)\n",
                    fam, max(hc$height)))
        summary_rows[[length(summary_rows) + 1]] <- data.frame(
          family = fam, n_input = 2, n_signatures = 2,
          stringsAsFactors = FALSE
        )
      }
    }, error = function(e) {
      # Fallback: keep both as separate signatures
      for (j in 1:2) {
        sig_counter <<- sig_counter + 1
        sig_name <- paste0("MOTIF", sig_counter)
        all_signatures[[sig_name]] <<- pfms_fam[[j]]
        cluster_table[[sig_name]] <<- names(fam_motifs)[j]
      }
      cat(sprintf("[WARN] %s: 2 motifs clustering failed (%s), keeping both\n",
                  fam, conditionMessage(e)))
      summary_rows[[length(summary_rows) + 1]] <<- data.frame(
        family = fam, n_input = 2, n_signatures = 2,
        stringsAsFactors = FALSE
      )
    })
    next
  }

  # n_motifs >= 3: full motifStack clustering
  pfms_fam <- mapply(fam_motifs, names(fam_motifs),
                     FUN = function(.pfm, .name) {
                       new("pfm", mat = fam_motifs[[.name]]@mat, name = .name)
                     })

  tryCatch({
    hc <- clusterMotifs(pfms_fam)
    phylog <- hclust2phylog(hc)
    leaves <- names(phylog$leaves)
    pfms_ordered <- pfms_fam[leaves]

    # Extract motif signatures with same cutoff as original
    motifSig <- motifSignature(pfms_ordered, phylog,
                               cutoffPval = 0.0001, min.freq = 1)
    sig <- signatures(motifSig)
    n_sig <- length(sig)

    for (j in 1:n_sig) {
      sig_counter <- sig_counter + 1
      sig_name <- paste0("MOTIF", sig_counter)

      # sig[[j]] is a pfm object with name = semicolon-joined members
      all_signatures[[sig_name]] <- sig[[j]]
      cluster_table[[sig_name]] <- sig[[j]]@name
    }

    cat(sprintf("[INFO] %s: %d motifs -> %d signatures\n", fam, n_motifs, n_sig))
    summary_rows[[length(summary_rows) + 1]] <- data.frame(
      family = fam, n_input = n_motifs, n_signatures = n_sig,
      stringsAsFactors = FALSE
    )
  }, error = function(e) {
    # Fallback: each motif becomes its own signature
    for (j in seq_along(fam_motifs)) {
      sig_counter <<- sig_counter + 1
      sig_name <- paste0("MOTIF", sig_counter)
      pfm_obj <- new("pfm", mat = fam_motifs[[j]]@mat,
                      name = names(fam_motifs)[j])
      all_signatures[[sig_name]] <<- pfm_obj
      cluster_table[[sig_name]] <<- names(fam_motifs)[j]
    }
    cat(sprintf("[WARN] %s: clustering failed (%s), keeping all %d as singletons\n",
                fam, conditionMessage(e), n_motifs))
    summary_rows[[length(summary_rows) + 1]] <<- data.frame(
      family = fam, n_input = n_motifs, n_signatures = n_motifs,
      stringsAsFactors = FALSE
    )
  })
}

cat(sprintf("\n[INFO] Total: %d signatures from %d motifs across %d families\n",
            length(all_signatures), length(At_Motif), length(families)))

## ---- Write output ----

# 1. Write MEME file
# Convert pfm list to universalmotif objects for write_meme
um_list <- lapply(names(all_signatures), function(sname) {
  pfm_obj <- all_signatures[[sname]]
  mat <- pfm_obj@mat  # 4 x width probability matrix
  # Create universalmotif object
  create_motif(mat, type = "PPM", name = pfm_obj@name,
               alphabet = "DNA")
})

write_meme(um_list, file.path(SIG_DIR, "At_Motif_SignatureDB.meme"), overwrite = TRUE)
cat(sprintf("[INFO] Wrote At_Motif_SignatureDB.meme (%d signatures)\n",
            length(um_list)))

# 2. Write cluster table
cluster_lines <- paste0(names(cluster_table), "\t",
                        unlist(cluster_table))
writeLines(cluster_lines, file.path(SIG_DIR, "At_MotifClusters.txt"))
cat(sprintf("[INFO] Wrote At_MotifClusters.txt (%d lines)\n",
            length(cluster_lines)))

# 3. Write summary
summary_df <- do.call(rbind, summary_rows)
write.table(summary_df, file.path(SIG_DIR, "v3_clustering_summary.tsv"),
            sep = "\t", row.names = FALSE, quote = FALSE)
cat(sprintf("[INFO] Wrote v3_clustering_summary.tsv\n"))

# 4. Save R objects for later inspection
Object_motifStack_v3 <- list(
  all_signatures = all_signatures,
  cluster_table = cluster_table,
  family_lookup = family_lookup,
  summary = summary_df
)
saveRDS(Object_motifStack_v3, file = "At_Full_Motif_SignatureDB_v3.rds")
cat(sprintf("[INFO] Wrote At_Full_Motif_SignatureDB_v3.rds\n"))

cat("\n[DONE] Per-family motif clustering complete.\n")
