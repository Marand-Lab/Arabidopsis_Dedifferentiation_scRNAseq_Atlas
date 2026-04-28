## evaluate remnant somatic programs ##

# arguments
args <- commandArgs(T)
celltype <- as.character(args[1])
cluster <- as.numeric(args[2])

# load libraries
library(Seurat)
library(mgcv)
library(scales)
library(parallel)

# functions
PTdegs <- function(counts,                  
                   pseudotime,              
                   family = c("nb", "gaussian"),
                   offset = NULL,           
                   nknots = 6,
                   grid_length = 100,
                   t0_quantile = 0.01,    
                   fdr = 0.05,
                   lfc_threshold = 0.25,
                   n_cores = 1,
                   verbose = TRUE) {
  
  # input
  family <- match.arg(family)
  n_genes <- nrow(counts)
  n_cells <- ncol(counts)
  
  # --- Setup ---
  t0 <- as.numeric(min(pseudotime))#, t0_quantile))
  grid <- seq(min(pseudotime), max(pseudotime), length.out = grid_length)
  
  newdata_grid <- data.frame(pt = grid)
  newdata_t0   <- data.frame(pt = t0)
  
  if (verbose) {
    message("Running ", n_genes, " genes on ", n_cores, " cores...")
  }
  
  # --- Worker function ---
  fit_one_gene <- function(g) {
    y <- as.numeric(counts[g, ])
    
    df <- data.frame(y = y, pt = pseudotime)
    
    fam <- switch(family,
                  nb = nb(),
                  gaussian = gaussian())
    
    fit <- try(
      gam(y ~ s(pt, k = nknots),
          data = df,
          family = fam,
          offset = offset),
      silent = TRUE
    )
    
    if (inherits(fit, "try-error")) {
      return(list(Delta = rep(NA, grid_length),
                  p = rep(NA, grid_length)))
    }
    
    # Design matrices
    X_grid <- predict(fit, newdata = newdata_grid, type = "lpmatrix")
    X_t0   <- predict(fit, newdata = newdata_t0,   type = "lpmatrix")
    
    beta <- coef(fit)
    V    <- vcov(fit)
    
    X_diff <- sweep(X_grid, 2, X_t0, FUN = "-")
    
    Delta <- as.numeric(X_diff %*% beta)
    
    XV <- X_diff %*% V
    var_Delta <- rowSums(XV * X_diff)
    se <- sqrt(pmax(var_Delta, 1e-8))
    
    z <- Delta / se
    p <- 2 * pnorm(-abs(z))
    
    list(Delta = Delta, p = p)
  }
  
  # --- Parallel execution ---
  res_list <- mclapply(seq_len(n_genes),
                         fit_one_gene,
                         mc.cores = n_cores)
  
  
  if (verbose) message("Combining results...")
  
  # --- Combine results ---
  Delta_mat <- do.call(rbind, lapply(res_list, `[[`, "Delta"))
  P_mat     <- do.call(rbind, lapply(res_list, `[[`, "p"))
  
  # --- Multiple testing ---
  P_adj <- apply(P_mat, 2, p.adjust, method = "BH")
  
  DE_mask <- (P_adj < fdr) & (abs(Delta_mat) > lfc_threshold)
  DEG_counts <- colSums(DE_mask, na.rm = TRUE)
  
  if (verbose) message("Done.")
  
  return(list(
    grid = grid,
    Delta = Delta_mat,
    pval = P_mat,
    padj = P_adj,
    DE_mask = DE_mask,
    DEG_counts = DEG_counts,
    t0 = t0
  ))
}

##########################################################################
## load data
##########################################################################

# load reverse pseudotime data
dir <- "/nfs/turbo/lsa-YOURNAME/YOURNAME_home/Arabidopsis_Protoplast_Dedifferentiation/step5_pseudocell_trajectory/probabilistic_walks/"
pw <- list.files(path=dir, pattern="PROB_WALKS*")
pwd <- lapply(pw, function(z){
  unique(unlist(readRDS(paste0(dir,z))$paths))
})
names(pwd) <- gsub("PROB_WALKS_","",pw)
names(pwd) <- gsub("cluster\\.","cluster_",names(pwd))
names(pwd) <- gsub("\\.rds","",names(pwd))

# meta data
m <- read.table("/nfs/turbo/lsa-YOURNAME/YOURNAME_home/Arabidopsis_Protoplast_Dedifferentiation/step5_pseudocell_trajectory/diffusion_pseudotime.metadata.all_cells.palantir.11.19.2025.knn30.real_time.cellfate.txt")
pt <- m$consensus_pseudotime
names(pt) <- rownames(m)

# gene annotation
ann <- read.table("/nfs/turbo/lsa-YOURNAME/shared_data/arabidopsis/reference_data/annotations/Arabidopsis_thaliana.TAIR10.58.annotation.txt", header=F)

# seurat object
obj <- readRDS("/nfs/turbo/lsa-YOURNAME/YOURNAME_home/Arabidopsis_Protoplast_Dedifferentiation/step5_pseudocell_trajectory/step1_palantir_obj_100DC_harmony_knn.15.10.21.2025.knn30.rds")
expr <- obj@assays$RNA$data


########################################################################### 
## model DEGs at discrete pseudotime points for each trajectory
########################################################################### 

# iterate over trajectories
z <- paste0("cluster_",cluster,".",celltype)
  
# verbose
message("- analyzing trajectory = ", z)

# select cells
cids <- pwd[[z]]
tpt <- m[cids,]$consensus_pseudotime
names(tpt) <- cids
tpt <- sort(tpt, decreasing=F)
texpr <- expr[,names(tpt)]
texpr <- texpr[Matrix::rowSums(texpr>0)>2,] # at least 3 cells express gene

# run DEG counts
res <- PTdegs(texpr, tpt, family="gaussian", n_cores=12)
saveRDS(res, file=paste0(z,".nDEGs_rt_grid.rds"))
