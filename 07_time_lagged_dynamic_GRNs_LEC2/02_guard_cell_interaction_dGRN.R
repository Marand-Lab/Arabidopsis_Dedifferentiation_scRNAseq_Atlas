#################################################
# LEC2 co-expression 
#################################################

# load libraries
library(pheatmap)
library(RColorBrewer)
library(viridis)
library(Seurat)
library(glmnet)
library(caret)
library(doMC)
library(RANN)
library(Matrix)
library(parallel)
library(irlba)
library(proxyC)
library(mclust)
library(qlcMatrix)
library(fgsea)
library(brms)
library(dplyr)
library(future)

# functions
smooth.data <- function(xx,
                        k=25,
                        step=3,
                        npcs=30,
                        df=NULL,
                        n.perms=10,
                        threads=1){
  
  # verbose
  message(" - imputing gene expression ...")
  
  # input
  x <- xx$data
  data.use <- x
  rds <- xx$embedding
  
  # hidden functions
  .markov_affinity <- function(com, dat.use, step=3, k=15){
    
    # get KNN
    knn.graph <- nn2(com, k=k, eps=0)$nn.idx
    j <- as.numeric(x = t(x = knn.graph))
    i <- ((1:length(x = j)) - 1) %/% k + 1
    edgeList = data.frame(i, j, 1)
    A = sparseMatrix(i = edgeList[,1], j = edgeList[,2], x = edgeList[,3])
    
    # Smooth graph
    A = A + t(A)
    A = A / Matrix::rowSums(A)
    step.size = step
    if(step.size > 1){
      for(i in 1:step.size){
        A = A %*% A
      }
    }
    
    # smooth data
    im.activity <- t(A %*% t(dat.use))
    colnames(im.activity) <- colnames(dat.use)
    rownames(im.activity) <- rownames(dat.use)
    
    # return sparse Matrix
    return(im.activity)
    
  }
  
  # verbose
  if(!is.null(rds)){
    if(npcs > ncol(rds)){
      npcs <- ncol(rds)
    }
    pcs <- rds[colnames(x),c(1:npcs)]
  }else{
    
    # verbose
    message("REDUCED DIMENSIONS REQUIRED...")
  }
  
  # check number of cells
  num.cells <- nrow(pcs)
  if(num.cells > 50000){
    message(" - number of cells > 50K, initiating multiple runs ...")
    max.val <- 10000
    x.vec <- seq(1:nrow(pcs))
    perms <- mclapply(seq(1:n.perms), function(z){
      message(" - initialized ", z, " runs ...")
      x.vec.r <- x.vec[sample(length(x.vec))]
      pcs.s <- split(rownames(pcs), ceiling(x.vec.r/max.val))
      outs <- lapply(pcs.s, function(ids){
        y <- pcs[ids,]
        dat <- data.use[,ids]
        .markov_affinity(y, dat, step=step, k=k)
      })
      outs <- do.call(cbind, outs)
      outs <- outs[,colnames(data.use)]
      return(outs)
    }, mc.cores=threads)
    message(" - averaging runs ...")
    imputed.activity <- Reduce("+", perms) / length(perms)
    return(imputed.activity)
    
  }else{
    imputed.activity <- .markov_affinity(pcs, data.use, step=step, k=k)
    return(imputed.activity)
  }
}
bayesMCMC <- function(){
  
  # prep model data
  make_gene_df <- function(gene){
    data.frame(y = as.numeric(pb_mat[gene, ]),
               t(lag_mat),
               condition = pb_meta$hormone,
               pseudotime = pb_meta$diffusion_pt,
               library_id = pb_meta$library)
  }
  
  # formula
  make_formula <- function(regulators){
    lag_terms <- paste(regulators, collapse = " + ")
    as.formula(
      paste0(
        "y ~ condition + ",
        "s(pseudotime, k = 6) + ",
        lag_terms, " + condition:(", lag_terms, ") + ",
        "(1 | library_id)"
      )
    )
  }
  
  # priors
  priors <- c(prior(normal(0, 1), class = "Intercept"),
              prior(horseshoe(df = 1), class = "b"),
              prior(exponential(1), class = "sd", group = "library_id"))
  
  # fit one gene functions
  fit_one_gene <- function(gene, lag_mat){
    brm(formula = make_formula(rownames(lag_mat)),
        data = make_gene_df(gene),
        family = gaussian(),
        prior = priors,
        chains = 4,
        iter = 2000,
        cores = 4,
        control = list(adapt_delta = 0.95),
        silent = TRUE)
  }
  
  # model LEC2 expression
  res <- fit_one_gene(lec2, lag_mat)
  
}
glmnetGRN <- function(cond=TRUE, pb_meta, pb_mat, lag_mat, alpha=1){
  
  # indices of samples in this condition
  idx <- which(pb_meta$hormone == cond)
  
  X <- t(lag_mat[, idx, drop=FALSE])  # samples × regulators
  X <- X[, apply(X, 2, var) > 1e-6, drop=FALSE]
  X <- as.matrix(scale(X))
  Y_mat <- pb_mat[, idx, drop=FALSE]  # genes × samples
  vars <- apply(Y_mat, 1, var)
  Y_mat <- Y_mat[vars > 1e-6,]
  
  # include pseudotime as additional predictor
  X <- cbind(X, pseudotime = pb_meta$diffusion_pt[idx])
  
  # optional weights = number of cells per pseudobulk
  weights <- pb_meta$cells[idx]
  
  # genes
  targets_to_fit <- rownames(Y_mat)
  
  # function
  fit_gene <- function(gene_idx) {
    if((gene_idx %% 100)==0){message("-- iterated over ",gene_idx," records...")}
    y <- as.numeric(Y_mat[gene_idx, ])
    if (length(unique(y)) < 4) return(NULL)
    nfolds_use <- min(5, length(y)-1)
    fit <- cv.glmnet(
      X, y,
      alpha = alpha,           # Lasso/Ridge/EN
      standardize = T,
      weights = weights,
      nfolds = nfolds_use
    )
    co <- coef(fit, s = "lambda.min")
    # return as named vector
    out <- co[,1]
    names(out) <- rownames(co)
    return(out)
  }
  fits <- lapply(seq_len(nrow(Y_mat)), fit_gene)
  names(fits) <- rownames(Y_mat)
  all.fits <- do.call(rbind, fits)
  
  return(all.fits)
  
}
test_condition_diff_glmnet <- function(y, 
                                       lag_mat, 
                                       condition, 
                                       pseudotime = NULL, 
                                       weights = NULL,
                                       alpha = 1, 
                                       nfolds = 5, 
                                       n_perm = 100, 
                                       seed = 1) {
  # set set
  set.seed(seed)
  
  # condition
  message("Setting up input data ...")
  condition <- factor(condition)
  stopifnot(nlevels(condition) == 2)
  
  # base predictors
  X_base <- t(lag_mat)
  colnames(X_base) <- make.names(colnames(X_base), unique = TRUE)
  
  # interaction predictors
  cond_bin <- as.numeric(condition == levels(condition)[2])
  X_int <- X_base * cond_bin
  colnames(X_int) <- paste0(colnames(X_base), "_cond")
  
  # full design matrix
  X <- cbind(X_base, X_int)
  if (!is.null(pseudotime)) {
    X <- cbind(X, pseudotime = pseudotime)
  }
  
  # remove zero-variance predictors
  keep <- apply(X, 2, var) > 1e-6
  X <- X[, keep, drop = FALSE]
  
  # skip degenerate responses
  if (var(y) < 1e-6 || length(unique(y)) < 3) {
    return(NULL)
  }
  
  # select folds
  nfolds_use <- min(nfolds, length(y)-1)
  
  # fit observed model
  message("Fitting regularized model ...")
  fit <- cv.glmnet(
    X, y,
    alpha = alpha,
    weights = weights,
    standardize = TRUE,
    nfolds = nfolds_use)
  
  co_obs <- coef(fit, s = "lambda.min")
  co_obs <- as.matrix(co_obs)
  
  # extract interaction terms
  int_cols <- grep("_cond$", rownames(co_obs))
  if (length(int_idx) == 0) return(NULL)
  obs_beta <- abs(co_obs[int_idx, , drop = FALSE])
  
  # permutation test
  message("Running permutation ...")
  perm_beta <- matrix(NA, nrow = length(int_idx), ncol = n_perm)
  
  for (b in seq_len(n_perm)) {
    perm_cond <- sample(cond_bin)
    
    Xp_int <- X_base * perm_cond
    colnames(Xp_int) <- colnames(X_int)
    
    Xp <- cbind(X_base, Xp_int)
    if (!is.null(pseudotime)) {
      Xp <- cbind(Xp, pseudotime = pseudotime)
    }
    
    Xp <- Xp[, keep, drop = FALSE]
    
    fit_p <- cv.glmnet(
      Xp, y,
      alpha = alpha,
      weights = weights,
      standardize = TRUE,
      nfolds = nfolds_use)
    
    co_p <- coef(fit_p, s = "lambda.min")
    co_p <- as.matrix(co_p)
    
    perm_beta[, b] <- abs(co_p[int_idx, , drop = FALSE])
  }
  
  # empirical p-values
  mu <- rowMeans(perm_beta)
  sigma <- apply(perm_beta, 1, sd)
  zscore <- abs((obs_beta[,1] - mu)/sigma)
  zscore[is.na(zscore)] <- 0
  p_value <- 2 * (1 - pnorm(zscore))
  
  data.frame(
    regulator = sub("_cond$", "", rownames(co_obs)[int_idx]),
    beta_diff = co_obs[int_idx, ],
    p_value = p_value,
    stringsAsFactors = FALSE
  )
}

# single grn
infer_lagged_grn_single_condition <- function(
    expr,
    meta,
    regulators,
    condition_value,            # specify which condition to analyze
    condition_col = "condition",
    time_col = "pseudotime",
    window_size = 7,
    threads = 1,
    lag = 1,
    n_perm = 200,
    alpha = 1,
    nfolds = 5,
    min_var = 1e-6,
    verbose = TRUE
) {
  
  ## -----------------------------
  ## 1. Checks
  ## -----------------------------
  stopifnot(window_size %% 2 == 1)
  stopifnot(lag == 1)
  stopifnot(condition_col %in% colnames(meta))
  stopifnot(time_col %in% colnames(meta))
  
  if (verbose) message("Subsetting condition: ", condition_value)
  
  ## -----------------------------
  ## 2. Subset condition
  ## -----------------------------
  keep_cells <- meta[[condition_col]] == condition_value
  expr <- expr[, keep_cells]
  meta <- meta[keep_cells, ]
  
  ## order by time
  ord <- order(meta[[time_col]])
  expr <- expr[, ord]
  meta <- meta[ord, ]
  
  ## -----------------------------
  ## 3. Sliding window smoothing
  ## -----------------------------
  if (verbose) message("Applying sliding-window smoothing...")
  
  half <- floor(window_size / 2)
  expr_sm <- expr * NA_real_
  n <- ncol(expr)
  
  for (i in seq_len(n)) {
    lo <- max(1, i - half)
    hi <- min(n, i + half)
    expr_sm[, i] <- rowMeans(expr[, lo:hi, drop = FALSE])
  }
  
  ## -----------------------------
  ## 4. Regulators / targets
  ## -----------------------------
  regulators <- intersect(regulators, rownames(expr_sm))
  targets <- rownames(expr_sm)
  
  if (verbose) {
    message("Using ", length(regulators), " regulators")
    message("Modeling ", length(targets), " target genes")
  }
  
  ## -----------------------------
  ## 5. Lag matrix
  ## -----------------------------
  if (verbose) message("Constructing lag matrix...")
  
  lag_mat <- matrix(
    NA_real_,
    nrow = length(regulators),
    ncol = ncol(expr_sm),
    dimnames = list(regulators, colnames(expr_sm))
  )
  
  for (i in seq_len(ncol(expr_sm))) {
    if (i == 1) next
    lag_mat[, i] <- expr_sm[regulators, i - 1]
  }
  
  keep <- colSums(is.na(lag_mat)) == 0
  expr_sm <- expr_sm[, keep]
  lag_mat <- lag_mat[, keep]
  meta <- meta[keep, ]
  
  ## -----------------------------
  ## 6. Design matrix
  ## -----------------------------
  X <- t(lag_mat)
  colnames(X) <- make.names(colnames(X), unique = TRUE)
  
  keep_cols <- apply(X, 2, var) > min_var
  X <- X[, keep_cols, drop = FALSE]
  
  reg_names <- colnames(X)
  
  ## -----------------------------
  ## 7. Safe glmnet
  ## -----------------------------
  fit_gene <- function(y, Xmat) {
    if (var(y) < min_var || length(unique(y)) < 3) return(NULL)
    suppressWarnings(
      cv.glmnet(
        Xmat, y,
        alpha = alpha,
        standardize = TRUE,
        nfolds = min(nfolds, length(y) - 1)
      )
    )
  }
  
  ## -----------------------------
  ## 8. Permutation test
  ## -----------------------------
  test_gene <- function(y) {
    
    fit <- fit_gene(y, X)
    if (is.null(fit)) return(NULL)
    
    co_obs <- as.matrix(coef(fit, s = "lambda.min"))
    idx <- intersect(reg_names, rownames(co_obs))
    if (length(idx) == 0) return(NULL)
    
    obs <- abs(co_obs[idx, 1])
    perm_vals <- matrix(NA, length(idx), n_perm)
    
    for (b in seq_len(n_perm)) {
      
      Xp <- X
      perm_order <- sample(nrow(Xp))
      Xp <- Xp[perm_order, , drop = FALSE]
      
      fit_p <- try(fit_gene(y, Xp), silent = TRUE)
      if (inherits(fit_p, "try-error") || is.null(fit_p)) next
      
      co_p <- as.matrix(coef(fit_p, s = "lambda.min"))
      perm_vals[, b] <- abs(co_p[idx, 1])
    }
    
    mu <- rowMeans(perm_vals, na.rm = TRUE)
    sigma <- apply(perm_vals, 1, sd, na.rm = TRUE)
    
    zscore <- (obs - mu) / sigma
    zscore[is.na(zscore)] <- 0
    
    pvals <- 1 - pnorm(zscore)
    
    epvals <- (1 + rowSums(perm_vals >= obs, na.rm = TRUE)) /
      (1 + n_perm)
    
    data.frame(
      regulator = idx,
      beta = co_obs[idx, 1],
      perm_mu = mu,
      perm_sigma = sigma,
      abs_zscore = zscore,
      p_value = pvals,
      e_pval = epvals,
      stringsAsFactors = FALSE
    )
  }
  
  ## -----------------------------
  ## 9. Run across targets
  ## -----------------------------
  if (verbose) message("Fitting models...")
  
  res_list <- parallel::mclapply(targets, function(g) {
    
    y <- as.numeric(expr_sm[g, ])
    tt <- test_gene(y)
    
    if (is.null(nrow(tt))) return(NULL)
    tt$target <- g
    tt
    
  }, mc.cores = threads)
  
  names(res_list) <- targets
  
  ## -----------------------------
  ## 10. Assemble GRN
  ## -----------------------------
  grn <- do.call(rbind, lapply(res_list, function(r) {
    if (is.null(r)) return(NULL)
    r
  }))
  
  if (is.null(grn)) {
    warning("No valid models fit.")
    return(NULL)
  }
  
  grn$padj <- p.adjust(grn$p_value, method = "BH")
  
  if (verbose) {
    message("Done. Significant edges (FDR < 0.05): ",
            sum(grn$padj < 0.05))
  }
  
  return(list(grn = grn, res = res_list))
}

# full grn compute
infer_lagged_grn_sc <- function(
    expr,                          # genes × cells (dgCMatrix or matrix)
    meta,                          # data.frame, rows = cells
    regulators,                    # character vector of TF genes
    condition_col = "condition",   # treatment
    time_col = "pseudotime",       # time column
    window_size = 7,               # sliding window size (odd integer)
    threads = 1,                   # number of cores
    lag = 1,                       # currently only lag = 1 supported
    n_perm = 200,                  # permutations for p-values
    alpha = 1,                     # lasso (1), ridge (0), or elastic net (0 < alpha < 1) 
    nfolds = 5,                    # CV folds
    min_var = 1e-6,                # variance cutoff for genes
    verbose = TRUE
) {
  
  ## -----------------------------
  ## 1. Checks
  ## -----------------------------
  stopifnot(window_size %% 2 == 1)
  stopifnot(lag == 1)
  
  stopifnot(condition_col %in% colnames(meta))
  stopifnot(time_col %in% colnames(meta))
  
  if (verbose) message("Processing single-cell data...")
  
  ## -----------------------------
  ## 3. Order cells
  ## -----------------------------
  ord <- order(meta[[condition_col]], meta[[time_col]])
  expr <- expr[, ord]
  meta <- meta[ord, ]
  
  cond <- factor(meta[[condition_col]])
  cond_bin <- as.numeric(cond == levels(cond)[2])
  
  ## -----------------------------
  ## 4. Sliding-window smoothing
  ## -----------------------------
  if (verbose) message("Applying sliding-window smoothing...")
  
  half <- floor(window_size / 2)
  expr_sm <- expr * NA_real_
  
  for (c in levels(cond)) {
    idx <- which(cond == c)
    n <- length(idx)
    
    for (i in seq_along(idx)) {
      lo <- max(1, i - half)
      hi <- min(n, i + half)
      expr_sm[, idx[i]] <- rowMeans(expr[, idx[lo:hi], drop = FALSE])
    }
  }
  
  ## -----------------------------
  ## 5. Regulators & targets
  ## -----------------------------
  regulators <- intersect(regulators, rownames(expr_sm))
  targets <- rownames(expr_sm)
  
  if (verbose) {
    message("Using ", length(regulators), " regulators")
    message("Modeling ", length(targets), " target genes")
  }
  
  ## -----------------------------
  ## 6. Lagged regulator matrix
  ## -----------------------------
  if (verbose) message("Constructing lagged design matrix...")
  
  lag_mat <- matrix(
    NA_real_,
    nrow = length(regulators),
    ncol = ncol(expr_sm),
    dimnames = list(regulators, colnames(expr_sm))
  )
  
  for (c in levels(cond)) {
    idx <- which(cond == c)
    for (i in seq_along(idx)) {
      if (i == 1) next
      lag_mat[, idx[i]] <- expr_sm[regulators, idx[i - 1]]
    }
  }
  
  ## Drop cells without lag
  keep <- colSums(is.na(lag_mat)) == 0
  expr_sm <- expr_sm[, keep]
  lag_mat <- lag_mat[, keep]
  meta <- meta[keep, ]
  cond_bin <- cond_bin[keep]
  
  ## -----------------------------
  ## 7. Base design matrix
  ## -----------------------------
  X_base <- t(lag_mat)
  colnames(X_base) <- make.names(colnames(X_base), unique = TRUE)
  
  X_int <- X_base * cond_bin
  colnames(X_int) <- paste0(colnames(X_base), "_cond")
  
  X <- cbind(
    X_base,
    X_int,
    time = meta[[time_col]]
  )
  
  ## variance filter ONCE
  keep_cols <- apply(X, 2, var) > min_var
  X <- X[, keep_cols, drop = FALSE]
  
  ## Track interaction columns safely
  int_cols <- intersect(
    paste0(colnames(X_base), "_cond"),
    colnames(X)
  )
  
  base_map <- sub("_cond$", "", int_cols)
  
  ## -----------------------------
  ## 7. Safe glmnet fit
  ## -----------------------------
  fit_gene <- function(y, Xmat) {
    if (var(y) < min_var || length(unique(y)) < 3) return(NULL)
    suppressWarnings(cv.glmnet(
      Xmat, y,
      alpha = alpha,
      standardize = TRUE,
      nfolds = min(nfolds, length(y)-1)
    ))
  }
  
  ## -----------------------------
  ## 9. Permutation test
  ## -----------------------------
  test_gene <- function(y) {
    
    fit <- fit_gene(y, X)
    if (is.null(fit)) return(NULL)
    
    co_obs <- as.matrix(coef(fit, s = "lambda.min"))
    idx <- intersect(int_cols, rownames(co_obs))
    if (length(idx) == 0) return(NULL)
    
    obs <- abs(co_obs[idx, 1])
    perm_vals <- matrix(NA, length(idx), n_perm)
    
    for (b in seq_len(n_perm)) {
      #message(b, " permutations complete")
      perm_cond <- sample(cond_bin)
      
      ## IMPORTANT: reuse X, only update interaction terms
      Xp <- X
      Xp[, idx] <- X_base[, base_map, drop = FALSE] * perm_cond
      
      fit_p <- try(fit_gene(y, Xp), silent = TRUE)
      if (inherits(fit_p, "try-error") || is.null(fit_p)) {
        #message("permutation ", b, " failed...")
        next
      }else{
        co_p <- as.matrix(coef(fit_p, s = "lambda.min"))
        perm_vals[, b] <- abs(co_p[idx, 1])
      }
      
    }
    
    # empirical p-values
    mu <- rowMeans(perm_vals, na.rm=T)
    sigma <- apply(perm_vals, 1, sd, na.rm=T)
    zscore <- (obs - mu)/sigma
    zscore[is.na(zscore)] <- 0
    pvals <- 1 - pnorm(zscore)
    epvals <- (1 + rowSums(perm_vals >= obs, na.rm = TRUE)) /
      (1 + n_perm)
    
    data.frame(
      regulator = base_map,
      perm_sigma = sigma,
      perm_mu = mu,
      abs_zscore = zscore,
      beta_diff = co_obs[idx, 1],
      p_value = pvals,
      e_pval = epvals,
      stringsAsFactors = FALSE
    )
  }

  ## -----------------------------
  ## 10. Run GRN inference
  ## -----------------------------
  if (verbose) message("Fitting models...")
  
  res_list <- mclapply(targets, function(g) {
    if((which(targets==g)%%100)==0){message("--- iterated over ",which(targets==g)," target genes...")}
    y <- as.numeric(expr_sm[g, ])
    tt <- test_gene(y)
    tt$target <- g
    if(is.null(nrow(tt))){
      return(NULL)
    }else{
      return(tt)  
    }
    
  }, mc.cores=threads)
  
  names(res_list) <- targets
  
  ## -----------------------------
  ## 11. Assemble GRN
  ## -----------------------------
  grn <- do.call(rbind, lapply(names(res_list), function(g) {
  r <- res_list[[g]]
  if (is.null(r)) return(NULL)
    cbind(target = g, r)
  }))
   
  grn$padj <- p.adjust(grn$p_value, method = "BH")
   
  if (verbose) {
    message("Done. Significant edges (FDR < 0.05): ", sum(grn$padj < 0.05))
  }

  # return
  return(list(grn=grn, res=res_list))
  
}


# load data
meta <- read.table("diffusion_pseudotime.metadata.all_cells.palantir.11.19.2025.knn30.real_time.cellfate.txt")
pc <- readRDS("step1_palantir_obj_100DC_harmony_knn.15.10.21.2025.knn30.rds")
lec2 <- "AT1G28300"
ann <- read.table("Arabidopsis_thaliana.TAIR10.58.annotation.txt")

#################################################
# prep data
#################################################

# subset
meta <- subset(meta, meta$seurat_clusters==18)
shared <- intersect(rownames(meta), colnames(pc))
meta <- meta[shared,]
meta <- meta[order(meta$diffusion_pt, decreasing=T),]

# expression
exprs <- pc@assays$RNA$data[,rownames(meta)]
exprs <- exprs[Matrix::rowMeans(exprs) > 0,]

# create pseudobulks
meta$age <- gsub("D","",meta$age)
meta$age <- as.numeric(meta$age)
meta$grpID <- paste(meta$age,meta$hormone,meta$library,sep=".")
groups <- with(meta, interaction(age, hormone, library, drop = TRUE))

# exprs
pb_mat <- exprs
pb_meta <- meta[,c("cellID","age","hormone","library","diffusion_pt")] 

# subset for dynamic genes
dynamic <- lapply(seq(1:nrow(pb_mat)), function(z){
  if((z %% 100)==0){message("-- iterated over ",z," records...")}
  df <- pb_meta
  df$exprs <- as.numeric(pb_mat[z,])
  mod <- lm(exprs~age*hormone, data=df)
  res <- summary(mod)
  return(data.frame(geneID=rownames(pb_mat)[z],
                    age=res$coefficients[2,4],
                    hormone=res$coefficients[3,4],
                    age_hormone=res$coefficients[4,4]))
})
dynamic <- do.call(rbind, dynamic)
sig <- dynamic[dynamic$age < 0.05 | dynamic$hormone < 0.05 | dynamic$age_hormone < 0.05,]

# keep significant across time/treatment?
pb_mat <- pb_mat[rownames(pb_mat) %in% sig$geneID,]

# add t0 for hormone
t0 <- pb_mat[,rownames(pb_meta[pb_meta$age==0,])]
ids <- colnames(t0)
t0_meta <- pb_meta[ids,]
t0_meta$hormone <- TRUE
rownames(t0_meta) <- paste0(rownames(t0_meta),"-h")
t0_meta$cellID <- rownames(t0_meta)
colnames(t0) <- rownames(t0_meta)
t0_meta$library <- paste0(t0_meta$library,"-h")

# updated matrix and metadata
pb_meta <- rbind(pb_meta, t0_meta)
pb_mat <- cbind(pb_mat, t0)

# select regulators
tfs <- read.table("TF_genes_TAIR10.txt", header=T)
tfIDs <- tfs$geneID
regulators <- intersect(tfIDs, rownames(pb_mat))  


##########################################################################
# run
##########################################################################

# interactive 
message("conditional GRN...")
res <- infer_lagged_grn_sc(expr=pb_mat,
			   meta=pb_meta,
			   regulators=regulators,
			   condition_value=T,
			   condition_col="hormone",
			   time_col="age",
			   alpha=0,
			   n_perm=100,
			   threads=36)

# save results
saveRDS(res, file="dGRN_glmnet_ridge.interactive.resuts.rds")



