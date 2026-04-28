# model expression

# load libraries
library(mgcv)
library(Seurat)
library(parallel)
library(lmtest)

# function
testTrajGN <- function(obj, pt, downsample=NULL, threads=1){
  
  # subset traj
  message(" - Sort cells ...")
  d0 <- subset(pt, pt$treatment=="D0")
  d0$hormone <- T
  d0$library <- paste0(d0$library, "-1")
  pt <- rbind(pt, d0)
  pt <- pt[!pt$library %in% c("YJ3","YJ4"),]
  
  # down-sample?
  if(! is.null(downsample)){
    new <- lapply(unique(pt$library), function(z){
      met <- subset(pt, pt$library==z)
      if(downsample > nrow(met)){
        return(met)
      }else{
        met[sample(nrow(met), downsample, replace=F),]
      }
    })
    new <- do.call(rbind, new)
    pt <- new
  }
  
  pt <- pt[order(pt$real_time, decreasing=F),]
  binary <- obj@assays$RNA$data[,pt$cellID]
  binary <- binary[,Matrix::colSums(binary)>0]
  binary <- binary[Matrix::rowSums(binary)>0,]
  pt$hormone <- as.factor(pt$hormone)
  pt$tech <- as.factor(pt$tech)
  pt$batch <- as.factor(pt$batch)
  pt$library <- as.factor(pt$library)
  
  # generalized additive model for logistic regression
  message(" - running generalized additive model ...")
  results <- mclapply(seq(1:nrow(binary)),function(x){
    if((x %% 500)==0){message("iterated over ",x, "/",nrow(binary), " genes...")}
    acc <- binary[x,]
    df <- cbind(pt, acc)
    mod1 <- gam(acc ~ tech + batch + log_umi + hormone + s(real_time, by=hormone) + s(library, bs="re"), data=df, method="REML")
    mod2 <- gam(acc ~ tech + batch + log_umi + s(real_time) + s(library, bs="re"), data=df, method="REML")
    res <- summary(mod1)
    mm1 <- update(mod1, method="ML")
    mm2 <- update(mod2, method="ML")
    glrt <- anova(mm1, mm2, test = "Chisq")[2,5]
    h.eff <- res$p.table[6,1]
    h.std <- res$p.table[6,2]
    h.pval <- res$p.pv[6]
    pt.h <- res$s.t[2,4]
    pt.nh <- res$s.t[1,4]
    pt.hf <- res$s.t[2,3]
    pt.nf <- res$s.t[1,3]
    return(data.frame(geneID=rownames(binary)[x],
                      hormone_beta=h.eff,
                      hormone_beta_se=h.std,
                      hormone_pval=h.pval,
                      pt_hormone_pval=pt.h,
                      pt_nonhormone_pval=pt.nh,
                      pt_hormone_F=pt.hf,
                      pt_nonhormone_F=pt.nf,
                      lrt=glrt,
                      row.names=rownames(binary)[x]))
    
  }, mc.cores=threads)
  results <- do.call(rbind, results)
  
  # return
  return(results)
  
}

# load data
meta <- read.table("diffusion_pseudotime.metadata.all_cells.palantir_pt_entrop.11.06.2025.knn30.inferred_age.txt")
obj <- readRDS("step1_palantir_obj_100DC_harmony_knn.15.10.21.2025.knn30.rds")

# subset meta?
meta <- subset(meta, meta$age != "D6")

#  get differential expressed genes
results <- testTrajGN(obj=obj,
                      pt=meta,
                      downsample=2000,
                      threads=24)

# save results
message(" - saving results to disk")
saveRDS(results, file="GAM_LMM_LRT_real_time_expression.D0_D4_ds2000.rds")
