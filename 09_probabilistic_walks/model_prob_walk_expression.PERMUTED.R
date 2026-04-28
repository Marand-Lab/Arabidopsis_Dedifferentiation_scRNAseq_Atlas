## model expression dynamics across time ##

# arguments
args <- commandArgs(T)
cluster <- as.numeric(args[2])
celltype <- as.character(args[1])

# load libraries
library(mgcv)
library(parallel)
library(Seurat)

# functions
model_expr_walks <- function(x, paths, m, pseudo, threads=1){
  
  # aggregate paths
  cids <- unique(unlist(paths$paths))
  pt <- pseudo[cids]
  r.pt <- pt[sample(length(pt))]
  names(r.pt) <- cids
  pt <- r.pt
  pt <- pt[order(pt, decreasing=F)]
  walk.exp <- x[,names(pt)]
  walk.exp <- walk.exp[Matrix::rowSums(walk.exp) > 0,]
  mm <- m[colnames(walk.exp),]
  df <- data.frame(pt=pt, 
                   logumi=mm$log_umi,
                   library=mm$library)
  df$library <- as.factor(df$library)
  
  # iterate over genes
  mod.exp <- mclapply(seq(1:nrow(walk.exp)), function(y){
    
    # verbose
    if((y %% 1000)==0){message("processed ",y,"/", nrow(walk.exp)," genes ...")}
    
    # input data
    gene_exp <- as.numeric(walk.exp[y,])
    df <- cbind(df, gene_exp)
    
    # GAM models
    mod1 <- gam(gene_exp ~ logumi + pt + s(pt, bs="cr"), data=df)
    mod2 <- gam(gene_exp ~ logumi, data=df)
    
    # model summaries
    res <- summary(mod1)
    mm1 <- update(mod1, method="ML")
    mm2 <- update(mod2, method="ML")
    glrt <- anova(mm1, mm2, test = "Chisq")[2,5]
    
    # extract statistics
    pt.fix <- res$p.pv["pt"]
    pt.beta <- res$p.coeff["pt"]
    pt.beta.se <- res$se["pt"]
    pt.smooth.p <- res$s.table[1,"p-value"]
    
    # return results
    return(data.frame(geneID=rownames(walk.exp)[y],
                      beta=pt.beta,
                      beta_se=pt.beta.se,
                      pval=pt.fix,
                      gam_pval=pt.smooth.p,
                      lrt=glrt))
  }, mc.cores=threads)
  
  # merge results
  mod.exp <- do.call(rbind, mod.exp)
  mod.exp$fdr <- p.adjust(mod.exp$pval, method="fdr")
  mod.exp$gam_fdr <- p.adjust(mod.exp$gam_pval, method="fdr")
  mod.exp$lrt_fdr <- p.adjust(mod.exp$lrt, method="fdr")
  
  # return results
  return(mod.exp)
  
} 

# load data
obj <- readRDS("step1_palantir_obj_100DC_harmony_knn.15.10.21.2025.knn30.rds")
meta <- read.table("diffusion_pseudotime.metadata.all_cells.palantir.11.19.2025.knn30.real_time.cellfate.txt")
pt <- readRDS("All_Palantir_results.11.26.2025.rds")
walk <- readRDS(paste0("PROB_WALKS_cluster.",cluster,".",celltype,".rds"))

# extract data
exprs <- obj@assays$RNA$data
pseudo <- pt[[paste0("cluster_",cluster)]]
pt <- pseudo$Pseudotime
names(pt) <- rownames(pseudo)
pt <- pt[order(pt, decreasing=F)]

# run gene modeling
results <- model_expr_walks(x=exprs, paths=walk, m=meta, pseudo=pt)

# save results
saveRDS(results, file=paste0("cluster_",cluster,".",celltype,".GAM_SUMMARY_STATS.PERMUTED.rds"))
