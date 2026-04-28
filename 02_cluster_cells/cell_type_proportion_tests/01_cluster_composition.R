## analyze cluster composition ##

# libraries
library(reshape2)
library(gtools)
library(RColorBrewer)
library(pheatmap)
library(viridis)

# load data
a <- read.table("All.YJ1_19.logNorm.seurat.metaData.txt")

# counts
cnts <- table(a$treatment, a$seurat_clusters)
exp.cnts <- table(a$seurat_clusters)
exp.freq <- prop.table(table(a$seurat_clusters))
treat.sum <- rowSums(cnts)

# chi-square
out <- lapply(seq(1:ncol(cnts)), function(z){
  
  # set up data
  treat.cl <- cnts[,z]
  treat.ncl <- treat.sum - treat.cl
  exp.cl <- exp.cnts[z]
  exp.ncl <- sum(exp.cnts)-exp.cl
  
  # iterate over samples
  sam <- lapply(seq(1:length(treat.cl)), function(x){
    mat <- matrix(c(treat.cl[x], treat.ncl[x],exp.cl, exp.ncl), nrow=2, byrow=F)
    treat.frq <- treat.cl[x]/(treat.cl[x]+treat.ncl[x])
    exp.frq <- exp.cl/(exp.cl+exp.ncl)
    ctest <- chisq.test(mat, rescale.p=T)
    return(data.frame(clusterID=paste0("cluster_",z-1),
                      sampleID=names(treat.cl)[x],
                      obs.freq=treat.frq,
                      exp.freq=exp.frq,
                      log2freq=log2((treat.frq+1e-10)/(exp.frq+1e-10)),
                      obs.cl.cnts=treat.cl[x],
                      obs.all.cnts=(treat.cl[x]+treat.ncl[x]),
                      exp.cl.cnts=exp.cl,
                      exp.all.cnts=sum(exp.cnts),
                      chi=ctest$statistic,
                      p.value=ctest$p.value))
           
  })
  sam <- do.call(rbind, sam)
  return(sam)
})
out <- do.call(rbind, out)
out$fdr <- p.adjust(out$p.value, method="fdr")
rownames(out) <- seq(1:nrow(out))
out$log10fdr <- -log10(out$fdr)

# create matrix
mat <- dcast(sampleID~clusterID, data=out, value.var="log2freq")
pmat <- dcast(sampleID~clusterID, data=out, value.var="log10fdr")
rownames(mat) <- mat$sampleID
mat$sampleID <- NULL
mat <- as.matrix(mat)
rownames(pmat) <- pmat$sampleID
pmat$sampleID <- NULL
pmat <- as.matrix(pmat)
pmat[is.infinite(pmat)] <- max(pmat[is.finite(pmat)])

cols <- colorRampPalette(c("#1F3E6F","#4B91CE","#C2D4D7","grey","#F78D25","#BE2426","#A41E22"))(100)
#cols <- magma(100)
mat <- mat[c("D0","D2_noH","D4_noH", "D2_wH", "D4_wH", "D6_wH"),]

pdf("log2fc_obs_exp.cell_comp.pdf", width=12, height=4)
pheatmap(mat, breaks=seq(from=-3.2, to=3.2, length.out=101), col=cols, cluster_rows=F)
dev.off()

