## test gene association with differentiation ##

# load libraries
library(lme4)
library(Seurat)
library(scales)
library(viridis)
library(RColorBrewer)
library(org.At.tair.db)
library(fgsea)

# functions
plot_cbd <- function(x1,x2,
                     ylim=c(min(x2),max(x2)),
                     xlim=c(min(x1),max(x1)),
                     xlab="",ylab="",main="",
                     cex=0.5, fit=F, bwf_x=5, bwf_y=5, nbin=300,
                     colP=NULL, rasterize=F){
  
  .normalize <- function(x){
    (x - min(x))/(max(x)-min(x))
  }
  if(is.null(colP)){
    
    colP <- colorRampPalette(c("grey75","grey70","darkorchid4","firebrick3","darkorange",
                               "gold1","yellow"), bias=1.5)(256)
    
  }else{
    colP <- colP(256)
  }
  bww <- (max(x1)-min(x1))/bwf_x
  bwh <- (max(x2)-min(x2))/bwf_y
  df <- data.frame(x1,x2)
  x <- densCols(x1,x2, colramp=colorRampPalette(c("black", "white")),
                nbin=nbin, bandwidth=c(bww, bwh))
  df$dens <- col2rgb(x)[1,] + 1L
  cols <- colP
  df$col <- cols[df$dens]
  df$opa <- .normalize(df$dens)
  
  # if raster
  if(rasterize){
    plot(x2~x1, data=df[order(df$dens),],
         ylim=ylim,xlim=xlim,pch=16,col=alpha(col,1),
         cex=cex,xlab=xlab,ylab=ylab,
         main=main, type="n")
    coords <- par("usr")
    gx <- grconvertX(coords[1:2], "user", "inches")
    gy <- grconvertY(coords[3:4], "user", "inches")
    width <- max(gx) - min(gx)
    height <- max(gy) - min(gy)
    tmp <- tempfile()
    png(tmp, width = width, height = height, units = "in", res = 300, bg = "transparent")
    par(mar=c(0,0,0,0))
    plot.new()
    plot.window(coords[1:2], coords[3:4], mar = c(0,0,0,0), xaxs = "i", yaxs = "i")
    points(x2~x1, data=df[order(df$dens),],
           pch=16,col=alpha(col,1),
           cex=cex)
    dev.off()
    panel <- readPNG(tmp)
    rasterImage(panel, coords[1], coords[3], coords[2], coords[4])
    
  }else{
    plot(x2~x1, data=df[order(df$dens),],
         ylim=ylim,xlim=xlim,pch=16,col=alpha(col,1),
         cex=cex,xlab=xlab,ylab=ylab,
         main=main)
  }
  grid(lty=1)
  
  # fit regression?
  if(fit){
    
    mod <- lm(x2~x1)
    abline(mod, col="firebrick3", lwd=2)
    
  }
  
  cors <- cor(x2,x1)
  mtext(paste0("PCC = ", signif(cors, digits=3)))
}


# load data
meta <- read.table("All.YJ1_19.metadata.filtered.detailed.06.11.2025.txt")
obj <- readRDS("All.YJ1_19.logNorm.seurat_object.rds")

# format counts data
counts <- obj@assays$RNA$data
shared <- intersect(rownames(meta), colnames(counts))
counts <- counts[,shared]
meta <- meta[shared,]
counts <- counts[Matrix::rowSums(counts > 1) > 50,]
df <- meta[,c("predcelltype","log_umi", "library", "tech", "batch", "cor.value")]

# run lmm
res <- lapply(seq(1:nrow(counts)), function(z){
  if((z %% 10)==0){message(" - iterated over ",z," genes...")}
  df$expr <- as.numeric(counts[z,])
  modf <- suppressMessages(lmer(cor.value~expr + batch + log_umi + (1 | library), data=df))
  modr <- suppressMessages(lmer(cor.value~batch + log_umi + (1 | library), data=df))
  a.res <- suppressMessages(anova(modf, modr))
  lrt <- a.res$`Pr(>Chisq)`[2]
  full <- summary(modf)
  return(data.frame(geneID=rownames(counts)[z],
                    beta=full$coefficients["expr",1],
                    beta_se=full$coefficients["expr",2],
                    tstat=full$coefficients["expr",3],
                    p.value=2*pnorm(abs(full$coefficients["expr",3]), lower.tail=F),
                    lrt=lrt))
})
res <- do.call(rbind, res)

# plot results
res$fdr <- p.adjust(res$p.value, method='bonferroni')
res$lrt.fdr <- p.adjust(res$lrt, method='bonferroni')
write.table(res, file="lmm_diff_score_gene_expression_summary_stats.no_predcelltype_effects.txt", quote=F, row.names=F, col.names=T, sep="\t")

# 
# res$lrt.fdr[res$lrt.fdr==0] <- 3.409053e-322
# 
# 
# eff <- 0.5

df <- read.table("perm_correlations.gene_diff_score.txt")

# ranked z-score lmm based
bb <- res$tstat
res$cols <- viridis(100)[cut(bb, breaks=101, include.lowest=T)]
res <- res[order(res$tstat, decreasing=T),]
pdf("lmm_results_ranked.v3.pdf", width=4, height=5)
plot(seq(1:nrow(res)), res$tstat, pch=16, cex=0.8, 
     col=alpha(ifelse(res$lrt.fdr < 0.05, res$cols, "grey75"),0.5),
      xlab="Ranked effects",
      ylab="Z-score")
grid(lty=1)
dev.off()

# ranked z-score permutation based
df <- read.table("perm_correlations.gene_diff_score.txt")
bb <- df$z.score
df$cols <- viridis(100)[cut(bb, breaks=101, include.lowest=T)]
df <- df[order(df$z.score, decreasing=T),]
pdf("perm_cor_results_ranked.pdf", width=3, height=5)
plot(seq(1:nrow(df)), df$z.score, pch=16, cex=0.8, 
     col=alpha(ifelse(df$z.score > 10 & df$fdr < 0.05, df$cols, 
                      ifelse(df$z.score < -10 & df$fdr < 0.0001, df$cols, "grey75")),0.5),
     xlab="Ranked effects",
     ylab="Z-score",
     ylim=c(-225,235))
abline(h=-10, lty=2)
abline(h=10, lty=2)
grid(lty=1)
dev.off()



### GSEA ####
zscores <- res$tstat
names(zscores) <- res$geneID
gmt1 <- gmtPathways("TAIR10.GO")
gmt <- lapply(gmt1, function(z){
  do.call(c, strsplit(z, ","))
})
names(gmt) <- names(gmt1)
results <- fgsea(pathways = gmt, 
                 stats    = zscores,
                 minSize  = 10,
                 maxSize  = 1000)
results <- results[order(results$pval, decreasing=F),]
results <- as.data.frame(results)
results$
write.table(results, file="GSEA_results_LMM_FDR05.txt", quote=F, row.names=F, col.names=T, sep="\t")
