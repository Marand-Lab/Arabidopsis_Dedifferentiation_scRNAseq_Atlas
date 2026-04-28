## Assess differentiation ##

# load libraries
library(Seurat)
library(qlcMatrix)
library(Matrix)
library(vioplot)
library(RColorBrewer)
library(viridis)
library(scales)
library(png)
library(MASS)
library(parallel)
library(reshape2)

# functions
plot_cbd <- function(x1,x2,
                     ylim=c(min(x2),max(x2)),
                     xlim=c(min(x1),max(x1)),
                     xlab="",ylab="",main="",
                     col="black",
                     cex=0.5){
  
    plot(x2~x1, 
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
    points(x2~x1, 
           pch=16,col=alpha(col,1),
           cex=cex)
    dev.off()
    panel <- readPNG(tmp)
    rasterImage(panel, coords[1], coords[3], coords[2], coords[4])
    
}

# load data
obj <- readRDS("All.YJ1_19.logNorm.seurat_object.rds")
d0 <- read.table("D0_cells/subcluster_cells/D0_cells.annotated.metadata.txt")

# create differentiated pseudocells
d0$celltypeID <- paste0(d0$celltype,"-",d0$seurat_clusters)
clusts <- lapply(sort(unique(d0$celltypeID)), function(x){
  df <- subset(d0, d0$celltypeID==x)
  Matrix::rowMeans(obj@assays$RNA$data[,colnames(obj@assays$RNA$data) %in% rownames(df)])
})
clusts <- do.call(cbind, clusts)
colnames(clusts) <- sort(unique(d0$celltypeID))
clusts <- Matrix(clusts, sparse=T)
meta <- obj@meta.data
cellID <- d0$celltypeID
names(cellID) <- rownames(d0)
meta$celltypeID <- cellID[rownames(meta)]

# correlations
pdc <- obj@assays$RNA$data
cors <- corSparse(pdc, clusts)
colnames(cors) <- colnames(clusts)
rownames(cors) <- colnames(pdc)
best.cluster <- apply(cors, 1, function(x){names(x)[which.max(x)]})
best.pcc <- apply(cors, 1, max)
aves.pcc <- rowMeans(cors)

# update meta
meta$cor.celltype <- best.cluster[rownames(meta)]
meta$cor.value <- best.pcc[rownames(meta)]
meta$ave.pcc <- aves.pcc[rownames(meta)]

# plot
pdf("correlation2d0cluster.pdf", width=5, height=5)
vioplot(cor.value~treatment, data=meta)
grid(lty=1)
dev.off()

# set up age & extract UMAP
meta$age <- gsub("_wH","",meta$treatment)
meta$age <- gsub("_noH","",meta$age)
meta$age <- factor(meta$age, levels=c("D0","D2","D4","D6"))
umapc <- Embeddings(obj, reduction="umap")
meta$umap1 <- umapc[,1]
meta$umap2 <- umapc[,2]

# colors
#cols <- colorRampPalette(rev(brewer.pal(11, "Spectral")))(100)
meta <- meta[order(meta$age, decreasing=T),]
meta$trim.cor <- meta$cor.value
q5 <- quantile(meta$trim.cor, c(0.25, 0.975))
meta$trim.cor[meta$trim.cor < q5[1]] <- q5[1]
meta$trim.cor[meta$trim.cor > q5[2]] <- q5[2]
meta <- meta[order(meta$trim.cor, decreasing=F),]
#cols <- colorRampPalette(c("grey70","grey75", brewer.pal(9, "RdPu")[2:9], bias=1))(100)
cols <- colorRampPalette(plasma(100), bias=0.75)(100)
v.cols <- cols[cut(meta$trim.cor, breaks=101)]
#dif.cols <- cols[cut(meta$trim.cor, breaks=101)]

# plot
pdf("correlation2differentiated.v3.pdf", width=10, height=10)
plot(meta$umap1, meta$umap2, cex=0.3, col=v.cols, pch=16)
dev.off()

# p-values
wilcox.test(meta$cor.value[meta$treatment=="D2_wH"],meta$cor.value[meta$treatment=="D2_noH"])$p.value
wilcox.test(meta$cor.value[meta$treatment=="D4_wH"],meta$cor.value[meta$treatment=="D4_noH"])$p.value


# model conditions
mod <- lm(cor.value~age+hormone, data=meta)


# model transcriptional complexity
exprs <- obj@assays$RNA$data
bexp <- exprs
bexp@x[bexp@x < 1] <- 0
bexp@x[bexp@x > 0] <- 1
bexp <- Matrix(bexp, sparse=T)
exp.genes <- Matrix::colSums(bexp)
meta$exp.genes <- exp.genes[rownames(meta)]
#mm <- meta[meta$library != "YJ5",]
#boxplot(log10(mm$exp.genes)~mm$treatment, outline=F)



meta$pOrg <- 1-meta$pTrx
meta$res.complexity <- residuals(lm(exp.genes~pOrg+batch+replicate+tech+nCount_RNA, data=meta))
pdf("transcriptional_complexity.age.pdf", width=5, height=5)
boxplot(meta$res.complexity~meta$age, outline=F)
dev.off()

pdf("transcriptional_complexity.treatment.pdf", width=5, height=5)
boxplot(meta$res.complexity~meta$treatment, outline=F)
dev.off()

# correlate differentiation score with expression patterns
diff.score <- meta[colnames(pdc),]$cor.value
cors <- corSparse(t(pdc), diff.score)
cors <- as.numeric(cors)
names(cors) <- rownames(pdc)
perms <- lapply(seq(1:1000), function(z){
  message(z)
  cors.p <- as.numeric(corSparse(t(pdc), diff.score[sample(length(diff.score))]))
  return(cors.p)
})
perms <- do.call(rbind,perms)
colnames(perms) <- rownames(pdc)
maxes <- apply(perms, 2, max, na.rm=T)
mins <- apply(perms, 2, min, na.rm=T)
maxes[is.infinite(maxes)] <- 0
mins[is.infinite(mins)] <- 0
z.score <- (cors - colMeans(perms, na.rm=T))/apply(perms, 2, sd, na.rm=T)
z.score[is.na(z.score)] <- 0
cors[is.na(cors)] <- 0
names(z.score) <- rownames(pdc)
z.score <- z.score[order(z.score, decreasing=T)]
cors <- cors[names(z.score)]
df <- data.frame(z.score=z.score, pcc=cors, row.names=names(z.score))
df$p.value <- 2*pnorm(abs(df$z.score), lower.tail=F)
df$fdr <- p.adjust(df$p.value, method="fdr")
ann <- read.table("/nfs/turbo/lsa-amarand/shared_data/arabidopsis/reference_data/annotations/Arabidopsis_thaliana.TAIR10.58.annotation.txt", header=F)
id <- ann$V2
names(id) <- ann$V1
func <- ann$V3
names(func) <- ann$V1
df$geneName <- id[rownames(df)]
df$func <- func[rownames(df)]
df$geneID <- rownames(df)
write.table(df, file="perm_correlations.gene_diff_score.txt", quote=F, row.names=T, col.names=T, sep="\t")

# check markers
prc2 <- c("AT4G02020","AT2G23380","AT1G02580", "AT5G51230","AT4G16845","AT2G35670","AT3G20740","AT5G58230")
lec2 <- c("AT1G28300", "AT3G26790", "AT2G24430", "AT3G62100")

# plot
cors <- cors[order(cors, decreasing=T)]
mins <- mins[names(cors)]
maxes <- maxes[names(cors)]
plot(seq(1:length(cors)), cors, pch=16, cex=0.5, type="n")
segments(seq(1:length(cors)), mins, seq(1:length(cors)), maxes, col="grey75", lwd=0.5)
points(seq(1:length(cors)), cors, pch=16, cex=0.5)

# 
s.df <- subset(df, df$fdr < 1)
cols <- colorRampPalette(c("#1F3E6F","#4B91CE","#C2D4D7","grey","#F78D25","#BE2426","#A41E22"))(100)
cc <- cols[cut(s.df$z.score, breaks=101)]
plot(seq(1:nrow(s.df)), s.df$z.score, pch=16, cex=pi*((abs(s.df$pcc)+0.3)^2), col=cc)

# marker genes
mg <- c("AT1G04880","AT4G17710","AT2G17950", "AT2G27250")

# cell cycle genes
cc <- read.table("/nfs/turbo/lsa-amarand/shared_data/arabidopsis/reference_data/annotations/cell_cycle_genes.txt", header=T)
cd <- read.table("/nfs/turbo/lsa-amarand/shared_data/arabidopsis/reference_data/annotations/cell_death_genes.txt", header=T)
meta <- read.table("All.YJ1_19.metadata.filtered.detailed.06.11.2025.txt")
exp.cc <- exprs[rownames(exprs) %in% cc$TAIR,]
exp.cd <- exprs[rownames(exprs) %in% cd$TAIR,]
gm.cc <- apply(exp.cc, 2, function(z){exp(mean(log(z+0.1)))})
gm.cd <- apply(exp.cd, 2, function(z){exp(mean(log(z+0.1)))})
meta$cellcycle_score <- gm.cc[rownames(meta)]
meta$celldeath_score <- gm.cd[rownames(meta)]

# plot proportion predicted cell type by treatment
cts <- d0$celltype
cts <- gsub("S-Phase","S/Phase", cts)
names(cts) <- rownames(d0)
meta$celltype <- cts[rownames(meta)]
meta$pcelltype <- ifelse(meta$cor.value > 0.5, meta$predcelltype, "Unknown")
meta$pcelltype <- ifelse(meta$age=="D0", meta$celltype, meta$pcelltype)
cell.cols <- c("#C6ADD3","#A6CEE1","#A588BE","#7BB3D5","#D9A599","#EC9554","#F48281",
               "#9E9C65","#F05B5A","#5D9E43","#6B3F98","#6ABD55","#E73232","#4B95A7",
               "#8962AA","#6A468E","#F89B5F","#94CB72","#A7D48C","#5A4770","grey75","#D4C0DD")
names(cell.cols) <- sort(unique(meta$pcelltype))
props <- t(prop.table(table(meta$treatment, meta$pcelltype), 1))
props <- props[order(props[,1], decreasing=F),]
props <- props[,c("D0","D2_noH","D4_noH","D2_wH","D4_wH","D6_wH")]
pdf("props_celltype_treatment.pdf", width=5, height=8)
barplot(props, beside=F, col=cell.cols[rownames(props)], border=NA)
dev.off()

# heatmap
nprops <- t(apply(props, 1, function(z){z/max(z)}))
lprops <- props+0.001
fcprops <- apply(lprops[,c(2:6)], 2, function(z){
  log2(z/lprops[,1])
})

pdf("log2fc_celltype_preds_treatment.spc.pdf", width=5, height=10)
pheatmap(fcprops, cluster_cols=F, 
         col=colorRampPalette(rev(brewer.pal(9, "Spectral")))(100), 
         scale="none", breaks=seq(from=-5,to=5, length.out=101))
dev.off()

# plot line graph
fcprops[fcprops > 5] <- 5
fcprops[fcprops < -5] <- -5
noh <- fcprops[,1:2]
wh <- fcprops[,3:5]
colnames(noh) <- c(1,2)
colnames(wh) <- c(1,2,3)
noh <- t(noh)
wh <- t(wh)

pdf("noH.fc_frequencies.pdf", width=5, height=5)
matplot(c(1,2),noh, type="l", col=cell.cols[colnames(noh)], xlim=c(1,3), lty=2)
dev.off()

pdf("wH.fc_frequencies.pdf", width=5, height=5)
matplot(c(1,2,3),wh, type="l", col=cell.cols[colnames(wh)], xlim=c(1,3), lty=1)
dev.off()

# chi-square
cnts <- table(meta$treatment, meta$pcelltype)
exp <- cnts[1,]
obs <- cnts[2:6,]
outs <- lapply(seq(1:ncol(cnts)), function(z){
  exp.ct <- exp[z]
  exp.all <- sum(exp) - exp.ct
  outs1 <- lapply(seq(1:nrow(obs)), function(x){
    obs.ct <- obs[x,z]
    obs.all <- sum(obs[x,])-obs.ct
    mat <- matrix(c(exp.ct, exp.all, obs.ct, obs.all), nrow=2, byrow=F)
    pval <- chisq.test(mat)$p.value
    return(data.frame(celltype=colnames(cnts)[z],
                      sampleID=rownames(obs)[x],
                      frac.D0=exp.ct/(exp.all+exp.ct),
                      frac.Sample=obs.ct/(obs.all+obs.ct),
                      p.value=pval))
  })
  outs1 <- do.call(rbind, outs1)
  return(outs1)
})
outs <- do.call(rbind, outs)
outs$fdr <- p.adjust(outs$p.value, method="fdr")
rownames(outs) <- seq(1:nrow(outs))


# evaluate "dedifferentiated cells"
dedif <- meta[meta$pcelltype=="Unknown",]
