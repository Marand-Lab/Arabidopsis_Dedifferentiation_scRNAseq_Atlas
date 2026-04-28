## scRNA-seq analysis ##

# load args
args <- commandArgs(T)
if(length(args) != 3){stop("Rscript clusterNuclei.seurat.R <RDS> <meta> <prefix>")}
prefix <- as.character(args[3])

# load libraries
library(Seurat)
library(Matrix)
library(ggplot2)
library(harmony)
library(uwot)
library(MAST)
library(glmGamPoi)
library(pheatmap)
library(reshape2)

# set future options
options(future.globals.maxSize = 8000 * 1024^2)

# load data
a <- readRDS(as.character(args[1]))
b <- read.table(as.character(args[2]))

# set global params
reduc <- "harmony"
num.dims <- 30

# protoplast genes
pgenes <- read.table("/nfs/turbo/lsa-YOURNAME/YOURNAME_home/Arabidopsis_Protoplast_Dedifferentiation/step2_cluster_cells/protoplasting_induced_genes_fc2_qval0.01.txt", header=T)

# align meta and sparse data (remove doublets)
b <- subset(b, b$dropletType=="Singlet" & b$treatment=="D0")
shared.cells <- intersect(rownames(b), colnames(a))
b <- b[shared.cells,]
a <- a[,shared.cells]

# remove chloroplast/mitochondrial genes
c.genes <- rownames(a)[grepl("ATC", rownames(a))]
m.genes <- rownames(a)[grepl("ATM", rownames(a))]
#a <- a[!rownames(a) %in% pgenes$Locus,]
a <- a[!rownames(a) %in% c.genes,]
a <- a[!rownames(a) %in% m.genes,]

# filter genes/cells
message(" - filtering low coverage cells and genes ...")
a <- a[Matrix::rowSums(a>0) > 0,]
a <- a[,Matrix::colSums(a>0) > 100]
a <- a[Matrix::rowSums(a) > 0,]
outliers <- quantile(Matrix::colSums(a), 0.99)
g.outliers <- quantile(Matrix::colSums(a > 0), 0.99)
a <- a[,Matrix::colSums(a) < as.numeric(outliers)]
a <- a[,Matrix::colSums(a > 0) < as.numeric(g.outliers)]
b <- b[colnames(a),]

# create seurat object
aa <- CreateSeuratObject(counts = a, meta.data = b)
aa@meta.data$log_umi <- log(aa@meta.data$nCount_RNA)

# normalize
message(" - normalizing counts ...")
aa <- NormalizeData(aa)
aa <- FindVariableFeatures(aa)
aa <- ScaleData(aa)

# reduce dimensions
message(" - running PCA ...")
aa <- RunPCA(aa, approx=FALSE, npcs=50)#, features=rownames(aa))

# integrate libraries
message(" - running Harmony ...")
aa <- RunHarmony(aa, c("library"), 
                 theta=c(2), 
                 nclust=100, 
                 max.iter=30, 
                 assay.use="RNA", 
                 dims.use=1:num.dims, 
                 project.dim=F)

# run UMAP with custom UWOT parameters
umap.cor <- umap(Embeddings(aa, reduction=reduc)[,1:num.dims], 
                 min_dist = 0.3, 
                 n_neighbors = 30,
                 metric="correlation",
                 verbose=T)

# add UMAP to Seurat Object
colnames(umap.cor) <- paste0("UMAP_", seq(1:ncol(umap.cor)))
rownames(umap.cor) <- rownames(Embeddings(aa, reduction=reduc))
aa[["umap"]] <- CreateDimReducObject(embeddings = umap.cor, key = "UMAP_", assay = DefaultAssay(aa))

# cluster
aa <- FindNeighbors(aa, reduction=reduc, k.param=30, dims=1:num.dims, prune.SNN=1/30, l2.norm=T)
aa <- FindClusters(aa, resolution=0.8, algorithm=1)
saveRDS(aa, file=paste0(prefix,".seurat_object.rds"))

# plot harmony UMAP
DimPlot(aa, reduction="umap", label = TRUE, pt.size=0.2, raster=FALSE) + NoLegend()
ggsave(filename = paste0(prefix,".Clusters_UMAP.pdf"), width=8, height=8, device = "pdf", units = "in")
DimPlot(aa, reduction = "umap", group.by = "library", pt.size=0.2, raster=FALSE)
ggsave(filename = paste0(prefix,".Libraries_UMAP.pdf"), width=8, height=8, device = "pdf", units = "in")
DimPlot(aa, reduction = "umap", group.by = "tech", pt.size=0.2, raster=FALSE)
ggsave(filename = paste0(prefix,".Technology_UMAP.pdf"), width=8, height=8, device = "pdf", units = "in")
DimPlot(aa, reduction = "umap", group.by = "batch", pt.size=0.2, raster=FALSE)
ggsave(filename = paste0(prefix,".Batches_UMAP.pdf"), width=8, height=8, device = "pdf", units = "in")
DimPlot(aa, reduction = "umap", group.by = "replicate", pt.size=0.2, raster=FALSE)
ggsave(filename = paste0(prefix,".replicates_UMAP.pdf"), width=8, height=8, device = "pdf", units = "in")
dev.off()

# save results
df <- aa@meta.data
pcs <- Embeddings(aa[[reduc]])
umapc <- Embeddings(aa[["umap"]])
colnames(umapc) <- c("umap1", "umap2")
df <- cbind(df, umapc)
write.table(df, file=paste0(prefix,".seurat.metaData.txt"), quote=F, row.names=T, col.names=T, sep="\t")
write.table(pcs, file=paste0(prefix,".seurat.harmonyPCs.txt"), quote=F, row.names=T, col.names=T, sep="\t")

# check per cell correlation
clust.aves <- lapply(unique(df$seurat_clusters), function(z){
  ddf <- subset(df, df$seurat_clusters==z)
  colMeans(pcs[rownames(ddf),])
})
clust.aves <- do.call(cbind, clust.aves)
colnames(clust.aves) <- paste0("cluster_", unique(df$seurat_clusters))
cors <- cor(t(pcs), clust.aves)
df$best.cl <- apply(cors, 1, function(x){names(x)[which.max(x)]})[rownames(df)]
df$best.cl <- as.numeric(gsub("cluster_","",df$best.cl))
test <- aa@meta.data
aa@meta.data <- df

# plot refinded
DimPlot(aa, reduction="umap", label = TRUE, group.by="best.cl", pt.size=0.2, raster=FALSE) + NoLegend()
ggsave(filename = paste0(prefix,".Clusters_refined_UMAP.pdf"), width=8, height=8, device = "pdf", units = "in")
cl.cors <- cor(clust.aves)
dev.off()

pdf("cluster_correlations.pdf", width=5, height=5)
pheatmap(cl.cors)
dev.off()
cl.corss <- melt(cl.cors)
cl.corss <- subset(cl.corss, cl.corss$value > 0.9)
cl.corss <- cl.corss[cl.corss$Var1!=cl.corss$Var2,]

# find DE genes
DEG <- FindAllMarkers(aa, only.pos=T)
DEG <- DEG[order(DEG$p_val, decreasing=F),]
top <- Reduce(rbind, by(DEG, DEG$cluster, head, 1))
top.genes <- as.character(unique(top$gene))
write.table(DEG, file=paste0(prefix,".DEG.raw.txt"), quote=F, row.names=T, col.names=T, sep="\t")

# plot top DEG
FeaturePlot(aa, features=top.genes, order=T)
ggsave(filename = paste0(prefix,".topDEG.pdf"), width=18, height=18, device = "pdf", units = "in")


###################################################################################################
# compare to data base
###################################################################################################
deg <- DEG

# filter counts
num.genes <- nrow(a)

# iterate over clusters
sig <- subset(deg, deg$p_val_adj < 0.05)
clusts <- sort(unique(sig$cluster))
cors <- lapply(clusts, function(z){
  message(" - comparing annotations for cluster = ",z)
  cl <- subset(sig, sig$cluster==z)
  cl$ranks <- seq(1:nrow(cl))
  rownames(cl) <- cl$gene
  
  # iterate over types
  outs <- lapply(types, function(y){
    df <- subset(markers, markers$celltypeID==y)
    rownames(df) <- df$gene
    num.sites <- min(c(nrow(df), nrow(cl)))
    cll <- cl[1:num.sites,]
    df <- df[1:num.sites,]
    shared <- intersect(cll$gene, df$gene)
    num.shared <- length(shared)
    num.cl <- nrow(cll)
    num.db <- nrow(df)
    if(num.shared > 0){
      cll <- cll[shared,]
      dff <- df[shared,]
      ranks <- data.frame(r1=cll$rank, r2=dff$rank)
      w <- (max(ranks$r1) - ranks$r1 + 1)/max(ranks$r1)
      corr <- cov.wt(ranks, wt=w, cor=T)$cor[1,2]
      pval <- phyper(num.shared, num.cl, num.genes-num.cl, num.db, lower.tail=F, log.p=F)
      frac <- num.shared/num.cl
      pval <- ifelse(pval==0, 2.225074e-308, pval)
      score <- sqrt(-log10(pval))*corr
      
    }else{
      corr <- -1
      pval <- 1
      frac <- 0
      score <- sqrt(-log10(pval))*corr
    }
    
    # generate permutations
    message(" - running permutations...")
    perms <- lapply(seq(1:100), function(i){
      clp <- cl[sample(nrow(cl)),]
      clp$rank <- seq(1:nrow(clp))
      clp <- clp[1:num.sites,]
      dfp <- df[1:num.sites,]
      shared <- intersect(clp$gene, dfp$gene)
      num.shared <- length(shared)
      num.cl <- nrow(clp)
      num.db <- nrow(dfp)
      if(num.shared > 0){
        clp <- clp[shared,]
        dfp <- dfp[shared,]
        ranks <- data.frame(r1=clp$rank, r2=dfp$rank)
        w <- (max(ranks$r1) - ranks$r1 + 1)/max(ranks$r1)
        corr <- cov.wt(ranks, wt=w, cor=T)$cor[1,2]
        pval <- phyper(num.shared, num.cl, num.genes-num.cl, num.db, lower.tail=F, log.p=F)
        pval <- ifelse(pval==0, 2.225074e-308, pval)
        score <- sqrt(-log10(pval))*corr
      }else{
        corr <- 0
        pval <- 1
        score <- sqrt(-log10(pval))*corr
      }
      return(data.frame(p.rank.cor=corr,
                        p.score=score))
    })
    perms <- do.call(rbind, perms)
    
    # get permutation metrics
    aves <- colMeans(perms)
    stds <- apply(perms, 2, sd)
    
    
    # return
    return(data.frame(cluster=z,
                      celltypeID=y,
                      num.cl.markers=num.cl,
                      num.db.markers=num.db,
                      num.shared.markers=num.shared,
                      rank.cor=corr,
                      frac_overlap=frac,
                      p.val=pval,
                      score=score,
                      perm.ave.rank.cor=aves[1],
                      perm.ave.score=aves[2],
                      perm.std.rank.cor=stds[1],
                      perm.std.score=stds[2],
                      perm.z.rank.cor=(corr-aves[1])/stds[1],
                      perm.z.score=(score-aves[2])/stds[2]))
  })
  outs <- do.call(rbind, outs)
  return(outs)
})
cors <- do.call(rbind, cors)
rownames(cors) <- seq(1:nrow(cors))
cors$perm.p.val <- pnorm(cors$perm.z.score, lower.tail=F)
cors$fdr <- p.adjust(cors$p.val, method="fdr")
cors$perm.fdr <- p.adjust(cors$perm.p.val, method="fdr")
cors <- cors[order(cors$cluster, cors$perm.p.val, decreasing=F),]
filt <- subset(cors, cors$perm.fdr < 0.05)
write.table(cors, file=paste0(prefix,".overlap_db_markers.txt"), quote=F, row.names=T, col.names=T, sep="\t")


# find best markers
m <- markers
ave.score <- aggregate(rank~gene+clusterName, data=m, FUN=mean)
num.mark <- aggregate(rank~gene+clusterName, data=m, FUN=length)
ave.score$counts <- num.mark$rank
ave.score <- ave.score[order(ave.score$counts, -1*ave.score$rank, decreasing=T),]
ave.score <- ave.score[!duplicated(ave.score$gene),]
top.m <- Reduce(rbind, by(ave.score, ave.score$clusterName, head, 5))
top.m <- top.m[!duplicated(top.m$gene),]

# plot marker heatmap
mat <- aa@assays$RNA$data
cl <- lapply(unique(aa@meta.data$seurat_clusters), function(z){
  Matrix::rowMeans(mat[,rownames(aa@meta.data[aa@meta.data$seurat_clusters==z,])])
})
cl <- do.call(cbind, cl)
colnames(cl) <- paste0("cluster_",unique(aa@meta.data$seurat_clusters))
cl <- cl[rowSums(cl)>0,]
z <- as.matrix(t(scale(t(cl))))

shared <- intersect(rownames(z), top.m$gene)
top.m <- top.m[top.m$gene %in% shared,]
z.m <- z[top.m$gene,]
z.m <- t(z.m)
#row.o <- apply(z.m, 1, which.max)
#z.m <- z.m[order(row.o, decreasing=F),]
ddf <- data.frame(celltype=top.m$clusterName, row.names=top.m$gene)

pdf("top_marker_genes.pdf", width=24, height=6)
pheatmap(z.m, cluster_rows=F, cluster_cols=F, annotation_col=ddf)
dev.off()

# plot select markers
shared <- intersect(spm$geneID, rownames(z))
spm <- spm[spm$geneID %in% shared,]
z.m <- z[spm$geneID,]
ddf <- data.frame(celltype=spm$type, row.names=spm$geneID)

pdf("select_marker_genes.pdf", width=10, height=6)
pheatmap(z.m, cluster_rows=F, cluster_cols=F, annotation_row=ddf)
dev.off()

# plot cluster correlations
cors <- cor(cl)
pdf("cluster_correlations.pdf", width=5, height=5)
pheatmap(cors, col=colorRampPalette(brewer.pal(9, "Blues"))(100))
dev.off()

# plot proportion of cells by cluster
cols <- c("#B1D7F2","#7FB5D8","#599EC5","#3A7E9C", "#236B85")
props <- prop.table(table(df$seurat_clusters, df$library), 2)
props <- props[,c(1,3,4,5,2)]
barplot(t(props), beside=T, col=cols, border=NA)
