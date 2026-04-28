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
library(FNN)

# functions
calcRestrictedKNNsplit <- function(x, 
                                   reduc="harmony",
                                   numPCs=10,
                                   k=100){
  
  # input data
  met <- x@meta.data
  emb <- Embeddings(x, reduction=reduc)[,1:numPCs]
  
  # iterate over principal factor
  age <- names(table(met$age_grp))
  struc <- list(D0_FALSE=c("D2_FALSE", "D2_TRUE"),
                D2_FALSE=c("D0_FALSE", "D2_FALSE", "D4_FALSE"),
                D2_TRUE=c("D0_FALSE", "D2_TRUE", "D4_TRUE"),
                D4_FALSE=c("D2_FALSE", "D4_FALSE"),
                D4_TRUE=c("D2_TRUE", "D4_TRUE", "D6_TRUE"),
                D6_TRUE=c("D4_TRUE", "D6_TRUE"))
  
  indexes <- list()
  dists <- list()
  its <- 0
  for(z in age){
    its <- its+1
    query <- met[met$age_grp==z,]
    subject <- met[met$age_grp %in% struc[[z]],]
    kknn <- get.knnx(emb[rownames(subject),], emb[rownames(query),], k=k)
    idx <- kknn$nn.index
    idx <- apply(idx, 2, function(z){rownames(subject)[z]})
    rownames(idx) <- rownames(query)
    colnames(idx) <- paste0("nn",seq(1:ncol(idx)))
    indexes[[its]] <- idx
    dists[[its]] <- kknn$nn.dist
  }
  indexes <- do.call(rbind, indexes)
  dists <- do.call(rbind, dists)
  rownames(dists) <- rownames(indexes)
  indexes <- indexes[rownames(emb),]
  dists <- dists[rownames(indexes),]
  vals <- seq(1:nrow(indexes))
  names(vals) <- rownames(indexes)
  indexes <- apply(indexes, 2, function(z){vals[z]})
  rownames(dists) <- NULL
  colnames(indexes) <- NULL
  return(list(idx=indexes, dist=dists))
  
}
calcRestrictedKNNdgM <- function(x, 
                                 reduc="harmony",
                                 numPCs=10,
                                 k=100){
  
  # input data
  met <- x@meta.data
  emb <- Embeddings(x, reduction=reduc)[,1:numPCs]
  
  # iterate over principal factor
  age <- names(table(met$age_grp))
  struc <- list(D0_FALSE=c("D2_FALSE", "D2_TRUE"),
                D2_FALSE=c("D0_FALSE", "D2_FALSE", "D4_FALSE"),
                D2_TRUE=c("D0_FALSE", "D2_TRUE", "D4_TRUE"),
                D4_FALSE=c("D2_FALSE", "D4_FALSE"),
                D4_TRUE=c("D2_TRUE", "D4_TRUE", "D6_TRUE"),
                D6_TRUE=c("D4_TRUE", "D6_TRUE"))
  
  indexes <- list()
  dists <- list()
  its <- 0
  for(z in age){
    its <- its+1
    query <- met[met$age_grp==z,]
    subject <- met[met$age_grp %in% struc[[z]],]
    kknn <- get.knnx(emb[rownames(subject),], emb[rownames(query),], k=k)
    idx <- kknn$nn.index
    idx <- apply(idx, 2, function(z){rownames(subject)[z]})
    rownames(idx) <- rownames(query)
    colnames(idx) <- paste0("nn",seq(1:ncol(idx)))
    indexes[[its]] <- idx
    dists[[its]] <- kknn$nn.dist
  }
  indexes <- do.call(rbind, indexes)
  dists <- do.call(rbind, dists)
  rownames(dists) <- rownames(indexes)
  lmat <- melt(indexes)
  dmat <- melt(dists)
  lmat$dist <- dmat$value
  lmat$Var2 <- NULL
  colnames(lmat)[1:2] <- c("cell1", "cell2")
  all.cells <- rownames(met)
  lmat$cell1 <- factor(lmat$cell1, levels=all.cells)
  lmat$cell2 <- factor(lmat$cell2, levels=all.cells)
  smat <- sparseMatrix(i=as.numeric(lmat$cell1),
                       j=as.numeric(lmat$cell2),
                       x=as.numeric(lmat$dist),
                       dimnames = list(levels(lmat$cell1), levels(lmat$cell2)))
  return(smat)
  
}
jaccard <- function(m) {
  ## common values:
  A = tcrossprod(m)
  ## indexes for non-zero common values
  im = which(A > 0, arr.ind=TRUE)
  ## counts for each row
  b = rowSums(m)
  
  ## only non-zero values of common
  Aim = A[im]
  
  ## Jacard formula: #common / (#i + #j - #common)
  J = sparseMatrix(
    i = im[,1],
    j = im[,2],
    x = Aim / (b[im[,1]] + b[im[,2]] - Aim),
    dims = dim(A)
  )
  
  return( J )
}

computeSNN <- function(knn, 
                       k = NULL, 
                       normalize = TRUE, 
                       symmetrize = TRUE) {
  if (!inherits(knn, "dgCMatrix")) {
    stop("Input must be a sparse dgCMatrix (e.g. from Seurat@graphs$RNA_nn)")
  }
  
  n <- nrow(knn)
  if (is.null(k)) {
    k <- round(mean(rowSums(knn != 0)))
    message("Inferred k = ", k)
  }
  
  # Shared neighbors = knn %*% t(knn)
  # Each entry (i,j) counts how many neighbors are shared between i and j
  shared_counts <- knn %*% t(knn)
  
  # Optionally normalize by k
  if (normalize) {
    shared_counts <- shared_counts / k
  }
  
  # Remove self-similarity if present
  diag(shared_counts) <- 0
  
  # Optionally symmetrize
  if (symmetrize) {
    shared_counts <- (shared_counts + t(shared_counts)) / 2
  }
  
  # Ensure it's sparse
  shared_counts <- as(shared_counts, "dgCMatrix")
  
  return(shared_counts)
}

# set future options
options(future.globals.maxSize = 8000 * 1024^2)

# load data
a <- readRDS(as.character(args[1]))
b <- read.table(as.character(args[2]))

# set global params
reduc <- "harmony"
num.dims <- 30

# protoplast genes
#pgenes <- read.table("/nfs/turbo/lsa-YOURNAME/YOURNAME_home/Arabidopsis_Protoplast_Dedifferentiation/step2_cluster_cells/protoplasting_induced_genes_fc2_qval0.01.txt", header=T)

# remove chloroplast/mitochondrial genes
c.genes <- rownames(a)[grepl("ATC", rownames(a))]
m.genes <- rownames(a)[grepl("ATM", rownames(a))]
a <- a[!rownames(a) %in% c.genes,]
a <- a[!rownames(a) %in% m.genes,]

# filter genes/cells
message(" - filtering low coverage cells and genes ...")
a <- a[,Matrix::colSums(a>0) > 100]
a <- a[Matrix::rowSums(a) > 0,]
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
aa <- RunHarmony(aa, c("replicate", "batch", "tech"), 
                 theta=c(1,1,1), 
                 lambda=NULL,
                 nclust=100, 
                 max_iter=30, 
                 assay.use="RNA", 
                 dims.use=1:num.dims, 
                 project.dim=F)

# custom knn graph
message(" - running custom KNN ...")
aa@meta.data$age <- gsub("_noH","",aa@meta.data$treatment)
aa@meta.data$age <- gsub("_wH","",aa@meta.data$age)
aa@meta.data$age_grp <- paste0(aa@meta.data$age, "_", aa@meta.data$hormone)
pc.knn <- calcRestrictedKNNsplit(aa, numPCs=10, k=150)

# run UMAP with custom UWOT parameters
message(" - running UMAP ...")
umap.cor <- umap(Embeddings(aa, reduction="harmony"),
                 nn_method = pc.knn,
                 min_dist = 0,
                 verbose=T)
colnames(umap.cor) <- paste0("UMAP_", seq(1:ncol(umap.cor)))
rownames(umap.cor) <- rownames(Embeddings(aa, reduction="harmony"))
aa[["umap"]] <- CreateDimReducObject(embeddings = umap.cor, key = "UMAP_", assay = DefaultAssay(aa))

# plots
DimPlot(aa, reduction = "umap", split.by = "treatment", pt.size=0.2, raster=FALSE)
ggsave(filename = paste0(prefix,".Split_Age_UMAP.timeseries.pdf"), width=24, height=6, device = "pdf", units = "in")
DimPlot(aa, reduction = "umap", group.by = "library", pt.size=0.2, raster=FALSE)
ggsave(filename = paste0(prefix,".Libraries_UMAP.timeseries.pdf"), width=8, height=8, device = "pdf", units = "in")
DimPlot(aa, reduction = "umap", group.by = "treatment", pt.size=0.2, raster=FALSE)
ggsave(filename = paste0(prefix,".Age_UMAP.timeseries.pdf"), width=8, height=8, device = "pdf", units = "in")
DimPlot(aa, reduction = "umap", group.by = "hormone", pt.size=0.2, raster=FALSE)
ggsave(filename = paste0(prefix,".Hormone_treatment_UMAP.timeseries.pdf"), width=8, height=8, device = "pdf", units = "in")
DimPlot(aa, reduction = "umap", group.by = "tech", pt.size=0.2, raster=FALSE)
ggsave(filename = paste0(prefix,".Technology_UMAP.timeseries.pdf"), width=8, height=8, device = "pdf", units = "in")
DimPlot(aa, reduction = "umap", group.by = "batch", pt.size=0.2, raster=FALSE)
ggsave(filename = paste0(prefix,".Batches_UMAP.timeseries.pdf"), width=8, height=8, device = "pdf", units = "in")
DimPlot(aa, reduction = "umap", group.by = "replicate", pt.size=0.2, raster=FALSE)
ggsave(filename = paste0(prefix,".replicates_UMAP.timeseries.pdf"), width=8, height=8, device = "pdf", units = "in")
dev.off()

# cluster
saveRDS(aa, file=paste0(prefix, ".seurat_object.pre_adjKNN.rds"))
message(" - running clustering ...")
aa@graphs$RNA_nn <- calcRestrictedKNNdgM(aa, numPCs=10, k=100)
bb <- aa@graphs$RNA_nn
bb@x <- rep(1, length(bb@x))
aa@graphs$RNA_snn <- jaccard(bb)
aa@graphs$RNA_snn@x[aa@graphs$RNA_snn@x < (1/15)] <- 0
rownames(aa@graphs$RNA_snn) <- rownames(aa@meta.data)
colnames(aa@graphs$RNA_snn) <- rownames(aa@meta.data)
aa@graphs$RNA_snn <- as.Graph(aa@graphs$RNA_snn)
aa@graphs$RNA_nn <- as.Graph(aa@graphs$RNA_nn)
aa <- FindClusters(aa, resolution=2.0, algorithm=1)
saveRDS(aa, file=paste0(prefix,".seurat_object.timeseries.rds"))

# plot harmony UMAP
DimPlot(aa, reduction="umap", label = TRUE, pt.size=0.2, raster=FALSE) + NoLegend()
ggsave(filename = paste0(prefix,".Clusters_UMAP.timeseries.pdf"), width=8, height=8, device = "pdf", units = "in")


# save results
df <- aa@meta.data
pcs <- Embeddings(aa[[reduc]])
umapc <- Embeddings(aa[["umap"]])
colnames(umapc) <- c("umap1", "umap2")
df <- cbind(df, umapc)
write.table(df, file=paste0(prefix,".seurat.metaData.timeseries.txt"), quote=F, row.names=T, col.names=T, sep="\t")
write.table(pcs, file=paste0(prefix,".seurat.harmonyPCs.timeseries.txt"), quote=F, row.names=T, col.names=T, sep="\t")
