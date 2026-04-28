## do doublet analysis ##

# load libraries
library(Seurat)
library(DoubletFinder)
library(umap)

# function
runDoublet <- function(id, df, sp){
  
  # verbose
  message("** running doublet finder for library = ", id)
  
  # select cells
  df <- subset(df, df$library==id)
  sp <- sp[,rownames(df)]
  sp <- sp[Matrix::rowSums(sp)>0,]
  sp <- sp[,Matrix::colSums(sp)>0]
  df <- df[colnames(sp),]
  message("** finished filtering cells by library = ", id)
  
  # create object
  obj <- CreateSeuratObject(counts=sp, meta.data=df)
  obj <- SCTransform(obj)
  obj <- RunPCA(obj)
  obj <- RunUMAP(obj, dims=1:50)
  obj <- FindNeighbors(obj, dims = 1:50, verbose = FALSE)
  obj <- FindClusters(obj, verbose = FALSE)
  message("** finished Seurat preprocessing for library = ", id)
  
  # pK identification
  sweep.res.list_obj <- paramSweep(obj, PCs = 1:50, sct = TRUE)
  sweep.stats_obj <- summarizeSweep(sweep.res.list_obj, GT = FALSE)
  bcmvn_obj <- find.pK(sweep.stats_obj)
  message("** finished parameter finding library = ", id)
  
  # homotypic doublets
  homotypic.prop <- modelHomotypic(obj@meta.data$seurat_clusters)
  nExp_poi <- round(0.1*nrow(obj@meta.data))
  nExp_poi.adj <- round(nExp_poi*(1-homotypic.prop))
  message("** classifying doublets for library = ", id)
  
  # classify
  obj <- doubletFinder(obj, PCs=1:50, pN=0.25, pK=0.09, nExp=nExp_poi.adj, sct=TRUE)
  return(obj@meta.data)
}

# load data
counts <- readRDS("cell_gene_counts.merged_sparseMatrix.rds")
meta <- read.table("cell_metadata.merged.txt")

# split by library
libs <- unique(meta$library)

# run doubletfinder
classification <- lapply(libs, runDoublet, meta, counts)
df <- lapply(classification, function(x){
  colnames(x)[(ncol(x)-1):ncol(x)] <- c("pANN", "dropletType")
  return(x)
})
df$nCount_SCT <- NULL
df$nFeature_SCT <- NULL
df$SCT_snn_res.0.8 <- NULL
df$seurat_clusters <- NULL
write.table(df, file="cell_metadata.merged.doubletFinder.txt",
            quote=F, row.names=T, col.names=T, sep="\t")
