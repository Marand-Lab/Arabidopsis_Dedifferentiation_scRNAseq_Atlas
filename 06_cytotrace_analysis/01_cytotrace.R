## cytotrace2 analysis ##

# use R 4.4 :::: ml R/4.4.0

# set python
Sys.setenv(RETICULATE_PYTHON="/sw/spack/bio/pkgs/gcc-10.3.0/python/3.9.7-rv5ybzg3/bin/python")

# libraries
library(CytoTRACE)
library(Seurat)
library(Matrix)

# load data
obj <- readRDS("step1_palantir_obj_100DC_harmony_knn.15.10.21.2025.knn30.rds")
exprs <- obj@assays$RNA$data

# process matrix
exprs <- exprs[Matrix::rowSums(exprs > 0) >= 100,]
exprs <- exprs[,Matrix::colSums(exprs)>0]
exprs <- as.matrix(exprs)

# run cytotrace
cyto <- CytoTRACE(exprs, batch=obj@meta.data$library, subsamplesize=5000, enableFast=T)

# save results
saveRDS(cyto, file='cytotrace_results.rds')
