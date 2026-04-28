## MILO analysis ##

# load libraries
library(Seurat)
library(SingleCellExperiment)
library(miloR)
library(miloDE)

# load data
message(" - load data")
obj <- readRDS("step1_palantir_obj_100DC_harmony_knn.15.10.21.2025.knn30.rds")
meta <- read.table("diffusion_pseudotime.metadata.all_cells.palantir_pt_entrop.10.21.2025.knn30.txt")

# reformat
obj@meta.data <- meta
a <- as.SingleCellExperiment(obj)

# assign neighborhoods
message(" - assigning neighborhoods")
a <- assign_neighbourhoods(a, k=20, order=2, filtering=T, reducedDim_name="HARMONY")
saveRDS(a, file="miloDE_obj.11.4.25.rds")

# differential expression
message(" - differential expression analysis")
de_stat <- de_test_neighbourhoods(a, 
                                  sample_id = "library", 
                                  design = ~hormone, 
                                  covariates = c("hormone"))
saveRDS(de_stat, file="miloDE_results.11.4.25.rds")
