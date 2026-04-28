#################################################
# dynamic time warping trajectories
#################################################

# load libraries
library(Seurat)
library(cellAlign)
library(RColorBrewer)
library(pheatmap)
library(ggplot2)


#################################################
# load and process
#################################################

# load data
obj <- readRDS("step1_palantir_obj_100DC_harmony_knn.15.10.21.2025.knn30.rds")
meta <- read.table("diffusion_pseudotime.metadata.all_cells.palantir.11.19.2025.knn30.real_time.cellfate.txt")
pt <- readRDS("All_Palantir_results.11.26.2025.rds")
walks <- list.files(pattern="PROB_WALK*")

# process
paths <- lapply(walks, function(z){readRDS(z)})
names(paths) <- gsub("PROB_WALKS_","",walks)
names(paths) <- gsub("\\.rds","",names(paths))
names(paths) <- gsub("cluster\\.","cluster_",names(paths))

# extract exp data
exprs <- obj@assays$RNA$data


#################################################
# interpolation and and scaling
#################################################

# parameters
numPts <- 300

# construct objects
interObj <- lapply(names(paths), function(z){
  
  # verbose
  message("processing ",z)
  
  # info
  id <- unlist(strsplit(z,"\\."))
  cl <- id[1]
  celltype <- id[2]
  
  # pseudotime
  pt.cl <- pt[[cl]]
  cell.ids <- rownames(pt.cl)
  pt.cl <- pt.cl$Pseudotime
  names(pt.cl) <- cell.ids
  
  # reduce to unique walk cells
  walk.cells <- unique(unlist(paths[[z]]$paths))
  pt.cl <- pt.cl[walk.cells]
  
  # exprs
  s.exprs <- exprs[,walk.cells]
  s.exprs <- s.exprs[Matrix::rowMeans(s.exprs)>0,]
  
  # interpolate
  smooth.exprs = cellAlign::interWeights(expDataBatch = s.exprs, 
                                         trajCond = pt.cl, 
                                         winSz = 0.1, 
                                         numPts = numPts)
  
  # scale
  scaled.exprs = cellAlign::scaleInterpolate(smooth.exprs)
  
  # return
  return(scaled.exprs)
  
})
names(interObj) <- names(paths)
obj <- lapply(interObj, function(z){t(apply(z$scaledData,1,rev))})
traj <- lapply(interObj, function(z){z$traj})

# get observed trajectory values
trajReal <- lapply(names(paths), function(z){
  
  # verbose
  message("processing ",z)
  
  # info
  id <- unlist(strsplit(z,"\\."))
  cl <- id[1]
  celltype <- id[2]
  
  # pseudotime
  pt.cl <- pt[[cl]]
  cell.ids <- rownames(pt.cl)
  pt.cl <- pt.cl$Pseudotime
  names(pt.cl) <- cell.ids
  
  # reduce to unique walk cells
  walk.cells <- unique(unlist(paths[[z]]$paths))
  pt.cl <- pt.cl[walk.cells]
  
  # return
  return(pt.cl)
  
})
names(trajReal) <- names(paths)

# add disparate genes
all_rows <- Reduce(union, lapply(obj, rownames))
interObj.full <- lapply(obj, function(m) {
  
  # create a zero matrix for all rows
  out <- matrix(0, nrow = length(all_rows), ncol = ncol(m),
                dimnames = list(all_rows, colnames(m)))
  
  # fill only matching rows
  out[rownames(m), ] <- m
  out
})
names(interObj.full) <- names(interObj)
names(traj) <- names(interObj)
saveRDS(interObj.full, file="interpolated_expression.rds")
saveRDS(traj, file="interpolated_trajectories.rds")