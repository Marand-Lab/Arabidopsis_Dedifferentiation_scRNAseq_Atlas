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


#################################################
# pairwise alignment
#################################################

# pairwise alignment
for(i in 1:(length(names(interObj.full))-1)){
  k <- i+1
  for(j in k:length(names(interObj.full))){
    
    # compute distances
    message("Computing distances between ",i, " & ", j)
    agg <- interObj.full[[i]] + interObj.full[[j]]
    gene.var <- rowSums(agg)
    gene.keep <- names(gene.var)[gene.var > 0]
    A <- calcDistMat(interObj.full[[i]][gene.keep,],interObj.full[[j]][gene.keep,], dist.method = 'Euclidean')
    id <- paste0("./cellAlign_output/",names(interObj.full)[i],"-",names(interObj.full[j]))
    
    # plot
    pdf(paste0(id,".aligned_trajectories.distance.pdf"), width=5, height=5)
    cols <- colorRampPalette(rev(brewer.pal(9, "YlGnBu")))(100)
    pheatmap(A, cluster_rows=F, cluster_cols=F, col=cols)
    dev.off()
    
    # perform global alignment
    message(" - running global alignment")
    gbl <- globalAlign(A, scores=list(query=traj[[i]], ref=traj[[j]], sigCalc=F, numPerm=20))
    
    # plot global alignment
    pdf(paste0(id,".aligned_trajectories.global.pdf"), width=5.5, height=5)
    plotAlign(gbl)
    dev.off()
    
    # mapping
    #message(" - mapping interpolated pseudotime back to observed pseudotime")
    #gmap <- mapRealDataGlobal(gbl, intTrajQuery=traj[[i]], realTrajQuery=trajReal[[i]],
    #                          intTrajRef=traj[[j]], realTrajRef=trajReal[[j]])
    
    # plot global alignment
    #pdf(paste0(id,".aligned_trajectories.global_mapping.pdf"), width=5, height=5)
    #plotMapping(gmap)
    #dev.off()
    
    # cluster genes by distance
    message(" - clustering genes with k=3")
    cl <- suppressMessages(pseudotimeClust(x=interObj.full[[i]][gene.keep,], y=interObj.full[[j]][gene.keep,], k=3))
    
    # per gene alignments
    message(" - per gene alignments")
    per.gene <- lapply(gene.keep, function(x){
      alignment = suppressMessages(globalAlign(interObj.full[[i]][x,], interObj.full[[j]][x,],
                              scores = list(query = traj[[i]],
                                            ref = traj[[j]]),
                              sigCalc = T, numPerm = 100))
      if(x==gene.keep[1]){
        pdf("test.singlegeneALN.pdf", width=6, height=5.5)
        plotAlign(alignment)
        dev.off()
      }
      return(alignment)
    })
    names(per.gene) <- gene.keep
    
    # save results
    saveRDS(A, file=paste0(id,".cellAlign_pairwise_distances.rds"))
    saveRDS(gbl, file=paste0(id,".cellAlign_pairwise_global_alignments.rds"))
    saveRDS(cl, file=paste0(id,".cellAlign_pairwise_shift_clustering.rds"))
    saveRDS(per.gene, file=paste0(id,".cellAlign_single_gene_alignments.rds"))
    
  }
}






