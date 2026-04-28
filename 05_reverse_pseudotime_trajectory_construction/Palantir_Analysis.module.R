## run palantir in R ##

# load arguments
args <- commandArgs(T)
cluster <- as.numeric(args[1])

# load libraries
library(Seurat)
library(reticulate)
library(qlcMatrix)
library(gtools)
library(uwot)
library(scales)
library(igraph)
library(matrixStats)
library(reshape2)

# import palantir
pl <- import("palantir")

# functions
runPalantir <- function(x, 
                        reduction="harmony", 
                        n_components=50, 
                        n.pcs=NULL, 
                        knn=15){
  
  # create temp dir
  subfolder <- tempdir()
  
  # number of input PCs
  if(is.null(n.pcs)){
    n.pcs <- ncol(Embeddings(x, reduction = reduction))
  }
  
  # save reduced dims
  csv_file_path <- file.path(subfolder, "dr.csv")
  write.csv(Embeddings(x, reduction = reduction)[,1:n.pcs], csv_file_path)
  
  # run python
  message(" - running Palantir...")
  code <- glue::glue("\nimport palantir\nimport pandas as pd\n\n# Read the PCA projections from the CSV file\npca_projections = pd.read_csv('{csv_file_path}', index_col=0)\n\n# Run diffusion maps and save the results\ndm_res = palantir.utils.run_diffusion_maps(pca_projections, knn={knn}, n_components={n_components})\ndm_res['EigenVectors'].to_csv('{subfolder}/dm_res.csv')\n\n# Determine multiscale space and save the results\nms_data = palantir.utils.determine_multiscale_space(dm_res)\nms_data.to_csv('{subfolder}/ms_data.csv')\n")
  py_run_string(code)
  message(" - finished running Palantir...")
  dm_res_path <- file.path(subfolder, "dm_res.csv")
  ms_data_path <- file.path(subfolder, "ms_data.csv")
  
  # lol
  dm_data <- read.csv(dm_res_path, row.names = 1)
  colnames(dm_data) <- paste0("DM_", 1:ncol(dm_data))
  x[["dm"]] <- CreateDimReducObject(embeddings = as.matrix(dm_data),
                                    key = "DM_", assay = DefaultAssay(x))
  ms_data <- read.csv(ms_data_path, row.names = 1)
  colnames(ms_data) <- paste0("MS_", 1:ncol(ms_data))
  x[["ms"]] <- CreateDimReducObject(embeddings = as.matrix(ms_data),
                                    key = "MS_", assay = DefaultAssay(x))
  return(x)
  
}
runPalantirPT <- function(x, start_cell, 
                          terminal_states=NULL, 
                          title="Pseudotime", 
                          n_jobs=1, 
                          n_waypoints=2500, 
                          knn=15){
  
  subfolder <- tempdir()
  csv_file_path <- file.path(subfolder, "ms.csv")
  write.csv(Embeddings(x, reduction = "ms"), csv_file_path)
  common_code <- glue::glue("\nimport palantir\nimport pandas as pd\n\n# Read the MS from the CSV file\nms_data = pd.read_csv('{csv_file_path}', index_col=0)\n\n# Define start_cell\nstart_cell = '{start_cell[1]}'\n")
  if (is.null(terminal_states)) {
    specific_code <- glue::glue("pr_res = palantir.core.run_palantir(ms_data, start_cell, num_waypoints={n_waypoints}, n_jobs={n_jobs}, use_early_cell_as_start=True, knn={knn})")
  }
  else {
    terminal_states_dict_str <- toString(sapply(names(terminal_states),
                                                function(name) sprintf("'%s': '%s'", terminal_states[name],
                                                                       name), USE.NAMES = FALSE), collapse = ", ")
    specific_code <- glue::glue("\n# Create the terminal_states pandas Series\nterminal_states_dict = {{ {terminal_states_dict_str} }}\nterminal_states = pd.Series(terminal_states_dict)\n\n# Run diffusion maps with start_cell and terminal_states and save the results\npr_res = palantir.core.run_palantir(ms_data, start_cell, num_waypoints={n_waypoints}, use_early_cell_as_start=True, terminal_states=terminal_states.index, n_jobs={n_jobs}, knn={knn})\npr_res.branch_probs.columns = terminal_states[pr_res.branch_probs.columns]\n")
  }
  code2 <- glue::glue("\n# Combine the results into one DataFrame\nresult_df = pd.DataFrame({{\n    'Pseudotime': pr_res.pseudotime,\n    'Entropy': pr_res.entropy\n}})\nresult_df = pd.concat([result_df, pr_res.branch_probs], axis=1)\n\n# Write the combined DataFrame to a CSV file\nresult_df.to_csv('{subfolder}/pr_res.csv')\n")
  code <- paste(common_code, specific_code, code2, sep = "\n")
  py_run_string(code)
  pr_res_path <- file.path(subfolder, "pr_res.csv")
  pr_data <- read.csv(pr_res_path, row.names = 1)
  x@misc[["Palantir"]][[title]] <- pr_data
  return(x)
  
}
getCellTips <- function(xx, 
                        start.cluster=18){
  
  # clusterID
  clusterID <- paste0("cluster",start.cluster)
  aa <- t(Embeddings(xx, reduction="harmony"))
  
  # get start and terminal cells
  message(" - getting cluster averages...")
  clust.aves <- lapply(unique(xx@meta.data$seurat_clusters), function(z){
    rowMeans(aa[,rownames(xx@meta.data[xx@meta.data$seurat_clusters==z,])])
  })
  clust.aves <- do.call(cbind, clust.aves)
  colnames(clust.aves) <- paste0("cluster",unique(xx@meta.data$seurat_clusters))
  cts <- unique(xx@meta.data$celltype)
  cts <- cts[!is.na(cts)]
  message(" - getting celltype D0 centroid...")
  celltype.aves <- lapply(cts, function(z){
    message(z)
    rowMeans(aa[,colnames(aa) %in% rownames(xx@meta.data[xx@meta.data$celltype==z & xx@meta.data$treatment=="D0",])])
  })
  celltype.aves <- do.call(cbind, celltype.aves)
  colnames(celltype.aves) <- paste0("celltype",cts)
  t.ids <- rownames(celltype.aves)
  
  # correlations
  celltype.cor <- corSparse(aa[t.ids,], celltype.aves[t.ids,])
  cluster.cor <- corSparse(aa[t.ids,], clust.aves[t.ids,])
  
  # rename
  rownames(celltype.cor) <- colnames(aa)
  rownames(cluster.cor) <- colnames(aa)
  colnames(celltype.cor) <- colnames(celltype.aves)
  colnames(cluster.cor) <- colnames(clust.aves)
  
  # select best start cell
  cluster.cor <- cluster.cor[order(cluster.cor[,clusterID], decreasing=T),]
  top.cluster <- apply(cluster.cor, 1, function(x){names(x)[which.max(x)]})
  top.cluster <- top.cluster[top.cluster==clusterID]
  top.cluster <- top.cluster[names(top.cluster) %in% rownames(xx@meta.data[xx@meta.data$seurat_clusters==start.cluster,])]
  start.cell <- names(top.cluster)[1]
  
  # select terminal cells
  top.cluster2 <- apply(cluster.cor, 1, function(x){names(x)[which.max(x)]})
  score.cluster2 <- apply(cluster.cor, 1, max)
  cluster2 <- data.frame(cluster=top.cluster2, score=score.cluster2, row.names=names(top.cluster2))
  cluster2 <- cluster2[order(cluster2$score, decreasing=T),]
  ucluster2 <- cluster2[!duplicated(cluster2$cluster),]
  top.celltype <- apply(celltype.cor, 1, function(x){names(x)[which.max(x)]})
  xx@meta.data$temp.celltype <- top.celltype[rownames(xx@meta.data)]
  xx@meta.data$temp.celltype <- gsub("celltype","",xx@meta.data$temp.celltype)
  xx@meta.data$temp.corval <- apply(celltype.cor, 1, max)
  mm <- xx@meta.data
  mm <- mm[mm$temp.celltype==mm$celltype,]
  mm <- mm[order(mm$temp.corval, decreasing=T),]
  mm <- subset(mm, mm$treatment=="D0")
  
  # ensure terminal cell is in correct cluster
  t.clusters <- list(Atrichoblast=8, Bundle_Sheath=17, `Columella/Lateral_Root_Cap`=8,
                     `Companion_Cell/Sieve_Elements`=17, Cortex=15, Endodermis=15,
                     Epidermal_initials=8, `G2/M_Cell`=20, Guard_Cell=18, Hydathode=14,
                     Meristematic_Cell=17, Palisade_Mesophyll=16, Pavement_Cell=23, Phloem_Parenchyma=17,
                     Phloem_Pole_Pericyle=17, Procambium=17, Protodermal_Cell=23, `S/Phase_Mesophyll`=22,
                     Spongy_Mesophyll=3, Trichoblast=8, Xylem_Pole_Pericyle=17)
  mm <- mm[mm$seurat_clusters==t.clusters[mm$celltype],]
  
  # filter
  rep.celltypes <- mm[!duplicated(mm$celltype),]
  rep.celltypes$clusterID <- paste0("cluster",rep.celltypes$og_seurat_clusters)
  ucluster2 <- ucluster2[!ucluster2$cluster %in% rep.celltypes$clusterID,]
  ucluster2 <- ucluster2[!ucluster2$cluster %in% c(paste0("cluster",start.cluster)),]
  terminal.states <- rownames(rep.celltypes) #, rownames(ucluster2))
  names(terminal.states) <- rep.celltypes$celltype #, ucluster2$cluster)
  terminal.states <- terminal.states[!is.na(names(terminal.states))]

  # return cell IDs
  return(list(start=start.cell, end=terminal.states))
  
}


###################################################################################################
## run Palantir on all cells
###################################################################################################

# load data
obj <- readRDS("step1_palantir_obj_100DC_harmony_knn.15.10.21.2025.knn30.rds")
meta <- read.table("diffusion_pseudotime.metadata.all_cells.palantir_pt_entrop.11.06.2025.knn30.inferred_age.txt")
meta$og_seurat_clusters <- meta$seurat_clusters
shared <- intersect(rownames(meta), rownames(obj@meta.data))
obj <- obj[,shared]
obj@meta.data <- meta[colnames(obj),]

# parameters
knn1 <- 15
knn2 <- 30
prefix <- paste0("cluster_",cluster)

# run diffusion
# obj <- runPalantir(obj, n_components=100, reduction="harmony", knn=knn1)

# get tips
tips <- getCellTips(obj, start.cluster=cluster)
saveRDS(tips, file=paste0(prefix,".tips.rds"))

# estimate pseudotime
results <- runPalantirPT(obj,
                          start_cell=tips$start,
                          terminal_states=tips$end, 
                          knn=knn2)

# save results
saveRDS(results@misc$Palantir$Pseudotime, file=paste0("PALANTIR_RESULTS.",prefix,".rds"))




