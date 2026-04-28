#-----------------------------
# Load required packages
#-----------------------------
library(igraph)
library(FNN)
library(Seurat)

# /nfs/turbo/lsa-amarand/alex_home/Arabidopsis_Protoplast_Dedifferentiation/step5_pseudocell_trajectory

# arguments
args <- commandArgs(T)
if(length(args) != 2){stop("Rscript Extract_Trajectories_Module.v3.R <terminal_celltype> <cluster>")}
t.celltype <- as.character(args[1])
cluster <- as.numeric(args[2])

# functions
selectTerminal <- function(x, xobj, reduc="harmony"){
  pcs <- Embeddings(xobj, reduction=reduc)
  centroid <- colMeans(pcs[x,])
  cors <- cor(t(pcs[x,]), centroid)
  cors <- cors[order(cors[,1], decreasing=T),]
  best <- names(cors)[1]
  return(best)
}
getshortestpaths <- function(x, 
                             pseudotime, 
                             cellids, 
                             nodes, 
                             reduction="harmony", 
                             k=15){
  
  # extract embedding
  rd <- x[[reduction]]@cell.embeddings[cellids,]
  pseudotime <- pseudotime[cellids]
  names(pseudotime) <- cellids
  pseudotime <- pseudotime[order(pseudotime, decreasing=F)]
  rd <- rd[names(pseudotime),]
  
  # get knn
  message("getting knn...")
  knn_res <- get.knn(rd, k = k)
  
  # Build directed edges constrained by pseudotime 
  message("building directed graph based on progressive pseudotime...")
  edges_list <- lapply(1:nrow(rd), function(i) {
    from_cell <- rownames(rd)[i]
    neigh_idx <- knn_res$nn.index[i, ]
    neigh_cells <- rownames(rd)[neigh_idx]
    neigh_dist <- knn_res$nn.dist[i, ]
    
    # Only keep neighbors with higher pseudotime
    valid <- (pseudotime[neigh_cells]) >= (pseudotime[from_cell])
    
    if (any(valid)) {
      data.frame(
        from = from_cell,
        to = neigh_cells[valid],
        weight = neigh_dist[valid]
      )
    } else {
      NULL
    }
  })
  edges_df <- do.call(rbind, edges_list)
  
  # Create directed, weighted igraph
  message("finding shortest path...")
  g <- graph_from_data_frame(edges_df, directed = TRUE, vertices = data.frame(name = rownames(rd)))
  
  # get shortest path
  sp <- shortest_paths(g, from=nodes[1], to=nodes[2], algorith="dijkstra")
  
  # get cells top 5% of cells within the shortest path
  message("returning results...")
  return(list(sp=sp, g=g))
  
}

probabilistic_constrained_walk <- function(g,
                                           start_node,
                                           terminal_node,
                                           node_values,
                                           n_walks = 1000,
                                           alpha = 2,
                                           max_steps = 1000,
                                           verbose = TRUE,
                                           saveAdjMat = FALSE,
                                           prefix = "adj_list",
                                           preAdjMat = NULL) {
  # --- Input checks ---
  stopifnot(is_directed(g))
  if (is.null(E(g)$weight)) stop("Graph edges must have a 'weight' attribute.")
  node_names <- V(g)$name
  if (is.null(names(node_values))) stop("node_values must be a named numeric vector.")
  if (!all(node_names %in% names(node_values)))
    stop("node_values must have entries for all vertices.")
  
  # --- Precompute adjacency lists & weights ---
  if(!is.null(preAdjMat)){
    adj_list <- preAdjMat$adj
    weight_list <- preAdjMat$weight
  }else{
    
    if(verbose){message("precomputing adjacency lists/weights ...")}
    adj_list <- lapply(node_names, function(v) neighbors(g, v, mode = "out")$name)
    names(adj_list) <- node_names
    weight_list <- lapply(node_names, function(v) {
      out_edges <- E(g)[.from(v)]
      if (length(out_edges) == 0) return(numeric(0))
      w <- E(g)[out_edges]$weight
      names(w) <- ends(g, out_edges)[, 2]
      w
    })
    names(weight_list) <- node_names
    
    # save
    if(saveAdjMat){
      saveRDS(weight_list, file=paste0(prefix,".weights.rds"))
      saveRDS(adj_list, file=paste0(prefix,".adjacency.rds"))
    }  
  }
  
  
  # --- Precompute reachability to terminal node ---
  reachable_to_terminal <- subcomponent(g, terminal_node, mode = "in")
  reachable_nodes <- intersect(node_names, names(reachable_to_terminal))
  if (!(start_node %in% reachable_nodes))
    stop("Start node cannot reach terminal node.")
  
  # --- Internal single walk (optimized) ---
  single_walk <- function() {
    current <- start_node
    visited <- current
    total_distance <- 0
    
    for (step in seq_len(max_steps)) {
      if (current == terminal_node) break
      
      neighbors <- adj_list[[current]]
      weights <- weight_list[[current]]
      weights <- weights[neighbors]
      if (length(neighbors) == 0){
        message("no neighbors found, breaking walk...") 
        break
      }
      
      # Constraints: increasing node value + must reach terminal
      valid_mask <- (node_values[neighbors]) >= (node_values[current]) &
        neighbors %in% reachable_nodes
      if (!any(valid_mask)){
        message("no valid neighbors found, breaking walk...") 
        break
      }
      valid_neighbors <- neighbors[valid_mask]
      valid_weights <- weights[valid_mask]
      
      inv_w <- 1 / (valid_weights + 1e-8)
      probs <- inv_w^alpha
      probs <- probs / sum(probs)
      
      next_node <- sample(valid_neighbors, 1, prob = probs)
      total_distance <- total_distance + as.numeric(valid_weights[next_node])
      #if(is.na(total_distance)){
      #  message(" mismatch distance/node name in step ",step)
      #}
      visited <- c(visited, next_node)
      current <- next_node
    }
    
    if (tail(visited, 1) != terminal_node) return(NULL)
    list(path = as.character(visited), total_distance = total_distance)
  }
  
  # --- Monte Carlo loop ---
  if(verbose){message("initializing monte carlo constrained walks ...")}
  if (verbose) pb <- txtProgressBar(0, n_walks, style = 3)
  paths <- vector("list", n_walks)
  total_distances <- numeric(n_walks)
  success <- logical(n_walks)
  
  for (i in seq_len(n_walks)) {
    res <- single_walk()
    if (!is.null(res)) {
      paths[[i]] <- res$path
      total_distances[i] <- res$total_distance
      success[i] <- TRUE
    }
    if (verbose && i %% 10 == 0) setTxtProgressBar(pb, i)
  }
  if (verbose) close(pb)
  
  # --- Summaries ---
  successful_paths <- paths[success]
  if (length(successful_paths) == 0)
    message("No successful walks reached the terminal node.")
  
  visit_counts <- table(unlist(successful_paths))
  visit_freq <- visit_counts / sum(visit_counts)
  
  list(
    n_walks = n_walks,
    success_rate = mean(success),
    avg_distance = mean(total_distances[success]),
    visit_freq = sort(visit_freq, decreasing = TRUE),
    paths = successful_paths,
    paths_distance = total_distances[success]
  )
}

#-----------------------------
# parameters
#-----------------------------
k <- 30
reduc <- "ms"


#-----------------------------
# Load input
#-----------------------------
message("LOADING input data...")
obj <- readRDS("step1_palantir_obj_100DC_harmony_knn.15.10.21.2025.knn30.rds")
meta <- read.table("diffusion_pseudotime.metadata.all_cells.palantir.11.19.2025.knn30.real_time.cellfate.txt")
tips <- readRDS("start_end_cells.rds")
pt.list <- readRDS("All_Palantir_results.11.26.2025.rds")


#-----------------------------
# Process inputs  
#-----------------------------
message("Subsetting cells for cluster ", cluster," trajectory...")
in.ids <- rownames(subset(meta, meta$celltype==t.celltype | meta$cellfate_type==t.celltype))
start.cell <- tips[[paste0("cluster_",cluster)]][["start"]]
term.cell <- tips[[paste0("cluster_",cluster)]][["end"]][t.celltype]
c.pt <- pt.list[[paste0("cluster_",cluster)]]$Pseudotime
names(c.pt) <- rownames(pt.list[[paste0("cluster_",cluster)]])
nodes <- c(start.cell, term.cell)
names(nodes) <- c("Totipotent", t.celltype)
new.ids <- unique(c(in.ids, start.cell, term.cell))
c.pt <- c.pt[new.ids]

# plot input data
message("Plotting input, start, and end cells ...")
pdf(paste0("UMAP_input_data_cluster_",cluster,".",t.celltype,".pdf"), width=7, height=7)
plot(meta$umap1, meta$umap2, pch=16, cex=0.3, col=ifelse(rownames(meta) %in% in.ids, "black", "grey90"))
points(meta[nodes,]$umap1, meta[nodes,]$umap2, pch=16, cex=1, col=c("dodgerblue1","firebrick1"))
dev.off()


#-----------------------------
# Build directed kNN graph with nearest neighbors
#-----------------------------
message("Construct graph for cluster ", cluster," trajectory...")
res <- getshortestpaths(obj, c.pt, new.ids, nodes, reduction=reduc, k=k)

# save graph
saveRDS(res, file=paste0("GRAPH_cluster.",cluster,".",t.celltype,".rds"))


#-----------------------------
# Run probabilistic walk
#-----------------------------
message("Starting probabilistic constrained walk...")
set.seed(123)
paths <- probabilistic_constrained_walk(res$g, 
                                        start_node = nodes[1],
                                        terminal_node = nodes[2],
                                        c.pt,
                                        n_walks = 1000,
                                        alpha = 1,
                                        max_steps = 1000,
                                        saveAdjMat = T,
                                        prefix = paste0("ADJ_data.",cluster, ".", t.celltype),
                                        verbose = T)

# save walks
saveRDS(paths, file=paste0("PROB_WALKS_cluster.",cluster,".",t.celltype,".rds"))


