###########################################################################
###########################################################################
##
## identify conserved early regulators 
##
###########################################################################
###########################################################################

# load libraries
library(scales)
library(gplots)
library(pheatmap)
library(RColorBrewer)
library(igraph)
library(cluster)
library(GO.db)
library(fgsea)
library(glmnet)
library(Matrix)
library(reshape2)
library(dplyr)
library(tidyverse)
library(reshape2)

# load functions
merge_named_vectors <- function(vec_list, fill = NA) {
  
  # union of all names
  all_names <- unique(unlist(lapply(vec_list, names)))
  
  # build matrix
  out <- sapply(vec_list, function(v) {
    v[match(all_names, names(v))]
  })
  
  rownames(out) <- all_names
  out[is.na(out)] <- fill
  
  out
}
fast_lagged_grn <- function(mat, targets, regulators, lag=1, threshold=0.8){
  
  # mat: genes x time
  # regulators: subset of genes
  
  # normalized
  mat <- as.matrix(t(scale(t(mat))))
  
  # introduce lag
  X <- mat[, 1:(ncol(mat)-lag), drop=FALSE]     # t
  Y <- mat[, (1+lag):ncol(mat), drop=FALSE]     # t+lag
  
  # find regulators
  reg_idx <- match(regulators, rownames(mat))
  reg_idx <- reg_idx[!is.na(reg_idx)]
  X_reg <- X[reg_idx, , drop=FALSE]
  
  # find targets
  tar_idx <- match(targets, rownames(mat))
  tar_idx <- tar_idx[!is.na(tar_idx)]
  Y_tar <- Y[tar_idx, ,drop=FALSE]
  
  # matrix multiplication = correlations
  cor_mat <- X_reg %*% t(Y_tar) / (ncol(X_reg) - 1)
  rownames(cor_mat) <- rownames(X_reg)
  colnames(cor_mat) <- rownames(Y_tar)
  
  # keep top targets per regulator
  df <- melt(cor_mat)
  if(!is.na(threshold)){
    df <- subset(df, abs(df$value)>threshold)
  }
  colnames(df) <- c("regulator", "target","cor")
  return(df)
}
fast_multilag_grn <- function(mat, regulators,
                              lags = c(1,2,3,5,10,20),
                              decay = 0.7,
                              threshold = 0.8){
  
  # scale matrix
  mat <- as.matrix(t(scale(t(mat))))
  
  reg_idx <- match(regulators, rownames(mat))
  reg_idx <- reg_idx[!is.na(reg_idx)]
  
  results_per_lag <- list()
  
  for(l in lags){
    
    if(ncol(mat) <= l + 1) next
    
    X <- mat[, 1:(ncol(mat)-l), drop=FALSE]
    Y <- mat[, (1+l):ncol(mat), drop=FALSE]
    
    X_reg <- X[reg_idx, , drop=FALSE]
    
    cor_mat <- X_reg %*% t(Y) / (ncol(X_reg)-1)
    
    results_per_lag[[as.character(l)]] <- cor_mat
  }
  
  # --- aggregate across lags ---
  lag_names <- names(results_per_lag)
  
  # convert to array: [reg, target, lag]
  cor_array <- simplify2array(results_per_lag)
  
  # weights (exponential decay)
  lag_vals <- as.numeric(lag_names)
  weights <- decay^(lag_vals - 1)
  weights <- weights / sum(weights)
  
  # weighted sum
  weighted_cor <- apply(cor_array, c(1,2), function(x){
    sum(x * weights, na.rm=TRUE)
  })
  
  rownames(weighted_cor) <- regulators
  colnames(weighted_cor) <- rownames(mat)
  
  # --- edge extraction ---
  df <- melt(weighted_cor)
  colnames(df) <- c("eReg", "allReg", "weight")
  df$effect <- ifelse(df$weight > 0, "pos", "neg")
  return(df)
  
  
}

###########################################################################
## data loading and processing
###########################################################################

# load expression data
dir <- "/nfs/turbo/lsa-amarand/alex_home/Arabidopsis_Protoplast_Dedifferentiation/step5_pseudocell_trajectory/probabilistic_walks/"
files <- list.files(path=dir, pattern="*.average_expression.rds")
a <- lapply(files, function(z){
  expr <- readRDS(paste0(dir,z))
  t(apply(expr, 1, function(x){x <- rev(x);x <- x-min(x);x/max(x)}))
})
names(a) <- gsub("\\.average_expression\\.rds","",files)
names(a) <- gsub("cluster\\.","cluster_",names(a))

# load reverse pseudotime data
pw <- list.files(path=dir, pattern="PROB_WALKS*")
pwd <- lapply(pw, function(z){
  unique(unlist(readRDS(paste0(dir,z))$paths))
})
names(pwd) <- gsub("PROB_WALKS_","",pw)
names(pwd) <- gsub("cluster\\.","cluster_",names(pwd))
names(pwd) <- gsub("\\.rds","",names(pwd))

# meta data
m <- read.table("/nfs/turbo/lsa-amarand/alex_home/Arabidopsis_Protoplast_Dedifferentiation/step5_pseudocell_trajectory/diffusion_pseudotime.metadata.all_cells.palantir.11.19.2025.knn30.real_time.cellfate.txt")
pt <- m$consensus_pseudotime
names(pt) <- rownames(m)

# k-means and alignment results
gs <- readRDS("kmean_gene_data.03.19.2026.rds")
alnG <- readRDS("traj_single_gene_alignments_classified.rds")

# annotation
ann <- read.table("/nfs/turbo/lsa-amarand/shared_data/arabidopsis/reference_data/annotations/Arabidopsis_thaliana.TAIR10.58.annotation.txt", header=F)


###########################################################################
## collect activation timing and variance
###########################################################################

# estimate center of mass
cm <- lapply(names(a), function(z){
  cells <- pwd[[z]]
  pt.cells <- pt[cells]
  
  # true pseudotime
  #pt.int <- seq(from=max(pt.cells), to=min(pt.cells), length.out=300)
  # proportional pseudotime
  pt.int <- seq(from=300, to=1)/300
  cm.g <- apply(a[[z]], 1, function(x){
    sum(x*pt.int)/sum(x)
  })
  return(cm.g)
})

# merge
cm <- merge_named_vectors(cm)
colnames(cm) <- names(a)

# mean, std, and conservation of activation timing
aves <- rowMeans(cm, na.rm=T)
stds <- apply(cm, 1, sd, na.rm=T)
cons <- rowMeans(!is.na(cm))

# plot
aves <- aves[order(aves, decreasing=T)]
stds <- stds[names(aves)]
stds[is.na(stds)] <- 0
cols <- colorRampPalette(rev(brewer.pal(11, "Spectral")))(100)

# pdf
pdf("Conservation_activation_timing.pdf", width=6, height=5)
layout(matrix(c(1:2), nrow=2), heights=c(1,3))
par(mar=c(1,5,1,3))
plot(seq(1:length(aves)),stds, type="h", xlab=NA, xaxt='n')
ss <- smooth.spline(seq(1:length(aves)), stds[names(aves)], spar=0.1)
lines(ss, col="dodgerblue3", lwd=2)
par(mar=c(5,5,0,3))
plot(seq(1:length(aves)),aves, type="h", cex=0.3,
     col=cols[cut(cons[names(aves)], breaks=101)],
     xlab="Rank",
     ylab="Average expression timing") 
dev.off()

# plot
cm <- cm[order(aves, decreasing=F),]
cm[is.na(cm)] <- 0
pheatmap(cm, cluster_rows=T, cluster_cols=T)

# save results
expr.timing <- data.frame(actTiming=aves,actTimingDev=stds, nTraj=cons[names(aves)])
saveRDS(expr.timing, file="Activation_timing.rds")


###########################################################################
## identify candidates with regulatory potential
###########################################################################

# go terms: transcription regulation, chromatin organization, signaling
go <- read.delim("/nfs/turbo/lsa-amarand/shared_data/arabidopsis/reference_data/annotations/TAIR10.GO", header=F)
gt <- read.delim("/nfs/turbo/lsa-amarand/shared_data/arabidopsis/reference_data/annotations/TAIR10_desc_GO_mapping.txt", header=F)
goterm <- gt$V2
names(goterm) <- gt$V1
df <- as.data.frame(do.call(rbind, strsplit(go$V1," "))[,1])
df$geneIDs <- go$V3
colnames(df)[1] <- "pathwayID"
df$goIDs <- goterm[df$pathwayID]

# TFs
tfgo <- c("GO:0003700","GO:0140110","GO:0001228","GO:0001227","GO:0000976",
          "GO:0000981")
tf_terms <- unlist(mget(tfgo, GOMFOFFSPRING, ifnotfound = NA))
tf_terms <- unique(c(tf_terms, tfgo))
tf_terms <- tf_terms[!is.na(tf_terms)]
tf_genes <- df[df$goIDs %in% unique(tf_terms),]
tfs <- unlist(strsplit(tf_genes$geneIDs,","))

# chromatin remodelers
crgo <- c("GO:0006338","GO:0140657","GO:0016570", "GO:0016573", "GO:0016571",
          "GO:0006306", "GO:0003682")
cr_terms <- unlist(mget(crgo, GOMFOFFSPRING, ifnotfound = NA))
cr_terms <- unique(c(cr_terms, crgo))
cr_terms <- cr_terms[!is.na(cr_terms)]
cr_genes <- df[df$goIDs %in% unique(cr_terms),]
crs <- unlist(strsplit(cr_genes$geneIDs,","))

# phosphorylation
phgo <- c("GO:0016301","GO:0004672","GO:0004674","GO:0004713","GO:0016791",
          "GO:0004721")
ph_terms <- unlist(mget(phgo, GOMFOFFSPRING, ifnotfound = NA))
ph_terms <- unique(c(ph_terms, phgo))
ph_terms <- ph_terms[!is.na(ph_terms)]
ph_genes <- df[df$goIDs %in% unique(ph_terms),]
phs <- unlist(strsplit(ph_genes$geneIDs,","))

# signal transduction
stgo <- c("GO:0007165","GO:0004871", "GO:0005102")
st_terms <- unlist(mget(stgo, GOMFOFFSPRING, ifnotfound = NA))
st_terms <- unique(c(st_terms, stgo))
st_terms <- st_terms[!is.na(st_terms)]
st_genes <- df[df$goIDs %in% unique(st_terms),]
sts <- unlist(strsplit(st_genes$geneIDs,","))

# merged
all_reg <- unique(c(tfs, crs, phs, sts))



###########################################################################
## trajectory-dependent GRNs
###########################################################################

# construct GRN
grns <- lapply(names(a), function(z){
  message("Generating lagged GRN for trajectory ",z)
  dgrn <- fast_lagged_grn(mat=a[[z]], regulators=all_reg, lag=1, threshold=0.8)
  dgrn$trajectory <- z
  return(dgrn)
})
names(grns) <- names(a)

# normalize
grns <- lapply(names(grns), function(z){
  grns[[z]]$cor <- grns[[z]]$cor/max(abs(grns[[z]]$cor))
  grns[[z]]
})
names(grns) <- names(a)
saveRDS(grns, file="dGRNs_normalized.rds")

# merge
combined_grn <- bind_rows(grns)

# summary
edge_summary <- combined_grn %>%
  group_by(regulator, target) %>%
  summarise(
    mean_weight = mean(cor),
    traj_support = n(),
    .groups="drop")
edge_summary$reg_target <- paste(edge_summary$regulator, edge_summary$target, sep="-")
edgeF <- subset(edge_summary, abs(edge_summary$mean_weight)>= 0.8 & edge_summary$traj_support >= 3)
saveRDS(edge_summary, file="dGRN_edgeSummary.rds")

# compute regulator conservation
regulator_stats <- edge_summary %>%
  group_by(regulator) %>%
  summarise(
    n_targets = n(),
    mean_weight = mean(mean_weight),
    traj_support = mean(traj_support),
    .groups="drop"
  )
saveRDS(regulator_stats, file="dGRN_regulator_conservation.rds")

# calculate regulatory influence (edge number/centrality)
reg.scores <- lapply(names(grns), function(z){
  df <- grns[[z]]
  df$regulator <- factor(df$regulator, levels=sort(unique(regulator_stats$regulator)))
  df$effect <- ifelse(df$cor > 0, "Pos", "Neg")
  props <- as.data.frame(table(df$regulator, df$effect))
  dprops <- dcast(Var1~Var2, data=props, value.var="Freq")
  dprops$trajectory <- z
  colnames(dprops)[1] <- "regulator"
  return(dprops)
})
names(reg.scores) <- names(grns)
reg.scores <- do.call(rbind, reg.scores)
saveRDS(reg.scores, file="dGRN_regulator_scores.rds")

# condense dgrns
cgrns <- lapply(names(grns), function(z){
  df <- grns[[z]]
  df$regulator <- factor(df$regulator, levels=sort(unique(regulator_stats$regulator)))
  df$effect <- ifelse(df$cor > 0, "Pos", "Neg")
  df$reg_target <- paste(df$regulator,df$target,sep="-")
  df$regulator <- NULL
  df$target <- NULL
  return(df)
})
cgrns <- do.call(rbind, cgrns)
cgrns <- cgrns[cgrns$reg_target %in% edgeF$reg_target,]
supp <- aggregate(trajectory~reg_target, data=cgrns, FUN=paste, collapse=",")
edgeF <- as.data.frame(edgeF)
rownames(edgeF) <- edgeF$reg_target
edgeF <- edgeF[supp$reg_target,]
edgeF$traj <- supp$trajectory
edgeF$Mtraj <- ifelse(grepl("Mesophyll", edgeF$traj), 1, 0)
edgeF$Htraj <- ifelse(grepl("Hydathode", edgeF$traj), 1, 0)
edgeF$Gtraj <- ifelse(grepl("Guard", edgeF$traj), 1, 0)
edgeF$numDiffOrigin <- edgeF$Mtraj + edgeF$Htraj + edgeF$Gtraj
edgeF2 <- subset(edgeF, edgeF$numDiffOrigin>1)
edgeF3 <- subset(edgeF, edgeF$numDiffOrigin>2)
#cgrnsF <- cgrns[cgrns$reg_target %in% edgeF3$reg_target,]
cgrnsF <- cgrns[cgrns$reg_target %in% edgeF2$reg_target,]
df <- as.data.frame(do.call(rbind, strsplit(cgrnsF$reg_target, "-")))
colnames(df) <- c("regulator", "target")
cgrnsF <- cbind(cgrnsF, df)
reg.summary <- as.data.frame(table(cgrnsF$regulator, cgrnsF$trajectory, cgrnsF$effect))
reg.stat <- aggregate(Freq~Var1+Var3, data=reg.summary, FUN=mean)
regstat <- dcast(Var1~Var3, data=reg.stat, value.var="Freq")
regstat$All <- regstat$Pos+regstat$Neg
exprT <- readRDS("Activation_timing.rds")
shared <- intersect(rownames(exprT), regstat$Var1)
exprT$regulator <- ifelse(rownames(exprT) %in% shared, 1, 0)
neg <- regstat$Neg
pos <- regstat$Pos
all <- regstat$All
names(neg) <- regstat$Var1
names(pos) <- regstat$Var1
names(all) <- regstat$Var1
exprT$regNeg <- neg[rownames(exprT)]
exprT$regPos <- pos[rownames(exprT)]
exprT$regAll <- all[rownames(exprT)]
exprT[is.na(exprT)] <- 0
exprT$isTF <- ifelse(rownames(exprT) %in% tfs, 1, 0)
exprT$score <- (exprT$actTiming/max(exprT$actTiming)) * (exprT$nTraj^2) * (sqrt(exprT$regAll)/sqrt(max(exprT$regAll)))
exprT <- exprT[order(exprT$score, decreasing=T),]
gName <- ann$V2
gDesc <- ann$V3
names(gName) <- ann$V1
names(gDesc) <- ann$V1
exprT$gName <- gName[rownames(exprT)]
exprT$gDesc <- gDesc[rownames(exprT)]

# plot results
exprT <- readRDS("ranked_regulators.timing.centrality.rds")
regs <- subset(exprT, exprT$regulator==1 & exprT$isTF==1)
pdf("Centrality_by_activation_timing.pdf", width=7, height=5)
plot(-1*regs$actTiming, regs$regAll, 
     col=colorRampPalette(brewer.pal(9, "Blues")[1:9])(100)[cut(regs$score, breaks=101)],
     pch=16,
     xlab="Average activation timing",
     ylab="# target genes")
grid(lty=1)
dev.off()

# plot reg score
pdf("Regulator_score_rankings.kr.v2.pdf", width=5, height=5)
plot(seq(1:nrow(regs)),
     regs$score,
     col=ifelse(regs$known==1,"firebrick3","grey75"),#colorRampPalette(brewer.pal(9, "Blues")[1:9])(100)[cut(regs$actTiming, breaks=101)],
     pch=16,
     cex=1,
     xlab="Regulator ranking",
     ylab="Dedifferentiation initiation score")
grid(lty=1)
dev.off()

# known regulators
kr <- read.table("known_regulators_chatgpt.txt")
regs$known <- ifelse(rownames(regs) %in% kr$V1, 1, 0)
pdf("enrichment_reg_score_known.pdf")
boxplot(regs$score~regs$known, outline=F)
dev.off()

# check GSEA of dedifferentiation potential genes
gmt1 <- gmtPathways("/nfs/turbo/lsa-amarand/shared_data/arabidopsis/reference_data/annotations/TAIR10.GO")
gmt <- lapply(gmt1, function(z){df <- do.call(c, strsplit(z, ",")); df[df %in% rownames(regs)]})
names(gmt) <- names(gmt1)
gmtt <- lapply(gmt, function(z){if(length(z)<1){return(NULL)}else{return(z)}})
gmt <- Filter(Negate(is.null), gmtt)
  
# run GSEA
regs$zscore <- as.numeric(scale(regs$score))
results <- fora(pathways = gmt,
                genes = rownames(regs)[regs$zscore > 1],
                universe = rownames(regs),
                minSize = 2,
                maxSize = 100)
                 
results <- as.data.frame(results)
results <- results[order(results$pval, decreasing=F),]
results$overlapGenes <- NULL
p.res <- subset(results, results$NES > 0)


###########################################################################
## trajectory-dependent GRNs (regulators only)
###########################################################################

# select regulators in the top 20% of activation timing
at <- readRDS("Activation_timing.rds")
at <- at[order(at$actTiming, decreasing=T),]
atr <- at[rownames(at) %in% all_reg,]
atr <- subset(atr, atr$nTraj >= (2/12) & atr$actTiming >= quantile(at$actTiming, 0.8))

# construct GRN
rns <- lapply(names(a), function(z){
  message("Generating lagged GRN for trajectory ",z)
  exprs <- a[[z]]
  exprs <- exprs[rownames(exprs) %in% all_reg,]
  regs <- rownames(atr)[rownames(atr) %in% rownames(exprs)]
  dgrn <- fast_multilag_grn(mat=exprs, regulators=regs)
  dgrn$trajectory <- z
  return(dgrn)
})
names(rns) <- names(a)

# normalize
rns <- lapply(names(rns), function(z){
  rns[[z]]$weight <- rns[[z]]$weight/max(abs(rns[[z]]$weight))
  rns[[z]]
})
names(rns) <- names(a)
saveRDS(rns, file="dGRNs_normalized.multilag.regulators.rds")

# merge rns
rns <- do.call(rbind, rns)
rownames(rns) <- seq(1:nrow(rns))
rns$reg_target <- paste(rns$eReg, rns$allReg, sep="-")
mm <- dcast(reg_target~trajectory, data=rns, value.var="weight")
rownames(mm) <- mm$reg_target
mm$reg_target <- NULL
mm <- as.matrix(mm)
mm[is.na(mm)] <- 0

# k-means cluster r/t
kk <- 30
wss <- sapply(seq(from=2,to=kk,by=2), function(k) {
  message(k)
  kmeans(mm, centers = k, nstart = 50)$tot.withinss
})
pdf("mm_kmeans.2_30.pdf", width=5, height=5)
plot(seq(from=2,to=kk,by=2), wss, type = "b",
     xlab = "Number of clusters (k)",
     ylab = "Total within-cluster SS")
dev.off()

kclust <- kmeans(mm, centers = 20, nstart = 50)

# plot
hcl <- hclust(dist(t(mm)))$order
mm <- mm[,hcl]
pdf("mm_sum_sort.heatmap.pdf", width=8, height=10)
heatmap.2(mm[order(rowSums(mm), decreasing=T),],
          dendrogram="none",
          trace="none",
          Rowv=F,
          Colv=F,
          col=colorRampPalette(rev(brewer.pal(9, "RdBu")))(100),
          useRaster=T)
dev.off()

# get row-means
row.ave <- rowMeans(mm)
row.f <- row.ave[abs(row.ave)>=0.8]

###########################################################################
## Conserved regulators (independent potential across trajectories)
###########################################################################


###########################################################################
## Dedifferentiation regulators are distinct to somatic identity genes
###########################################################################
ctm <- read.table("top_arabidopsis_marker_genes.txt", header=T)
conctm <- readRDS("/nfs/turbo/lsa-amarand/alex_home/Arabidopsis_Protoplast_Dedifferentiation/step5_pseudocell_trajectory/misc/Consensus_marker_genes.Arabidopsis.rds")
conf <- subset(conctm, conctm$adjScore > 0.5)
cts <- c("Guard_cell","Hydathodes","Leaf_epidermis","Leaf_guard_cell","Leaf_pavement_cell",
         "Mesophyll","Palisade_mesophyll","Spongy_mesophyll")
conf <- conf[conf$clusterName %in% cts,]
all.ctm <- unique(c(conf$gene, ctm$geneID))
exprT <- readRDS("ranked_regulators.timing.centrality.rds")
exprT$ctm <- ifelse(rownames(exprT) %in% ctm$geneID, 1, 0)
exprT$conctm <- ifelse(rownames(exprT) %in% conf$gene, 1, 0)
exprT$ctmAny <- exprT$ctm + exprT$conctm
exprT$ctmAny <- ifelse(exprT$ctmAny > 0, 1, 0)
regs <- exprT[exprT$regulator==1 & exprT$isTF==1,]
pdf("somatic_markers.dediff_potential.pdf", width=5, height=5)
boxplot(score~ctmAny, data=regs, outline=F)
dev.off()

regs$top <- ifelse(as.numeric(scale(regs$score))>1, 1, 0)

###########################################################################
## Integrated evidence of master dediff regulators (contrast with known factors)
###########################################################################

# score = timing + traj_conservation + centrality + regulatory_potential -1 * (correlation with known somatic identity genes)
# groundtruth = WIND1







