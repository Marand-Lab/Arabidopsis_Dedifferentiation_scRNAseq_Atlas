## cluster aligned expression profiles ##

# load libraries
library(cluster)
library(gplots)
library(RColorBrewer)
library(viridis)
library(fgsea)

# word cloud libs
library(dplyr)
library(tidytext)
library(wordcloud)
library(ggplot2)
library(stringr)
library(rrvgo)
library(org.At.tair.db)
library(GO.db)
library(AnnotationDbi)
library(scales)

# load functions

# load data
gs <- readRDS("traj_single_gene_alignments_classified.rds")
dir <- "/nfs/turbo/lsa-YOURNAME/YOURNAME_home/Arabidopsis_Protoplast_Dedifferentiation/step5_pseudocell_trajectory/probabilistic_walks"
dat <- list.files(path=dir, pattern="*.average_expression.rds")
expr <- lapply(dat, function(z){readRDS(paste0(dir,"/",z))})

# update trajectory names
names(expr) <- gsub("\\.average_expression\\.rds","",dat)
names(expr) <- gsub("cluster\\.","cluster_",names(expr))

# process
aln.genes <- unique(gs$geneID)
expr.filt <- lapply(expr, function(z){
  z <- z[rownames(z) %in% aln.genes,]
  message("num genes = ", nrow(z))
  return(z[,c(ncol(z):1)])
})
commonIDs <- Reduce(intersect, lapply(expr.filt, rownames))
expr.filt <- lapply(expr.filt, function(z){z[commonIDs,]})

# average trajectories and standardize
ave.expr <- Reduce("+", expr.filt)/length(expr.filt)
ave.z <- as.matrix(t(scale(t(ave.expr))))
std.expr1 <- lapply(expr.filt, function(z){(z-ave.z)^2})
std.expr <- sqrt(Reduce("+", std.expr1)/length(expr.filt))
ave.expr.0 <- ave.expr - min(ave.expr)
cv.expr <- ave.expr.0/std.expr
  
#################################################
# cluster
#################################################
set.seed(1)

# within cluster sum of squares (k=8)
kk <- 30
wss <- sapply(1:kk, function(k) {
  kmeans(ave.z, centers = k, nstart = 50)$tot.withinss
})
plot(1:kk, wss, type = "b",
     xlab = "Number of clusters (k)",
     ylab = "Total within-cluster SS")

# silhouette (k=8)
avg_sil <- sapply(2:15, function(k) {
  km <- kmeans(ave.z, centers = k, nstart = 50)
  ss <- silhouette(km$cluster, dist(ave.z))
  mean(ss[, 3])
})
plot(2:15, avg_sil, type = "b",
     xlab = "k",
     ylab = "Average silhouette width")

# gap statistic
gap <- clusGap(ave.z,
               FUN = kmeans,
               nstart = 50,
               K.max = 15,
               B = 100)
plot(gap)

# choose K=8
k <- 8
kclust <- kmeans(ave.z, centers = k, nstart = 50, iter.max=100)

# order clusters by average position
ave.shift <- pmax(ave.z, 0)
t <- 1:ncol(ave.shift)
df <- lapply(seq(1:k), function(z){
  geneIDs <- names(kclust$cluster[kclust$cluster==z])
  #mean(rowSums(ave.shift[geneIDs,] * matrix(t, length(geneIDs), ncol(ave.shift), byrow = TRUE)) /
  #  rowSums(ave.shift[geneIDs,]))
  ave <- colMeans(ave.z[geneIDs,])
  wm <- which.max(ave)
  mm <- max(ave)
  return(data.frame(wm=wm, mm=mm, k=z))
})
df <- do.call(rbind, df)
df <- df[order(df$wm, df$mm*ifelse(df$wm < 150,-1,1), decreasing=F),]
gIDs <- names(kclust$cluster)
kclust$cluster <- as.numeric(factor(kclust$cluster, levels=df$k))
names(kclust$cluster) <- gIDs
ave.z.k <- ave.z[names(sort(kclust$cluster, decreasing=F)),]

# plot kmeans plot
pdf(paste0("03.13.2026.Mean_zscore_all_traj.aligned_genes.",k,".pdf"), width=6, height=9)
r.cols <- colorRampPalette(brewer.pal(12, "Paired"))(k)
heatmap.2(ave.z.k, Rowv=F, Colv=F, trace='none', dendrogram='none',
          col=colorRampPalette(c("grey85",brewer.pal(9,"YlGnBu")[2:9]),bias=0.5)(100),
          breaks=seq(from=-2.5,to=2.5,length.out=101),
          useRaster=T, labRow = F, labCol = F,
          RowSideColors=r.cols[kclust$cluster[rownames(ave.z.k)]])
dev.off()

pdf(paste0("03.13.2026.Std_dev_all_traj.aligned_genes.",k,".pdf"), width=6, height=9)
r.cols <- colorRampPalette(brewer.pal(12, "Paired"))(k)
heatmap.2(std.expr[rownames(ave.z.k),], Rowv=F, Colv=F, trace='none', dendrogram='none',
          col=colorRampPalette(c("grey85",brewer.pal(9,"YlGnBu")[2:9]),bias=0.5)(100),
          breaks=seq(from=0,to=3,length.out=101),
          useRaster=T, labRow = F, labCol = F,
          RowSideColors=r.cols[kclust$cluster[rownames(ave.z.k)]])
dev.off()

pdf(paste0("03.13.2026.CV_all_traj.aligned_genes.",k,".pdf"), width=6, height=9)
r.cols <- colorRampPalette(brewer.pal(12, "Paired"))(k)
heatmap.2(cv.expr[rownames(ave.z.k),], Rowv=F, Colv=F, trace='none', dendrogram='none',
          col=colorRampPalette(c("grey85",brewer.pal(9,"YlGnBu")[2:9]),bias=0.5)(100),
          breaks=seq(from=0,to=15,length.out=101),
          useRaster=T, labRow = F, labCol = F,
          RowSideColors=r.cols[kclust$cluster[rownames(ave.z.k)]])
dev.off()

# gene-wise entropy
expr_array <- simplify2array(expr.filt)
expr_array <- pmax(expr_array, 0)
K <- dim(expr_array)[3]
gene_entropy_pt <- apply(expr_array, c(1,2), function(x) {
  
  p <- x / sum(x)
  
  -sum(p * log(p + 1e-12))
})
gene_entropy_pt <- (gene_entropy_pt / log(K))

pdf(paste0("03.13.2026.Gene_pseudo_entropy_all_traj.aligned_genes.",k,".pdf"), width=6, height=9)
r.cols <- colorRampPalette(brewer.pal(12, "Paired"))(k)
heatmap.2(gene_entropy_pt[rownames(ave.z.k),], Rowv=F, Colv=F, trace='none', dendrogram='none',
          col=colorRampPalette(c("grey85",brewer.pal(9,"YlGnBu")[2:9]),bias=0.5)(100),
          breaks=seq(from=0,to=1,length.out=101),
          useRaster=T, labRow = F, labCol = F,
          RowSideColors=r.cols[kclust$cluster[rownames(ave.z.k)]])
dev.off()

plot(colMeans(gene_entropy_pt[rownames(ave.z.k),]), type='l')

# get number of trajectories
gss <- gs[gs$geneID %in% rownames(ave.z.k),]
t.cnts <- table(gss$geneID)
pdf("Alignment_frequency.pdf", width=5, height=5)
plot(density(t.cnts))
dev.off()
plot(t.cnts[rownames(ave.z.k)], type='h')


##########################################################################
## GO enrichment
##########################################################################

# get pseudotime rank
ranks <- apply(ave.z.k, 1, which.max)

# k-means data
k.genes <- data.frame(kclust=kclust$cluster[rownames(ave.z.k)], 
                      ave.entropy=rowMeans(gene_entropy_pt[rownames(ave.z.k),]),
                      rank=ranks,
                      row.names=rownames(ave.z.k))

# plot per gene entropy for each cluster
pdf("per_gene_entropy.boxplot.pdf", width=5, height=5)
boxplot(ave.entropy~kclust, data=k.genes, outline=F, 
        col=r.cols)
dev.off()

# GO Enrichment for each k-means cluster
k.genes <- readRDS("kmean_gene_data.03.09.2026.rds")
ptag <- read.table("/nfs/turbo/lsa-YOURNAME/YOURNAME_home/Arabidopsis_Protoplast_Dedifferentiation/step5_pseudocell_trajectory/probabilistic_walks/GAM_FDR0.01_dedifferentiation_trajectories.txt", header=T)

# GSEA inputs
gt <- read.delim("/nfs/turbo/lsa-YOURNAME/shared_data/arabidopsis/reference_data/annotations/TAIR10_desc_GO_mapping.txt", header=F)
gmt1 <- gmtPathways("/nfs/turbo/lsa-YOURNAME/shared_data/arabidopsis/reference_data/annotations/TAIR10.GO")
gmt <- lapply(gmt1, function(z){df <- do.call(c, strsplit(z, ",")); df[df %in% ptag$geneID]})
names(gmt) <- names(gmt1)

# estimate cluster metrics
centroids <- rowsum(ave.z.k, kclust$cluster[rownames(ave.z.k)]) / as.vector(table(kclust$cluster))
gene_stat <- t(sapply(1:nrow(ave.z.k), function(i) {
  outs <- unlist(lapply(1:max(k.genes$kclust), function(k){
    centroid <- centroids[k, ]
    -sqrt(sum((ave.z.k[i,] - centroid)^2))  
  }))
  return(outs)
}))
rownames(gene_stat) <- rownames(ave.z.k)
colnames(gene_stat) <- paste0("km_",seq(1:ncol(gene_stat)))
statz <- as.matrix(t(scale(t(gene_stat))))

# iterate over each cluster
go.outs <- lapply(seq(1:ncol(statz)), function(z){
  
  # verbose
  message(" - running GSEA for k-means group = ",z)
  
  # run GSEA
  score <- statz[,z]
  names(score) <- rownames(statz)
  results <- fgsea(pathways = gmt,
                   stats    = score,
                   minSize  = 5,
                   maxSize  = 400,
                   nPermSimple = 10000)
  results <- as.data.frame(results)
  results <- results[order(results$NES, decreasing=T),]
  results$kmeans <- z
  results$leadingEdge <- NULL
  return(results)
})
go.outs <- do.call(rbind, go.outs)

# filter
sig.go <- subset(go.outs, go.outs$padj < 0.05 & go.outs$NES > 0)
table(sig.go$kmeans)

# descriptions
goterm <- gt$V2
names(goterm) <- gt$V1

# reduce
fgseaRes <- go.outs
fgseaRes$goTerm <- goterm[fgseaRes$pathway]
scores <- -log10(fgseaRes$padj)
names(scores) <- fgseaRes$goTerm

# simplify
simMatrix <- calculateSimMatrix(fgseaRes$goTerm,
                                orgdb="org.At.tair.db",
                                ont="BP")

reduced <- reduceSimMatrix(simMatrix,
                           score = scores,
                           orgdb = "org.At.tair.db")

fgsea_reduced <- fgseaRes %>%
  inner_join(reduced, by=c("goTerm"="go"))

fgsea_reduced <- fgsea_reduced[order(fgsea_reduced$NES, decreasing=T),]
fgsea_reduced <- fgsea_reduced[!duplicated(fgsea_reduced[,c("kmeans","parentTerm")]),]

# word cloud results
for(i in 1:8){
  fgseaResKmeans <- subset(fgsea_reduced, fgsea_reduced$kmeans==i)
  
  df <- fgseaResKmeans %>%
    mutate(term = term) %>%
    mutate(term = str_replace_all(term, "_", " "))
  
  words <- df %>%
    unnest_tokens(word, term)
  
  data("stop_words")
  
  extra_stop <- tibble(word = c(
    "process","regulation","positive","negative",
    "activity","response","cell","protein", "binding", 
    "acting", "specific", "cellular", "metabolic",
    "post", "independent", "establishment", "processing",
    "biosynthetic", "complex", "region", "family",
    "acceptor", "donors", "coupled",  "synthesis",
    "substances", "compound", "v0","3", "4", "network",
    "mediated", "system", "type", "subcompartment",
    "structure", "pathway", "domain", "anatomical", 
    "substrate", "ii", "structural", "constituent",
    "molecule", "stimulus", "endogenous","levels",
    "acid"
  ))
  
  words <- words %>%
    anti_join(stop_words) %>%
    anti_join(extra_stop)
  
  words <- words %>%
    mutate(weight = NES)
  
  word_freq <- words %>%
    group_by(word) %>%
    summarise(weight = sum(weight)) %>%
    arrange(desc(weight))
  
  set.seed(1)
  
  pdf(paste0("kmean_word_cloud_",i,".pdf"), width=10, height=10)
  par(mar=c(5,5,5,5))
  wordcloud(words = word_freq$word,
    freq = word_freq$weight,
    min.freq = 1,
    max.words = 50,
    random.order = FALSE,
    main = paste("K-mean = ", i),
    colors = RColorBrewer::brewer.pal(11, "Spectral")[c(1:4,8:11)])  
  dev.off()
}


#========================================================
# Base R Bubble Plot for FGSEA Results by K-means Cluster
#========================================================
fgseaRes <- subset(fgsea_reduced, fgsea_reduced$padj < 0.05 & fgsea_reduced$NES > 0)
fgseaRes$logpadj <- -log10(fgseaRes$padj)
#fgseaRes <- fgseaRes[fgseaRes$padj < 0.01, ]

# top N pathways per k-means cluster
topN <- 50 
fgsea_filtered <- do.call(rbind, lapply(split(fgseaRes, fgseaRes$kmeans), function(cluster_df) {
  cluster_df[order(-cluster_df$logpadj), ][1:min(topN, nrow(cluster_df)), ]
}))

# reorder
pathways <- unique(fgsea_filtered$parentTerm)
clusters <- rev(sort(unique(fgsea_filtered$kmeans)))

fgsea_filtered$x <- match(fgsea_filtered$parentTerm, pathways)
fgsea_filtered$y <- match(fgsea_filtered$kmeans, clusters)

# colors
ncol <- 100
cols <- colorRampPalette(brewer.pal(9,"YlOrRd"))(ncol)
color_index <- cut(fgsea_filtered$logpadj, breaks = ncol, labels = FALSE)
fgsea_filtered$col <- cols[color_index]
fgsea_filtered$col <- ifelse(fgsea_filtered$padj > 0.05, "white", fgsea_filtered$col)
fgsea_filtered$border <- ifelse(fgsea_filtered$padj > 0.05, "white", "black")

# scale size
fgsea_filtered$size <- rescale(abs(fgsea_filtered$NES), c(1, 3))

############################
# plot
#############################
pdf("03.13.2026.GSEA_bubble_plot.pdf", width=22.5, height=6)
par(las=2, mar=c(10,5,4,2))  # rotate x-axis labels
plot(fgsea_filtered$x, fgsea_filtered$y,
     pch=21,
     bg=fgsea_filtered$col,
     cex=fgsea_filtered$size,
     col=fgsea_filtered$border,
     xaxt="n",
     yaxt="n",
     xlab="",
     ylab="K-means cluster",
     main="FGSEA Bubble Plot by K-means Cluster")

axis(1, at=1:length(pathways), labels=pathways, las=2, cex.axis=0.7)
axis(2, at=1:length(clusters), labels=clusters)

abline(h = 1:length(clusters), col = "grey90", lty = 1)  # horizontal grid
abline(v = 1:length(pathways), col = "grey90", lty = 1)  # vertical grid

legend_vals <- seq(min(fgsea_filtered$logpadj), max(fgsea_filtered$logpadj), length.out=5)
legend_cols <- cols[cut(legend_vals, breaks = ncol, labels = FALSE)]

legend("topright",
       legend = round(legend_vals,2),
       fill = legend_cols,
       title = "-log10(padj)",
       cex = 0.8)
dev.off()