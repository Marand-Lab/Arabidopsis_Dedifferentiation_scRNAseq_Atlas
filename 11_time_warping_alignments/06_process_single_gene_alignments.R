################################################# 
# collect aligned genes 
#################################################

# load libraries
library(pheatmap)
library(RColorBrewer)
library(MASS)
library(Matrix)
library(wesanderson)
library(reshape2)


# load functions
getDists <- function(x){
  gbl_dist <- unlist(lapply(names(x), function(z){
    x[[z]]$normalizedDistance
  }))
  return(data.frame(geneID=names(x),nd=gbl_dist))
}

# load data
galn.files <- list.files(pattern="*.cellAlign_single_gene_alignments.rds")
perm.files <- list.files(pattern="*.cellAlign_single_gene_alignments.PERMUTED.rds")
sig.genes <- read.table("../probabilistic_walks/GAM_FDR0.01_dedifferentiation_trajectories.txt", header=T)

# iterate over pair-wise alignments
ndists <- lapply(galn.files, function(y){
  message("reading ",y)
  a <- readRDS(y)
  message("getting normalized distances")
  getDists(a)
})
permdists <- lapply(perm.files, function(y){
  a <- readRDS(y)
  data.frame(geneID=names(a), nd=unlist(a))
})

# names
names(ndists) <- gsub("\\.cellAlign_single_gene_alignments\\.rds","",galn.files)
names(permdists) <- gsub("\\.cellAlign_single_gene_alignments\\.PERMUTED\\.rds","",perm.files)

# save distances
saveRDS(ndists, file="gene_level_normalized_distances.pairwise_trajectories.rds")
saveRDS(permdists, file="gene_level_normalized_distances.pairwise_trajectories.PERMUTED.rds")

# empirical false discovery filtering
eFDR <- 0.05
sig.aln <- lapply(names(ndists), function(z){
  threshold <- quantile(permdists[[z]]$nd, eFDR)
  df <- subset(ndists[[z]], ndists[[z]]$nd <= threshold)
  df$comparison <- z
  message(z, " | cutoff = ", threshold, " | number of genes = ", nrow(df))
  return(df)
})
sig.aln <- do.call(rbind, sig.aln)

# parse conserved and non-conserved alignments
for(i in galn.files){

  # load
  message("Processing sample = ", i)
  a <- readRDS(i)
  a <- a[names(a) %in% unique(sig.genes$geneID)]

  # process
  id <- gsub("\\.cellAlign_single_gene_alignments\\.rds","",i)
  df <- subset(sig.aln, sig.aln$comparison==id)
  aa <- a[df$geneID]
  bb <- a[!names(a) %in% df$geneID]

  # iterate
  con <- lapply(aa, function(z){
    xvals <- z$align[[1]]$index1
    yvals <- z$align[[1]]$index2
    data.frame(xvals=xvals, yvals=yvals)
  })
  con <- do.call(rbind, con)
  div <- lapply(bb, function(z){
    xvals <- z$align[[1]]$index1
    yvals <- z$align[[1]]$index2
    data.frame(xvals=xvals, yvals=yvals)
  })
  div <- do.call(rbind, div)

  # return
  saveRDS(con, file=paste0(id,".Aligned_single_genes.rds"))
  saveRDS(div, file=paste0(id,".Diverged_single_genes.rds"))
}


#################################################
# load parsed alignments
#################################################

# load parsed alignments
message("load data")
aln.files <- list.files(pattern="*.Aligned_single_genes.rds")
div.files <- list.files(pattern="*.Diverged_single_genes.rds")
num.genes <- c()
keep.names <- c()
aln <- lapply(aln.files, function(z){
  
  # check that clusters and somatic identities are distinct
  id <- gsub("\\.Aligned_single_genes\\.rds","",z)
  tj <- unlist(strsplit(id, "-"))
  tj1 <- unlist(strsplit(tj[1],"\\."))
  tj2 <- unlist(strsplit(tj[2],"\\."))
  
  if(tj1[1] == tj2[1] | tj1[2] == tj2[2]){
    return(NULL)
  }
  
  message("loading file: ",z)
  keep.names <<- c(keep.names, z)
  df <- readRDS(z)
  df2 <- df
  df2$xvals <- df$yvals
  df2$yvals <- df$xvals
  df <- rbind(df, df2)
  df$id <- paste(df$xvals,df$yvals, sep="_")
  counts <- table(df$id)
  ddf <- as.data.frame(do.call(rbind, strsplit(names(counts),"_")))
  ddf$V3 <- as.numeric(counts)
  ddf$V1 <- as.numeric(ddf$V1)
  ddf$V2 <- as.numeric(ddf$V2)
  mat <- sparseMatrix(i=ddf$V1,j=ddf$V2,x=ddf$V3,dimnames=list(seq(1:300),seq(1:300)))
  df$geneID <- do.call(rbind, strsplit(rownames(df),"\\."))[,1]
  num.genes <<- c(num.genes, length(unique(df$geneID)))
  df$geneID <- NULL
  return(mat)
})
aln.norm <- Reduce("+", Filter(Negate(is.null), aln)) / sum(num.genes)

# iterate over diverged sites
div.names <- gsub("Aligned_single","Diverged_single", keep.names)
names(num.genes) <- div.names
div <- lapply(div.files, function(z){
  
  #check that clusters and somatic identities are distinct
  id <- gsub("\\.Diverged_single_genes\\.rds","",z)
  tj <- unlist(strsplit(id, "-"))
  tj1 <- unlist(strsplit(tj[1],"\\."))
  tj2 <- unlist(strsplit(tj[2],"\\."))
  
  if(tj1[1] == tj2[1] | tj1[2] == tj2[2]){
    return(NULL)
  }
  
  message("loading file: ",z)
  df <- readRDS(z)
  num.gene.keep <- num.genes[z]
  df$geneID <- do.call(rbind, strsplit(rownames(df),"\\."))[,1]
  df$n.gene <- as.numeric(factor(df$geneID, levels=sample(unique(df$geneID))))
  df <- subset(df, df$n.gene <= num.gene.keep)
  df$geneID <- NULL
  df$n.gene <- NULL
  df2 <- df
  df2$xvals <- df$yvals
  df2$yvals <- df$xvals
  df <- rbind(df, df2)
  df$id <- paste(df$xvals,df$yvals, sep="_")
  counts <- table(df$id)
  ddf <- as.data.frame(do.call(rbind, strsplit(names(counts),"_")))
  ddf$V3 <- as.numeric(counts)
  ddf$V1 <- as.numeric(ddf$V1)
  ddf$V2 <- as.numeric(ddf$V2)
  mat <- sparseMatrix(i=ddf$V1,j=ddf$V2,x=ddf$V3,dimnames=list(seq(1:300),seq(1:300)))
  return(mat)
})
div.norm <- Reduce("+", Filter(Negate(is.null), div)) / sum(num.genes)

# estimate conserved density
#message("esimate density aligned...")
#den.aln <- kde2d(aln$xvals, aln$yvals, n=300)
#saveRDS(den.aln, file="conserved_density.rds")
#rm(aln)
#gc()

# color palettes
cols <- colorRampPalette(brewer.pal(9, "Greys"))(100)
cols2 <- colorRampPalette(c("grey85", "#FEF7BF", "#FECF67", "#D05427", "#1E2556"))(100)
cols3 <- wes_palette("Zissou1", 100, type="continuous")
norm <- (aln.norm+1e-4)/(div.norm+1e-4)

# plot conserved
pdf("Conserved_over_diverged_alignments.unique_somatic_stem_cell_traj.pdf", width=5.5, height=5)
pheatmap(log2(as.matrix(norm)), 
         cluster_rows=F, cluster_cols=F, 
         show_rownames=F, show_colnames=F,
         col=cols3, breaks=seq(from=-4, to=4, length.out=101))#breaks=c(seq(from=-4, to=0, length.out=51),seq(from=1e-8, to=4, length.out=50)))
dev.off()
#rm(den.aln)


# investigate aligned genes
genescores <- lapply(aln.files, function(z){
  
  # check that clusters and somatic identities are distinct
  id <- gsub("\\.Aligned_single_genes\\.rds","",z)
  tj <- unlist(strsplit(id, "-"))
  tj1 <- unlist(strsplit(tj[1],"\\."))
  tj2 <- unlist(strsplit(tj[2],"\\."))
  
  if(tj1[1] == tj2[1] | tj1[2] == tj2[2]){
    return(NULL)
  }
  
  # verbose
  message("loading file: ",z)
  df <- readRDS(z)
  
  # process
  df$geneID <- do.call(rbind, strsplit(rownames(df),"\\."))[,1]
  
  # gene distance from diagonal
  genes <- unique(df$geneID)
  outs <- lapply(genes, function(x){
    ddf <- subset(df, df$geneID==x)
    ddf$residual <- abs(ddf$xvals - ddf$yvals)/sqrt(2)
    ave.res1 <- mean(ddf$residual[ddf$xvals <= 100])
    ave.res2 <- mean(ddf$residual[ddf$xvals > 100 & ddf$xvals <= 200])
    ave.res3 <- mean(ddf$residual[ddf$xvals > 200])
    return(data.frame(geneID=x, trajID=id, early=ave.res1, middle=ave.res2, late=ave.res3))
  })
  outs <- do.call(rbind, outs)
  return(outs)
})
genescores <- do.call(rbind, genescores)
saveRDS(genescores, file="traj_single_gene_alignments_classified.rds")

# load expression info
res <- read.table("/nfs/turbo/lsa-YOURNAME/YOURNAME_home/Arabidopsis_Protoplast_Dedifferentiation/step5_pseudocell_trajectory/probabilistic_walks/GAM_FDR0.01_dedifferentiation_trajectories.txt", header=T)
gs <- melt(genescores[,2:5])

pdf("residual_block.pdf", width=5, height=5)
boxplot(gs$value~gs$variable, las=2)
dev.off()

eaves <- aggregate(early~geneID, data=genescores, FUN=mean)
maves <- aggregate(middle~geneID, data=genescores, FUN=mean)
laves <- aggregate(late~geneID, data=genescores, FUN=mean)
counts <- table(genescores$geneID)
all <- data.frame(geneID=eaves$geneID,
                  early=eaves$early,
                  middle=maves$middle,
                  late=laves$late,
                  ntraj=as.numeric(counts[eaves$geneID]))
z <- as.matrix(t(scale(t(all[,c(2:4)]))))
z[is.na(z)] <- 0
pdf("alignment_heatmap.pdf", width=5, height=6.5)
pheatmap(z, col=colorRampPalette(brewer.pal(9, "RdBu"))(100),
         show_rownames=F)
dev.off()
mall <- melt(all[,2:5], id.vars=c("ntraj"))










