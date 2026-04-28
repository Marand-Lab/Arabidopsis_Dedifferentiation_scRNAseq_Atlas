## Analyze DEG per window ##

# R version R/4.4.0

# load libraries
library(RColorBrewer)
library(gtools)
library(mgcv)
library(Seurat)
library(scales)
library(pheatmap)
library(reshape2)
library(viridis)
library(gplots)

##########################################################################
# Initial data loading and processing 
##########################################################################

# load data
#files <- list.files(pattern="*rt_grid.rds")
files <- list.files(pattern="*rpt_grid.rds")
a <- lapply(files, function(z){
  readRDS(z)
})
names(a) <- gsub("\\.nDEGs_rpt_grid\\.rds","",files)
#names(a) <- gsub("\\.nDEGs_rt_grid\\.rds","",files)

# process
delta <- lapply(names(a), function(z){
  a[[z]]$Delta
})
ndeg <- lapply(names(a), function(z){
  data.frame(grid=a[[z]]$grid,
             ndegs=a[[z]]$DEG_counts,
             traj=z)
  
})
names(delta) <- names(a)
names(ndeg) <- names(a)
all <- do.call(rbind, ndeg)


##########################################################################
# plot trajectory DEGs
##########################################################################

# plot
pdf("Number_DEGs_trajectory.rt.pdf", width=5, height=5)
smcols <- colorRampPalette(brewer.pal(9, "Greens")[3:9])(10)
names(smcols) <- mixedsort(names(a)[grepl("Spongy",names(a))])
ocols <- c("dodgerblue4","deeppink4")
names(ocols) <- c("cluster_14.Hydathode","cluster_18.Guard_Cell")
cols <- c(smcols, ocols)
for(i in unique(all$traj)){
  df <- subset(all, all$traj==i)
  if(i == all$traj[1]){
    plot(df$grid, df$ndegs, type="l",col=cols[i], ylim=range(all$ndegs),
         xlab="% of trajectory",
         ylab="# of DEGs vs. T0")
  }else{
    lines(df$grid,df$ndegs, col=cols[i])
  }  
}
grid(lty=1)
#legend("topleft", legend=names(cols), fill=cols)
dev.off()

# estimate derivatives for rate of change
fd <- lapply(names(a), function(z){
  df <- ndeg[[z]]
  t <- df$grid
  N <- df$ndegs
  fit <- gam(N ~ s(t, k = 6))
  deriv <- derivatives(fit, select = "s(t)", n = 100)
  deriv$traj <- z
  return(deriv)
})
names(fd) <- names(a)
fdd <- do.call(rbind, fd)

# plot
ids <- mixedsort(names(a))
ids <- ids[c(1:7,9,12,10,8,11)]
pdf("All_dDEG_drt.pdf", width=24, height=8)
layout(matrix(c(1:12), nrow=2, byrow=T))
for(i in ids){
  df <- subset(fdd, fdd$traj==i)
  #if(i == all$traj[1]){
    plot(df$t, df$.derivative, type="l",col=cols[i], ylim=range(fdd$.derivative),
         xlab="% of trajectory",
         ylab="d(DEG)/dt")
    lines(df$t, df$.lower_ci, col=cols[i], lwd=0.5)
    lines(df$t, df$.upper_ci, col=cols[i], lwd=0.5)
  #}else{
  #  lines(df$t,df$.derivative, col=cols[i])
  #  lines(df$t, df$.lower_ci, col=cols[z], lwd=0.5)
  #  lines(df$t, df$.upper_ci, col=cols[z], lwd=0.5)
  #}  
    grid(lty=1)
}
dev.off()


##########################################################################
## Add gene IDs to delta map
##########################################################################

# load reverse pseudotime data
dir <- "/nfs/turbo/lsa-YOURNAME/YOURNAME_home/Arabidopsis_Protoplast_Dedifferentiation/step5_pseudocell_trajectory/probabilistic_walks/"
pw <- list.files(path=dir, pattern="PROB_WALKS*")
pwd <- lapply(pw, function(z){
  unique(unlist(readRDS(paste0(dir,z))$paths))
})
names(pwd) <- gsub("PROB_WALKS_","",pw)
names(pwd) <- gsub("cluster\\.","cluster_",names(pwd))
names(pwd) <- gsub("\\.rds","",names(pwd))

# meta data
m <- read.table("/nfs/turbo/lsa-YOURNAME/YOURNAME_home/Arabidopsis_Protoplast_Dedifferentiation/step5_pseudocell_trajectory/diffusion_pseudotime.metadata.all_cells.palantir.11.19.2025.knn30.real_time.cellfate.txt")
pt <- m$consensus_pseudotime
names(pt) <- rownames(m)

# seurat object
obj <- readRDS("/nfs/turbo/lsa-YOURNAME/YOURNAME_home/Arabidopsis_Protoplast_Dedifferentiation/step5_pseudocell_trajectory/step1_palantir_obj_100DC_harmony_knn.15.10.21.2025.knn30.rds")
expr <- obj@assays$RNA$data

# select cells
for(z in names(pwd)){
  cids <- pwd[[z]]
  tpt <- rescale(1-pt[cids], c(0,1)) # invert pseudotime and rescale
  tpt <- sort(tpt, decreasing=F)
  texpr <- expr[,names(tpt)]
  texpr <- texpr[Matrix::rowSums(texpr>0)>2,] 
  rownames(delta[[z]]) <- rownames(texpr)
}

# plot delta examples
cl6 <- delta[["cluster_6.Spongy_Mesophyll"]]
cl7 <- delta[["cluster_7.Spongy_Mesophyll"]]
fc6 <- rownames(cl6[rowSums(abs(cl6)>0.25)>0,])
fc7 <- rownames(cl7[rowSums(abs(cl7)>0.25)>0,])
deg <- union(fc6, fc7)
shared <- intersect(deg,rownames(cl6))
shared <- intersect(shared,rownames(cl7))
cl6 <- cl6[shared,]
cl7 <- cl7[shared,]
ave <- (cl6+cl7)/2
row.o <- hclust(dist(ave))$order
#km <- kmeans(ave, center=9)
cl6 <- cl6[row.o,]
cl7 <- cl7[rownames(cl6),]
#cl6 <- cl6[names(sort(km$cluster, decreasing=F)),]
#cl7 <- cl7[rownames(cl6),]

pdf("cl6_delta.pdf", width=6, height=6)
heatmap.2(cl6, trace='n',dendrogram='n',
          Rowv=F, Colv=F,
          col=colorRampPalette(rev(brewer.pal(9, "RdBu")))(100),
          useRaster=T,
          labRow=F,
          labCol=F,
          breaks=seq(from=-2,to=2,length.out=101))
dev.off()

pdf("cl7_delta.pdf", width=6, height=6)
heatmap.2(cl7, trace='n',dendrogram='n',
          Rowv=F, Colv=F,
          col=colorRampPalette(rev(brewer.pal(9, "RdBu")))(100),
          useRaster=T,
          labRow=F,
          labCol=F,
          breaks=seq(from=-2,to=2,length.out=101))
dev.off()

##########################################################################
## Compare signs
##########################################################################
sm <- names(cols)[1:10]
signs <- lapply(seq(1:(length(sm)-1)), function(z){
  t1 <- delta[[sm[z]]][,2:100]
  colnames(t1) <- paste0("pt",seq(1:ncol(t1)))
  t1id <- sm[z]
  outs <- lapply(seq(from=(z+1), to=length(sm)), function(x){
    t2 <- delta[[sm[x]]][,2:100]
    colnames(t2) <- paste0("pt",seq(1:ncol(t2)))
    shared <- intersect(rownames(t1), rownames(t2))
    oo <- unlist(lapply(seq(1:ncol(t2)), function(y){
      mean(sign(t1[shared,y])==sign(t2[shared,y]))
    }))
    return(oo)
  })
  outs <- do.call(cbind, outs)
  colnames(outs) <- paste0(t1id,":",sm[seq(from=(z+1), to=length(sm))])
  return(outs)
})
signs <- do.call(cbind, signs)
signs <- t(signs)
zsign <- as.matrix(t(scale(t(signs))))

pdf("sign_heatmap.pdf", width=6, height=6)
pheatmap(zsign, cluster_rows=T, cluster_cols=F, col=colorRampPalette(brewer.pal(9, "Greys"))(100))
dev.off()

aves <- colMeans(zsign)
sds <- apply(zsign, 2, var)

pdf("metaplot_pairwise_scaled_sign.pdf", width=6, height=5)
plot(aves, type="l", ylim=range(c(aves+sds, aves-sds)))
lines(aves+sds, col="grey75")
lines(aves-sds, col="grey75")
grid(lty=1)
dev.off()

# add pairwise info
pwc <- gsub("\\.Spongy_Mesophyll","",rownames(zsign)[hclust(dist(zsign))$order])
info <- as.data.frame(do.call(rbind, strsplit(pwc,":")))
i.mat <- matrix(NA, nrow=nrow(info), ncol=length(cols)-2)
i.mat[is.na(i.mat)] <- 0
ccc <- cols[1:10]
names(ccc) <- gsub("\\.Spongy_Mesophyll","",names(ccc))
colnames(i.mat) <- gsub("\\.Spongy_Mesophyll","",names(cols)[1:10])
for(i in 1:nrow(i.mat)){
  id1 <- info$V1[i]
  id2 <- info$V2[i]
  i.mat[i,id1] <- which(names(ccc)==id1)
  i.mat[i,id2] <- which(names(ccc)==id2)
}
pdf("pairwise_sign_legend.pdf", width=4, height=5)
pheatmap(i.mat, col=c("white",ccc), cluster_rows=F, cluster_cols=F)
dev.off()


##########################################################################
## Cluster trajectory log2fc 
##########################################################################

# top markers
mm <- read.table("top_arabidopsis_marker_genes.txt", header=T)
conctm <- readRDS("/nfs/turbo/lsa-YOURNAME/YOURNAME_home/Arabidopsis_Protoplast_Dedifferentiation/step5_pseudocell_trajectory/misc/Consensus_marker_genes.Arabidopsis.rds")
ctsIDs <- c("Spongy_mesophyll","Palisade_mesophyll","Mesophyll","Leaf_guard_cell","Hydathodes", "Guard_cell")
conctm <- conctm[conctm$clusterName %in% ctsIDs,]
conf <- subset(conctm, conctm$adjScore > 0.5)
conf$celltype <- ifelse(conf$clusterName=="Leaf_guard_cell", "Guard_cell",
                        ifelse(conf$clusterName=="Palisade_mesophyll", "Mesophyll",
                               ifelse(conf$clusterName=="Spongy_mesophyll", "Mesophyll", conf$clusterName)))

# estimate linear beta
betas <- lapply(names(delta), function(z){
  
  # verbose
  message("Getting summary stats for log2fc patterns of trajectory = ",z)
  
  # log2fc
  df <- delta[[z]]
  
  # per gene model
  outs <- lapply(seq(1:nrow(df)), function(x){
    if((x %% 1000)==0){message("   - iterated over ",x, " records...")}
    dd <- data.frame(logfc=as.numeric(df[x,]),
                     pt=seq(1:ncol(df)))
    mod <- summary(lm(logfc~pt, data=dd))
    return(data.frame(trajID=z,
                      geneID=rownames(df)[x],
                      beta=mod$coefficients[2,1],
                      betaSe=mod$coefficients[2,2],
                      zscore=mod$coefficients[2,3],
                      pval=mod$coefficients[2,4]))
  })
  outs <- do.call(rbind, outs)
  return(outs)
})
betas <- do.call(rbind, betas)
mat <- dcast(geneID~trajID, value.var="beta", data=betas)
rownames(mat) <- mat$geneID
mat$geneID <- NULL
mat <- as.matrix(mat)
mat[is.na(mat)] <- 0

mat.m <- data.frame(row.names=rownames(mat),
                    celltype=ifelse(rownames(mat) %in% conf$gene, conf$celltype, "Non-marker"))
thresh <- quantile(as.numeric(mat), c(0.01, 0.99))
pheatmap(mat, 
         annotation_row=mat.m, 
         show_rownames=F,
         col=colorRampPalette(rev(brewer.pal(9, "RdBu")))(100),
         breaks=seq(from=thresh[1],to=thresh[2],length.out=101))

# iterate over cell types
mat.cts <- c("Mesophyll","Mesophyll","Hydathodes",
             "Mesophyll","Guard_cell","Mesophyll",
             "Mesophyll","Mesophyll","Mesophyll",
             "Mesophyll","Mesophyll","Mesophyll")

m$treatment <- factor(m$hormone)
outs <- lapply(seq(1:ncol(mat)), function(z){
  traj <- colnames(mat)[z]
  cells <- pwd[[traj]]
  treat <- table(m[cells,]$treatment)
  exprs <- mat[,z]
  ctid <- mat.cts[z]
  conf.cts <- subset(mat.m, mat.m$celltype==ctid)
  cts.exprs <- exprs[names(exprs) %in% rownames(conf.cts)]
  ncts.exprs <- exprs[!names(exprs) %in% names(cts.exprs)]
  
  # permute
  perms <- unlist(lapply(seq(1:1000), function(x){
    pp <- sample(ncts.exprs, length(cts.exprs))
    mean(pp)
  }))
  obs <- mean(cts.exprs)
  z.score <- (obs - mean(perms))/sd(perms)
  e.pal <- (1+sum(abs(perms) > abs(obs)))/(length(perms)+1)
  return(data.frame(trajID=colnames(mat)[z],
                    celltypeMarkers=mat.cts[z],
                    obsX=obs,
                    nullX=mean(perms),
                    nullSD=sd(perms),
                    Zscore=z.score,
                    ePval=e.pal,
                    hormone=treat[2],
                    nonhormone=treat[1]))
})
outs <- do.call(rbind, outs)
rownames(outs) <- outs$trajID

# plot
outs <- outs[names(cols),]
pdf("marker_dynamics.rt.pdf", width=5, height=5)
plot(seq(1:nrow(outs)), outs$obsX, pch=16, cex=abs(outs$Zscore), col=cols)#, ylim=c(-0.0025, 0.00125))
points(seq(1:nrow(outs)), outs$nullX, pch=16, cex=1, col="black")
segments(x0=seq(1:nrow(outs)),
         y0=outs$nullX-outs$nullSD,
         x1=seq(1:nrow(outs)),
         y1=outs$nullX+outs$nullSD,
         col="grey75")
grid(lty=1)
dev.off()


##########################################################################
## compare delta at each timepoint between mesophyll lineages
##########################################################################
sm <- names(cols)[1:10]
cors <- lapply(seq(1:(length(sm)-1)), function(z){
  t1 <- delta[[sm[z]]][,2:100]
  colnames(t1) <- paste0("pt",seq(1:ncol(t1)))
  t1id <- sm[z]
  outs <- lapply(seq(from=(z+1), to=length(sm)), function(x){
    t2 <- delta[[sm[x]]][,2:100]
    colnames(t2) <- paste0("pt",seq(1:ncol(t2)))
    shared <- intersect(rownames(t1), rownames(t2))
    diag(cor(t1[shared,],t2[shared,]))
  })
  outs <- do.call(cbind, outs)
  colnames(outs) <- paste0(t1id,":",sm[seq(from=(z+1), to=length(sm))])
  return(outs)
})
cors <- do.call(cbind, cors)
cors <- t(cors)
n.cors <- as.matrix(t(scale(t(cors))))
cl <- hclust(dist(n.cors))$order
n.cors <- n.cors[cl,]
saveRDS(rownames(n.cors), file="pwc_rowids.rds")
saveRDS(n.cors, file="pwc_per_timepoint.rds")

# individual pairwise results
pdf("pairwise_normalized_PCC.pdf", width=6, height=5)
pheatmap(n.cors, cluster_rows=F, cluster_cols=F,
         show_colnames=F,
         show_rownames=F,
         col=colorRampPalette(brewer.pal(9, "Greys"))(100))
dev.off()

# add pairwise info
pwc <- gsub("\\.Spongy_Mesophyll","",rownames(n.cors))
info <- as.data.frame(do.call(rbind, strsplit(pwc,":")))
i.mat <- matrix(NA, nrow=nrow(info), ncol=length(cols)-2)
i.mat[is.na(i.mat)] <- 0
ccc <- cols[1:10]
names(ccc) <- gsub("\\.Spongy_Mesophyll","",names(ccc))
colnames(i.mat) <- gsub("\\.Spongy_Mesophyll","",names(cols)[1:10])
for(i in 1:nrow(i.mat)){
  id1 <- info$V1[i]
  id2 <- info$V2[i]
  i.mat[i,id1] <- which(names(ccc)==id1)
  i.mat[i,id2] <- which(names(ccc)==id2)
}
pdf("pairwise_sPCC_legend.pdf", width=4, height=5)
pheatmap(i.mat, col=c("white",ccc), cluster_rows=F, cluster_cols=F)
dev.off()
  
  
aves <- colMeans(n.cors)
sds <- apply(n.cors, 2, var)

pdf("metaplot_pairwise_scaled_pcc.pdf", width=6, height=5)
plot(aves, type="l", ylim=range(c(aves+sds, aves-sds)))
lines(aves+sds, col="grey75")
lines(aves-sds, col="grey75")
grid(lty=1)
dev.off()


##########################################################################
## compare hormone treatments across pseudotime block
##########################################################################

# 