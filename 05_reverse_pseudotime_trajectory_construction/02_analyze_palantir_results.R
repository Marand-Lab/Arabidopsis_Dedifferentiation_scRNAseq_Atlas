#################################################
## plot palantir results 
#################################################

# libraries
library(RColorBrewer)
library(viridis)
library(pheatmap)
library(scales)
library(png)
library(vioplot)
library(gtools)
library(reshape2)
library(mgcv)

# functions
celltypeCols <- function(){
  cell.cols <- c("#C6ADD3","#A6CEE1","#A588BE","#7BB3D5","#D9A599","#EC9554","#F48281",
                 "#9E9C65","#F05B5A","#5D9E43","#6B3F98","#6ABD55","#E73232","#4B95A7",
                 "#8962AA","#6A468E","#F89B5F","#94CB72","#A7D48C","#5A4770","#D4C0DD")
  names(cell.cols) <- sort(unique(meta$celltype))
  return(cell.cols)
}
plot_cbd <- function(x1,x2,
                     ylim=c(min(x2),max(x2)),
                     xlim=c(min(x1),max(x1)),
                     xlab="",ylab="",main="",
                     cex=0.5, fit=F, bwf_x=5, bwf_y=5, nbin=300,
                     colP=NULL, rasterize=F){
  
  .normalize <- function(x){
    (x - min(x))/(max(x)-min(x))
  }
  if(is.null(colP)){
    
    colP <- colorRampPalette(c("grey75","grey70","darkorchid4","firebrick3","darkorange",
                               "gold1","yellow"), bias=1.5)(256)
    
  }else{
    colP <- colP(256)
  }
  bww <- (max(x1)-min(x1))/bwf_x
  bwh <- (max(x2)-min(x2))/bwf_y
  df <- data.frame(x1,x2)
  x <- densCols(x1,x2, colramp=colorRampPalette(c("black", "white")),
                nbin=nbin, bandwidth=c(bww, bwh))
  df$dens <- col2rgb(x)[1,] + 1L
  cols <- colP
  df$col <- cols[df$dens]
  df$opa <- .normalize(df$dens)
  
  # if raster
  if(rasterize){
    plot(x2~x1, data=df[order(df$dens),],
         ylim=ylim,xlim=xlim,pch=16,col=alpha(col,1),
         cex=cex,xlab=xlab,ylab=ylab,
         main=main, type="n")
    coords <- par("usr")
    gx <- grconvertX(coords[1:2], "user", "inches")
    gy <- grconvertY(coords[3:4], "user", "inches")
    width <- max(gx) - min(gx)
    height <- max(gy) - min(gy)
    tmp <- tempfile()
    png(tmp, width = width, height = height, units = "in", res = 300, bg = "transparent")
    par(mar=c(0,0,0,0))
    plot.new()
    plot.window(coords[1:2], coords[3:4], mar = c(0,0,0,0), xaxs = "i", yaxs = "i")
    points(x2~x1, data=df[order(df$dens),],
           pch=16,col=alpha(col,1),
           cex=cex)
    dev.off()
    panel <- readPNG(tmp)
    rasterImage(panel, coords[1], coords[3], coords[2], coords[4])
    
  }else{
    plot(x2~x1, data=df[order(df$dens),],
         ylim=ylim,xlim=xlim,pch=16,col=alpha(col,1),
         cex=cex,xlab=xlab,ylab=ylab,
         main=main)
  }
  grid(lty=1)
  
  # fit regression?
  if(fit){
    
    mod <- lm(x2~x1)
    abline(mod, col="firebrick3", lwd=2)
    
  }
  
  cors <- cor(x2,x1)
  mtext(paste0("PCC = ", signif(cors, digits=3)))
}
plotPT <- function(mm, pt, prefix="", 
                   adj.quant=T, 
                   quan=c(0.01,0.99)){
  
  mm$pt <- pt[rownames(mm),]$Pseudotime
  
  if(adj.quant){
    rr <- quantile(mm$pt, quan)
    mm$pt[mm$pt > rr[2]] <- rr[2]
    mm$pt[mm$pt < rr[1]] <- rr[1]
  }
  
  
  #mm$sortorder <- ifelse(!mm$hormone & mm$treatment!="D0", 0, mm$pt)
  mm <- mm[order(mm$pt, decreasing=F),]
  cols <- colorRampPalette(rev(brewer.pal(11, "Spectral")))(100)
  plot(mm$umap1, mm$umap2, 
       pch=16, cex=0.3, 
       axes=F, bty="n", 
       xlab="", ylab="",
       col=cols[cut(mm$pt, breaks=seq(from=min(mm$pt)-1e-8, to=max(mm$pt)+1e-8, length.out=101))], 
       main=prefix)  
  
  
}
plotCellFate <- function(mm, pt, prefix,
                         adj.quant=T, 
                         quan=c(0.99)){
  
  # prep
  cf <- pt[,3:ncol(pt)]
  colnames(cf) <- gsub("\\.","/", colnames(cf))
  cf <- cf[,mixedorder(colnames(cf), decreasing=F)]
  cf <- as.matrix(cf[rownames(mm),])
  
  if(adj.quant){
    rr <- quantile(cf, quan)
    cf[cf > rr[1]] <- rr[1]
  }
  
  # plot parameters
  png(paste0(prefix,".cell_fate_map.png"), width=21, height=10, unit="in", res=300)
  layout(matrix(c(1:21), nrow=3, byrow=T))
  par(mar=c(3,3,1,1))
  
  # plot
  for(i in colnames(cf)){
    
    fate <- cf[,i]
    cols <- colorRampPalette(c("grey75",brewer.pal(9, "YlGnBu")))(100)
    col <- cols[cut(fate, breaks=seq(from=-1e-8, to=max(cf)+1e-8, length.out=101))]
    plot(meta$umap1, meta$umap2, 
         pch=16, cex=0.3, 
         col=col, 
         main=i, 
         xlab="", ylab="", 
         axes=F, bty='n')
  }
  dev.off()
  
}
frobenius_dist <- function(A, B) {
  sqrt(sum((A - B)^2))
}
rel_fro <- function(A, B) {
  sqrt(sum((A - B)^2)) / sqrt(sum(A^2))
}
RV <- function(A, B) {
  num <- sum((A %*% t(A)) * (B %*% t(B)))
  den <- sqrt(sum((A %*% t(A))^2) * sum((B %*% t(B))^2))
  num / den
}


#################################################
# load data
#################################################
meta <- read.table("../diffusion_pseudotime.metadata.all_cells.palantir.11.19.2025.knn30.real_time.cellfate.txt")
res.list <- list.files(pattern="PALANTIR_RESULTS*")
res <- lapply(res.list, function(z){
  readRDS(z)
})
names(res) <- gsub("PALANTIR_RESULTS\\.","",res.list)
names(res) <- gsub("\\.rds","", names(res))
ctcols <- celltypeCols()

#################################################
# plot pseudotime
#################################################
png("palantir_pseudotime_UMAP.png", width=20, height=8, res=300, units="in")
layout(matrix(c(1:length(res)), nrow=2, byrow=T))
for(i in names(res)){
  plotPT(meta, res[[i]], prefix=i)
}
dev.off()


#################################################
# plot cell fate
#################################################
for(i in names(res)){
  plotCellFate(meta, res[[1]], prefix=i, quan=0.999)
}


#################################################
# plot cell fate probabilities by start cell cluster
#################################################
ctcols <- celltypeCols()
names(ctcols) <- gsub("\\/",".",names(ctcols))
new.res <- lapply(1:length(res), function(i){
  id <- names(res)[i]
  c.id <- as.numeric(gsub("cluster_","",id))
  df <- res[[i]][,3:ncol(res[[i]])]
  df <- df[,names(ctcols)]
  cell.ids <- rownames(meta[meta$seurat_clusters==c.id & meta$age!="D0",])
  df.cids <- df[cell.ids,]
  df.long <- melt(as.matrix(df.cids))
  df.long$cl <- id
  return(df.long)
})
new.res <- do.call(rbind, new.res)

pdf("cell_fate_distributions.pdf", width=28, height=12)
layout(matrix(c(1:21), nrow=3, byrow=T))
for(i in names(ctcols)){
  ddf <- subset(new.res, new.res$Var2==i)
  vioplot(value~cl, data=ddf, col=ctcols[i], xlab="", ylab="", main=i, ylim=c(0,1), las=2)
}
dev.off()


#################################################
# get consensus pseudotime
#################################################
pts <- lapply(res, function(z){
  z$Pseudotime
})
pts <- do.call(cbind, pts)
rownames(pts) <- rownames(res[[1]])
con.pt <- rowMeans(pts)
min.pt <- apply(pts, 1, min)
var.pt <- apply(pts, 1, var)
df1 <- data.frame(Pseudotime=con.pt, Pt.var=var.pt, row.names=names(con.pt))
df2 <- data.frame(Pseudotime=min.pt, Pt.var=var.pt, row.names=names(con.pt))
cpt <- df2$Pseudotime
cpt.var <- df2$Pt.var 
names(cpt) <- rownames(df1)
names(cpt.var) <- rownames(df1)
meta$consensus_pseudotime <- cpt[rownames(meta)]
meta$consensus_pseudotime.var <- cpt.var[rownames(meta)]

# plot
png("Consensus_pseudotime_palantir.png", width=7, height=7, res=300, units="in")
plotPT(meta, df1, prefix="Consensus pseudotime", quan=c(0.01, 0.99))
dev.off()

png("Min_dist_pseudotime_palantir.png", width=7, height=7, res=300, units="in")
plotPT(meta, df2, prefix="Minimum pseudotime", quan=c(0.01, 0.99))
dev.off()


#################################################
# compare pseudotime with cytoTRACE
#################################################
pts <- as.data.frame(pts)
pts$consensus <- cpt[rownames(pts)]
pts$cytotrace <- meta[rownames(pts),]$cytotrace
pts$real_time <- meta[rownames(pts),]$real_time
pts$differentiation <- meta[rownames(pts),]$cor.value
pts$diffusion_pt <- meta[rownames(pts),]$diffusion_pt

cors <- cor(pts)
pdf("pseudotime_correlations.pdf", width=6, height=6)
pheatmap(cors, col=colorRampPalette(rev(brewer.pal(9, "RdBu")))(100), 
         breaks=c(seq(from=min(cors)-1e-16,to=0,length.out=50),
                  seq(from=1e-16, to=max(cors)+1e-16,length.out=51)))
dev.off()

# consensus pseudotime and cytotrace
pdf("Cytotrace_consensus_pseudotime.pdf", width=5, height=5)
plot_cbd(meta$cytotrace, meta$consensus_pseudotime, 
         bwf_x=20, bwf_y=20, cex=0.5,
         xlab="Cytotrace differentiation status",
         ylab="Consensus pseudotime",
         rasterize=T)
mod <- gam(consensus_pseudotime~s(cytotrace, bs="cr"), data=meta)
new.dat <- data.frame(cytotrace=seq(from=min(meta$cytotrace),to=max(meta$cytotrace), length.out=100))
preds <- predict(mod, new.dat, se.fit=T)
upr <- preds$fit + (2*preds$se.fit)
lwr <- preds$fit - (2*preds$se.fit)
lines(new.dat$cytotrace, preds$fit, lwd=1.5)
lines(new.dat$cytotrace, lwr, lwd=1.5)
lines(new.dat$cytotrace, upr, lwd=1.5)
dev.off()


#################################################
# get highest cell fate probability
#################################################
names(ctcols) <- gsub("\\/",".",names(ctcols))
calls <- lapply(1:length(res), function(i){
  id <- names(res)[i]
  c.id <- as.numeric(gsub("cluster_","",id))
  df <- res[[i]][,3:ncol(res[[i]])]
  df <- df[,names(ctcols)]
  cell.ids <- rownames(meta[meta$seurat_clusters==c.id & meta$age!="D0",])
  df.cids <- df[cell.ids,]
  outs <- apply(df.cids, 1, function(z){
    type <- names(z)[which.max(z)]
    prob <- max(z)
    data.frame(type=type, prob=prob, cluster=id)
  })
  outs <- do.call(rbind, outs)
  return(outs)
})
calls <- do.call(rbind, calls)
num.calls <- table(calls$type)
calls <- calls[calls$type %in% names(num.calls)[num.calls > 10],]


#################################################
# plot most likely cell type origin (UMAP)
#################################################

# cols
meso <- colorRampPalette(c("grey90",brewer.pal(9, "Greens")[3:9]))(100)
gc <- colorRampPalette(c("grey90",brewer.pal(9, "RdPu")[3:9]))(100)
epi <- colorRampPalette(c("grey90",brewer.pal(9, "Blues")[3:9]))(100)

# scales
#calls$m.col <- meso[cut(calls$prob, breaks=seq(from=-1e-8, to=max(calls$prob[calls$type=="Spongy_Mesophyll"]), length.out=101))]
#calls$g.col <- gc[cut(calls$prob, breaks=seq(from=-1e-8, to=max(calls$prob[calls$type=="Guard_Cell"]), length.out=101))]
#calls$e.col <- epi[cut(calls$prob, breaks=seq(from=-1e-8, to=max(calls$prob[calls$type=="Hydathode"]), length.out=101))]
calls$m.col <- meso[cut(calls$prob, breaks=seq(from=-1e-8, to=1, length.out=101))]
calls$g.col <- gc[cut(calls$prob, breaks=seq(from=-1e-8, to=1, length.out=101))]
calls$e.col <- epi[cut(calls$prob, breaks=seq(from=-1e-8, to=1, length.out=101))]

# new meta
mm <- meta[,c("umap1","umap2")]
tt <- calls$type
pp <- calls$prob
mcol <- calls$m.col
gcol <- calls$g.col
ecol <- calls$e.col
names(tt) <- rownames(calls)
names(pp) <- rownames(calls)
names(mcol) <- rownames(calls)
names(gcol) <- rownames(calls)
names(ecol) <- rownames(calls)
mm$type <- tt[rownames(mm)]
mm$prob <- pp[rownames(mm)]
mm$m.col <- mcol[rownames(mm)]
mm$g.col <- gcol[rownames(mm)]
mm$e.col <- ecol[rownames(mm)]
mm$prob[is.na(mm$prob)] <- 0
mm <- mm[order(mm$prob, decreasing=F),]

# plot
pdf("Celltype_origin_UMAP.pdf", width=8, height=8)
plot(mm$umap1, mm$umap2,
     pch=16,
     cex=0.3,
     axes=F,
     ylab="",xlab="",
     col=ifelse(is.na(mm$type), "grey90",
                ifelse(mm$type=="Spongy_Mesophyll",mm$m.col,
                       ifelse(mm$type=="Guard_Cell", mm$g.col,
                              ifelse(mm$type=="Hydathode",mm$e.col,"grey90")))))
dev.off()


#################################################
# plot most likely cell origin (donut)
#################################################
new.cols <- c(meso[100],gc[100],epi[100])
names(new.cols) <- c("Spongy_Mesophyll","Guard_Cell","Hydathode")
freq <- prop.table(table(calls$type))
pdf("pie_most_likely_celltype.pdf", width=5, height=5)
pie(freq, col=new.cols[names(freq)])
dev.off()


#################################################
# update metadata
#################################################
mm <- mm[rownames(meta),]
meta$cellfate_prob <- mm$prob
meta$cellfate_type <- mm$type
write.table(meta, file="../diffusion_pseudotime.metadata.all_cells.palantir.11.19.2025.knn30.real_time.cellfate.txt",quote=F, row.names=T, col.names=T, sep="\t")











