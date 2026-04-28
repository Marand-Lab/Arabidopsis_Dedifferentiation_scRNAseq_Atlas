## Analyze miloR abundance results ##

# load libraries
library(SingleCellExperiment)
library(miloR)
library(RColorBrewer)
library(ggplot2)
library(scales)
library(RColorBrewer)
library(MASS)
library(mgcv)
library(png)
library(dplyr)

# functions
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
    
    vals <- data.frame(x1=seq(from=min(x1),to=max(x1), length.out=100))
    mod <- gam(x2~s(x1, bs="cr"))
    preds <- predict(mod, vals, se.fit=T)
    lines(vals$x1, preds$fit, col="firebrick4", lwd=2)
    lines(vals$x1, preds$fit+(2*preds$se.fit), col="firebrick3", lwd=2)
    lines(vals$x1, preds$fit-(2*preds$se.fit), col="firebrick3", lwd=2)
    
  }
  
  cors <- cor(x2,x1)
  mtext(paste0("PCC = ", signif(cors, digits=3)))
}

# load data
res <- readRDS("milo_abundance_results.11.7.25.rds")
obj <- readRDS("milo_abundance_object.11.7.25.rds")
meta <- read.table("diffusion_pseudotime.metadata.all_cells.palantir_pt_entrop.11.06.2025.knn30.inferred_age.txt")

# get graph
obj <- buildNhoodGraph(obj)

# significant
sig <- subset(res, res$SpatialFDR < 0.05)

# cols
minn <- "#9b9635"
maxx <- "#893061"

# new cols
#cols1 <- c("grey90", "grey80", "#fff1e9", "#fbae92", "#fa6749", "#d42022", "#930a13","#5e0000")
#cols2 <- rev(brewer.pal(8, "Blues"))
#cols <- rev(colorRampPalette(c(cols2,cols1))(100))
#res$cols <- cols[cut(res$logFC, breaks=seq(from=min(res$logFC)-0.1,to=max(res$logFC)+0.1, length.out=101))]

# plot
nh_graph_pl <- plotNhoodGraphDA(obj, res, layout="UMAP",alpha=0.05, res_column="logFC") 
nh_graph_pl
ggsave(filename="DA_nhood_graph_milo.new_cols.pdf", units="in", device="pdf", width=6, height=4)

# plot volcano
cols <- colorRampPalette(c(minn, "grey70", maxx))(100)
ccs <- cols[cut(res$logFC, breaks=c(seq(from=min(res$logFC)-1e-8, to=0, length.out=50),seq(from=1e-8, to=max(res$logFC)+1e-8, length.out=51)))]
pdf("volcano_abundance.pdf", width=5, height=5)
plot(res$logFC, -log10(res$SpatialFDR), pch=16, cex=rescale(res$logCPM, c(0.2, 2)),
     xlim=c(-8.1, 8.1),
     col=alpha(ifelse(res$SpatialFDR < 0.05 & abs(res$logFC) > 2, ccs, "grey75"), 0.2))
grid(lty=1)
abline(h=-log10(0.05), lty=2)
dev.off()

# extract cells in each neighborhood
nhoods_sce = nhoods(obj)
ave.age <- lapply(seq(1:ncol(nhoods_sce)), function(z){
  if((z %% 1000)==0){message("iterated over ",z," neighborhoods...")}
  ids <- rownames(nhoods_sce)[nhoods_sce[,z] > 0]
  ave <- mean(meta[ids,]$inferred_age)
  med <- median(meta[ids,]$inferred_age)
  return(data.frame(ave_age=ave, median_age=med))
})
ave.age <- do.call(rbind, ave.age)
ave.age$Nhood <- seq(1:ncol(nhoods_sce))
res <- res[order(res$Nhood, decreasing=F),]
res$ave.age <- ave.age$ave_age
res$med.age <- ave.age$median_age

# plot
pdf("age_vs_logfc.pdf", width=5,4)
plot_cbd(res$ave.age, res$logFC, 
         fit=T, bwf_x=10, bwf_y=10,
         cex=1,
         colP=colorRampPalette(c("white", "grey75", brewer.pal(9, "YlGnBu")), bias=1.5),
         rasterize=T)
dev.off()

# get counts
res$age_bin <- ntile(res$ave.age, 20)
res$pos <- ifelse(res$logFC > 0, "pos", "neg")
prop <- table(res$pos, res$age_bin)
prop[1,] <- prop[1,]*-1
pdf("num_cell_states.pdf", width=5, height=4)
barplot(prop, beside=T)
dev.off()
barplot(prop, beside=T)
