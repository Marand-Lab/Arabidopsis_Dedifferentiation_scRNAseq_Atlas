## clean cells ##

# load args
args <- commandArgs(T)
if(length(args) != 3){stop("Rscript filter_nuclei.R <metadata> <sparse.rds> <prefix>")}
mm <- as.character(args[1])
ss <- as.character(args[2])
prefix <- as.character(args[3])

# load libraries
library(Matrix)
library(MASS)
library(viridis)
library(RColorBrewer)

# load functions
convert2Matrix <- function(x){
  x$V1 <- as.factor(x$V1)
  x$V2 <- as.factor(x$V2)
    x <- sparseMatrix(i=as.numeric(x$V1),
                      j=as.numeric(x$V2),
                      x=as.numeric(x$V3),
                      dimnames=list(levels(x$V1),levels(x$V2)))
    return(x)
}
plotDensity <- function(x, column="pMt", main=""){
    
    # get densities
    den <- kde2d(log10(x$total), x[,column], n=300)
    
    # plot
    image(den, col=colorRampPalette(c("white","grey75",rev(magma(18))))(100), useRaster=T, 
          bty="n", main=main, ylim=c(0,1))
    box()
}
filterCells <- function(x, column="pMt", threshold=0, direction="greater", hard=NULL){
    
    # choose method
    if(is.null(hard)){

        # estimate z-score
        x$zscore <- (x[,column]-mean(x[,column], na.rm=T))/sd(x[,column],na.rm=T)
    
        # greater than
        if(direction=="greater"){
            xx <- subset(x, x$zscore >= threshold)
            xx$zscore <- NULL
        }else{
            xx <- subset(x, x$zscore <= threshold)
            xx$zscore <- NULL
        }
        return(xx)

    }else{
	if(direction=="greater"){
	    xx <- subset(x, x[,column] > hard)
	}else{
	    xx <- subset(x, x[,column] < hard)
	}
    }
    
}

# load data
message(" - loading data ...")
meta <- read.table(mm)
all <- readRDS(ss)

# convert to sparseMatrix
message(" - filtering barcodes ...")
ids <- intersect(rownames(meta), colnames(all))
all <- all[,ids]
meta <- meta[ids,]

# count number of genes per cell
meta$n.genes <- Matrix::colSums(all > 0)

# count number of transcripts per cell
meta$n.trx <- Matrix::colSums(all)

# filter cells
meta <- subset(meta, meta$total > 1000 & meta$n.genes >= 100 & meta$n.trx >= 700)
message(" - identified ",nrow(meta), " potential cells ...")

# prop mito, chloro, genic
meta$pMt <- meta$Mt/meta$total
meta$pPt <- meta$Pt/meta$total
meta$pNuc <- meta$nuclear/meta$total
meta$pTrx <- meta$n.trx/meta$total

# estimate densities
l1 <- meta

# plot individually - mitocondrial proportion
pdf(paste0(prefix,".Mitocondrial_distributions.pdf"), width=8, height=8)
plotDensity(l1, main=prefix)
dev.off()

# chloroplast proportion
pdf(paste0(prefix,".Chloroplast_distributions.pdf"), width=8, height=8)
plotDensity(l1, column="pPt", main=prefix)
dev.off()

# nuclear proportion
pdf(paste0(prefix,".Nuclear_distributions.pdf"), width=8, height=8)
plotDensity(l1, column="pNuc", main=prefix)
dev.off()

# proportion transcript
pdf(paste0(prefix,".transcript_distributions.pdf"), width=8, height=8)
plotDensity(l1, column="pTrx", main=prefix)
dev.off()

# plot gene by umi
pdf(paste0(prefix,".transcripts_per_gene.pdf"), width=8, height=8)
plot(l1$n.genes, l1$n.trx, pch=16, cex=0.8, xlab="Num. genes", ylab="Num. UMI")
dev.off()

# filtering individual
l1.f <- filterCells(l1, column="pMt", threshold=2, direction="less")
l1.f <- filterCells(l1.f, column="pPt", threshold=2, direction="less")
l1.f <- filterCells(l1.f, column="pTrx", threshold=-2, direction="greater")

# specify keepers
meta$pass <- ifelse(rownames(meta) %in% rownames(l1.f), 1, 0)
num.pass <- nrow(subset(meta, meta$pass==1))
if(num.pass > 16000){
  top <- meta[meta$pass==1,]
  top <- top[order(top$n.trx, decreasing=T),]
  topp <- top[1:16000,]
  meta$pass <- ifelse(rownames(meta) %in% rownames(topp), 1, 0)
  num.pass <- 16000
}

# filter
message(" - final filter, outputting cleaned data for ", num.pass, " cells ...")
f.meta <- meta[meta$pass > 0,]
all <- all[,rownames(f.meta)]
write.table(f.meta, file=paste0(prefix,".filtered.meta.txt"), quote=F, row.names=T, col.names=T, sep="\t")
