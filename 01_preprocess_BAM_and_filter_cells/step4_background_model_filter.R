# remove bad cells #

# load args
args <- commandArgs(T)
if(length(args) != 4){stop("Rscript filter_low_qual_cells.R <rds> <meta> <filt> <prefix>")}
input.dat <- as.character(args[1])
meta.dat <- as.character(args[2])
filt.dat <- as.character(args[3])
prefix <- as.character(args[4])

# load libraries
library(Matrix)
library(qlcMatrix)

# functions
RowVar <- function(x) {
  spm <- t(x)
  stopifnot(methods::is(spm, "dgCMatrix"))
  ans <- sapply(base::seq.int(spm@Dim[2]), function(j) {
    if (spm@p[j + 1] == spm@p[j]) {
      return(0)
    }
    mean <- base::sum(spm@x[(spm@p[j] + 1):spm@p[j +
                                                   1]])/spm@Dim[1]
    sum((spm@x[(spm@p[j] + 1):spm@p[j + 1]] - mean)^2) +
      mean^2 * (spm@Dim[1] - (spm@p[j + 1] - spm@p[j]))
  })/(spm@Dim[1] - 1)
  names(ans) <- spm@Dimnames[[2]]
  ans
}

# load data
message(" - loading data")
a <- readRDS(input.dat)
b <- read.table(meta.dat)
a <- list(counts=a, meta=b)
filt <- read.table(filt.dat)

# align ids with counds
shared <- intersect(rownames(a$meta), colnames(a$counts))
a$counts <- a$counts[,shared]
a$meta <- a$meta[shared,]

# order
a$meta$umi <- Matrix::colSums(a$counts)
a$meta$ngenes <- Matrix::colSums(a$counts > 0)
a$meta <- a$meta[order(a$meta$trx, decreasing=T),]

# set initial thresholds
message(" - setting filters")
a$meta$qc_check <- ifelse(rownames(a$meta) %in% rownames(filt), 1, 0)

# parse
message(" - parsing initial boundaries")
good.cells <- rownames(subset(a$meta, a$meta$qc_check==1))
bad.cells <- rownames(subset(a$meta, a$meta$qc_check==0))
gg <- a$counts[,colnames(a$counts) %in% good.cells]
gg <- gg[,Matrix::colSums(gg > 0) > 100]
a$meta$qc_check <- ifelse(rownames(a$meta) %in% colnames(gg), 1, 0)
bb <- a$counts[,! colnames(a$counts) %in% colnames(gg)]
sites <- Matrix::rowMeans(gg > 0)
sites <- sites[order(sites, decreasing=T)]
num.sites <- max(a$meta$ngenes)
if(length(sites) < num.sites){
	num.sites <- length(sites)
}
gg <- gg[names(sites)[1:num.sites],]
gg <- gg[,Matrix::colSums(gg) > 0]
bb <- bb[rownames(gg),]
bb <- bb[,Matrix::colSums(bb) > 0]
shared <- intersect(rownames(gg), rownames(bb))
gg <- gg[shared,]
bb <- bb[shared,]
gg <- gg[,Matrix::colSums(gg) > 0]
bb <- bb[,Matrix::colSums(bb) > 0]

# ensure that bad cell have less than or equal to 500 umis
bb <- bb[,Matrix::colSums(bb)<=500]

# Do not use more than 250,000 'bad' cells
if(ncol(bb) > 250000){
    top <- Matrix::colSums(bb)
    top <- top[order(top, decreasing=F)]
    top <- names(top)[1:250000] # skip cells at the boundary
}else{
    top <- colnames(bb)
}

# clean up ref
bb <- bb[,top]
bb <- bb[Matrix::rowSums(bb)>0,]
shared.sites <- intersect(rownames(bb), rownames(gg))
bb <- bb[shared.sites,]
gg <- gg[shared.sites,]
bb <- bb[,Matrix::colSums(bb)>0]
gg <- gg[,Matrix::colSums(gg)>0]

# make references
num.good <- floor(ncol(gg)*(1/10))
if(num.good < 500){
	num.good <- min(c(ncol(gg), 500))
}
top.gg <- Matrix::colSums(gg)
top.gg <- top.gg[order(top.gg, decreasing=T)]
top.gg.ids <- names(top.gg)[1:num.good]

message(" - normalizing distributions and creating references")
sub.counts <- cbind(gg,bb)
all.res <- sub.counts %*% Diagonal(x=1e4/Matrix::colSums(sub.counts))
colnames(all.res) <- colnames(sub.counts)
all.res@x <- log(all.res@x)
bb.norm <- all.res[,colnames(bb)]
gg.norm <- all.res[,colnames(gg)]

# pick sites
res.ave <- Matrix::rowMeans(gg.norm)
res.res <- RowVar(gg.norm)
resis <- loess(res.res~res.ave)$residuals
names(resis) <- rownames(gg.norm)
top.sites <- names(resis[resis > 0])
bb.norm <- bb.norm[top.sites,]
gg.norm <- gg.norm[top.sites,]

# make references
bad.ref <- Matrix(Matrix::rowMeans(bb.norm), sparse=T)
good.ref <- Matrix(Matrix::rowMeans(gg.norm[,top.gg.ids]), sparse=T) #gg.norm[,top.gg.ids]

# check each cell against ref
message(" - estimating correlations")
b.ref <- corSparse(gg.norm, bad.ref)
g.ref <- corSparse(gg.norm, good.ref)
b.bad <- corSparse(bb.norm, bad.ref)
g.bad <- corSparse(bb.norm, good.ref)
refs <- data.frame(cbind(b.ref, g.ref))
bads <- data.frame(cbind(b.bad, g.bad))
colnames(refs) <-c("bREF", "gREF")
colnames(bads) <-c("bREF", "gREF")
refs$call <- ifelse(refs$bREF > refs$gREF, 0, 1)
bads$call <- ifelse(bads$bREF > bads$gREF, 0, 1)
rownames(refs) <- colnames(gg.norm)
rownames(bads) <- colnames(bb.norm)
all.refs <- rbind(refs,bads)
shared <- intersect(rownames(a$meta), rownames(all.refs))
meta <- a$meta[shared,]
all.refs <- all.refs[shared,]
all.refs <- cbind(meta, all.refs)
nonrefs <- a$meta[!rownames(a$meta) %in% rownames(all.refs),]
nonrefs$bREF <- NA
nonrefs$gREF <- NA
nonrefs$call <- NA
test <- rbind(all.refs, nonrefs)

# get new call
test$dif <- test$gREF-test$bREF
top <- test$dif[test$qc_check==1]
bottom <- test$dif[test$qc_check==0]
cut.off <- median(bottom, na.rm=T)
test$pass <- ifelse(test$qc_check == 1 & test$gREF > test$bREF, 1, 0)
print(table(test$pass))
write.table(test, file=paste0(prefix,".metadata.filtered.v2.txt"), quote=F, row.names=T, col.names=T, sep="\t")
