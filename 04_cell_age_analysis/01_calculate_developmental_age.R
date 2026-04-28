## train model on cell age ##

# library
library(glmnet)
library(caret)
library(vioplot)
library(RColorBrewer)
library(viridis)
library(mgcv)
library(parallel)
library(scales)
library(gplots)

# functions
plotTrajGN <- function(obj, pt, prefix="temp", threads=1){
  
  # subset traj
  message(" - Sort cells ...")
  pt <- pt[order(pt$inferred_age, decreasing=F),]
  binary <- obj[,rownames(pt)]
  binary <- binary[,Matrix::colSums(binary)>0]
  binary <- binary[Matrix::rowSums(binary)>0,]
  
  # generalized additive model for logistic regression
  message(" - running generalized additive model ...")
  results <- list()
  newdat <- data.frame(p.time=seq(from=min(pt$inferred_age, na.rm=T), to=max(pt$inferred_age, na.rm=T), length.out=100))
  fit <- mclapply(seq(1:nrow(binary)),function(x){
    df <- data.frame(acc=as.numeric(binary[x,]), p.time=pt$inferred_age)
    mod <- gam(acc~s(p.time, bs="cr"), data=df)
    pred <- predict(mod, newdat, type="response")
    zscore <- (pred-mean(pred, na.rm=T))/sd(pred, na.rm=T)
    res <- summary(mod) 
    results[[x]] <<- res$s.table[1,4]
    return(zscore)
  }, mc.cores=threads)
  names(fit) <- rownames(binary)
  fit <- do.call(rbind, fit)
  write.table(fit, file=paste0(prefix,".Gene_pt.txt"), quote=F, row.names=T, col.names=T, sep="\t")
  
  # filter by variances
  fit <- t(apply(fit, 1, function(x){
    rescale(x, c(0,1))
  }))
  
  # reformat output
  row.o <- apply(fit, 1, which.max)
  fit <- fit[order(row.o, decreasing=F),]
  
  # plot
  message(" - plotting cell trajectory ...")
  cols <- colorRampPalette(rev(brewer.pal(9, "RdBu")))(100)
  pdf(paste0(prefix,".trajectoryGene.pdf"), width=10, height=10)
  heatmap.2(fit, trace="none", col=cols, Colv=NA, Rowv=NA, dendrogram="none",
            scale="none", labRow = NA, labCol=NA, useRaster=T,
            ylab=paste("Genes", paste0("(n=",nrow(fit),")"), sep=" "))
  dev.off()
  
  # return
  return(results)
  
}

# load data
obj <- readRDS("step1_palantir_obj_100DC_harmony_knn.15.10.21.2025.knn30.rds")
meta <- read.table("diffusion_pseudotime.metadata.all_cells.palantir_pt_entrop.10.21.2025.knn30.txt")
obj@meta.data <- meta

# split data into hormone and non-hormone treated cells
exprs <- obj@assays$RNA$data
d0 <- subset(meta, meta$age=="D0")
h <- subset(meta, meta$hormone==T)
h <- rbind(d0, h)
n <- subset(meta, meta$hormone==F)
h.exp <- exprs[,rownames(h)]
n.exp <- exprs[,rownames(n)]

# count cells from each stage
h.counts <- table(h$age)
n.counts <- table(n$age)
min.h <- min(h.counts)
min.n <- min(n.counts)

# select equal numbers of cells
h.sub <- lapply(unique(h$age), function(z){
  message(z)
  df <- subset(h, h$age==z)
  df[sample(nrow(df), min.h),]
})
n.sub <- lapply(unique(n$age), function(z){
  message(z)
  df <- subset(n, n$age==z)
  df[sample(nrow(df), min.n),]
})
h.sub <- do.call(rbind, h.sub)
n.sub <- do.call(rbind, n.sub)

# subset
sub.hexp <- h.exp[,rownames(h.sub)]
sub.nexp <- n.exp[,rownames(n.sub)]
sub.hexp <- sub.hexp[Matrix::rowSums(sub.hexp)>0,]
sub.nexp <- sub.nexp[Matrix::rowSums(sub.nexp)>0,]
shared <- intersect(rownames(sub.hexp), rownames(sub.nexp))
sub.hexp <- sub.hexp[shared,]
sub.nexp <- sub.nexp[shared,]

# set dev time
h.sub$devtime <- as.numeric(factor(h.sub$age, levels=c("D0","D2","D4","D6")))
n.sub$devtime <- as.numeric(factor(n.sub$age, levels=c("D0","D2","D4")))

# split
h.trainidx <- createDataPartition(h.sub$age, p=10/11, list=F, times=1)
n.trainidx <- createDataPartition(n.sub$age, p=10/11, list=F, times=1)
h.train <- h.sub[h.trainidx,]
h.test <- h.sub[-h.trainidx,]
n.train <- n.sub[n.trainidx,]
n.test <- n.sub[-n.trainidx,]

# train
h.model <- cv.glmnet(t(sub.hexp[,rownames(h.train)]), h.train$devtime)
n.model <- cv.glmnet(t(sub.nexp[,rownames(n.train)]), n.train$devtime)
saveRDS(h.model, file="hormone_model.rds")
saveRDS(n.model, file="nonhormone_model.rds")

# test
h.pred <- as.data.frame(predict(h.model, newx=t(sub.hexp[,rownames(h.test)]), s='lambda.min', type="response"))
n.pred <- as.data.frame(predict(n.model, newx=t(sub.nexp[,rownames(n.test)]), s='lambda.min', type="response"))
h.pred$obs <- h.test$devtime
n.pred$obs <- n.test$devtime

# plot test boxplot
pdf("vioplot_obs_inferred_age.hormone.test.pdf", width=5, height=5)
vioplot(h.pred$lambda.min~h.pred$obs, xlab="Predicted developmental time", ylab="Observed developmental time", main="Hormone")
dev.off()

# plot test
pdf("scatter_obs_inferred.hormone.test.pdf", width=5, height=5)
plot(h.pred$lambda.min, h.pred$obs, pch=16, xlab="Predicted developmental time", ylab="Observed developmental time", main="Hormone")
dev.off()

# plot test boxplot
pdf("vioplot_obs_inferred_age.NONhormone.test.pdf", width=5, height=5)
vioplot(n.pred$lambda.min~n.pred$obs, xlab="Predicted developmental time", ylab="Observed developmental time", main="Non-hormone")
dev.off()

# plot test
pdf("scatter_obs_inferred.NONhormone.test.pdf", width=5, height=5)
plot(n.pred$lambda.min, n.pred$obs, pch=16, xlab="Predicted developmental time", ylab="Observed developmental time", main="Non-hormone")
dev.off()

# collect dev time for all cells
h.age <- as.data.frame(predict(h.model, newx=t(h.exp[shared,rownames(h)]), s='lambda.min', type="response"))
n.age <- as.data.frame(predict(n.model, newx=t(n.exp[shared,rownames(n)]), s='lambda.min', type="response"))
h$inferred_age <- h.age[rownames(h),1]
n$inferred_age <- n.age[rownames(n),1]
h$devtime <- as.numeric(factor(h$age, levels=c("D0","D2","D4","D6")))
n$devtime <- as.numeric(factor(n$age, levels=c("D0","D2","D4")))

# plot all predictions
pdf("scatter_obs_inferred.hormone.ALL.pdf", width=5, height=5)
plot(h$devtime, h$inferred_age, pch=16, xlab="Obs dev age", ylab="Inferred dev age", main="Hormone")
dev.off()

pdf("inferred_vs_observed_developmental_time.hormone.ALL.pdf", width=5, height=5)
boxplot(h$inferred_age~factor(h$devtime), outline=F)
dev.off()

pdf("scatter_obs_inferred.NONhormone.ALL.pdf", width=5, height=5)
plot(n$devtime, n$inferred_age, pch=16, xlab="Obs dev age", ylab="Inferred dev age", main="Non-hormone")
dev.off()

pdf("inferred_vs_observed_developmental_time.NONhormone.ALL.pdf", width=5, height=5)
boxplot(n$inferred_age~factor(n$devtime), outline=F)
dev.off()

# merge
n.meta <- rbind(h, n)
n.meta <- n.meta[!duplicated(n.meta[,c("cellID","library")]),]





#####################################################################
# plot trajectory data
#####################################################################

# coefficients ----------------------------------
h.coef <- coefficients(h.model, s='lambda.min')
h.coe <- h.coef[-1,1]
h.coe <- h.coe[order(h.coe, decreasing=F)]
h.coe <- h.coe[h.coe != 0]

pdf("hormone_age_coefficients.pdf", width=5, height=5)
cols <- colorRampPalette(rev(brewer.pal(9, "Spectral")))(100)
plot(seq(1:length(h.coe)), h.coe, pch=16, cex=0.5, col=cols[cut(h.coe, breaks=seq(from=min(h.coe)-0.1, to=max(h.coe)+0.1, length.out=101))])
grid(lty=1)
dev.off()

# coefficients ----------------------------------
n.coef <- coefficients(n.model, s='lambda.min')
n.coe <- n.coef[-1,1]
n.coe <- n.coe[order(h.coe, decreasing=F)]
n.coe <- n.coe[h.coe != 0]

pdf("non_hormone_age_coefficients.pdf", width=5, height=5)
cols <- colorRampPalette(rev(brewer.pal(9, "Spectral")))(100)
plot(seq(1:length(n.coe)), n.coe, pch=16, cex=0.5, col=cols[cut(n.coe, breaks=seq(from=min(n.coe)-0.1, to=max(n.coe)+0.1, length.out=101))])
grid(lty=1)
dev.off()

# plot dev age gene programs --------------------
h.tests <- plotTrajGN(h.exp, h, prefix="Inferred_AGE_hormone", threads=10)
n.tests <- plotTrajGN(n.exp, n, prefix="Inferred_AGE_nonhormon", threads=10)
saveRDS(h.tests, file="hormone_inferred_age.tests.rds")
saveRDS(n.tests, file="non-hormone_inferred_age.tests.rds")

