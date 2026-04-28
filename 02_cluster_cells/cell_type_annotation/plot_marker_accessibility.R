###################################################################################################
## plot marker accessibility scores
###################################################################################################

# load arguments
args <- commandArgs(trailingOnly=T)
if(length(args) != 4){stop("Rscript plot_marker_accessibility.R [obj] [markers.bed] [threads] [prefix]")}

#args
obj <- as.character(args[1])
mark <- as.character(args[2])
threads <- as.numeric(args[3])
prefix <- as.character(args[4])

# load functions
source("functions.plot_marker_accessibility.R")

# load data
dat <- loadData(obj, mark)
b.meta <- dat$b
activity.all <- dat$activity
h.pcs1 <- dat$h.pcs
marker.info.dat <- dat$marker.info

# match ids
activity.all <- activity.all[,rownames(b.meta)]
activity.all <- activity.all[Matrix::rowSums(activity.all)>0,]
h.pcs1 <- h.pcs1[rownames(b.meta),]
marker.info.dat <- marker.info.dat[rownames(marker.info.dat) %in% rownames(activity.all),]

# iterate over each major cluster
out <- runMajorPriori(b.meta, activity.all, h.pcs1, marker.info.dat, threads=threads, output=prefix, smooth.markers=T)
