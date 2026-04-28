## pairwise alignments ##

# load libraries
library(RColorBrewer)
library(MASS)
library(pheatmap)

# load data
files <- list.files(pattern="*cellAlign_pairwise_global_alignments.rds")

# process
xvals <- c()
yvals <- c()

# filter alignments
ffiles <- unlist(lapply(files, function(z){
  id <- gsub("\\.cellAlign_pairwise_global_alignments\\.rds","",z)
  tj <- unlist(strsplit(id, "-"))
  tj1 <- unlist(strsplit(tj[1],"\\."))
  tj2 <- unlist(strsplit(tj[2],"\\."))
  if(tj1[2] != tj2[2] & tj1[1] != tj2[1]){
    return(z)
  }else{
    return(NULL)
  }
}))

# iterate over alignments
for(i in ffiles){
  
  # verbose
  message("reading ",i)
  
  # read
  obj <- readRDS(i)
  
  # concatenate
  xvals <- c(xvals, obj$align[[1]]$index1, obj$align[[1]]$index2)
  yvals <- c(yvals, obj$align[[1]]$index2, obj$align[[1]]$index1)
  
}

# get global alignment density
den <- kde2d(xvals, yvals, n=300)
cols <- colorRampPalette(brewer.pal(9, "Greys"))(100)

# plot
pdf("pairwise_global_alignment.unique_somatic_origins.unique_stem_cells.pdf", width=5.25, height=5)
pheatmap(den$z, col=cols, cluster_rows=F, cluster_cols=F, show_rownames=F, show_colnames=F)
dev.off()
