## estimate dedifferentiation rates ##

# ml R/4.5.1

# load libraries
library(mgcv)
library(gratia)
library(scales)

# load functions

# load data
meta <- read.table("../diffusion_pseudotime.metadata.all_cells.palantir.11.19.2025.knn30.real_time.cellfate.txt")
pt <- readRDS("../All_Palantir_results.11.26.2025.rds")

# walk data
files <- read.table("walk_files.txt")
res <- lapply(files$V1, function(z){
  walks <- readRDS(z)
  unique(unlist(walks$paths))
})
output <- gsub("PROB_WALKS_","", files$V1)
output <- gsub("\\.rds","",output)
names(res) <- output

# ids
tIDs <- c("cluster.1.Spongy_Mesophyll","cluster.2.Spongy_Mesophyll","cluster.4.Spongy_Mesophyll","cluster.6.Spongy_Mesophyll",
          "cluster.7.Spongy_Mesophyll", "cluster.9.Spongy_Mesophyll", "cluster.11.Spongy_Mesophyll", "cluster.14.Spongy_Mesophyll",
          "cluster.21.Spongy_Mesophyll","cluster.18.Guard_Cell","cluster.14.Hydathode","cluster.18.Spongy_Mesophyll")

# iterate over walks
all.r <- c()
num.cells <- c()
outs <- lapply(tIDs, function(z){
  cellIDs <- res[[z]]
  trajm <- meta[cellIDs,]
  d0cells <- subset(trajm, trajm$age=="D0")
  d0cells$hormone <- T
  rownames(d0cells) <- paste0(rownames(d0cells),"-H")
  trajm <- rbind(trajm, d0cells)
  trajm$hormoneVar <- factor(as.numeric(trajm$hormone))
  trajm$library <- factor(trajm$library)
  num.cells <<- c(num.cells, nrow(trajm))
  message(z)
  print(table(trajm$hormone))
  
  # fit GAM
  m <- gam(consensus_pseudotime ~ hormoneVar + s(real_time, by=hormoneVar) + s(library, bs="re"), data=trajm, method = "REML")
  
  # extract velocity
  vel_n <- as.data.frame(derivatives(m, term = "s(real_time):hormoneVar0", type = "central"))
  vel_n <- subset(vel_n, vel_n$real_time < max(trajm$real_time[trajm$hormone==F]))
  vel_h <- as.data.frame(derivatives(m, term = "s(real_time):hormoneVar1", type = "central"))
  all <- range(c(range(vel_h$`.lower_ci`), range(vel_n$`.lower_ci`),
                 range(vel_h$`.upper_ci`), range(vel_n$`.upper_ci`)))
  all.r <<- c(all.r, all)
  return(list(vel_n=vel_n, vel_h=vel_h))
})
names(outs) <- tIDs
saveRDS(outs, file="dedifferentiation_velocity.rds")

# plot treatment proportions
 names(num.cells) <- tIDs
 num.cells <- rescale(num.cells, c(1, 5))
 pdf("Dedifferentiation_velocity.PIECHARTs.02.26.2026.pdf", width=24, height=8)
 layout(matrix(c(1:12), nrow=2, byrow=T))
 for(z in tIDs){
   cellIDs <- res[[z]]
   trajm <- meta[cellIDs,]
   d0cells <- subset(trajm, trajm$age=="D0")
   d0cells$hormone <- T
   rownames(d0cells) <- paste0(rownames(d0cells),"-H")
   trajm <- rbind(trajm, d0cells)
   trajm$hormoneVar <- factor(as.numeric(trajm$hormone))
   cols <- c("#c4af2f","#692d89")
   pie(table(trajm$hormoneVar), col=cols, labels=NA, border="white", radius=num.cells[z])
 }
 dev.off()


# plot
pdf("Dedifferentiation_velocity.02.26.2026.pdf", width=24, height=8)
layout(matrix(c(1:12), nrow=2, byrow=T))
for(z in tIDs){
  
  # extract velocity
  vel_h <- outs[[z]]$vel_h
  vel_n <- outs[[z]]$vel_n
  
  # hormone
  all <- range(all.r[1:22])
  plot(vel_h$real_time, vel_h$`.derivative`, type="l", col="darkorchid4", 
       ylim=all,
       xlim=range(meta$real_time),
       xlab="Real time",
       ylab="Dedifferentiation velocity (dPt/dT)",
       main=z)
  lines(vel_h$real_time, vel_h$`.lower_ci`, type="l", col="darkorchid1")
  lines(vel_h$real_time, vel_h$`.upper_ci`, type="l", col="darkorchid1")
  
  # non-hormone
  lines(vel_n$real_time, vel_n$`.derivative`, type="l", col="gold4")
  lines(vel_n$real_time, vel_n$`.lower_ci`, type="l", col="gold2")
  lines(vel_n$real_time, vel_n$`.upper_ci`, type="l", col="gold2")
  grid(lty=1)
}
dev.off()
