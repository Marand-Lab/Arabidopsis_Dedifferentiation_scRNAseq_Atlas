## diffusion graphs ##

# load libraries
library(Seurat)
library(destiny)
library(RColorBrewer)
library(uwot)

# load data
a <- readRDS("../All.YJ1_19.logNorm.seurat_object.rds")
b <- read.table("../All.YJ1_19.logNorm.seurat.metaData.txt")

# reformat
b$age <- gsub("_wH","",b$treatment)
b$age <- gsub("_noH","",b$age)
b$age.n <- as.numeric(factor(b$age, levels=c("D0","D2","D4","D6")))
b <- b[order(b$age.n, decreasing=F),]

# diffusion with pcs
pcs <- Embeddings(a, reduction="harmony")
dm.pcs <- DiffusionMap(pcs, n_eigs=50)
saveRDS(dm.pcs, file="All.YJ1_19.DCs.harmony.rds")
ev.pcs <- dm.pcs@eigenvectors
ev.pcs <- ev.pcs[rownames(b),]

# find dcs correlated with age
cors <- cor(ev.pcs, b$age.n)
ids <- names(cors[abs(cors[,1]) > 0.1,])

# umap
umap.dm <- umap(ev.pcs[,ids], min_dist=0.01, n_neighbor=100)
umap.dm <- umap.dm[rownames(b),]

# plot results
cols <- brewer.pal(5, "YlGnBu")[2:5]

# DC 1 vs DC 2
pdf("DC1.2.harmony.pdf", width=5, height=5)
plot(ev.pcs[,1], ev.pcs[,2], pch=16, cex=0.5, col=cols[b$age.n])
grid(lty=1)
legend("topleft", legend=c("D0","D2","D4","D6"), col=cols, border=NA, pch=16)
dev.off()

pdf("dcUMAP.harmony.pdf", width=5, height=5)
plot(umap.dm[,1], umap.dm[,2], pch=16, cex=0.3, col=cols[b$age.n])
legend("topleft", legend=c("D0","D2","D4","D6"), col=cols, border=NA, pch=16)
grid(lty=1)
dev.off()


