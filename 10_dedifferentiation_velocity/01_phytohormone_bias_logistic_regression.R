## test hormone effect over reverse pseudotime ##

# libraries
library(RColorBrewer)
library(pheatmap)
library(mgcv)
library(parallel)
library(scales)
library(MASS)

# arguments
files <- read.table("walk_files.txt")
res <- lapply(files$V1, function(z){
  walks <- readRDS(z)
  walks$paths
})
output <- gsub("PROB_WALKS_","", files$V1)
output <- gsub("\\.rds","",output)
names(res) <- output

# load data
meta <- read.table("../diffusion_pseudotime.metadata.all_cells.palantir.11.19.2025.knn30.real_time.cellfate.txt")
pt <- readRDS("../All_Palantir_results.11.26.2025.rds")

# diffusion
difpt <- meta$cor.value
names(difpt) <- rownames(meta)

# walks
pseudorange <- c()
for(z in names(res)){

  # extract pseudotime
  id <- unlist(strsplit(z,"\\."))
  cl <- paste0(id[1],"_",id[2])
  celltype <- id[3]
  
  # pseudotime
  pt.cl <- pt[[cl]]
  cell.ids <- rownames(pt.cl)
  pt.cl <- pt.cl$Pseudotime
  names(pt.cl) <- cell.ids
  
  # time effect on hormone treated cells
  cellIDs1 <- res[[z]]
  ptt <- pt.cl[unlist(cellIDs1)]
  pseudorange <- c(pseudorange, range(ptt))
  outs2 <- mclapply(seq(1:length(cellIDs1)), function(x){
    
    # verbose
    message(" -- testing walk ID = ",x, " | trajectory = ", z)
    
    # set up
    cellIDs <- cellIDs1[[x]]
    if(length(cellIDs) < 3){
      return(NULL)
    }else{
      mmeta <- meta[cellIDs,]
      d0 <- subset(mmeta, mmeta$age=="D0")
      d0$hormone <- T
      rownames(d0) <- paste0("h",rownames(d0))
      all.meta <- rbind(mmeta, d0)
      tt <- all.meta$hormone
      pttt <- all.meta$consensus_pseudotime
      
      # gam model
      mod <- tryCatch({glm(as.numeric(tt)~pttt, family=binomial())}, error=function(e){NULL})
      if(is.null(mod)){
        return(NULL)
      }else{
        ress <- summary(mod)
        #stat <- ress$s.table[1,3]
        #pval <- ress$s.table[1,4]
        stat <- ress$coefficients[2,3]
        pval <- ress$coefficients[2,4]
        
        # permute
        nulls <- unlist(lapply(seq(1:100), function(xx){
          p.pt <- sample(pttt)
          mod1 <- tryCatch({glm(as.numeric(tt)~p.pt, family=binomial())}, error=function(e){NULL})
          if(is.null(mod1)){
            return(NULL)
          }
          res1 <- summary(mod1)
          stat1 <- res1$coefficients[2,3]
          stat1
        }))
        
        # eFDR 
        mu <- mean(nulls, na.rm = TRUE)
        sigma <- sd(nulls, na.rm=T)
        ezscore <- (stat - mu) / sigma
        ezscore[is.na(ezscore)] <- 0
        epval <- 1 - pnorm(abs(ezscore))
        emp <- (1+sum(abs(nulls) >= abs(stat)))/(length(nulls)+1)
        
        # return
        return(data.frame(walkID=x, trajID=z, stat=stat, pval=pval, enpval=epval, epval=emp))
      }
    }
  }, mc.cores=24)
  outs2 <- do.call(rbind, outs2)
  saveRDS(outs2, file=paste0(z, ".hormone_GAM_tests.rds"))
}

# get predictions
predictions <- lapply(names(res), function(z){
  
  # extract pseudotime
  id <- unlist(strsplit(z,"\\."))
  cl <- paste0(id[1],"_",id[2])
  celltype <- id[3]
  
  # pseudotime
  pt.cl <- pt[[cl]]
  cell.ids <- rownames(pt.cl)
  pt.cl <- pt.cl$Pseudotime
  names(pt.cl) <- cell.ids
  
  # time effect on hormone treated cells
  cellIDs1 <- res[[z]]
  ptt <- pt.cl[unlist(cellIDs1)]
  outs2 <- mclapply(seq(1:length(cellIDs1)), function(x){
    
    # verbose
    message(" -- testing walk ID = ",x, " | trajectory = ", z)
    
    # set up
    cellIDs <- cellIDs1[[x]]
    if(length(cellIDs) < 3){
      return(NULL)
    }else{
      mmeta <- meta[cellIDs,]
      d0 <- subset(mmeta, mmeta$age=="D0")
      d0$hormone <- T
      rownames(d0) <- paste0("h",rownames(d0))
      all.meta <- rbind(mmeta, d0)
      tt <- all.meta$hormone
      pttt <- all.meta$consensus_pseudotime
      
      # gam model
      mod <- tryCatch({glm(as.numeric(tt)~pttt, family=binomial())}, error=function(e){NULL})
      if(is.null(mod)){
        return(NULL)
      }else{
        
        # predictions
        preds <- predict(mod, newdata=data.frame(pttt=seq(from=min(pttt),to=max(pttt), length.out=100)), type="response")
        return(data.frame(walkID=x, trajID=z, x=seq(from=min(pttt),to=max(pttt), length.out=100), y=preds))
        
      }
    }
  }, mc.cores=24)
  outs2 <- do.call(rbind, outs2)
  return(outs2)
  
})
predictions <- do.call(rbind, predictions)

# save results
saveRDS(pseudorange, file="seudorange.rds")
saveRDS(predictions, file="all_predictions_df.rds")

# load results
test.files <- list.files(pattern="*.hormone_GAM_tests.rds")
treat <- lapply(test.files, function(z){
  df <- readRDS(z)
  print(range(df$pval))
  return(df)
})
treat <- do.call(rbind, treat)
saveRDS(treat, file="hormone_dediff_traj_tests.rds")

# mulitple test correction
treat$fdr <- p.adjust(treat$pval, method="fdr")
treat$enfdr <- p.adjust(treat$enpval, method="fdr")
treat$efdr <- p.adjust(treat$epval, method="fdr")
treat$pass <- ifelse(treat$enfdr < 0.05, 1, 0)
treat$log10enFDR <- -log10(treat$pval)
treat$log10enFDR[is.infinite(treat$log10enFDR)] <- max(treat$log10enFDR[is.finite(treat$log10enFDR)])

# iterate over predictions
cols <- colorRampPalette(brewer.pal(9, "Greys")[3:9])(100)
treat$col <- cols[cut(treat$log10enFDR, breaks=seq(from=0, to=max(treat$log10enFDR)+1e-8, length.out=101))]
treat <- treat[order(treat$trajID, treat$log10enFDR, decreasing=F),]

# plot parameters
pdf("hormone_cells_by_rPt.02.26.2026.pdf", width=24, height=8)
layout(matrix(c(1:12), nrow=2, byrow=T))

tIDs <- c("cluster.1.Spongy_Mesophyll","cluster.2.Spongy_Mesophyll","cluster.4.Spongy_Mesophyll","cluster.6.Spongy_Mesophyll",
          "cluster.7.Spongy_Mesophyll", "cluster.9.Spongy_Mesophyll", "cluster.11.Spongy_Mesophyll", "cluster.14.Spongy_Mesophyll",
          "cluster.21.Spongy_Mesophyll","cluster.18.Guard_Cell","cluster.14.Hydathode","cluster.18.Spongy_Mesophyll")
for(i in tIDs){
  tj <- subset(treat, treat$trajID==i)
  plot.new()
  plot.window(xlim=range(pseudorange), ylim=c(0,1))
  axis(1)
  axis(2)
  box()
  grid(lty=1)
  title(xlab="Reverse pseudotime", ylab="Phytohormone treatment probability", main=i)
  for(j in 1:nrow(tj)){
    
    df <- subset(predictions, predictions$trajID==i & predictions$walkID==tj$walkID[j])  
    lines(df$x, df$y, col=tj$col[j])
    
  }
}
dev.off()



##############################
## DENSITY PLOT ##############
##############################

# get ranges
tIDs <- c("cluster.1.Spongy_Mesophyll","cluster.2.Spongy_Mesophyll","cluster.4.Spongy_Mesophyll","cluster.6.Spongy_Mesophyll",
          "cluster.7.Spongy_Mesophyll", "cluster.9.Spongy_Mesophyll", "cluster.11.Spongy_Mesophyll", "cluster.14.Spongy_Mesophyll",
          "cluster.21.Spongy_Mesophyll","cluster.18.Guard_Cell","cluster.14.Hydathode","cluster.18.Spongy_Mesophyll")

denrange <- c()
for(i in tIDs){
  
  # verbose
  message(i)
  
  # subset
  tj <- subset(treat, treat$trajID==i)
  
  # extract data
  df <- subset(predictions, predictions$trajID==i)
  den <- kde2d(df$x, df$y, n=300, lims=c(range(pseudorange), c(0,1)))
  denrange <- c(denrange, range(den$z))
}

# plot
pdf("hormone_cells_by_density.02.26.2026.rr.pdf", width=24, height=8)
layout(matrix(c(1:12), nrow=2, byrow=T))
for(i in tIDs){
  
  # verbose
  message(i)
  
  # subset
  tj <- subset(treat, treat$trajID==i)
  
  # plot parameters
  #plot.new()
  #plot.window(xlim=range(pseudorange), ylim=c(0,1))
  #axis(1)
  #axis(2)
  ##box()
  #grid(lty=1)
  #title(xlab="Reverse pseudotime", ylab="Phytohormone treatment probability", main=i)
  
  # extract data
  df <- subset(predictions, predictions$trajID==i)
  den <- kde2d(df$x, df$y, n=300, lims=c(range(pseudorange), c(0,1)))
  denrange <- c(denrange, range(den$z))
  cols <- colorRampPalette(c("white",brewer.pal(9, "Greys")), bias=2)(100)
  image(den, col=cols, breaks=seq(from=min(denrange), to=max(denrange), length.out=101),
        xlab="Reverse pseudotime", ylab="Phytohormone treatment probability", main=i, useRaster=T)
  grid(lty=1)
  box()
}
dev.off()

# proportions
treat$trajID <- factor(treat$trajID, levels=tIDs)
treat$dir <- ifelse(treat$stat < 0, "hormone", "non_hormone")
treat$type <- ifelse(treat$pass==1 & treat$dir=="hormone", "hormone_FDR<05",
                     ifelse(treat$pass==1 & treat$dir=="non_hormone", "non_hormone_FDR<05", "FDR>05"))
props <- prop.table(table(treat$trajID, treat$type),1)
pdf("proportion_walks_significant.pdf", width=8, height=4)
barplot(t(props), col=c("grey75","darkorchid4","gold4"), beside=F, border=NA)
dev.off()

# save results
#saveRDS(treat, file=paste0(output,".hormone_props_reverse_pseudotime.rds"))