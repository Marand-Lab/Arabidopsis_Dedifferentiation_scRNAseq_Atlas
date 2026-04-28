#!/bin/bash

## submission properties

#SBATCH --partition=standard
#SBATCH --account=YOURACCT
#SBATCH --job-name=UMAP_markers
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=15
#SBATCH --time=3-00:00:00
#SBATCH --mem=150g
#SBATCH --output=ASSESS_marker_expression.%j.log
#SBATCH --error=ASSESS_marker_expression.%j.err

# set env
cd $SLURM_SUBMIT_DIR
source ~/.zshrc

# vars
data=D0_cells.seurat_object.rds

# run for each
Rscript plot_marker_accessibility.R $data top_arabidopsis_marker_genes.txt 15 D0_cells.MARKERS
