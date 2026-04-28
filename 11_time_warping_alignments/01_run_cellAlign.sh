#!/bin/bash

## submission properties

#SBATCH --partition=standard
#SBATCH --account=YOURNAME1
#SBATCH --job-name=cellAlign
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=7-00:00:00
#SBATCH --mem=100g
#SBATCH --output=LOG_cellAlign.%j.log
#SBATCH --error=LOG_cellAlign.%j.err

# set env
cd $SLURM_SUBMIT_DIR

# run
Rscript cellAlign.R
