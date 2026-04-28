#!/bin/bash

## submission properties

#SBATCH --partition=standard
#SBATCH --account=YOURNAME1
#SBATCH --job-name=Palantir_pairwise
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=7-00:00:00
#SBATCH --mem=25g
#SBATCH --output=FIT_PSEUDOTIME_PATHS_palantir.%j.log
#SBATCH --error=FIT_PSEUDOTIME_PATHS_palantir.%j.err
#SBATCH --array=1-10

# set env
cd $SLURM_SUBMIT_DIR
source ~/.zshrc

# cell types to check
cts=$(sed -n "${SLURM_ARRAY_TASK_ID}p" < dedifferentiated_clusters.txt)

# function
runTraj(){
	echo "Running palantir for start cell derived from $1"
	Rscript Palantir_Analysis.module.R $1
}

# run for each
runTraj $cts
