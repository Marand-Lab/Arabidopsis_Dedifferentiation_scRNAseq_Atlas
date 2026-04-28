#!/bin/bash

## submission properties

#SBATCH --partition=standard
#SBATCH --account=YOURACCT
#SBATCH --job-name=PSEUDOTIME_PATHS_ALL_CLUSTERS
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=7-00:00:00
#SBATCH --mem=150g
#SBATCH --output=FIT_PSEUDOTIME_PATHS_ALL_COMBOS.%j.log
#SBATCH --error=FIT_PSEUDOTIME_PATHS_ALL_COMBOS.%j.err
#SBATCH --array=1-12

# set env
cd $SLURM_SUBMIT_DIR

# cell types to check
cts=$( sed -n "${SLURM_ARRAY_TASK_ID}p" < celltype_walk.config )
arg1=$( echo $cts | cut -f 1 )
arg2=$( echo $cts | cut -f 2 )
arg2=$( echo "$arg2" | tr -d '\n' )

# function
runTraj(){
	echo "Running probabilistic random walks between cluster $1 and cell type $2"
	Rscript Extract_Trajectories_Module.v3.R $1 $2
}

# run for each
runTraj $arg1 $arg2
