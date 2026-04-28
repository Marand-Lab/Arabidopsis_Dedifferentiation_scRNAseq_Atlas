#!/bin/bash

## submission properties

#SBATCH --partition=standard
#SBATCH --account=amarand1
#SBATCH --job-name=PERMUTED_GAM_PATHS
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=7-00:00:00
#SBATCH --mem=50g
#SBATCH --output=FIT_PERMUTED_GAM_PATHS.%j.log
#SBATCH --error=FIT_PERMUTED_GAM_PATHS.%j.err
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
	Rscript model_prob_walk_expression.PERMUTED.R $1 $2
}

# run for each
runTraj $arg1 $arg2
