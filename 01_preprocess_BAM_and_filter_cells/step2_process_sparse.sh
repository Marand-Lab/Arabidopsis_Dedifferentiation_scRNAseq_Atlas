#!/bin/bash

#SBATCH --partition=standard
#SBATCH --account=amarand1
#SBATCH --job-name=process_sparse
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1-00:00:00
#SBATCH --mem=20g
#SBATCH --output=LOGS_PROCESS_scATAC_sparse.%j.log
#SBATCH --error=LOGS_PROCESS_scATAC_sparse.%j.err

# set env
cd $SLURM_SUBMIT_DIR

# threads
threads=15
qual=1

# load modules
ml picard-tools/2.8.1

# set dir
source ~/.zshrc

# functions
doCall(){

	# input
	base=$1
	
	# clean sparse
	perl cleanSparse.pl $base.sparse > $base.clean.sparse

}
export -f doCall

# run processing (change sample1 to the file name without the .bam suffix)
doCall YJ1 $qual $threads
doCall YJ2 $qual $threads
doCall YJ3 $qual $threads
doCall YJ4 $qual $threads
doCall YJ5 $qual $threads
doCall YJ6 $qual $threads
doCall YJ7 $qual $threads
doCall YJ8 $qual $threads
doCall YJ9 $qual $threads
doCall YJ10 $qual $threads
doCall YJ11 $qual $threads
doCall YJ12 $qual $threads
doCall YJ13 $qual $threads
doCall YJ14 $qual $threads
doCall YJ15 $qual $threads
doCall YJ16 $qual $threads
doCall YJ17 $qual $threads
doCall YJ18 $qual $threads
doCall YJ19 $qual $threads
