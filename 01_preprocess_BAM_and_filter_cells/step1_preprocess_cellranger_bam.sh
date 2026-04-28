#!/bin/bash

#SBATCH --partition=standard
#SBATCH --account=YOURNAME1
#SBATCH --job-name=process_scRNA_bams
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=7-00:00:00
#SBATCH --mem=100g
#SBATCH --output=LOGS_PROCESS_scATAC_BAMs.%j.log
#SBATCH --error=LOGS_PROCESS_scATAC_BAMs.%j.err

# set env
cd $SLURM_SUBMIT_DIR

# threads
threads=5
qual=1

# load modules
ml picard-tools/2.8.1

# set dir
source ~/.zshrc

# functions
doCall(){

	# input
	base=$1
	qual=$2
	threads=$3
	
	# merge UMI and bc
	perl fixBCname.pl $base.raw.bam | samtools view -bShq 1 - > $base.mq$qual.bam

	# run picard
	echo "removing dups - $base ..."
	PicardCommandLine MarkDuplicates \
		MAX_FILE_HANDLES_FOR_READ_ENDS_MAP=1000 \
		REMOVE_DUPLICATES=true \
		METRICS_FILE=$base.metrics \
		I=$base.mq$qual.bam \
		O=$base.mq$qual.rmdup.bam \
		BARCODE_TAG=MB \
		ASSUME_SORT_ORDER=coordinate

	# collect counts
	perl extractCountsBAM.pl $base.mq$qual.rmdup.bam

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
