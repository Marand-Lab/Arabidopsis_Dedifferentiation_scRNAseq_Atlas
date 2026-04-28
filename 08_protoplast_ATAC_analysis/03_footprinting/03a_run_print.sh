#!/bin/bash
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=80G
#SBATCH --job-name=v4_02c_FP
#SBATCH --partition=standard
#SBATCH --account=YOURNAME1
#SBATCH --output=_logs/%x_%A_%a.log
#SBATCH --mail-user=gomezcan@umich.edu
#SBATCH --mail-type=FAIL,END
#SBATCH --array=0-1

# v4_02c — Run scPrinter FP on merged condition BAMs.
# Array: 0=leaf, 1=proto
#
# Dependency: run after v4_01b (fragment generation) completes.
# Run: sbatch v4/v4_02c_run_print.sh

set -euo pipefail
mkdir -p _logs v4/3_PRINT

# Tame native threads & HDF5 locking
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export HDF5_USE_FILE_LOCKING=FALSE

# Env
source ~/home_turbo/fabio_home/LocalInstall/miniconda3/etc/profile.d/conda.sh
conda activate scprinter-cpu

CONDITIONS=(leaf proto)
CONDITION="${CONDITIONS[${SLURM_ARRAY_TASK_ID}]}"

echo "[INFO] Array task ${SLURM_ARRAY_TASK_ID}: ${CONDITION}"
python -u final/03_footprinting/03a_run_print.py "$CONDITION"
