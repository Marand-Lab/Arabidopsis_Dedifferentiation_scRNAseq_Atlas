#!/bin/bash
#SBATCH --job-name=v3_08c_wrky_nuc
#SBATCH --account=YOURNAME1
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=80G
#SBATCH --time=08:00:00
#SBATCH --output=_logs/v3_wrky_nuc_%j.log



set -euo pipefail

echo "Node: $(hostname), Start: $(date)"

# Environment
export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

source ~/home_turbo/fabio_home/LocalInstall/miniconda3/etc/profile.d/conda.sh
conda activate scprinter-cpu

python -u v3_08c_wrky_nucleosome_extract.py \
    --tmp-dir /tmp
