#!/bin/bash --login
#SBATCH --job-name=LogNormal_T
#SBATCH --time=96:00:00
#SBATCH --array=1-19
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output test_serial-job_%j.out
#SBATCH --error test_serial-job_%j.err


module load Miniconda3
conda activate myconda

python3 Truncation_Distribution.py $SLURM_ARRAY_TASK_ID LogNormal