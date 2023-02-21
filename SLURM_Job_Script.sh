#!/bin/bash --login
#SBATCH --job-name=DEV_LOOR
#SBATCH --time=6:00:00
#SBATCH --array=1-60
#SBATCH --cpus-per-task=2 #normally one CPU is enough, only a tiny part of the code uses multiple CPUs
#SBATCH --mem-per-cpu=4G #if using one CPU-core, thinking about providing at least 6GB (depending on data export config)
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output test_serial-job_%j.out
#SBATCH --error test_serial-job_%j.err


module load Miniconda3
conda activate myconda
 
python3 exp_batch_manager.py $SLURM_ARRAY_TASK_ID

# This is a sample job script for SLURM usage
# A defined conda environment providing the needed requirements is needed
# Please see, that the determination of operating curves needs a closed experiment with changing utilisation
# This rows have to be deleted.