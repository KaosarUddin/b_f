#!/bin/bash
#SBATCH --job-name=az8s_dis_job
#SBATCH --partition=bigmem2  # Change the partition to bigmem2
#SBATCH --output=/mmfs1/home/mzu0014/az8s_dis_output.txt  # Ensure full path for output
#SBATCH --error=/mmfs1/home/mzu0014/az8s_dis_output.txt   # Redirect error to the same file
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=4  # Request 4 CPUs
#SBATCH --time=720:00:00
#SBATCH --mem=32000M  # Increase memory allocation if needed (32GB)
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mzu0014@auburn.edu

module load python

python /mmfs1/home/mzu0014/az8s_dis.py

echo "Job finished on $(date)"
