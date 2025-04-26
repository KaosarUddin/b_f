#!/bin/bash
#SBATCH --job-name=az_div_wm1_job
#SBATCH --output=az_div_wm1_output.txt
#SBATCH --ntasks=1
#SBATCH --time=240:00:00
#SBATCH --mem=16000
#SBATCH --mail-type=ALL              
#SBATCH --mail-user=mzu0014@auburn.edu

module load python 

python /mmfs1/home/mzu0014/az_div_wm1.py
