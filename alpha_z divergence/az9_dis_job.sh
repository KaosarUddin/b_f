#!/bin/bash
#SBATCH --job-name=az9_dis_job
#SBATCH --output=az9_dis_output.txt
#SBATCH --ntasks=1
#SBATCH --time=240:00:00
#SBATCH --mem=8000
#SBATCH --mail-type=ALL              
#SBATCH --mail-user=mzu0014@auburn.edu

module load python 

python /mmfs1/home/mzu0014/az9_dis.py
