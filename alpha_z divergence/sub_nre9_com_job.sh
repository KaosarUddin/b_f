#!/bin/bash
#SBATCH --job-name=sub_nre9_com_job
#SBATCH --output=sub_nre9_com_output.txt
#SBATCH --ntasks=3
#SBATCH --time=720:00:00
#SBATCH --mem=8000
#SBATCH --mail-type=ALL              
#SBATCH --mail-user=mzu0014@auburn.edu

module load python 

python /mmfs1/home/mzu0014/sub_nre9_com.py
