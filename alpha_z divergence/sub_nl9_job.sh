#!/bin/bash
#SBATCH --job-name=sub_nl9_job
#SBATCH --output=sub_nl9_output.txt
#SBATCH --ntasks=3
#SBATCH --time=720:00:00
#SBATCH --mem=8000
#SBATCH --mail-type=ALL              
#SBATCH --mail-user=mzu0014@auburn.edu

module load python 

python /mmfs1/home/mzu0014/sub_nl9.py
