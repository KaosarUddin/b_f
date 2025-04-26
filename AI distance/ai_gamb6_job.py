#!/bin/bash
#SBATCH --job-name=ai_gamb6_job
#SBATCH --output=ai_gamb6_output.txt
#SBATCH --ntasks=3
#SBATCH --time=720:00:00
#SBATCH --mem=16000
#SBATCH --mail-type=ALL              
#SBATCH --mail-user=mzu0014@auburn.edu

module load python 

python /mmfs1/home/mzu0014/ai_gamb6.py
