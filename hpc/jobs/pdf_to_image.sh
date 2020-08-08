#!/bin/bash -x

#PBS -l nodes=1:ppn=28
#PBS -l walltime=47:59:59
#PBS -q normal_q
#PBS -A waingram_lab
#PBS -W group_list=newriver
#PBS -M sampanna@vt.edu
#PBS -m bea

#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sampanna@vt.edu
#SBATCH -t 47:59:59
#SBATCH -p normal_q
#SBATCH -A waingram_lab

module load Anaconda/5.2.0
source activate deepfigures_3
python /home/sampanna/deepfigures-open/gold_standard/pdf_to_image.py
echo "Done."
