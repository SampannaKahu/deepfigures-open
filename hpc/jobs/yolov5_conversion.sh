#!/bin/bash -x

#PBS -l nodes=1:ppn=28
#PBS -l walltime=23:59:59
#PBS -q normal_q
#PBS -A waingram_lab
#PBS -W group_list=newriver
#PBS -M sampanna@vt.edu
#PBS -m bea

#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sampanna@vt.edu
#SBATCH -t 23:59:59
#SBATCH -p normal_q
#SBATCH -A waingram_lab

CONDA_ENV=deepfigures_3

if [ "$HOSTNAME" = "ir.cs.vt.edu" ]; then
  PYTHON=/home/sampanna/anaconda3/envs/"$CONDA_ENV"/bin/python
  conda activate "$CONDA_ENV"
  SOURCE_CODE=/home/sampanna/deepfigures-open
  SCRATCH_DIR=/tmp
  ZIP_DIR=$DEEPFIGURES_RESULTS/pregenerated_training_data/377269
  YOLOV5_WORK_DIR=/home/sampanna/yolov5
elif [ "$SYSNAME" = "cascades" ]; then
  module purge
  module load Anaconda/5.1.0
  PYTHON=/home/sampanna/.conda/envs/"$CONDA_ENV"/bin/python
  source activate "$CONDA_ENV"
  DEEPFIGURES_RESULTS=/work/cascades/sampanna/deepfigures-results
  SOURCE_CODE=/home/sampanna/deepfigures-open
  SCRATCH_DIR=$TMPRAM # 311 GB on v100 nodes. 331 MBPS.
  ZIP_DIR=$DEEPFIGURES_RESULTS/pregenerated_training_data/377269
  YOLOV5_WORK_DIR=/work/cascades/sampanna/yolov5
elif [ "$SYSNAME" = "newriver" ]; then
  module purge
  module load Anaconda/5.2.0
  PYTHON=/home/sampanna/.conda/envs/"$CONDA_ENV"/bin/python
  source activate "$CONDA_ENV"
  DEEPFIGURES_RESULTS=/work/cascades/sampanna/deepfigures-results
  SOURCE_CODE=/home/sampanna/deepfigures-open
  SCRATCH_DIR=$TMPFS # 429 GB on p100 nodes. 770 MBPS.
  ZIP_DIR=$DEEPFIGURES_RESULTS/pregenerated_training_data/377269
  YOLOV5_WORK_DIR=/work/cascades/sampanna/yolov5
elif [ "$SYSNAME" = "dragonstooth" ]; then
  module purge
  module load Anaconda/5.2.0
  PYTHON=/home/sampanna/.conda/envs/"$CONDA_ENV"/bin/python
  source activate "$CONDA_ENV"
  DEEPFIGURES_RESULTS=/work/cascades/sampanna/deepfigures-results
  SOURCE_CODE=/home/sampanna/deepfigures-open
  SCRATCH_DIR=$TMPFS # 429 GB on p100 nodes. 770 MBPS.
  ZIP_DIR=$DEEPFIGURES_RESULTS/pregenerated_training_data/377269
  YOLOV5_WORK_DIR=/work/cascades/sampanna/yolov5
elif [ "$HOSTNAME" = "xps15" ]; then
  PYTHON=/home/sampanna/anaconda3/envs/"$CONDA_ENV"/bin/python
  conda activate "$CONDA_ENV"
  DEEPFIGURES_RESULTS=/home/sampanna/workspace/bdts2/deepfigures-results
  SOURCE_CODE=/home/sampanna/workspace/bdts2/deepfigures-open
  SCRATCH_DIR=/tmp
  ZIP_DIR=$DEEPFIGURES_RESULTS/pregenerated_training_data/377269
  YOLOV5_WORK_DIR=/tmp
else
  PYTHON=/home/sampanna/anaconda3/envs/"$CONDA_ENV"/bin/python
  conda activate "$CONDA_ENV" || source activate "$CONDA_ENV"
  DEEPFIGURES_RESULTS=/home/sampanna/deepfigures-results
  SOURCE_CODE=/home/sampanna/deepfigures-open
  SCRATCH_DIR=/tmp
  ZIP_DIR=$DEEPFIGURES_RESULTS/pregenerated_training_data/377269
  YOLOV5_WORK_DIR=/tmp
fi

YOLOV5_DATA_DIR="$YOLOV5_WORK_DIR"/377269

python $SOURCE_CODE/temporary/convert_data_for_yolov5.py \
  --zip_dir="$ZIP_DIR" \
  --tmp_dir="$SCRATCH_DIR" \
  --output_dir="$YOLOV5_DATA_DIR" \
  --num_zips_to_process=1000 \
  --random_seed=42

echo "Job ended. Job ID: $SLURM_JOBID"
exit
