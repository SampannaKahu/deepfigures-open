#!/bin/bash -x

#PBS -l nodes=1:ppn=28:gpus=1
#PBS -l walltime=143:59:59
#PBS -q p100_normal_q
#PBS -A waingram_lab
#PBS -W group_list=newriver
#PBS -M sampanna@vt.edu
#PBS -m bea

#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sampanna@vt.edu
#SBATCH -t 143:59:59
#SBATCH -p v100_normal_q
#SBATCH -A waingram_lab

EXPERIMENT_NAME=debug_2

current_timestamp() {
  date +"%Y-%m-%d_%H-%M-%S"
}
ts=$(current_timestamp)

#if [ -z ${CUDA_VISIBLE_DEVICES+x} ]; then
#  echo "CUDA_VISIBLE_DEVICES is not set. Defaulting to 0."
#  CUDA_VISIBLE_DEVICES=0
#fi

if [ "$HOSTNAME" = "ir.cs.vt.edu" ]; then
  PYTHON=/home/sampanna/anaconda3/envs/deepfigures_3/bin/python
  DEEPFIGURES_RESULTS=/home/sampanna/deepfigures-results
  SOURCE_CODE=/home/sampanna/deepfigures-open
  CUDA_VISIBLE_DEVICES=0
  SCRATCH_DIR=/tmp
  ZIP_DIR=$DEEPFIGURES_RESULTS/pregenerated_training_data/377266
elif [ "$SYSNAME" = "cascades" ]; then
  module purge
  module load gcc/7.3.0
  module load cuda/9.0.176
  module load cudnn/7.1
  PYTHON=/home/sampanna/.conda/envs/deepfigures_3/bin/python
  DEEPFIGURES_RESULTS=/work/cascades/sampanna/deepfigures-results
  SOURCE_CODE=/home/sampanna/delete_this/deepfigures-open
  SCRATCH_DIR=$TMPRAM # 311 GB on v100 nodes. 331 MBPS.
  ZIP_DIR=$DEEPFIGURES_RESULTS/pregenerated_training_data/377266
elif [ "$SYSNAME" = "newriver" ]; then
  module purge
  module load gcc/6.1a.0
  module load cuda/9.0.176
  module load cudnn/7.1
  PYTHON=/home/sampanna/.conda/envs/deepfigures_3/bin/python
  DEEPFIGURES_RESULTS=/work/cascades/sampanna/deepfigures-results
  SOURCE_CODE=/home/sampanna/delete_this/deepfigures-open
  SCRATCH_DIR=$TMPFS # 429 GB on p100 nodes. 770 MBPS.
  ZIP_DIR=$DEEPFIGURES_RESULTS/pregenerated_training_data/377266
elif [ "$HOSTNAME" = "xps15" ]; then
  PYTHON=/home/sampanna/anaconda3/envs/deepfigures_3/bin/python
  DEEPFIGURES_RESULTS=/home/sampanna/workspace/bdts2/deepfigures-results
  SOURCE_CODE=/home/sampanna/workspace/bdts2/deepfigures-open
  CUDA_VISIBLE_DEVICES=0
  SCRATCH_DIR=/tmp
  ZIP_DIR=$DEEPFIGURES_RESULTS/pregenerated_training_data/377266
else
  PYTHON=/home/sampanna/anaconda3/envs/deepfigures_3/bin/python
  DEEPFIGURES_RESULTS=/home/sampanna/deepfigures-results
  SOURCE_CODE=/home/sampanna/deepfigures-open
  CUDA_VISIBLE_DEVICES=0
  SCRATCH_DIR=/tmp
  ZIP_DIR=$DEEPFIGURES_RESULTS/pregenerated_training_data/377266
fi

WEIGHTS_PATH=$DEEPFIGURES_RESULTS/weights/save.ckpt-500000
HYPES_PATH=$SOURCE_CODE/models/sample_hypes.json
MAX_ITER=10000000
LOG_DIR=$DEEPFIGURES_RESULTS/model_checkpoints
DATASET_DIR=$DEEPFIGURES_RESULTS/arxiv_coco_dataset
TRAIN_IDL_PATH=$DATASET_DIR/figure_boundaries_train.json
TRAIN_IMAGES_DIR=$DATASET_DIR/images
TEST_IDL_PATH=$DATASET_DIR/figure_boundaries_test.json
TEST_IMAGES_DIR=$DATASET_DIR/images
MAX_CHECKPOINTS_TO_KEEP=100
TEST_SPLIT_PERCENT=20

$PYTHON $SOURCE_CODE/vendor/tensorboxresnet/tensorboxresnet/train.py \
  --weights "$WEIGHTS_PATH" \
  --gpu="$CUDA_VISIBLE_DEVICES" \
  --hypes="$HYPES_PATH" \
  --max_iter="$MAX_ITER" \
  --logdir="$LOG_DIR" \
  --experiment_name="$EXPERIMENT_NAME" \
  --train_idl_path="$TRAIN_IDL_PATH" \
  --test_idl_path="$TEST_IDL_PATH" \
  --train_images_dir="$TRAIN_IMAGES_DIR" \
  --test_images_dir="$TEST_IMAGES_DIR" \
  --max_checkpoints_to_keep="$MAX_CHECKPOINTS_TO_KEEP" \
  --timestamp="$ts" \
  --scratch_dir="$SCRATCH_DIR" \
  --zip_dir="$ZIP_DIR" \
  --test_split_percent="$TEST_SPLIT_PERCENT"

echo "Job ended. Job ID: $SLURM_JOBID"
exit
