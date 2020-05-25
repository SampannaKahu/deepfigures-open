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

if [ "$SYSNAME" = "cascades" ]; then
  module purge
  module load gcc/7.3.0
  module load cuda/10.2.89
  module load cudnn/7.5
fi

if [ "$SYSNAME" = "newriver" ]; then
  module purge
  module load gcc/6.1a.0
  module load cuda/10.1.168
  module load cudnn/7.1
fi

current_timestamp() {
  date +"%Y-%m-%d_%H-%M-%S"
}
ts=$(current_timestamp)

if [ -z ${CUDA_VISIBLE_DEVICES+x} ]; then
  echo "CUDA_VISIBLE_DEVICES is not set. Defaulting to 0."
  CUDA_VISIBLE_DEVICES=0
fi

EXPERIMENT_NAME=debug_1
DEEPFIGURES_RESULTS=/work/cascades/sampanna/deepfigures-results
SOURCE_CODE=/home/sampanna/delete_this/deepfigures-open
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

/home/sampanna/.conda/envs/deepfigures_2/bin/python $SOURCE_CODE/vendor/tensorboxresnet/tensorboxresnet/train.py \
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
  --timestamp="$ts"

echo "Job ended. Job ID: $SLURM_JOBID"
exit
