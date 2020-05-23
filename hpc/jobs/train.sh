#!/bin/bash -x

#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sampanna@vt.edu
#SBATCH -t 143:59:59
#SBATCH -p v100_normal_q
#SBATCH -A waingram_lab

module purge
module load gcc/7.3.0
module load cuda/10.2.89
module load cudnn/7.5

current_timestamp() {
  date +"%Y-%m-%d_%H-%M-%S"
}
ts=$(current_timestamp)

if [ -z ${CUDA_VISIBLE_DEVICES+x} ]; then
  echo "CUDA_VISIBLE_DEVICES is not set. Defaulting to 0."
  CUDA_VISIBLE_DEVICES=0
fi

WEIGHTS_PATH=/home/sampanna/deepfigures-results/weights/save.ckpt-500000
HYPES_PATH=/home/sampanna/deepfigures-open/models/sample_hypes.json
MAX_ITER=10000000
LOG_DIR=/home/sampanna/deepfigures-results
EXPERIMENT_NAME=arxiv_experiment
TRAIN_IDL_PATH=/home/sampanna/deepfigures-results/figure_boundaries_train.json
TRAIN_IMAGES_DIR=/home/sampanna/deepfigures-results/images
TEST_IDL_PATH=/home/sampanna/deepfigures-results/figure_boundaries_test.json
TEST_IMAGES_DIR="$TRAIN_IMAGES_DIR"
MAX_CHECKPOINTS_TO_KEEP=100

/home/sampanna/.conda/envs/deepfigures/bin/python /home/sampanna/deepfigures-open/vendor/tensorboxresnet/tensorboxresnet/train.py \
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
