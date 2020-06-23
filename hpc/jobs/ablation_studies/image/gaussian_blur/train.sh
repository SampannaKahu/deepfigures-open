#!/bin/bash -x

#PBS -l nodes=1:ppn=10:gpus=1
#PBS -l walltime=72:00:00
#PBS -q p100_normal_q
#PBS -A waingram_lab
#PBS -W group_list=newriver
#PBS -M sampanna@vt.edu
#PBS -m bea

#SBATCH --nodes=1
#SBATCH --ntasks=10
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sampanna@vt.edu
#SBATCH -t 72:00:00
#SBATCH -p v100_normal_q
#SBATCH -A waingram_lab

EXPERIMENT_NAME=377269_0_to_4_arxiv_ablation_all_latex_transformations_enabled_gaussian_blur_excluded
CONDA_ENV=deepfigures_3

current_timestamp() {
  date +"%Y-%m-%d_%H-%M-%S"
}
ts=$(current_timestamp)

if [ "$HOSTNAME" = "ir.cs.vt.edu" ]; then
  PYTHON=/home/sampanna/anaconda3/envs/"$CONDA_ENV"/bin/python
  conda activate "$CONDA_ENV"
  DEEPFIGURES_RESULTS=/home/sampanna/deepfigures-results
  SOURCE_CODE=/home/sampanna/deepfigures-open
  CUDA_VISIBLE_DEVICES=0
  SCRATCH_DIR=/tmp
  ZIP_DIR=$DEEPFIGURES_RESULTS/pregenerated_training_data/377269_0_to_4
elif [ "$SYSNAME" = "cascades" ]; then
  module purge
  module load Anaconda/5.1.0
  module load gcc/7.3.0
  module load cuda/9.0.176
  module load cudnn/7.1
  PYTHON=/home/sampanna/.conda/envs/"$CONDA_ENV"/bin/python
  source activate "$CONDA_ENV"
  DEEPFIGURES_RESULTS=/work/cascades/sampanna/deepfigures-results
  SOURCE_CODE=/home/sampanna/deepfigures-open
  SCRATCH_DIR=$TMPRAM # 311 GB on v100 nodes. 331 MBPS.
  ZIP_DIR=$DEEPFIGURES_RESULTS/pregenerated_training_data/377269_0_to_4
elif [ "$SYSNAME" = "newriver" ]; then
  module purge
  module load Anaconda/5.2.0
  module load gcc/6.1a.0
  module load cuda/9.0.176
  module load cudnn/7.1
  PYTHON=/home/sampanna/.conda/envs/"$CONDA_ENV"/bin/python
  source activate "$CONDA_ENV"
  DEEPFIGURES_RESULTS=/work/cascades/sampanna/deepfigures-results
  SOURCE_CODE=/home/sampanna/deepfigures-open
  SCRATCH_DIR=$TMPFS # 429 GB on p100 nodes. 770 MBPS.
  ZIP_DIR=$DEEPFIGURES_RESULTS/pregenerated_training_data/377269_0_to_4
elif [ "$HOSTNAME" = "xps15" ]; then
  PYTHON=/home/sampanna/anaconda3/envs/"$CONDA_ENV"/bin/python
  conda activate "$CONDA_ENV"
  DEEPFIGURES_RESULTS=/home/sampanna/workspace/bdts2/deepfigures-results
  SOURCE_CODE=/home/sampanna/workspace/bdts2/deepfigures-open
  CUDA_VISIBLE_DEVICES=0
  SCRATCH_DIR=/tmp
  ZIP_DIR=$DEEPFIGURES_RESULTS/pregenerated_training_data/377269_0_to_4
else
  PYTHON=/home/sampanna/anaconda3/envs/"$CONDA_ENV"/bin/python
  conda activate "$CONDA_ENV" || source activate "$CONDA_ENV"
  DEEPFIGURES_RESULTS=/home/sampanna/deepfigures-results
  SOURCE_CODE=/home/sampanna/deepfigures-open
  CUDA_VISIBLE_DEVICES=0
  SCRATCH_DIR=/tmp
  ZIP_DIR=$DEEPFIGURES_RESULTS/pregenerated_training_data/377269_0_to_4
fi

WEIGHTS_PATH=$DEEPFIGURES_RESULTS/weights/save.ckpt-500000
HYPES_PATH=$SOURCE_CODE/hpc/jobs/ablation_studies/image/gaussian_blur/sample_hypes.json
MAX_ITER=20000
LOG_DIR=$DEEPFIGURES_RESULTS/model_checkpoints
DATASET_DIR=$DEEPFIGURES_RESULTS/arxiv_coco_dataset
TRAIN_IDL_PATH=$DATASET_DIR/figure_boundaries_train.json
TRAIN_IMAGES_DIR=$DATASET_DIR/images
TEST_IDL_PATH=$DATASET_DIR/figure_boundaries_test.json
TEST_IMAGES_DIR=$DATASET_DIR/images
GOLD_STANDARD_DATASET_DIR=$DEEPFIGURES_RESULTS/gold_standard_dataset
HIDDEN_IDL_PATH=$GOLD_STANDARD_DATASET_DIR/figure_boundaries.json
HIDDEN_IMAGES_DIR=$GOLD_STANDARD_DATASET_DIR/images
MAX_CHECKPOINTS_TO_KEEP=200
TEST_SPLIT_PERCENT=20
USE_GLOBAL_STEP_FOR_LR=False

$PYTHON -m tensorboxresnet.train \
  --weights "$WEIGHTS_PATH" \
  --gpu="$CUDA_VISIBLE_DEVICES" \
  --hypes="$HYPES_PATH" \
  --max_iter="$MAX_ITER" \
  --logdir="$LOG_DIR" \
  --experiment_name="$EXPERIMENT_NAME" \
  --train_idl_path="$TRAIN_IDL_PATH" \
  --test_idl_path="$TEST_IDL_PATH" \
  --hidden_idl_path="$HIDDEN_IDL_PATH" \
  --train_images_dir="$TRAIN_IMAGES_DIR" \
  --test_images_dir="$TEST_IMAGES_DIR" \
  --hidden_images_dir="$HIDDEN_IMAGES_DIR" \
  --max_checkpoints_to_keep="$MAX_CHECKPOINTS_TO_KEEP" \
  --timestamp="$ts" \
  --scratch_dir="$SCRATCH_DIR" \
  --zip_dir="$ZIP_DIR" \
  --test_split_percent="$TEST_SPLIT_PERCENT" \
  --use_global_step_for_lr "$USE_GLOBAL_STEP_FOR_LR"

echo "Job ended. Job ID: $SLURM_JOBID $PBS_JOBID"
exit
