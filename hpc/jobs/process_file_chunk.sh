#!/bin/bash -x

# Example Slurm submission script for Cascades

# NOTE: You will need to edit the Walltime, Resource Request, Queue, and Module lines
# to suit the requirements of your job. You will also, of course have to replace the example job
# commands below with those that run your job.

#### Resource Request: ####
# Cascades has the following hardware:
#   a. 190 32-core, 128 GB Intel Broadwell nodes
#   b.   4 32-core, 512 GB Intel Broadwell nodes with 2 Nvidia K80 GPU
#   c.   2 72-core,   3 TB Intel Broadwell nodes
#   d.  39 24-core, 376 GB Intel Skylake nodes with 2 Nvidia V100 GPU
#
# Resources can be requested by specifying the number of nodes, cores, memory, GPUs, etc
# Examples:
#   Request 4 cores (on any number of nodes)
#   #SBATCH --ntasks=4
#   Request exclusive access to all resources on 2 nodes
#   #SBATCH --nodes=2
#   #SBATCH --exclusive
#   Request 4 cores (on any number of nodes)
#   #SBATCH --ntasks=4
#   Request 2 nodes with 12 tasks running on each
#   #SBATCH --nodes=2
#   #SBATCH --ntasks-per-node=12
#   Request 12 tasks with 20GB memory per core
#   #SBATCH --ntasks=12
#   #SBATCH --mem-per-cpu=20G
#   Request 5 nodes and spread 50 tasks evenly across them
#   #SBATCH --nodes=5
#   #SBATCH --ntasks=50
#   #SBATCH --spread-job
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gres=gpu:2
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sampanna@vt.edu

#### Walltime ####
# Set the walltime, which is the maximum time your job can run in HH:MM:SS
# Note that if your job exceeds the walltime estimated during submission, the scheduler
# will kill it. So it is important to be conservative (i.e., to err on the high side)
# with the walltime that you include in your submission script.
#SBATCH -t 48:00:00

#### Queue ####
# Queue name. Cascades has five queues:
#   normal_q        for production jobs on all Broadwell nodes
#   largemem_q      for jobs on the two 3TB, 60-core Ivy Bridge servers
#   dev_q           for development/debugging jobs. These jobs must be short but can be large.
#   v100_normal_q   for production jobs on Skylake/V100 nodes
#   v100_dev_q      for development/debugging jobs on Skylake/V100 nodes
#SBATCH -p k80_q

#SBATCH -A waingram_lab

# Add any modules you might require. This example removes all modules and then adds
# the Intel compiler and mvapich2 MPI modules module. Use the module avail command
# to see a list of available modules.
module purge
module load singularity/3.3.0
module load parallel/20180222

# Below here enter the commands to start your job. A few examples are provided below.
# Some useful variables set by the job:
#  $SLURM_SUBMIT_DIR   Directory from which the job was submitted
#  $PBS_NODEFILE       File containing list of cores available to the job
#  $PBS_GPUFILE        File containing list of GPUs available to the job
#  $SLURM_JOBID        Job ID (e.g., 107619.master.cluster)
#  $SLURM_NTASKS       Number of cores allocated to the job
# You can run the following (inside a job) to see what environment variables are available:
#  env | grep SLURM
#
# Some useful storage locations (see ARC's Storage documentation for details):
#  $HOME     Home directory. Use for permanent files.
#  $WORK     Work directory. Use for fast I/O.
#  $TMPFS    File system set up in memory for this job. Use for very fast, small I/O
#  $TMPDIR   Local disk (hard drive) space set up for this job

current_timestamp() {
  date +"%Y-%m-%d_%H-%M-%S"
}

# Change to the directory from which the job was submitted
cd "$WORK"/deepfigures-results || exit
if [ -z ${SLURM_ARRAY_TASK_ID+x} ]; then
  echo "SLURM_ARRAY_TASK_ID is not set. Stopping the job."
  exit
fi

i=$SLURM_ARRAY_TASK_ID

CPU_IMAGE="$WORK/singularity/vt_cs_6604_digital_libraries_sha256.fcf57738a51202fe7e13350f423b920d6425db9d3de6e54f80d2c1ce31e38a5e.sif"
GPU_IMAGE="$WORK/singularity/vt_cs_6604_digital_libraries_sha256.7bfe02ec818baab43ed974653e2214868525b7e234543d7d053663ffe6af2a38.sif"
NUM_CPUS=$(lscpu | grep "CPU(s)" | head -1 | awk -F' ' '{print $2}')
NUM_CPUS_TIMES_2=$((NUM_CPUS * 2))
NUM_GPUS=$(nvidia-smi | grep -ic tesla)

echo "Starting batch $i"
echo "-------------"
echo "Number of CPUs : $NUM_CPUS"
echo "Number of GPUs : $NUM_GPUS"
echo "Singularity CPU image path: $CPU_IMAGE"
echo "Singularity GPU image path: $GPU_IMAGE"

# Copy the model weights generated by the previous iteration to a safe location.
if ls "$WORK"/deepfigures-results/pmctable_arxiv_combined* 1>/dev/null 2>&1; then
  echo "Previous model exists."
  LATEST_MODEL_DIR=$(ls -dt $WORK/deepfigures-results/pmctable_arxiv_combined* | head -1)

  # Copying the weights (meta) from the previous checkpoint to the weights directory.
  LATEST_META_FILE=$(ls -t "$LATEST_MODEL_DIR/*.meta" | head -1)
  echo "Copying the weights from $LATEST_META_FILE to $WORK/deepfigures-results/weights/save.ckpt-500000.meta"
  rm -f "$WORK/deepfigures-results/weights/*.meta"
  [ -f "$LATEST_META_FILE" ] && cp "$LATEST_META_FILE" "$WORK"/deepfigures-results/weights/save.ckpt-500000.meta

  # Copying the weights (index) from the previous checkpoint to the weights directory.
  LATEST_INDEX_FILE=$(ls -t "$LATEST_MODEL_DIR/*.index" | head -1)
  echo "Copying the weights from $LATEST_INDEX_FILE to $WORK/deepfigures-results/weights/save.ckpt-500000.index"
  rm -f "$WORK/deepfigures-results/weights/*.index"
  [ -f "$LATEST_INDEX_FILE" ] && cp "$LATEST_INDEX_FILE" "$WORK"/deepfigures-results/weights/save.ckpt-500000.index

  # Copying the weights (data) from the previous checkpoint to the weights directory.
  LATEST_DATA_FILE=$(ls -t "$LATEST_MODEL_DIR/*.data*" | head -1)
  echo "Copying the weights from $LATEST_DATA_FILE to $WORK/deepfigures-results/weights/save.ckpt-500000.data-00000-of-00001"
  rm -f "$WORK"/deepfigures-results/weights/*.data*
  [ -f "$LATEST_DATA_FILE" ] && cp "$LATEST_DATA_FILE" "$WORK"/deepfigures-results/weights/save.ckpt-500000.data-00000-of-00001

  echo "Moving the previous model to the checkpoints directory. Model name: $LATEST_MODEL_DIR"
  mv "$LATEST_MODEL_DIR" "$WORK"/deepfigures-results/model_checkpoints
  echo "Previous model moved successfully."

  echo "Moving the figure jsons to the model checkpoint directory of the corresponding model."
  [ -f "$WORK"/deepfigures-results/figure_boundaries.json ] && mv "$WORK"/deepfigures-results/figure_boundaries.json "$LATEST_MODEL_DIR"
  [ -f "$WORK"/deepfigures-results/figure_boundaries_train.json ] && mv "$WORK"/deepfigures-results/figure_boundaries_train.json "$LATEST_MODEL_DIR"
  [ -f "$WORK"/deepfigures-results/figure_boundaries_test.json ] && mv "$WORK"/deepfigures-results/figure_boundaries_test.json "$LATEST_MODEL_DIR"
else
  echo "No previous model found."
  rm -f "$WORK"/deepfigures-results/figure_boundaries.json
  rm -f "$WORK"/deepfigures-results/figure_boundaries_train.json
  rm -f "$WORK"/deepfigures-results/figure_boundaries_test.json
fi

# clean up the files generated by the previous iterations.
echo "Zipping $WORK/deepfigures-results/arxiv_data_output and $WORK/deepfigures-results/arxiv_data_output using GNU parallel."
rm -f "$WORK"/deepfigures-results/to_be_zipped.txt
ls -d "$WORK"/deepfigures-results/arxiv_data_output/diffs_100dpi/* "$WORK"/deepfigures-results/arxiv_data_output/figure-jsons/* "$WORK"/deepfigures-results/arxiv_data_output/modified_src/* "$WORK"/deepfigures-results/arxiv_data_output/src/* "$WORK"/deepfigures-results/arxiv_data_temp/* >"$WORK"/deepfigures-results/to_be_zipped.txt
#parallel -j "$NUM_CPUS" --progress --no-notice -a "$WORK"/deepfigures-results/to_be_zipped.txt rm -rf
parallel -j "$NUM_CPUS_TIMES_2" --progress --no-notice -a "$WORK"/deepfigures-results/to_be_zipped.txt 'var="{}"; zip -rmq "$var".zip $var'
cho "Again calling the zip -rm command to do the final packing."
ts=$(current_timestamp)
zip -rm "$WORK"/deepfigures-results/arxiv_data_temp_"$i"_"$ts".zip "$WORK"/deepfigures-results/arxiv_data_temp
zip -rm "$WORK"/deepfigures-results/arxiv_data_temp_"$i"_"$ts".zip "$WORK"/deepfigures-results/arxiv_data_output
echo "Again calling the rm -rf command just to be sure."
rm -rf "$WORK"/deepfigures-results/arxiv_data_temp/*
rm -rf "$WORK"/deepfigures-results/arxiv_data_output/*
echo "Creating output and temp dirs just to be sure."
mkdir -p "$WORK"/deepfigures-results/arxiv_data_temp
mkdir -p "$WORK"/deepfigures-results/arxiv_data_output

# Preparing the list of files for training.
echo "Copying /home/sampanna/df/files_$i.json to $WORK/deepfigures-results/files_$i.json"
cp /home/sampanna/df/files_"$i".json "$WORK"/deepfigures-results/files_"$i".json
echo "Renaming $WORK/deepfigures-results/files_0.json to $WORK/deepfigures-results/files.json"
cp "$WORK"/deepfigures-results/files_0.json "$WORK"/deepfigures-results/files.json

# Generate the data.
singularity exec -B "$WORK"/deepfigures-results:/work/host-output -B "$WORK"/deepfigures-results:/work/host-input "$CPU_IMAGE" python /work/deepfigures/data_generation/arxiv_pipeline.py

# Prepare the figure_boundaries.json file.
singularity exec -B "$WORK"/deepfigures-results:/work/host-output -B "$WORK"/deepfigures-results:/work/host-input "$CPU_IMAGE" python /work/figure_json_transformer.py
singularity exec -B "$WORK"/deepfigures-results:/work/host-output -B "$WORK"/deepfigures-results:/work/host-input "$CPU_IMAGE" python /work/figure_boundaries_train_test_split.py

# Trigger the training.
# The timeout command will kill the training after certain time.
timeout --preserve-status 12h singularity exec --nv -B "$WORK"/deepfigures-results:/work/host-output -B "$WORK"/deepfigures-results:/work/host-input "$GPU_IMAGE" python /work/train.py --hypes /work/host-input/weights/hypes.json --gpu "$CUDA_VISIBLE_DEVICES" --logdir /work/host-output

exit
