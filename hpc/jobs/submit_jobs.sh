sbatch --array=0-26%1 process_file_chunk.sh

sudo ./ir_job.sh 2>&1 | tee -a output.log

## For pregenerating training data

# Default command
sbatch --array=0-39 pregenerate_data.sh

# For jobs that timed out (from 0 to 23)
sbatch --array=0,4,5,6,8,12,14,16,17,20 pregenerate_data.sh

# For jobs that timed out (from 24 to 39)
sbatch --array=24,25,28,31,32,33,34,35,37,38,39 pregenerate_data.sh

# For jobs that faced the stale file handle error
sbatch --array=8,16,20 pregenerate_data.sh

# For debugging training_data_generator
DELETE_THIS="/tmp/delete_this"
mkdir -p $DELETE_THIS
i=0
ZIP_SAVE_DIR=$DELETE_THIS
ZIP_DEST_DIR="$DELETE_THIS"/dest
NUM_CPUS_TIMES_2=1
WORK_DIR_PREFIX=$DELETE_THIS
ARXIV_DATA_TEMP=$DELETE_THIS
DOWNLOAD_CACHE=$DELETE_THIS
ARXIV_DATA_OUTPUT=$DELETE_THIS
mkdir -p $ZIP_DEST_DIR
cat ~/deepfigures-open/hpc/files_random_40/files_"$i".json | grep tar | awk -F '/' '{print $3"_"$4"_"$5}' | awk -F '"' '{print "/work/cascades/sampanna/deepfigures-results/download_cache/"$1}' | xargs cp -t "$DOWNLOAD_CACHE"
/home/sampanna/.conda/envs/deepfigures/bin/python /home/sampanna/deepfigures-open/deepfigures/data_generation/training_data_generator.py \
  --file_list_json /home/sampanna/deepfigures-open/hpc/files_random_40/files_"$i".json \
  --images_per_zip=2 \
  --zip_save_dir="$ZIP_SAVE_DIR" \
  --zip_dest_dir="$ZIP_DEST_DIR" \
  --n_cpu=$NUM_CPUS_TIMES_2 \
  --work_dir_prefix "$WORK_DIR_PREFIX" \
  --arxiv_tmp_dir "$ARXIV_DATA_TEMP" \
  --arxiv_cache_dir "$DOWNLOAD_CACHE" \
  --arxiv_data_output_dir "$ARXIV_DATA_OUTPUT" \
  --delete_tar_after_extracting True \
  --augment_typewriter_font True \
  --augment_line_spacing_1_5 True

# For training the model
# cascades
sbatch --ignore-pbs train.sh
# newriver
qsub -k oe train.sh
