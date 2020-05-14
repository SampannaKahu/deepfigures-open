sbatch --array=0-26%1 process_file_chunk.sh

sudo ./ir_job.sh 2>&1 | tee -a output.log



## For pregenerating training data

# Default command
sbatch --array=0-39 pregenerate_data.sh

# For jobs that timed out
sbatch --array=0,4,5,6,8,12,14,16,17,20 pregenerate_data.sh