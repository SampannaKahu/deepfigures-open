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
