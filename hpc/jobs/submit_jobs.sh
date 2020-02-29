sbatch --array=0-26%1 process_file_chunk.sh

./ir_job.sh 2>&1 | tee -a output.log