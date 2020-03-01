#!/bin/bash -x

current_timestamp() {
  date +"%Y-%m-%d_%H-%M-%S"
}

CPU_IMAGE="sampyash/vt_cs_6604_digital_libraries@sha256:4f11459fb5e6df40fecc08a1a15f4d49fb061604c5898b59d1fab21925bce5d8"
GPU_IMAGE="sampyash/vt_cs_6604_digital_libraries@sha256:eadba541198726e02750586232eb0498bc5df7f307da53ed86375da6bf29a37f"
NUM_CPUS=$(lscpu | grep "CPU(s)" | head -1 | awk -F' ' '{print $2}')
NUM_CPUS_TIMES_2=$((NUM_CPUS * 2))
NUM_GPUS=1

#conda init bash
#conda activate deepfigures

WORK=/home/sampanna

# Create the necessary directories, if they do not exist.
mkdir -p "$WORK"/deepfigures-results/arxiv_data_output
mkdir -p "$WORK"/deepfigures-results/arxiv_data_temp
mkdir -p "$WORK"/deepfigures-results/model_checkpoints
mkdir -p "$WORK"/deepfigures-results/weights

# Start the iterations.
for i in {0..26}; do
  echo ""
  echo ""
  echo "Starting batch $i"
  echo "-------------"
  echo "Number of CPUs : $NUM_CPUS"
  echo "Number of GPUs : $NUM_GPUS"
  echo "CPU image path: $CPU_IMAGE"
  echo "GPU image path: $GPU_IMAGE"

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

    echo "Moving the figure jsons to the model checkpoint directory of the corresponding model."
    [ -f "$WORK"/deepfigures-results/figure_boundaries.json ] && mv "$WORK"/deepfigures-results/figure_boundaries.json "$LATEST_MODEL_DIR"
    [ -f "$WORK"/deepfigures-results/figure_boundaries_train.json ] && mv "$WORK"/deepfigures-results/figure_boundaries_train.json "$LATEST_MODEL_DIR"
    [ -f "$WORK"/deepfigures-results/figure_boundaries_test.json ] && mv "$WORK"/deepfigures-results/figure_boundaries_test.json "$LATEST_MODEL_DIR"

    echo "Moving the previous model to the checkpoints directory. Model name: $LATEST_MODEL_DIR"
    mv "$LATEST_MODEL_DIR" "$WORK"/deepfigures-results/model_checkpoints
    echo "Previous model moved successfully."
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
  echo "Again calling the zip -rm command to do the final packing."
  ts=$(current_timestamp)
  zip -rm "$WORK"/deepfigures-results/arxiv_data_temp_"$i"_"$ts".zip "$WORK"/deepfigures-results/arxiv_data_temp
  zip -rm "$WORK"/deepfigures-results/arxiv_data_output_"$i"_"$ts".zip "$WORK"/deepfigures-results/arxiv_data_output
  scp "$WORK"/deepfigures-results/arxiv_data_temp_"$i"_"$ts".zip cascades2.arc.vt.edu:~/ir_backup
  scp "$WORK"/deepfigures-results/arxiv_data_output_"$i"_"$ts".zip cascades2.arc.vt.edu:~/ir_backup
  echo "Again calling the rm -rf command just to be sure."
  rm -rf "$WORK"/deepfigures-results/arxiv_data_temp/*
  rm -rf "$WORK"/deepfigures-results/arxiv_data_output/*
  echo "Creating output and temp dirs just to be sure."
  mkdir -p "$WORK"/deepfigures-results/arxiv_data_temp
  mkdir -p "$WORK"/deepfigures-results/arxiv_data_output
  echo "Cleaning up the download_cache"
  rm -rf "$WORK"/deepfigures-results/download_cache
  mkdir -p "$WORK"/deepfigures-results/download_cache

  # Preparing the list of files for training.
  echo "Copying /home/sampanna/df/files_$i.json to $WORK/deepfigures-results/files.json"
  cp /home/sampanna/df/files_"$i".json "$WORK"/deepfigures-results/files.json

  # Download the files over SSH.
  cat "$WORK"/deepfigures-results/files.json | grep arXiv | awk -F'.' '{print $1}' | awk -F'/' '{print "arxiv_src_"$5".tar"}' >/tmp/rsync.txt
  rsync -Pauv -e 'ssh -i sshkey' --files-from=/tmp/rsync.txt cascades2.arc.vt.edu:/work/cascades/sampanna/deepfigures-results/download_cache/ "$WORK"/deepfigures-results/download_cache
  rm -f /tmp/rsync.txt

  # Generate the data.
  docker run --gpus all -it --volume "$WORK"/deepfigures-results:/work/host-output --volume "$WORK"/deepfigures-results:/work/host-input $CPU_IMAGE python deepfigures/data_generation/arxiv_pipeline.py

  # Prepare the figure_boundaries.json file.
  docker run --gpus all -it --volume "$WORK"/deepfigures-results:/work/host-output --volume "$WORK"/deepfigures-results:/work/host-input $CPU_IMAGE python figure_json_transformer.py
  docker run --gpus all -it --volume "$WORK"/deepfigures-results:/work/host-output --volume "$WORK"/deepfigures-results:/work/host-input $CPU_IMAGE python figure_boundaries_train_test_split.py

  # Trigger the training.
  # The timeout command will kill the training after certain time.
  timeout --preserve-status 12h docker run --gpus all -it --volume "$WORK"/deepfigures-results:/work/host-output --volume "$WORK"/deepfigures-results:/work/host-input $GPU_IMAGE python /work/vendor/tensorboxresnet/tensorboxresnet/train.py --hypes /work/host-input/weights/hypes.json --gpu 0 --logdir /work/host-output

done
