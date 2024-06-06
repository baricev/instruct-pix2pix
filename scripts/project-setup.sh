#!/bin/bash
set -e

# This script sets up the project environment for training the instruct-pix2pix model on a Google Cloud TPU Pod Slice

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT_DIR=$(dirname "$SCRIPT_DIR")

# Project directories
DATASET_DIR="$PROJECT_ROOT_DIR/dataset"
JAX_DIR="$PROJECT_ROOT_DIR/jax-files"
PIPELINE_DIR="$PROJECT_ROOT_DIR/diffusers-pipeline"
TEST_IMAGES_DIR="$PROJECT_ROOT_DIR/test-images"

PROJECT_DIRS=("$DATASET_DIR" "$JAX_DIR" "$PIPELINE_DIR" "$TEST_IMAGES_DIR")
UPLOAD_SCRIPT="tpu-upload-dir.sh"
ENV_FILE="vars.txt"

# (1) Load the environment variables from the ENV_FILE
source $ENV_FILE
[ -z "$TPU_NAME" ] && printf "TPU_NAME not set. Exiting." && exit 1

# Functions

function run() {
  # Partial gcloud CLI commands to run a command on the TPU workers
  local CMD="source $REMOTE_HOME/$VENV_PATH/bin/activate && cd $REMOTE_WORKING_DIR && "
  gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --worker=all --command="$CMD $1"
}

function run_home() {
  # Partial gcloud CLI commands to run a command on the TPU workers from the home directory
  local CMD_HOME_DIR="source $REMOTE_HOME/$VENV_PATH/bin/activate && "
  gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --worker=all --command="$CMD_HOME_DIR $1"
}

# (2) Copy the training script and other necessary files to the TPU workers

for dir in "${PROJECT_DIRS[@]}"; do
    [ ! -d "$dir" ] && printf "Directory $dir does not exist. Exiting." && exit 1
    cp "$UPLOAD_SCRIPT" "$dir" # Copy upload script to any dir we expect to upload
done

bash "$DATASET_DIR/$UPLOAD_SCRIPT" "$ENV_FILE"
bash "$JAX_DIR/$UPLOAD_SCRIPT" "$ENV_FILE"
bash "$PIPELINE_DIR/$UPLOAD_SCRIPT" "$ENV_FILE"
bash "$TEST_IMAGES_DIR/$UPLOAD_SCRIPT" "$ENV_FILE"

# NOTE: To upload a directory to TPU outside of this script run:
# bash path/to/dir/upload_script.sh path/to/env_file.txt


# (3) Git clone the diffusers repository and install the required packages
run_home "git clone https://github.com/ivgtech/diffusers.git diffusers"
run_home "cd diffusers && pip3 install -e '.[dev]' "

# (4) Update the diffusers repository with the new pipeline
run_home "bash $PIPELINE_DIR/update_src.diffusers.pipelines.sh"

# (5) Build the Flax UNet model (optional as the model is available on Hugging Face Model Hub)
# run_home "python $PIPELINE_DIR/create_flax_unet_model.py"

# NOTE: Only use the following if you are creating your own dataset. Otherwise, please use cloud storage (like Google Cloud Storage, Azure Blob Storage or AWS S3).

# (6) Distribute and download the dataset
run_home "python dataset/map.py"
printf "Starting reduce step.\n"
run_home "bash dataset/reduce.sh > /dev/null 2>&1"
run_home "python dataset/shuffle.py"

# (7) Convert Parquet files to TFRecords
run_home "python dataset/tf_parquet.py"

sleep 1
printf "Project setup complete.\n"
