#!/bin/bash
set -e

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

ENV_FILE="vars.txt"
source "$SCRIPTS_DIR/$ENV_FILE"

[ -z "$TPU_NAME" ] && {
  echo "Error: Expected TPU_NAME to be set"
  exit 1
}

# FUNCTIONS

function run() {
  # Partial gcloud CLI command
  local CMD="source $REMOTE_HOME/$VENV_PATH/bin/activate && cd $REMOTE_WORKING_DIR && "
  gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --worker=all --command="$CMD $1"
}

function copy_to() {
  # Copy a file or directory to the TPU
  local source=$1
  local dest=$2
  gcloud compute tpus tpu-vm scp --recurse "$source" $TPU_NAME:"$dest" --worker=all --zone=$ZONE
}

function copy_images() {
  PARENT_DIR=$(dirname "$SCRIPTS_DIR")
  local IMAGES_DIR="$PARENT_DIR/images"
  local HOSTS_FILE="$SCRIPTS_DIR/hosts.txt"
  local N_THREADS=16
  mkdir -p "$IMAGES_DIR"

  cat $HOSTS_FILE | parallel -j $N_THREADS scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null  -r -i /Users/v/.ssh/google_compute_engine "v@{}:$REMOTE_IMAGES_DIR/*" $IMAGES_DIR/ # recursively copy all subdirectories

}

export -f run
export -f copy_to
export -f copy_images
