#!/bin/bash
set -e

# Usage bash /path/to/foo/this-script.sh <env-file>
# This script copies the directory it resides in ('foo') to the remote machine's $HOME directory.

# Check Usage
[[ "$#" -ne 1 ]] && printf "Usage: $0 ENV_FILE. Exiting." && exit 1

# Source the vars text file to extract TPU_NAME, ZONE, and REMOTE_HOME
ENV_FILE=$1
source $ENV_FILE

# Check if the required variables were set
[ -z "$TPU_NAME" ] && printf "TPU_NAME not set. Exiting." && exit 1
[ -z "$ZONE" ] && printf "ZONE not set. Exiting." && exit 1
[ -z "$REMOTE_HOME" ] && printf "REMOTE_HOME not set. Exiting." && exit 1

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
printf "Copying $SCRIPT_DIR to $TPU_NAME:$REMOTE_HOME...\n"

# Upload the directory to TPU
gcloud compute tpus tpu-vm scp --recurse $SCRIPT_DIR $TPU_NAME:$REMOTE_HOME --worker=all --zone=$ZONE
