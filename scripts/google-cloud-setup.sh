#!/bin/bash
set -e

# This script creates a Google Cloud TPU and sets up a JAX and python environment

# Script files
CREATE_TPU_SCRIPT="tpu-create-vm.sh"
GET_IPS_SCRIPT="tpu-get-ips.sh"

# Log files to be created by the scripts
WORKERS_FILE="workers.txt"
HOSTS_FILE="hosts.txt"
ENV_FILE="vars.txt"

# (1) Create a TPU Pod Slice, run the setup script and write all environment variables used to create the TPU and project to a vars.txt file.
if ! bash $CREATE_TPU_SCRIPT; then
  echo "TPU creation failed. Exiting."
  exit 1
fi

# (2) Load the environment variables from the ENV_FILE

source $ENV_FILE
[ -z "$TPU_NAME" ] && printf "TPU_NAME not set. Exiting." && exit 1

# (3) Find the worker ids and external and internal IP addresses and write them to a 'hosts' and 'workers' text file

bash $GET_IPS_SCRIPT "$TPU_NAME" "$ZONE"
[ -z "$WORKERS_FILE" ] && printf "WORKERS_FILE not set. Exiting." && exit 1
[ -z "$HOSTS_FILE" ] && printf "HOSTS_FILE not set. Exiting." && exit 1

sleep 1
printf "TPU setup complete.\n"
