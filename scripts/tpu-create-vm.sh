#!/bin/bash
set -e

# Script to setup and initialize TPU resources on Google Cloud.

# (1) Variables for resource identification

export PROJECT_ID="??"
export TPU_PREFIX="??"
export TPU_TYPE="??"
export ZONE="us-central2-b"
export VERSION="tpu-ubuntu2204-base"

DATE_UPPERCASE=$(date +'%B-%d')         # NOTE: Resource names should be all lowercase
export DATE=${DATE_UPPERCASE,,}         # Convert to all lowercase (",," is a bash lowercase command)
export TPU_NAME="${TPU_PREFIX}-${DATE}" # Add month and day as unique TPU ID
export QR_ID="${TPU_NAME}"
export QR_TYPE="--best-effort"

export GCE_PRIVATE_KEY="$HOME/.ssh/google_compute_engine"
export PUBLIC_KEY="$HOME/.ssh/id_rsa.pub"

export REMOTE_USER="v"
export REMOTE_HOME="/home/v"
export VENV_PATH="venv"
export STARTUP_SCRIPT="tpu-setup.sh"
export ENV_FILE="vars.txt"

export WORKING_DIR="diffusers/examples/instruct_pix2pix"
export IMAGES_DIR="images"
export TENSORBOARD_LOGS_DIR="tensorboard_logs"

# Allow for time to abort TPU creation
echo -e "Creating TPU_NAME=$TPU_NAME on $TPU_TYPE in $ZONE\n"
sleep 5

# Write resource identification variables to a text file for use with other scripts or on the command line
save_vars_to() {
  {
    echo 'export STARTUP_SCRIPT="'$STARTUP_SCRIPT'"'
    echo 'export PROJECT_ID="'$PROJECT_ID'"'
    echo 'export TPU_PREFIX="'$TPU_PREFIX'"'
    echo 'export ZONE="'$ZONE'"'
    echo 'export TPU_TYPE="'$TPU_TYPE'"'
    echo 'export VERSION="'$VERSION'"'
    echo 'export DATE="'$DATE'"'
    echo 'export TPU_NAME="'$TPU_NAME'"'
    echo 'export QR_ID="'$QR_ID'"'
    echo 'export QR_TYPE="'$QR_TYPE'"'

    echo 'export REMOTE_USER="'$REMOTE_USER'"'
    echo 'export GCE_PRIVATE_KEY="'$GCE_PRIVATE_KEY'"'
    echo 'export PUBLIC_KEY="'$PUBLIC_KEY'"'
    echo 'export HOME="'$HOME'"'
    echo 'export REMOTE_HOME="'$REMOTE_HOME'"'
    echo 'export VENV_PATH="'$VENV_PATH'"'
    echo 'export ENV_FILE="'$ENV_FILE'"'

    echo 'export WORKING_DIR="'$WORKING_DIR'"'
    echo 'export IMAGES_DIR="'$IMAGES_DIR'"'
    echo 'export TENSORBOARD_LOGS_DIR="'$TENSORBOARD_LOGS_DIR'"'
    echo 'export REMOTE_WORKING_DIR="'$REMOTE_HOME/$WORKING_DIR'"'
    echo 'export REMOTE_IMAGES_DIR="'$REMOTE_HOME/$IMAGES_DIR'"'
    echo 'export REMOTE_TENSORBOARD_LOGS_DIR="'$REMOTE_HOME/$TENSORBOARD_LOGS_DIR'"'
  } >$1
}

# Save variables to a text file for use with other scripts or on the command line
save_vars_to "$ENV_FILE"

# Function to check TPU VM readiness
check_tpu_ready() {
  echo "$TPU_NAME"
  # gcloud compute tpus describe "${tpu_name}" --zone "${ZONE}" | grep -q 'HEALTHY'
  gcloud compute tpus queued-resources list --zone "${ZONE}" | grep -q "${TPU_NAME}.*ACTIVE"
  return $?
}

# Function to create an on-demand queued resource (for preemptible use: --best-effort. On-demand is the default: no flags required.)
create_tpu_resource_fn() { # NOTE: this is an async command
  local cmd="gcloud alpha compute tpus queued-resources create ${TPU_NAME} \
    --node-id=${QR_ID} \
    --zone=${ZONE} \
    --accelerator-type=${TPU_TYPE} \
    --runtime-version=${VERSION}\
    ${QR_TYPE}"

  printf "Running command: %s\n" "$cmd"
  while ! $cmd; do
    printf "Error creating VM, retrying in 15 seconds...\n"
    sleep 15
  done
}

# (2) Create TPUs        #<=========
create_tpu_resource_fn   #<=========

printf "Waiting for the VM to become ready...\n"
while ! check_tpu_ready "${TPU_NAME}"; do
  printf "VM not ready, checking again in 60 seconds...\n"
  sleep 60
done

printf "VM is now ready. Proceeding with setup...\n"

# (3) Add SSH keys
printf "Adding SSH keys...\n"
ssh-add $GCE_PRIVATE_KEY

# (4) Add Public Key to all TPU VMs using the gcloud command
gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --worker=all --command="echo $(cat $PUBLIC_KEY) >> ~/.ssh/authorized_keys"

# (5) Run TPU Setup Script
# Copy the startup script to the TPU VM and execute it
gcloud compute tpus tpu-vm scp "${STARTUP_SCRIPT}" "${TPU_NAME}:~/${STARTUP_SCRIPT}" --zone="${ZONE}" --worker=all
gcloud compute tpus tpu-vm ssh "${QR_ID}" --zone="${ZONE}" --worker=all --command="bash ${STARTUP_SCRIPT}"
printf "Setup script executed successfully.\n"

# (6) VM IPs
printf "\n"
printf "To setup this vm's ips and configure ssh use:\n"
printf "bash tpu-get-ips.sh ${TPU_NAME} ${ZONE} \n\n"

# (7) Deletion using the web UI does not remove the associated QR
printf "To delete this tpu vm and queued-resource use:\n"
printf "gcloud alpha compute tpus queued-resources delete $QR_ID  --project $PROJECT_ID  --zone us-central2-b  --force  --async\n\n"
