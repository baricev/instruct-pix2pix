#!/bin/bash
set -e

# Check
if [[ "$#" -ne 2 ]]; then # echo $# returns number of args used with script
  printf "Usage: $0 \$TPU_NAME \$ZONE \n"
  printf "Exiting...\n"
  exit 1
fi
# Export TPU_NAME and ZONE
TPU_NAME=$1
ZONE=$2

# Script files
WHOAMI_SCRIPT="tpu-whoami.sh"
# Files to be created by the script
TPU_HOSTS_FILE="hosts.txt"
TPU_WORKERS_FILE="workers.txt"

# (1) Remove the workers and hosts files if they exist
rm -f $TPU_WORKERS_FILE $TPU_HOSTS_FILE

# (2) Fetch TPU VM details using the gcloud CLI
printf "Finding external and internal IP addresses for %s ... " "$TPU_NAME"
output=$(gcloud compute tpus tpu-vm describe --zone "$ZONE" "$TPU_NAME")
readarray -t external_ips < <(echo "$output" | awk '/externalIp:/ { print $2 }')
readarray -t internal_ips < <(echo "$output" | awk '/ipAddress:/ { print $2 }')
printf "done.\n"

# (3) Extract the first IP from the external_ips array
EXTERNAL_IP=${external_ips[0]}

# (4) Write the external and internal IPs to a workers text file
printf "Writing $TPU_WORKERS_FILE file..."
paste <(printf "%s\n" "${external_ips[@]}") <(printf "%s\n" "${internal_ips[@]}") >"$TPU_WORKERS_FILE"
printf "done.\n"

# (5) Write a 'hosts' text file
printf "Writing $TPU_HOSTS_FILE file..."
paste <(printf "%s\n" "${external_ips[@]}") >"$TPU_HOSTS_FILE"
printf "done.\n"

# (6) Copy workers text and whoami script file to all remote machines
printf "Copying $TPU_WORKERS_FILE and $WHOAMI_SCRIPT to all devices..."
gcloud compute tpus tpu-vm scp $TPU_WORKERS_FILE $WHOAMI_SCRIPT $TPU_NAME:~/ --worker=all --zone=$ZONE
printf "done.\n"

# (7) Get worker ids
printf "Running $WHOAMI_SCRIPT ...\n"
gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --worker=all --command="bash $WHOAMI_SCRIPT $TPU_WORKERS_FILE" >$TPU_WORKERS_FILE
printf "done.\n"

cat $TPU_WORKERS_FILE
