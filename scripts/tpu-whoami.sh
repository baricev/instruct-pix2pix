#!/bin/bash
set -e

# Check if the TPU workers file is provided
if [[ "$#" -ne 1 ]]; then
  printf "Usage: $0 TPU_WORKERS_FILE\n"
  printf "Exiting...\n"
  exit 1
fi

TPU_WORKERS_FILE=$1

# (1) Remove any old worker id files

# (note: the checks are necessary as the script uses the "set -e" option and removing non-existent files raises an error)
if [ -f "internal-ip-*" ]; then
  rm internal-ip-*
fi

if [ -f "external-ip-*" ]; then
  rm external-ip-*
fi

if [ -f "worker-*" ]; then
  rm worker-*
fi

# (2) Source the python environment (only one with a JAX install)
source venv/bin/activate

# (3) Record each worker's external and internal IPs

# Execute Python code and capture the output in a Bash variable
process_index=$(python -c "import jax; idx = jax.process_index(); print(idx);")

# Then Create the file with the worker id
touch "worker-$process_index"

# Create a file with the internal IP (found using the hostname Linux utility)
ip=$(hostname -I | awk '{print $1}')
touch "internal-ip-$ip"

# And Extract the corresponding external IP from workers text file using grep
external_ip=$(grep "$ip" $TPU_WORKERS_FILE | awk '{print $1}')

# Finally create the file with the external IP
touch "external-ip-$external_ip"
printf "%-12s %-16s %-16s\n" "worker-$process_index" "$external_ip" "$ip"
