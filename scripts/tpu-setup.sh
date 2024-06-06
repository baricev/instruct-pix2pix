#!/bin/bash
# This script was inspired by https://github.com/google/maxtext/blob/main/setup.sh

# Enable "exit immediately if any command fails" option
set -e

# ENVIRONMENT VARIABLES

export DEBIAN_FRONTEND=noninteractive
export NEEDRESTART_SUSPEND=1
export NEEDRESTART_MODE=l
export DEVICE="tpu"
export VENV_PATH="venv"
export WORKING_DIR=$(pwd)

# FUNCTIONS

check_venv_fn() {
    printf "Checking for existing virtual environment"
    if [ -d "$VENV_PATH" ]; then
        printf "Virtual environment exists at $VENV_PATH."
    else
        printf "No virtual environment found at $VENV_PATH."
    fi
}

remove_venv_fn() {
    printf "Removing virtual environment if it exists"

    deactivate 2>/dev/null || true

    if [ -d "$VENV_PATH" ]; then
        rm -rf "$VENV_PATH"
        printf "Virtual environment removed from $VENV_PATH."
    else
        printf "No virtual environment found at $VENV_PATH to remove."
    fi
}

assert_in_venv_fn() {
    if [[ -z "$VIRTUAL_ENV" ]]; then
        printf "Virtual environment is not activated."
        exit 1
    fi
}

create_venv_fn() {
    local venv_path="$1"
    python3 -m venv "$venv_path"
    echo "Created virtual environment at $venv_path"
}
remove_existing_installs_fn() {
    printf "Removing existing installations of jax, jaxlib, and libtpu-nightly and failing silently"
    (pip3 show jax && pip3 uninstall -y jax) || true
    (pip3 show jaxlib && pip3 uninstall -y jaxlib) || true
    (pip3 show libtpu-nightly && pip3 uninstall -y libtpu-nightly) || true
    local libtpu_path="$HOME/custom_libtpu/libtpu.so"
    ([ -e "$libtpu_path" ] && rm "$libtpu_path") || true
}

install_stable_fn() {
    printf "Installing stable jax, jaxlib and libtpu stable"
    pip3 install jax[tpu] -U -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
}

install_nightly_fn() {
    printf "Installing jax, jaxlib, and libtpu-nightly"
    pip3 install --pre -U jax -f https://storage.googleapis.com/jax-releases/jax_nightly_releases.html
    pip3 install --pre -U jaxlib -f https://storage.googleapis.com/jax-releases/jaxlib_nightly_releases.html
    pip3 install --pre -U libtpu-nightly -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
    printf "Installing nightly tensorboard plugin profile"
    pip3 install -U tbp-nightly
}

track_one_pip_install_fn() {
    printf "Tracking changes made by a single pip3install using strace"
    strace -f -e trace=%file,%process pip3install "$pip_package" -vvv 2>&1 | tee changes.log
}

track_pip_installs_fn() {
    printf "Tracking changes made by pip install commands using strace"
    local log_file="$1"
    shift
    local install_commands=("$@") # Remaining arguments are treated as complete pip install commands

    for cmd in "${install_commands[@]}"; do
        strace -f -e trace=%file,%process bash -c "$cmd" 2>&1 | tee -a "$log_file"
    done
}

# REQUIREMENTS TEXT FILE

# Write a requirements.txt so we don't have to upload it
FILE="requirements.txt"

/usr/bin/cat <<EOM >$FILE
# jax>=0.4.23
# jaxlib>=0.4.23
orbax-checkpoint>=0.5.2
absl-py
array-record
aqtp
cloud-tpu-diagnostics
crc32c # tensorboardX speedup
google-cloud-storage
grain-nightly
# flax>=0.8.0
ml-collections
numpy
optax
# protobuf>=4.25.3
pylint
pytest
pytype
sentencepiece==0.1.97
tensorflow-text>=2.13.0
tensorflow>=2.13.0
tensorflow-datasets
tensorboardx
tensorboard-plugin-profile
torch
torchvision
git+https://github.com/mlperf/logging.git
git+https://github.com/huggingface/diffusers
transformers
datasets
jupyter
ipykernel
ipywidgets
ipython
matplotlib
pillow
huggingface_hub 
accelerate
pyarrow
natsort
tensorflow_io
git-lfs

EOM

# SYSTEM
printf "Installing system dependencies"

(sudo bash || bash) <<'EOF'
apt update  &&  \
apt install -y numactl &&  \
apt install -y lsb-release  &&  \
apt install -y gnupg  &&  \
apt install -y curl  &&  \
apt install -y aria2 && \
apt install -y jq && \
apt install -y strace && \
apt install -y python3.10-venv
export GCSFUSE_REPO=gcsfuse-`lsb_release -c -s`
echo "deb https://packages.cloud.google.com/apt $GCSFUSE_REPO main" | tee /etc/apt/sources.list.d/gcsfuse.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -
apt update -y && apt -y install gcsfuse
rm -rf /var/lib/apt/lists/*
EOF

# BASE INSTALL STEPS

# (1) Remove the virtual environment if it exists
remove_venv_fn

# (2) Create a venv env, and source it
create_venv_fn "venv"
source "venv/bin/activate"

# (3) Upgrade pip *before* attempting any other installs
pip3 install -U pip

# (4) Always check if we are in a virtual environment before proceeding with any python installs
assert_in_venv_fn

# (5) Install dependencies from the requirements.txt file first
# as other libraries may downgrade JAX (leading to PJRT API and PJRT framework mis-matches) 
pip3 install -r requirements.txt

# (6) Remove existing JAX libraries
remove_existing_installs_fn

# (7) Install JAX nightly. Options: install_stable_fn, install_nightly_fn
install_nightly_fn

# (8) or install and track JAX with strace:
# track_pip_installs_fn "pip_changes.log" \
#     "pip3 install --pre -U jax -f https://storage.googleapis.com/jax-releases/jax_nightly_releases.html" \
#     "pip3 install --pre -U jaxlib -f https://storage.googleapis.com/jax-releases/jaxlib_nightly_releases.html" \
#     "pip3 install --pre -U libtpu-nightly -f https://storage.googleapis.com/jax-releases/libtpu_releases.html"
