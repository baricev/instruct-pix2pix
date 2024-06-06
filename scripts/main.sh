#!/bin/bash
set -e

# This script runs the TPU setup and main project setup scripts.

# It is the main entry point for setting up the project environment for training the pix2pix model on a Google Cloud TPU Pod Slice

GOOGLE_CLOUD_SCRIPT="google-cloud-setup.sh"
PROJECT_SCRIPT="project-setup.sh"

# Set the script paths

if ! bash $GOOGLE_CLOUD_SCRIPT; then
    echo "$GOOGLE_CLOUD_SCRIPT failed. Exiting."
    exit 1
fi

if ! bash $PROJECT_SCRIPT; then
    echo "$PROJECT_SCRIPT failed. Exiting."
    exit 1
fi
