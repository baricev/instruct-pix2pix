#!/bin/bash

# NOTE: To register the new pipeline we need to update the __init__.py files at three locations:
# diffusers/src/diffusers/__init__.py
# diffusers/src/diffusers/pipelines/__init__.py
# diffusers/src/diffusers/pipelines/stable_diffusion/__init__.py

# (1) First, make copies of the changed files and move them to examples/instruct_pix2pix:

# cp src/diffusers/__init__.py   diffusers.src.diffusers.__init__.py 
# cp src/diffusers/pipelines/__init__.py   diffusers.src.diffusers.pipelines.__init__.py 
# cp src/diffusers/pipelines/stable_diffusion/__init__.py   diffusers.src.diffusers.pipelines.stable_diffusion.__init__.py

# (2) Run this script on TPU to update the diffusers repository

REMOTE_HOME="/home/v"

cd "$REMOTE_HOME/pipeline-flax"
TO="/home/v/diffusers/src/diffusers"

# Overwrite the current init files
cp diffusers.src.diffusers.__init__.py "$TO/__init__.py" 
printf "Copied to: $TO/__init__.py \n" 

cp diffusers.src.diffusers.pipelines.__init__.py  "$TO/pipelines/__init__.py"  
printf "Copied to: $TO/pipelines/__init__.py \n"  

cp diffusers.src.diffusers.pipelines.stable_diffusion.__init__.py "$TO/pipelines/stable_diffusion/__init__.py"  
printf "Copied to: $TO/pipelines/stable_diffusion/__init__.py \n"  

cp pipeline_flax_stable_diffusion_instruct_pix2pix.py "$TO/pipelines/stable_diffusion/pipeline_flax_stable_diffusion_instruct_pix2pix.py"
printf "Copied to: $TO/pipelines/stable_diffusion/pipeline_flax_stable_diffusion_instruct_pix2pix.py \n"

exit 0
