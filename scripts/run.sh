source scripts/functions.sh

FILE="jax-files/inference.py"
copy_to $FILE $REMOTE_WORKING_DIR 

FILE="jax-files/config.py"
copy_to $FILE $REMOTE_WORKING_DIR 

FILE="jax-files/train.py"
copy_to $FILE $REMOTE_WORKING_DIR 

# Train!
FILE="train.py"
run "python $FILE"

# Run inference on all devices
FILE="inference.py"
run "python $FILE"

# Copy created images to local folder 
copy_images

# Save training run in time-stamped run directory
FILE="scripts/save_training_run.py"
python $FILE

# Push model to hub
FILE="jax-files/hf_hub.py"
copy_to $FILE $REMOTE_WORKING_DIR 

FILE="hf_hub.py"
run "export HUGGING_FACE_HUB_WRITE_TOKEN=$HF_WRITE_TOKEN; python $FILE --output_dir "$REMOTE_HOME/instruct-pix2pix-model"  --repo_id flax-june3-tfboardx"

