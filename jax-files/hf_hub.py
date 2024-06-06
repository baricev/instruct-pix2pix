import argparse
import os
from pathlib import Path
import shutil
from huggingface_hub import create_repo, upload_folder
import jax

def get_tokens():
    # Retrieve hugging face token environment variables
    write_token = os.getenv('HUGGING_FACE_HUB_WRITE_TOKEN')
    read_token = os.getenv('HUGGING_FACE_HUB_READ_TOKEN')
    return { 'write_token': write_token, 'read_token': read_token , }

def get_hf_repo_id(repo_name, write_token=None):
    return create_repo(repo_id=repo_name, exist_ok=True, token=write_token).repo_id

def upload_checkpoint(repo_id, output_dir, write_token=None, commit_message="End of training",
                      ignore_patterns=["step_*", "epoch_*"], ):

    upload_folder(repo_id=repo_id,
                  folder_path=output_dir,
                  commit_message=commit_message,
                  ignore_patterns=ignore_patterns,
                  token=write_token
                  )


def parse_arguments():
    # Create the parser
    parser = argparse.ArgumentParser(description='Process some input arguments.')

    # Define the arguments
    parser.add_argument('--repo_id', type=str, default='flax-june1-v3', required=False, 
                        help='Hugging Face repository name (can be a new or existing repo')

    parser.add_argument('--output_dir', default='instruct-pix2pix-model', type=str, required=False,
                        help='Model output directory')

    parser.add_argument('--worker_all', default=True, action="store_false", required=False,
                        help='Flag to indicate whether script is run on all devices (True equivalent to `worker=all` )')

    # Parse the arguments
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    repo_id = args.repo_id 
    output_dir = args.output_dir  
    worker_all = args.worker_all

    # Copy tensorboardX logs to model directory
    logs = os.path.join(os.path.expanduser('~'), 'tensorboard_logs')
    shutil.move(logs, output_dir)

    write_token = get_tokens()['write_token']

    if not write_token:
        print("HF write token not set. Exiting.")
        return 1
    else:
        print(f"HF write token set {write_token}")


    if worker_all: # Running script on all machines
        if jax.process_index() == 0:

            id = get_hf_repo_id(repo_id, write_token)
            res = upload_checkpoint(id, output_dir, write_token)
    
    else:
        # Running script on a single machine
        id = get_hf_repo_id(repo_id, write_token)
        upload_checkpoint(id, output_dir, write_token)

    print("Upload completed.")


if __name__ == "__main__":
    main()
