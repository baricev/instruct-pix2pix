import os
import shutil
import importlib.util
import dataclasses
from datetime import datetime

def get_project_root():
    """Returns the absolute path to the project root directory."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def create_directory(path):
    """Create directory if it does not exist."""
    if not os.path.exists(path):
        os.makedirs(path)

def move_and_copy_files(images_dir, run_dir, files_to_copy, training_dir, scripts_dir):
    """Move images directory and copy specified files to the run directory."""
    shutil.move(images_dir, run_dir)
    print(f"'images' directory has been moved to '{run_dir}'.")
    
    for file_name in files_to_copy:
        source_path = os.path.join(training_dir if file_name in ['train.py', 'config.py'] else scripts_dir, file_name)
        if os.path.exists(source_path):
            shutil.copy(source_path, run_dir)

def read_configurations(config_path):
    """Dynamically import settings from a config.py file and extract configurations."""

    module_name = 'config_module'
    spec = importlib.util.spec_from_file_location(module_name, config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    
    args = getattr(config_module, 'args', None)
    combined_args = getattr(config_module, 'combined_args', {})
    
    # List of Configuration keys we wish to track 
    keys = ['conditioning_dropout_prob', 'max_train_steps', 'mixed_precision', 'cache_dir', 'seed', 'pretrained_model_name_or_path', 'revision']
    output = {}
    
    if args and dataclasses.is_dataclass(args):
        for field in dataclasses.fields(args):
            if field.name in keys:
                output[field.name] = getattr(args, field.name, None)
    
    # Additional checks for combined_args which is a regular dictionary
    for key in keys:
        if key in combined_args and key not in output:
            output[key] = combined_args[key]
    
    return output





def write_markdown_file(config, run_dir, project_name, current_time):
    """Generate and write Markdown file with project and training details."""
    md_content = f"""
# Title

Date: {current_time.strftime('%Y-%m-%d')}
Time: {current_time.strftime('%H:%M:%S')}
Project Name: {project_name}

## Training Configuration
"""
    for key, value in config.items():
        md_content += f"{key} {value}\n"

    with open(os.path.join(run_dir, 'details.md'), 'w') as md_file:
        md_file.write(md_content)

def main():
    base_dir = get_project_root()
    results_dir = os.path.join(base_dir, 'training-runs')
    run_dir = f"{results_dir}/run-{int(datetime.now().timestamp())}"
    training_dir = os.path.join(base_dir, 'jax-files')
    scripts_dir = os.path.join(base_dir, 'scripts')
    images_dir = os.path.join(base_dir, 'images')
    files_to_copy = ['train.py', 'config.py', 'vars.txt', 'output.txt']
    
    create_directory(results_dir)
    create_directory(run_dir)
    move_and_copy_files(images_dir, run_dir, files_to_copy, training_dir, scripts_dir)
    
    config_file_path = os.path.join(training_dir, 'config.py')
    config = read_configurations(config_file_path)
    
    project_name = os.path.basename(base_dir)
    current_time = datetime.now()
    write_markdown_file(config, run_dir, project_name, current_time)

if __name__ == "__main__":
    main()

