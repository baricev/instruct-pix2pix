import dataclasses
from dataclasses import dataclass, field
from typing import Optional
from dataclasses import dataclass
import jax
import jax.numpy as jnp

from flax import struct
import jax
from typing import Any, Callable


# Extracted from the Diffusers PyTorch training script: diffusers/examples/instruct_pix2pix/train_instruct_pix2pix.py

default_args = {
    "pretrained_model_name_or_path": "runwayml/stable-diffusion-v1-5",
    "revision": "null",
    "variant": "null",
    "dataset_name": "fusing/instructpix2pix-1000-samples",
    "dataset_config_name": "null",
    "train_data_dir": "null",
    "original_image_column": "input_image",
    "edited_image_column": "edited_image",
    "edit_prompt_column": "edit_prompt",
    "val_image_url": "null",
    "validation_prompt": "null",
    "num_validation_images": "4",
    "validation_epochs": "1",
    "max_train_samples": "null",
    "output_dir": "instruct-pix2pix-model",
    "cache_dir": "null",
    "seed": "42",
    "resolution": "256",
    "center_crop": "False",
    "random_flip": "True",
    "train_batch_size": "4",
    "num_train_epochs": "100",
    "max_train_steps": "15000",
    "gradient_accumulation_steps": "4",
    "gradient_checkpointing": "True",
    "learning_rate": "5e-05",
    "scale_lr": "False",
    "lr_scheduler": "constant",
    "lr_warmup_steps": "0",
    "conditioning_dropout_prob": "0.05",
    "use_8bit_adam": "False",
    "allow_tf32": "False",
    "use_ema": "False",
    "non_ema_revision": "null",
    "dataloader_num_workers": "0",
    "adam_beta1": "0.9",
    "adam_beta2": "0.999",
    "adam_weight_decay": "0.01",
    "adam_epsilon": "1e-08",
    "max_grad_norm": "1.0",
    "push_to_hub": "True",
    "hub_token": "null",
    "hub_model_id": "null",
    "logging_dir": "logs",
    "mixed_precision": "fp16",
    "report_to": "tensorboard",
    "local_rank": "-1",
    "checkpointing_steps": "5000",
    "checkpoints_total_limit": "1",
    "resume_from_checkpoint": "null",
    "enable_xformers_memory_efficient_attention": "True",
}


# Convert to type
def convert_to_type(value):
    if value.lower() == "true":
        return True
    elif value.lower() == "false":
        return False
    elif value.lower() == "null":
        return None
    else:
        try:
            return int(value)
        except ValueError:
            try:
                return float(value)
            except ValueError:
                return value


# Define a `static` flax.struct.dataclass

@struct.dataclass
class StaticConfig:
    pretrained_model_name_or_path: str
    revision: Optional[str]
    variant: Optional[str]
    dataset_name: str
    dataset_config_name: Optional[str]
    train_data_dir: Optional[str]
    original_image_column: str
    edited_image_column: str
    edit_prompt_column: str
    val_image_url: Optional[str]
    validation_prompt: Optional[str]
    num_validation_images: int
    validation_epochs: int
    max_train_samples: Optional[int]
    output_dir: str
    cache_dir: Optional[str]
    seed: int
    resolution: int
    center_crop: bool
    random_flip: bool
    train_batch_size: int
    num_train_epochs: int
    max_train_steps: int
    gradient_accumulation_steps: int
    gradient_checkpointing: bool
    learning_rate: float
    scale_lr: bool
    lr_scheduler: str
    lr_warmup_steps: int
    conditioning_dropout_prob: float
    use_8bit_adam: bool
    allow_tf32: bool
    use_ema: bool
    non_ema_revision: Optional[str]
    dataloader_num_workers: int
    adam_beta1: float
    adam_beta2: float
    adam_weight_decay: float
    adam_epsilon: float
    max_grad_norm: float
    push_to_hub: bool
    hub_token: Optional[str]
    hub_model_id: Optional[str]
    logging_dir: str
    mixed_precision: str
    report_to: str
    local_rank: int
    checkpointing_steps: int
    checkpoints_total_limit: int
    resume_from_checkpoint: bool
    enable_xformers_memory_efficient_attention: bool
    ema_decay_power: float = 0.6666666
    max_ema_decay: float = 0.9999
    min_ema_decay: float = 0.5
    ema_inv_gamma: float = 1.0
    start_ema_update_after_n_steps: int = 10
    update_ema_every: int = 1
    use_ema_warmup: bool = True
    regularization: float = 0.1
    from_pt: bool = False
    streaming: bool = False
    all_files: bool = False
    parquet: bool = False
    load_from_disk: Optional[str] = None


# Convert a dictionary to a dataclass instance with type conversion
def from_dict_to_dataclass(d, cls):
    # Convert a dictionary to a dataclass instance with type conversion.
    field_meta = {f.name: f.metadata.get("convert", lambda x: x) for f in dataclasses.fields(cls)}
    converted_data = {k: field_meta[k](v) if k in field_meta else v for k, v in d.items()}
    return cls(**converted_data)


# Convert values to the correct type
default_args = {key: convert_to_type(value) for key, value in default_args.items()}

# "from_pt": False,  # Leave as False
# As from_pt is Only used when SAVING trained Pipelines, Diffusers Pipelines do not use the from_pt argument, and their submodules fail to load when from_pt=True

# Combine the default and new arguments into a single dictionary
combined_args = {**default_args}

# Update the combined_args with the new values

N=8 # n processes
M=4 # n local devices
W = 8*4 # n devices
B = BATCH = 8 # variable (batch size)
DS = 2 # variable (data set multiple)
P = B*M
G = B*W
DATASET_SIZE = None # G * DS # (B=8*W=32) = (256, ...) = global batch size

combined_args["conditioning_dropout_prob"] = 0.10
combined_args["dataset_name"] = "timbrooks/instructpix2pix-clip-filtered"
combined_args["max_train_steps"] = 15000
combined_args["gradient_accumulation_steps"] = 1
combined_args["num_train_epochs"] = 100 
combined_args["train_batch_size"] = B # 32 x 4 (local devices) x 8 (workers) = 1024
combined_args["mixed_precision"] = "bf16"
combined_args["revision"] = "bf16"
combined_args["streaming"] = False
combined_args["parquet"] = True
combined_args["all_files"] = True
combined_args["train_data_dir"] = "/home/v/data"
combined_args["max_train_samples"] = DATASET_SIZE

args = from_dict_to_dataclass(combined_args, StaticConfig)
