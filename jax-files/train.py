import argparse
import io
import logging
import math
import os
import random
import time
from pathlib import Path
from matplotlib import pyplot as plt

import jax
import jax.numpy as jnp
import numpy as np
import optax
import torch
import torch.utils.checkpoint
import transformers
from datasets import load_dataset, load_from_disk
from flax import jax_utils
from flax.core.frozen_dict import unfreeze
from flax.training import train_state
from flax.training.common_utils import shard, shard_prng_key
from huggingface_hub import create_repo, upload_folder
from PIL import Image, PngImagePlugin
from torch.utils.data import IterableDataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTokenizer, FlaxCLIPTextModel, set_seed
from functools import partial
from tensorboardX import SummaryWriter


import copy
from typing import Any
from flax import jax_utils
from flax.training.train_state import TrainState
from flax.training.common_utils import shard
from flax.core.frozen_dict import FrozenDict
from flax import core

from transformers import CLIPImageProcessor, CLIPTokenizer, FlaxCLIPTextModel, set_seed
from diffusers import (
    FlaxAutoencoderKL,
    FlaxDDPMScheduler,
    FlaxPNDMScheduler,
    FlaxStableDiffusionPipeline,
    FlaxUNet2DConditionModel,
)
from diffusers.pipelines.stable_diffusion import FlaxStableDiffusionSafetyChecker


from config import args

logger = logging.getLogger(__name__)

import tensorflow as tf

OPTIONS = tf.data.Options()
OPTIONS.deterministic = True
OPTIONS.autotune.enabled = True

# --------------------------------------------------------------------------------


def visualize_image(batch, n, tokenizer=None):
    # {'key': batched array -> (B,3,256,256): 'input_ids': batched array (B,77), 'prompt': batched array (B,77)}

    # Get the nth sample from the batch
    original_image= batch["original_pixel_values"][n]
    edited_image= batch["edited_pixel_values"][n]
    prompt= batch["input_ids"][n]

    # Transpose the images from (3, 256, 256) -> (256, 256, 3) for visualization
    original_image = original_image.transpose(1, 2, 0)
    edited_image = edited_image.transpose(1, 2, 0)
    
    assert original_image.shape == (256, 256, 3) 
    assert edited_image.shape == (256, 256, 3) 

    # Reverse the preprocessing step:
    #   images = 2 * (images / 255) - 1
    # which set the pixel values in the range [-1, 1]. PIL Images are in the range [0,255] range and np.uint8 type
    original_image = Image.fromarray(((original_image + 1) * 127.5).astype(np.uint8))
    edited_image = Image.fromarray(((edited_image + 1) * 127.5).astype(np.uint8))

    # Display images
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(original_image)

    axes[0].imshow(original_image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(edited_image)
    axes[1].set_title("Edited Image")
    axes[1].axis("off")

    plt.show()
    
    # Print the prompt, without special tokens and padding
    print("Prompt:", tokenizer.decode(prompt, skip_special_tokens=True))


def collate_fn(examples):
    original_pixel_values = torch.stack([example["original_pixel_values"] for example in examples])
    original_pixel_values = original_pixel_values.to(memory_format=torch.contiguous_format).float()

    edited_pixel_values = torch.stack([example["edited_pixel_values"] for example in examples])
    edited_pixel_values = edited_pixel_values.to(memory_format=torch.contiguous_format).float()

    input_ids = torch.stack([example["input_ids"] for example in examples])

    batch = {
        "original_pixel_values": original_pixel_values,
        "edited_pixel_values": edited_pixel_values,
        "input_ids": input_ids,
    }
    batch = {k: v.numpy() for k, v in batch.items()}
    return batch


def make_train_dataset(args, batch_size=None):

    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    DATASET_NAME_MAPPING = {
        "fusing/instructpix2pix-1000-samples": ("input_image", "edit_prompt", "edited_image"),
        "timbrooks/instructpix2pix-clip-filtered": (
            "original_image",
            "edit_prompt",
            "edited_image",
        ),
    }
    def get_image_from_parquet(images_dict):
        if "bytes" not in images_dict:
            raise ValueError( "The provided image is a dictionary but does not contain a 'bytes' key.")

        return Image.open(io.BytesIO(images_dict["bytes"]))

    def convert_to_np(image, resolution):
        if isinstance(image, dict):
            image = get_image_from_parquet(image)
        
        image = image.convert("RGB").resize((resolution, resolution))
        return np.array(image).transpose(2, 0, 1)

    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    if args.parquet is False and args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=args.cache_dir,
            streaming=args.streaming,
        )
    else:
        if args.train_data_dir is not None:

            assert  args.dataset_name == "timbrooks/instructpix2pix-clip-filtered"
            assert args.streaming == False  # Not supported for ParquetDataset

            if args.load_from_disk:
                dataset = load_from_disk(
                    args.train_data_dir,
                )
            else:
                dataset = load_dataset(
                    args.train_data_dir,
                    cache_dir=args.cache_dir,
                )
        # See more about loading custom images at
        # https://huggingface.co/docs/datasets/v2.0.0/en/dataset_script

    def get_column_names(dataset):

        # Preprocessing the datasets.
        # We need to tokenize inputs and targets.

        # Set column names for the dataset.
        if isinstance(dataset["train"], IterableDataset):
            column_names = next(iter(dataset["train"])).keys()
        else:
            column_names = dataset["train"].column_names

        # print("Column names: ", column_names)

        # Map column names for provided dataset.
        if args.dataset_name == "timbrooks/instructpix2pix-clip-filtered":
            ARGS_ORIGINAL_IMAGE_COLUMN = "original_image"
            ARGS_EDIT_PROMPT_COLUMN = "edit_prompt"
            ARGS_EDITED_IMAGE_COLUMN = "edited_image"
        else:
            assert args.dataset_name == "fusing/instructpix2pix-1000-samples"
            ARGS_ORIGINAL_IMAGE_COLUMN = args.original_image_column
            ARGS_EDIT_PROMPT_COLUMN = args.edit_prompt_column
            ARGS_EDITED_IMAGE_COLUMN = args.edited_image_column

        # 6. Get the column names for input/target.
        dataset_columns = DATASET_NAME_MAPPING.get(args.dataset_name, None)

        if ARGS_ORIGINAL_IMAGE_COLUMN is None:
            original_image_column = (
                dataset_columns[0] if dataset_columns is not None else column_names[0]
            )
        else:
            original_image_column = ARGS_ORIGINAL_IMAGE_COLUMN
            if original_image_column not in column_names:
                raise ValueError(
                    f"--original_image_column' value '{ARGS_ORIGINAL_IMAGE_COLUMN}' needs to be one of: {', '.join(column_names)}"
                )
        if ARGS_EDIT_PROMPT_COLUMN is None:
            edit_prompt_column = (
                dataset_columns[1] if dataset_columns is not None else column_names[1]
            )
        else:
            edit_prompt_column = ARGS_EDIT_PROMPT_COLUMN
            if edit_prompt_column not in column_names:
                raise ValueError(
                    f"--edit_prompt_column' value '{ARGS_EDIT_PROMPT_COLUMN}' needs to be one of: {', '.join(column_names)}"
                )
        if ARGS_EDITED_IMAGE_COLUMN is None:
            edited_image_column = (
                dataset_columns[2] if dataset_columns is not None else column_names[2]
            )
        else:
            edited_image_column = ARGS_EDITED_IMAGE_COLUMN
            if edited_image_column not in column_names:
                raise ValueError(
                    f"--edited_image_column' value '{ARGS_EDITED_IMAGE_COLUMN}' needs to be one of: {', '.join(column_names)}"
                )

        return original_image_column, edit_prompt_column, edited_image_column


    # 6. Get the column names for input/target.
    original_image_column, edit_prompt_column, edited_image_column = get_column_names(dataset)

    def tokenize_captions_instruct_pix2pix(captions):
        inputs = tokenizer(
            captions,
            max_length=tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return inputs.input_ids

    # Torchvision Transforms
    train_transforms = transforms.Compose(
        [
            (
                transforms.CenterCrop(args.resolution)
                if args.center_crop
                else transforms.RandomCrop(args.resolution)
            ),
            (
                transforms.RandomHorizontalFlip()
                if args.random_flip
                else transforms.Lambda(lambda x: x)
            ),
        ]
    )

    def preprocess_images(examples):
        original_images = np.concatenate(
            [convert_to_np(image, args.resolution) for image in examples[original_image_column]]
        )
        edited_images = np.concatenate(
            [convert_to_np(image, args.resolution) for image in examples[edited_image_column]]
        )
        # We need to ensure that the original and the edited images undergo the same
        # augmentation transforms.
        images = np.concatenate([original_images, edited_images])
        images = torch.tensor(images)
        images = 2 * (images / 255) - 1
        return train_transforms(images)  # NOTE: this returns a torch tensor

    def preprocess_train(examples):
        # Preprocess images.
        preprocessed_images = preprocess_images(examples)  # NOTE: returns torch tensors

        # Since the original and edited images were concatenated before
        # applying the transformations, we need to separate them and reshape
        # them accordingly.
        original_images, edited_images = preprocessed_images.chunk(2)
        original_images = original_images.reshape(-1, 3, args.resolution, args.resolution)
        edited_images = edited_images.reshape(-1, 3, args.resolution, args.resolution)

        # Collate the preprocessed images into the `examples`.
        examples["original_pixel_values"] = original_images
        examples["edited_pixel_values"] = edited_images

        # Preprocess the captions.
        captions = list(examples[edit_prompt_column])
        examples["input_ids"] = tokenize_captions_instruct_pix2pix(captions)
        return examples



    if args.max_train_samples is not None:
        if args.streaming:
            dataset["train"] = dataset["train"].shuffle(seed=args.seed).take(args.max_train_samples)
        else:
            dataset["train"] = dataset["train"].shuffle(seed=args.seed).select(range(args.max_train_samples))
    # Set the training transforms
    if args.streaming:
        train_dataset = dataset["train"].map(
            preprocess_train,
            batched=True,
            batch_size=batch_size,
            remove_columns=list(dataset["train"].features.keys()),
        )
    else:
        train_dataset = dataset["train"].with_transform(preprocess_train)



    assert ( args.dataset_name == "timbrooks/instructpix2pix-clip-filtered")
    assert args.streaming == False  # Not supported for ParquetDataset
    dataset_download_path = os.path.join(os.path.expanduser("~"), "data")



    return train_dataset



def get_params_to_save(params):
    return jax.device_get(jax.tree_util.tree_map(lambda x: x[0], params))


def get_nparams(params: FrozenDict) -> int:
    nparams = 0
    for item in params:
        if isinstance(params[item], FrozenDict) or isinstance(params[item], dict):
            nparams += get_nparams(params[item])
        else:
            nparams += params[item].size
    return nparams


def tokenize_captions(captions, tokenizer, return_tensors):
    """Returns a BatchEncoding object with input_ids, attention_mask, and token_type_ids (if applicable)"""
    batch_encoding = tokenizer(
        captions,
        max_length=tokenizer.model_max_length,
        padding="max_length",
        truncation=True,
        return_tensors=return_tensors,
    )
    return batch_encoding


# EMA Update implementation


class ExtendedTrainState(TrainState):
    ema_params: core.FrozenDict[str, Any]

    def apply_gradients(self, grads: core.FrozenDict[str, Any]) -> "ExtendedTrainState":
        updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)
        return self.replace(params=new_params, opt_state=new_opt_state, ema_params=self.ema_params)


def get_decay(
    step: int,
    max_ema_decay: float = 0.9999,
    min_ema_decay: float = 0.0,
    ema_inv_gamma: float = 1.0,
    ema_decay_power: float = 2 / 3,
    use_ema_warmup: bool = True,
    start_ema_update_after_n_steps: float = 10.0,
):
    # Adjust step to consider the start update offset
    adjusted_step = jnp.maximum(step - start_ema_update_after_n_steps, 0)

    # Compute base decay
    if use_ema_warmup:
        decay = 1.0 - (1.0 + adjusted_step / ema_inv_gamma) ** -ema_decay_power
    else:
        initial_steps = jnp.where(
            start_ema_update_after_n_steps == 0, 10.0, start_ema_update_after_n_steps
        )
        decay = (1.0 + adjusted_step) / (initial_steps + adjusted_step)

    # Ensure decay starts changing only after certain steps
    decay = jnp.where(step > start_ema_update_after_n_steps, decay, min_ema_decay)

    # Clip the decay to ensure it stays within the specified bounds
    return jnp.clip(decay, min_ema_decay, max_ema_decay)


@jax.jit
def ema_update(new_params, ema_params, ema_decay):
    upadated_ema_params = jax.tree.map(
        lambda ema, p: ema * ema_decay + (1 - ema_decay) * p, ema_params, new_params
    )
    return upadated_ema_params


# Conditioning dropout implementation
def apply_conditioning_dropout(
    encoder_hidden_states,
    original_image_embeds,
    dropout_rng,
    bsz,
    conditioning_dropout_prob,
    tokenizer,
    text_encoder,
):

    def tokenize_captions(captions):
        inputs = tokenizer(
            captions,
            max_length=tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="np",
        )
        return inputs.input_ids

    # Generating a random tensor `random_p` with shape (bsz,)
    random_p = jax.random.uniform(dropout_rng, (bsz,))

    # Generating the prompt mask
    prompt_mask = random_p < 2 * conditioning_dropout_prob
    prompt_mask = prompt_mask.reshape(bsz, 1, 1)  # Reshape to match dimensions for broadcasting

    null_text_conditioning = text_encoder(
        tokenize_captions([""]),
        params=text_encoder.params,
        train=False,
    )[0]

    # Applying null conditioning using the prompt mask
    updated_encoder_hidden_states = jnp.where(
        prompt_mask, null_text_conditioning, encoder_hidden_states
    )

    # Generating the image mask
    image_mask_dtype = original_image_embeds.dtype
    image_mask = 1 - (
        (random_p >= conditioning_dropout_prob).astype(image_mask_dtype)
        * (random_p < 3 * conditioning_dropout_prob).astype(image_mask_dtype)
    )
    image_mask = image_mask.reshape(bsz, 1, 1, 1)

    # Final image conditioning.
    original_image_embeds = image_mask * original_image_embeds

    return updated_encoder_hidden_states, original_image_embeds


# Training function
def _train_step(
    state, 
    text_encoder_params,
    vae_params, batch, 
    train_rng,
    unet=None,
    vae=None,
    text_encoder=None,
    noise_scheduler=None,
    noise_scheduler_state=None,
    tokenizer=None,
    args=None,
    ):

    # Training function
    dropout_rng, sample_rng, new_train_rng = jax.random.split(train_rng, 3)

    def compute_loss(params):
        # Convert images to latent space
        vae_outputs = vae.apply(
            {"params": vae_params},
            batch["edited_pixel_values"],
            deterministic=True,
            method=vae.encode,
        )
        latents = vae_outputs.latent_dist.sample(sample_rng)
        latents = (
            jnp.einsum("ijkl->iljk", latents) * vae.config.scaling_factor
        )  # (NHWC) -> (NCHW)
        noise_rng, timestep_rng = jax.random.split(sample_rng)

        # Sample noise that we'll add to the latents
        noise = jax.random.normal(noise_rng, latents.shape)

        # Sample a random timestep for each image
        bsz = latents.shape[0]
        timesteps = jax.random.randint(
            timestep_rng,
            (bsz,),
            0,
            noise_scheduler.config.num_train_timesteps,
        )
        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = noise_scheduler.add_noise(
            noise_scheduler_state, latents, noise, timesteps
        )

        # Get the text embedding for conditioning
        encoder_hidden_states = text_encoder(
            batch["input_ids"],
            params=text_encoder_params,
            train=False,
        )[0]

        # Get the additional image embedding for conditioning.
        # Instead of getting a diagonal Gaussian here, we simply take the mode.
        vae_image_outputs = vae.apply(
            {"params": vae_params},
            batch["original_pixel_values"],
            deterministic=True,
            method=vae.encode,
        )
        original_image_embeds = vae_image_outputs.latent_dist.mode()
        original_image_embeds = jnp.einsum(
            "ijkl->iljk", original_image_embeds
        )  # (NHWC) -> (NCHW)

        # (7) Classifier-Free Guidance (conditioning dropout)
        if args.conditioning_dropout_prob > 0.0:
            encoder_hidden_states, original_image_embeds = apply_conditioning_dropout(
                encoder_hidden_states,
                original_image_embeds,
                dropout_rng,
                bsz,
                args.conditioning_dropout_prob,
                tokenizer,
                text_encoder,
            )

        # Concatenate the noisy latents with the original image embeddings
        concatenated_noisy_latents = jnp.concatenate(
            [noisy_latents, original_image_embeds], axis=1
        )

        # Predict the noise residual and compute loss
        model_pred = unet.apply(
            {"params": params},
            concatenated_noisy_latents,
            timesteps,
            encoder_hidden_states,
            train=True,
        ).sample

        # Get the target for loss depending on the prediction type
        if noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif noise_scheduler.config.prediction_type == "v_prediction":
            target = noise_scheduler.get_velocity(
                noise_scheduler_state, latents, noise, timesteps
            )
        else:
            raise ValueError(
                f"Unknown prediction type {noise_scheduler.config.prediction_type}"
            )

        loss = (target - model_pred) ** 2
        loss = loss.mean()

        return loss

    grad_fn = jax.value_and_grad(compute_loss)

    # Backprop to get gradients
    loss, grad = grad_fn(state.params)

    # behind the scenes JAX does the allreduce for us here
    grad = jax.lax.pmean(grad, "batch")

    # update weights by taking a step in the direction of the gradient
    new_state = state.apply_gradients(grads=grad)

    # (8) Decay rate for current step
    decay = get_decay(
        state.step,
        args.max_ema_decay,
        args.min_ema_decay,
        args.ema_inv_gamma,
        args.ema_decay_power,
        args.use_ema_warmup,
        args.start_ema_update_after_n_steps,
    )
    # (9) EMA update
    new_ema_params = ema_update(new_state.params, state.ema_params, decay)
    new_state = new_state.replace(ema_params=new_ema_params)

    metrics = {"loss": loss}
    # behind the scenes JAX does the allreduce for us here
    metrics = jax.lax.pmean(metrics, axis_name="batch")

    # finally return the new state, metrics and the new random number generator
    return new_state, metrics, new_train_rng



def main():
    # -----------------------------------------------------------------------------
    # (1) Set up logging

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    
    # Setup logging, we only want one process per machine to log things on the screen.
    
    # logger.setLevel(logging.INFO if jax.process_index() == 0 else logging.ERROR)
    logger.setLevel(logging.INFO if jax.process_index() == 0 else logging.ERROR)
    if jax.process_index() == 0:
        # transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.set_verbosity_error()
    else:
        transformers.utils.logging.set_verbosity_error()


    # -----------------------------------------------------------------------------
    # (2) TensorBoardX Writer Initialization

    log_dir = os.path.join(os.path.expanduser("~"), "tensorboard_logs")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)



    # -----------------------------------------------------------------------------
    # (3) Set up the random number generators

    if args.seed is not None:
        set_seed(args.seed)

    # -----------------------------------------------------------------------------
    # (4) Datasets!
    
    if args.dataset_name is None:
        raise ValueError("No dataset specified.")

    num_processes = jax.process_count()
    num_global_devices = jax.device_count()
    num_local_devices = jax.local_device_count()
    # Total train batch size across all devices
    total_train_batch_size = args.train_batch_size * num_local_devices

    # -----------------------------------------------------------------------------

    # (5) Data Loader

    # Load the tokenizer and add the placeholder token as a additional special token
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
    )


    # -----------------------------------------------------------------------------



    MAX_DATASET_LEN = 39424
    MIN_DATASET_LEN = 37888
    DATASET_LEN = 37888 

    N = BLOCKS  = 8  # = jax.process_count()
    M = THREADS = 4  # = jax.local_device_count()
    W = WORKERS = GRID = M*N # = jax.device_count()
    
    batch_size = 1
    P = M * batch_size  # = total_train_batch_size    = per_process_batch_size
    B = batch_size      # = args.training_batch_size  = per_replica_batch_size
    G = B * W           # = global_batch_size         = per_replica_batch_size * jax.device_count(

   
    weight_dtype = jnp.bfloat16
    _tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32", dtype=weight_dtype)


    # (5.1) List all TFRecord files in the specified directory
    home_dir = os.getenv('HOME')
    data_dir = os.path.join(home_dir, 'data')
    tfrecord_files = [os.path.join(data_dir, fname) for fname in os.listdir(data_dir) if fname.endswith('.tfrecord')]

    # (5.2) Create a TensorFlow dataset from TFRecord files
    raw_dataset = tf.data.TFRecordDataset(tfrecord_files)


    # (5.3) Define the feature description dictionary based on discovered features
    FEATURE_DESCRIPTION = {
        'original_image': tf.io.FixedLenFeature([], tf.string),
        'edited_image': tf.io.FixedLenFeature([], tf.string),
        'edit_prompt': tf.io.FixedLenFeature([], tf.string),
    }



    # Normalize data 
    # Scales image to a guassian by transforming image values from [0,255] uint8 to [-1,1] float32
    def normalize_image(image):
        # return (tf.cast(image, tf.float32) - 127.5) / 127.5
        return 2 * (tf.cast(image, tf.float32) / 255) - 1

    def denormalize_image(image):
        return (image + 1) * 127.5

    def train_transforms_pre_norm(image, resolution=args.resolution, center_crop=args.center_crop, random_flip=args.random_flip):
        # Assumes `image` is a 3D tensor of shape (H, W, C) and values in [0,255] uint8
        if center_crop:
            image = tf.image.central_crop(image, central_fraction=resolution / min(image.shape[:2]))
        else:
            image = tf.image.random_crop(image, size=[resolution, resolution, image.shape[-1]])

        if random_flip:
            image = tf.image.random_flip_left_right(image)

        return image

    # Resize images to ensure they have the same shape
    resolution = args.resolution
    def resize_image(image, size=(resolution, resolution)):
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, size)
        image = tf.cast(image, tf.uint8)
        return image

    def _parse_and_resize_function(proto):
        parsed_features = tf.io.parse_single_example(proto, FEATURE_DESCRIPTION)
        original_image = resize_image(parsed_features['original_image'])
        edited_image = resize_image(parsed_features['edited_image'])
        edit_prompt = parsed_features['edit_prompt']
        return {
            'original_image': original_image,
            'edited_image': edited_image,
            'edit_prompt': edit_prompt
        }


    def _transform_and_normalize_function(example):
        original_image = example['original_image']
        edited_image = example['edited_image']

        # Concat the two images (of shape (256, 256, 3)) and appy the same transform to both.
        image = tf.concat([original_image, edited_image], axis=2)
        image = train_transforms_pre_norm(image)
        image = normalize_image(image)

        # Split the images back into original and edited
        original_image, edited_image = tf.split(image, num_or_size_splits=2, axis=2)

        # Transpose image  (H,W,C) -> (C,H,W)
        original_image = tf.transpose(original_image, [2, 0, 1])
        edited_image = tf.transpose(edited_image, [2, 0, 1])

        example["original_image"] = original_image
        example["edited_image"] = edited_image
        
        return example


    def _numpy_preprocess(examples):
        # Takes a list of NumPy arrays ('batch') 
        
        original_images = examples['original_image'].squeeze(1) # should remove extra dims here
        edited_images = examples['edited_image'].squeeze(1)
        # print(f"np shape {original_images.shape}")

        # Convert the list of bytes to a list of strings
        nested_byte_np_array = examples['edit_prompt'] # should remove extra dims here
        byte_list = list(nested_byte_np_array.flatten()) 
        string_list = [b.decode('utf-8') for b in byte_list]

        # Preprocess the captions.
        input_ids= _tokenizer(
            string_list,
            max_length=tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="np",
        )['input_ids']

        # print(f"np ids shape {input_ids.shape}")
        # Collate the preprocessed images into the `examples`.
        return {
                "original_pixel_values": original_images,
                "edited_pixel_values": edited_images,
                "input_ids": input_ids,
        }



    # --------------------------------------------------------
    # (5.4) Set a minimum dataset length so all devices receive the same number of batches
    subset_size = args.max_train_samples if args.max_train_samples else MIN_DATASET_LEN
    raw_dataset = raw_dataset.take(subset_size)

    assert subset_size % jax.process_count() == 0, "Dataset [{subset_size}] size must be integer divisible by process_count() [{jax.process_count()}]"

    G = global_batch_size = B * jax.device_count()
    assert subset_size % global_batch_size == 0, "Dataset size [{subset_size}] must be integer divisible by the global batch size [{global_batch_size}]"



    # (5.5) Calculate training values

    # Convenience functions for dataset and generator lengths
    def _dataset_length(dataset):
        length = dataset.reduce(0, lambda x, _: x + 1).numpy()
        return length

    def _generator_length(dataset_length, batch_size):
        return (dataset_length + batch_size - 1) // batch_size

    # len_train_dataset = args.max_train_samples # _dataset_length(raw_dataset)
    len_train_dataset = _dataset_length(raw_dataset)
    len_train_dataloader = _generator_length(len_train_dataset, batch_size)



    steps_per_epoch = (
        math.ceil(args.max_train_samples // total_train_batch_size)
        if args.streaming or args.max_train_samples
        else math.ceil(len_train_dataset // total_train_batch_size)
    )

    max_train_samples = args.max_train_samples * num_processes if args.streaming else len_train_dataset * num_processes
    local_train_samples = args.max_train_samples if args.streaming else len_train_dataset

    max_steps = args.max_train_steps
    if max_steps is None:
        max_steps = args.num_train_epochs * steps_per_epoch
    

    num_train_epochs = max_steps // steps_per_epoch


    # --------------------------------------------------------
    # (5.6) Dataset 

    ds = raw_dataset.map(_parse_and_resize_function).map(_transform_and_normalize_function)

    shuffle_buffer_size = 128
    
    repeat = num_train_epochs
    ds = ds.repeat(repeat)

    if shuffle_buffer_size is not None:
        ds = ds.shuffle(shuffle_buffer_size)

    ds = ds.batch(jax.local_device_count(), drop_remainder=True)
    ds = ds.with_options(OPTIONS)
    ds = ds.prefetch(1)

    # (5.7) Numpy iterator

    train_iter = iter(ds)

    # -----------------------------------------------------------------------------
    # (6) Models and state
    # modified_unet_save_dir = os.path.join(os.path.expanduser("~"), "modified_unet")

    weight_dtype = jnp.float32
    if args.mixed_precision == "fp16":
        weight_dtype = jnp.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = jnp.bfloat16

    # Load models and create wrapper for stable diffusion
    text_encoder = FlaxCLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        dtype=weight_dtype,
        revision=args.revision,
        from_pt=args.from_pt,
    )
    vae, vae_params = FlaxAutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        revision=args.revision,
        subfolder="vae",
        dtype=weight_dtype,
        from_pt=args.from_pt,
    )

    unet, unet_params = FlaxUNet2DConditionModel.from_pretrained(
        "baricevic/flax-instruct-pix2pix",
        dtype=weight_dtype,
        # revision=args.revision,
        from_pt=args.from_pt,
    )

    noise_scheduler = FlaxDDPMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
    )
    noise_scheduler_state = noise_scheduler.create_state()

    # -----------------------------------------------------------------------------
    # (7) Optimizer and learning rate schedule

    if args.scale_lr:
        args.learning_rate = args.learning_rate * total_train_batch_size

    # Learning rate scheduler
    constant_scheduler = optax.constant_schedule(args.learning_rate)

    # Optimizer
    adamw = optax.adamw(
        learning_rate=constant_scheduler,
        b1=args.adam_beta1,
        b2=args.adam_beta2,
        eps=args.adam_epsilon,
        weight_decay=args.adam_weight_decay,
    )

    optimizer = optax.chain(
        optax.clip_by_global_norm(args.max_grad_norm),
        adamw,
    )

    # Initialize our training
    rng = jax.random.PRNGKey(args.seed)
    train_rngs = jax.random.split(rng, jax.local_device_count())

    # Initialize EMA params with original model params
    ema_params = copy.deepcopy(unet_params)

    # Prepare optimizer and state
    state = ExtendedTrainState.create(
        apply_fn=unet.__call__, params=unet_params, ema_params=ema_params, tx=optimizer
    )

    # -----------------------------------------------------------------------------
    # (8) Replicate the state and model params

    # Replicate the train state on each global device (each host has 4 local devices * 4 hosts = 16 global devices)
    state = jax_utils.replicate(state)
    text_encoder_params = jax_utils.replicate(text_encoder.params)
    vae_params = jax_utils.replicate(vae_params)

    # -----------------------------------------------------------------------------
    # (9) Training step function

    train_step = partial(
        _train_step,
        unet=unet,
        vae=vae,
        text_encoder=text_encoder,
        noise_scheduler=noise_scheduler,
        noise_scheduler_state=noise_scheduler_state,
        tokenizer=tokenizer,
        args=args,
    )

    # Create parallel version of the train step
    p_train_step = jax.pmap(train_step, "batch", donate_argnums=(0,))

    # -----------------------------------------------------------------------------
    # (10)  Training Loop Setup

    # Train!

    logger.info(f"  =================================================")
    logger.info(f"  Num unet parameters = { get_nparams(unet_params)}")
    logger.info(f"  Conditioning dropout prob = { args.conditioning_dropout_prob }")
    logger.info(f"  Dataset subset size= { subset_size}")
    logger.info(f"  Global dataset length = {len_train_dataset  * num_processes}")
    logger.info(f"  Local dataset length = {len_train_dataset }")
    logger.info(f"  Global dataloader length = {len_train_dataloader  * num_processes}")
    logger.info(f"  Local dataloader length = {len_train_dataloader }")

    logger.info(f"  Batch size per device = {args.train_batch_size}")
    logger.info(f"  Local total batch size (w. distributed) = {total_train_batch_size}")
    logger.info(f"  Global total batch size (w. parallel & distributed) = {total_train_batch_size * num_processes}")

    logger.info(f"  Global Num examples = {max_train_samples}")
    logger.info(f"  Local Num examples = {local_train_samples}")
    
    logger.info(f"  Steps per epoch = {steps_per_epoch}")

    logger.info(f"  Configuration values:")
    logger.info(f"  Max num optimization steps = {max_steps}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Calculated values:")
    logger.info(f"  Max num optimization steps = {num_train_epochs * steps_per_epoch}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  =================================================")
    

    global_step = step0 = 0
    epochs = tqdm(
        range(args.num_train_epochs), # set to absolute num
        desc="Epoch ... ",
        position=0,
        disable=jax.process_index() > 0,
    )

    
    # -----------------------------------------------------------------------------
    # (11)  Training Loop

    for epoch in epochs:
        # ======================== Training ================================

        train_metrics = []
        train_metric = None


        train_step_progress_bar = tqdm(
            total=steps_per_epoch,
            desc="Training...",
            position=1,
            leave=False,
            disable=jax.process_index() > 0,
        )
        # train
        for _ in range(steps_per_epoch):

            next_element = next(train_iter)
            batch = {k: v.numpy() for k,v in next_element.items() }
            batch = _numpy_preprocess(batch)
 
            batch = shard(batch)
            with jax.profiler.StepTraceAnnotation("train", step_num=global_step):
                state, train_metric, train_rngs = p_train_step(
                    state, text_encoder_params, vae_params, batch, train_rngs
                )
            train_metrics.append(train_metric)

            # Log metrics to TensorBoard
            writer.add_scalar(
                "Training/Loss",
                jax_utils.unreplicate(train_metric["loss"]),
                global_step,
            )


            train_step_progress_bar.update(1)
            global_step += 1
            if global_step >= max_steps:
                break

        train_step_progress_bar.close()
        train_metric = jax_utils.unreplicate(train_metric)
        # epochs.write(f"Epoch... ({epoch + 1}/{args.num_train_epochs} | Loss: {train_metric['loss']})")

        avg_loss = np.mean(
            [jax_utils.unreplicate(m["loss"]) for m in train_metrics]
        )
        writer.add_scalar("Epoch/Loss", avg_loss, epoch)
        epochs.write(f"Epoch {epoch + 1}/{args.num_train_epochs} | Loss: {avg_loss}")



    # -----------------------------------------------------------------------------
    # Training complete!

    # -----------------------------------------------------------------------------
    # Close the TensorBoardX writer

    if writer:
        writer.close()

    # -----------------------------------------------------------------------------
    # Save trained model

    # Create the pipeline using using the trained modules and save it.
    scheduler = FlaxPNDMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        skip_prk_steps=True,
    )
    safety_checker = FlaxStableDiffusionSafetyChecker.from_pretrained(
        "CompVis/stable-diffusion-safety-checker", from_pt=True
    )
    pipeline = FlaxStableDiffusionPipeline(
        text_encoder=text_encoder,
        vae=vae,
        unet=unet,
        tokenizer=tokenizer,
        scheduler=scheduler,
        safety_checker=safety_checker,
        feature_extractor=CLIPImageProcessor.from_pretrained(
            "openai/clip-vit-base-patch32"
        ),
    )

    output_dir = os.path.join(os.path.expanduser("~"), args.output_dir)

    pipeline.save_pretrained(
        output_dir,
        params={
            "text_encoder": get_params_to_save(text_encoder_params),
            "vae": get_params_to_save(vae_params),
            "unet": get_params_to_save(state.ema_params),
            "safety_checker": safety_checker.params,
        },
    )

if __name__ == "__main__":
    main()

