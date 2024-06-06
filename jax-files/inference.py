import os
from natsort import natsorted, ns
import requests
from io import BytesIO
from PIL import Image
import jax
import numpy as np
import jax.numpy as jnp
from flax.jax_utils import replicate
from flax.training.common_utils import shard
from flax.serialization import to_bytes, from_bytes
from flax.core.frozen_dict import FrozenDict, unfreeze, freeze
from diffusers.utils import make_image_grid
from diffusers import (
    FlaxStableDiffusionInstructPix2PixPipeline,
)

from jax.experimental.compilation_cache import compilation_cache as cc

cc.set_cache_dir("/tmp/sd_cache")

HOME_DIR = os.path.expanduser("~")
images_dir = f"{HOME_DIR}/images"

if not os.path.exists(images_dir):
    os.makedirs(images_dir)


def download_image(url):
    response = requests.get(url)
    return Image.open(BytesIO(response.content)).convert("RGB")


def create_key(seed=0):
    return jax.random.PRNGKey(seed)


pipeline, pipeline_params = FlaxStableDiffusionInstructPix2PixPipeline.from_pretrained(
    f"{HOME_DIR}/instruct-pix2pix-model",
    dtype=jnp.bfloat16,
    safety_checker=None,
)

worker_id = jax.process_index()
rng = create_key(1371 + worker_id)

image_dir = os.path.join(os.path.expanduser("~"), "test-images")
image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.jpg')]
image_paths = natsorted(image_paths)

test_images = []
for i in range(len(image_paths)):
    image = Image.open(image_paths[i])
    image = image.convert("RGB")
    test_images.append(image)

with open(f"{image_dir}/prompts.txt", 'r') as file:
    test_prompts = [line.strip() for line in file]


prompt = test_prompts[worker_id]
image = test_images[worker_id].resize((512, 512))

num_samples = jax.local_device_count()
rng = jax.random.split(rng, jax.local_device_count())

prompt_ids, processed_image = pipeline.prepare_inputs(
    prompt=[prompt] * num_samples, image=[image] * num_samples
)

p_params = replicate(pipeline_params)
prompt_ids = shard(prompt_ids)
processed_image = shard(processed_image)


def get_images(output):
    return pipeline.numpy_to_pil(np.asarray(output.reshape((num_samples,) + output.shape[-3:])))


def save_run(images, run_name, images_dir=images_dir, worker_id=worker_id, starting_index=0):
    if not run_name:
        run_name = "temp"
    if not os.path.exists(f"{images_dir}/{run_name}"):
        os.makedirs(f"{images_dir}/{run_name}")
    for i, img in enumerate(images):
        img.save(f"{images_dir}/{run_name}/output_{worker_id + starting_index}_{i}.png")

## Test-images

run_name = "reference-test-images"
N, G, I = 100, 7.5, 1.2

output = pipeline(
    prompt_ids=prompt_ids,
    image=processed_image,
    params=p_params,
    prng_seed=rng,
    num_inference_steps=N,
    guidance_scale=G,
    image_guidance_scale=I,
    height=512,
    width=512,
    jit=True, # include for img2img
).images

images = get_images(np.array(output))
save_run(images, run_name)


run_name = "diffusers-test-images"
N, G, I = 20, 10.0, 1.5

output = pipeline(
    prompt_ids=prompt_ids,
    image=processed_image,
    params=p_params,
    prng_seed=rng,
    num_inference_steps=N,
    guidance_scale=G,
    image_guidance_scale=I,
    height=512,
    width=512,
    jit=True, # include for img2img
).images

images = get_images(np.array(output))
save_run(images, run_name)


run_name = "flax-test-images"
N, G, I = 50, 7.5, 1.5

output = pipeline(
    prompt_ids=prompt_ids,
    image=processed_image,
    params=p_params,
    prng_seed=rng,
    num_inference_steps=N,
    guidance_scale=G,
    image_guidance_scale=I,
    height=512,
    width=512,
    jit=True, # include for img2img
).images

images = get_images(np.array(output))
save_run(images, run_name)

print("Inference test completed.")
