# Code derived from: https://github.com/huggingface/diffusers/issues/604 from a snippet provided by https://github.com/patil-suraj
import os
import json
import tempfile

import jax
import jax.numpy as jnp
import numpy as np
import torch
from diffusers import FlaxUNet2DConditionModel, UNet2DConditionModel

# Instruct model
# (1) Use a 8 in-channel instruct-pix2pix model to test the conversion process.
instruct_model = UNet2DConditionModel.from_pretrained(
    "timbrooks/instruct-pix2pix",
    subfolder="unet",
    torch_dtype=torch.float32,
    revision="main",
    from_pt=True,
)

# (2) First, download the stable-diffusion-v1-5 unet checkpoint from the Hugging Face Hub and load it as a PyTorch UNet2DConditionModel model
# and add 4 additional in-channels to the model configuration.
extended_model = UNet2DConditionModel.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    subfolder="unet",
    in_channels=8,
    low_cpu_mem_usage=False,
    ignore_mismatched_sizes=True,
    revision="main",
    from_pt=True,
    torch_dtype=torch.float32,
)

# (3) Set both PyTorch models to evaluation mode
extended_model = extended_model.eval()
instruct_model = instruct_model.eval()


# (3) Next save the extended PyTorch model
with tempfile.TemporaryDirectory() as tmpdirname:
    extended_model.save_pretrained(
        tmpdirname,
        safe_serialization=False,
        revision="main",
        torch_dtype=torch.float32,
    )

    # (4) Then load it as a Flax model.
    # Under the hood the #from_pretrained method performs the standard Flax init and apply methods
    # Giving us a modified Flax model with the correct shape and the pre-trained and randomly initialized weights.
    flax_model, flax_params = FlaxUNet2DConditionModel.from_pretrained(
        tmpdirname,
        from_pt=True,
        variant="bf16",
        dtype=jnp.float32,
    )

# (5) Compare the outputs of the PyTorch and Flax models to ensure that the conversion process was successful.
sample = torch.rand(
    1,
    extended_model.config.in_channels,
    extended_model.config.sample_size,
    extended_model.config.sample_size,
)
time = 1
text_emb = torch.rand(1, 1, extended_model.config.cross_attention_dim)

print(f"Input shape: {sample.shape}")
torch_output = extended_model(sample, time, text_emb).sample
instruct_torch_output = instruct_model(sample, time, text_emb).sample
assert torch_output.shape == instruct_torch_output.shape

flax_sample = jnp.array(sample.numpy())
flax_text_emb = jnp.array(text_emb.numpy())
flax_output = flax_model.apply({"params": flax_params}, flax_sample, time, flax_text_emb).sample
converted_flax_output = torch.from_numpy(np.array(flax_output))
torch.testing.assert_close(converted_flax_output, torch_output, rtol=4e-03, atol=4e-03)

# Check that the EMA-only weights are in float32
assert flax_model.dtype == jax.numpy.float32


# (6) Save the Flax model's parameters and configuration file

save_dir = os.path.join(os.path.expanduser("~"), "modified_unet")
config_path = os.path.join(save_dir, "config.json")

# Save the model parameters
params_path = os.path.join(save_dir, "flax_model_params.msgpack")
flax_model.save_pretrained(save_dir, variant="bf16", params=flax_params, _internal_call=True)

# Save the model configuration
with open(config_path, "w") as f:
    # json.dump(unfreeze(flax_model.config).to_dict(), f)
    json.dump(flax_model.config, f)

print(f"Model, parameters, and configuration saved in directory: {save_dir}")

# Remove the model and its parameters from memory
del flax_model, flax_params

# (7) Lastly, check that the model was saved correctly by loading it back in
flax_model, flax_params = FlaxUNet2DConditionModel.from_pretrained(save_dir)
assert flax_model.config.in_channels == 8
assert flax_model.config.block_out_channels == [320, 640, 1280, 1280]
