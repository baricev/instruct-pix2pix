# InstructPix2Pix Diffusion Models for JAX and Cloud TPUs
A full JAX/ Flax implementation of Brook et. al.'s 2023 InstructPix2Pix paper.

<img src="images/grid_cyborg.png" width="800" height="800">


## JAX InstructPix2Pix
This repository contains a JAX implementation of Brook et. al.'s 2023 paper `InstructPix2Pix: Learning to Follow Image Editing Instructions` and includes:

  * a JAX port of the EMA Model model used in the Diffuser’s implementation. 
  * a JAX conditioning dropout method based on the same
  * a simple distributed map-shuffle-reduce data-loader and dataset which leverages existing Linux utilities, Parquet and TFRecord datasets to efficiently serve large image data in a distributed TPU setting.
  * code to automate training, inference, TPU pod slice creation, VM setup and IP discovery.


To recreate the original model on the full training set follow the instructions below. If you would like to experiment with the toy data set from Hugging Face, please change the configuration args in `config.py`. Note: a v3-32 TPU Pod slice takes approximately 30 mins to finish training on the the smaller dataset. A full training run using v4 TPUs can be completed in a few hours, depending on the number of TPUs used.

### Setting up a Cloud TPU VM
Use scripts/google-cloud-setup.sh to create a TPU pod slice.

```
cd scripts
bash google-cloud-setup.sh 
```

### Training and inference with the toy dataset
Use scripts/project-setup.sh to clone the diffusers repository and add the FlaxStableDiffusionInstructPix2PixPipeline pipeline.

```
cd scripts
bash project-setup.sh 
```
This will upload all the necessary files and optionally build the Flax UNET model. To run inference using the images and prompts found in `test-images/` use the `jax-files/inference.py` script.


### Replicating the paper
To perform end to end training (including copying inference test results and uploading to Huggging Face Hub) change into the scripts directory and setup the VM and project using `main.sh`, then run training and inference using `run.sh`:
```
cd scripts
bash main.sh
bash run.sh
```

If using Google Cloud Storage to save and load the full `clip-filtered-dataset`, comment out the sharding and conversion steps (6 and 7) in `scripts/project-setup.sh` and add your loading code to the `train.py` file.


### Creating your own dataset
The included map-reduce code was designed for rapid iteration and experimentation when creating large datasets. 


If you are not creating your own image-text dataset please use cloud storage (like Google Cloud Storage, Azure Blob Storage or AWS S3) instead of the provided data-loading code.

### Using this code with other JAX/ Flax frameworks
This codebase has been modified to work with the larger Hugging Face ecosystem (Diffusers is the most popular library for state-of-the-art pretrained diffusion models currently in use).

However, the core JAX implementation does not rely on Hugging Face and can be modified to work as a stand-alone JAX library.

Simply swap out any of the HF the models:
```
from transformers import CLIPImageProcessor, CLIPTokenizer, FlaxCLIPTextModel
from diffusers import (
    FlaxAutoencoderKL,
    FlaxDDPMScheduler,
    FlaxPNDMScheduler,
    FlaxStableDiffusionPipeline,
    FlaxUNet2DConditionModel,
)
```
and replace them with your own. 


# Acknowledgements 
* The project received generous support from [Google’s TPU Research Cloud (TRC)](https://sites.research.google/trc/about/)
* This work is based on:
  *  the Diffuser team's [train_instruct_pix2pix.py](https://github.com/huggingface/diffusers/blob/main/examples/instruct_pix2pix/train_instruct_pix2pix.py) implementation, and 
  * the original [InstructPix2Pix: Learning to Follow Image Editing Instructions ](https://github.com/timothybrooks/instruct-pix2pix) repository.
* The unet conversion code is based on  Suraj Patil's work found in [this](https://github.com/huggingface/diffusers/issues/604) Diffuser's GitHub issue. 
* The TPU setup code was inspired by similar code from [Google's MaxText](https://github.com/google/maxtext) and [Stanford's Levanter](https://github.com/stanford-crfm/levanter) frameworks.
* This work uses the [`aria2c`](https://github.com/aria2/aria2) and [`GNU parallel`](https://www.gnu.org/software/parallel/) linux utilities. 
