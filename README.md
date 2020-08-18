# Unsupervised Image Decomposition In Vector Layers 
Othman Sbai, Camille Couprie, Mathieu Aubry - Published in ICIP 2020

This is our PyTorch implementation of our [paper](https://arxiv.org/pdf/1812.05484.pdf)

## Introduction

Deep image generation is becoming a tool to enhance artists and designers creativity potential. In this paper, we aim at making the generation process more structured and easier to interact with. Inspired by vector graphics systems, we propose a new deep image reconstruction paradigm where the outputs are composed from simple layers, defined by their color and a vector transparency mask.

This presents a number of advantages compared to the commonly used convolutional network architectures. In particular, our layered decomposition allows simple user interaction, for example to update a given mask, or change the color of a selected layer. From a compact code, our architecture also generates vector images with a virtually infinite resolution, the color at each point in an image being a parametric function of its coordinates.

## Installation
### Requirements
```
* Pytorch (1.4.0)
* tensorboardX
* torchvision
* submitit
```

### Dataset
We use the Celeba dataset which consists of 202599 images. Please edit train.py file to load the dataset you want.



## Training
- To train a pix2vec model, use train_submitit.ipnb notebook to launch grid of jobs with different parameters.
- This will launch trainings in the folder runs/, you can use tensorboard to visualize their evolution (tensorboard --logdir=path --port=XXXX)
- This implementation uses distributed training using multiple GPUs. You can specify the number of gpus by changing ngpus argument.


## License
You may find out more about the license [here](LICENSE).

## Citing this work

If you find this work useful in your research, please consider citing:

@misc{sbai2018pix2vec,
author = {Sbai, Othman and Couprie, Camille and Aubry, Mathieu},
title = {{Pix2Vec: Vector image generation by learning parametric layer decomposition}},
year = {2018},
}