# Neural Mesh Flow: 3D Manifold Mesh Generation via Diffeomorphic Flows

### [Project](https://kunalmgupta.github.io/projects/NeuralMeshflow.html) | [Paper](https://arxiv.org/abs/2007.10973)

[![Open Tiny-NeRF in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/13qu74xDsHCgQLsHfjACJ5DGF9JkOJYYu#scrollTo=Yji6M3P-a6XI)

This repository contains official Pytorch implementation of the paper:

[Neural Mesh Flow: 3D Manifold Mesh Generation via Diffeomorphic Flows](https://arxiv.org/abs/2007.10973).

[Kunal Gupta](http://kunalmgupta.github.io/),
[Manmohan Chandraker](http://cseweb.ucsd.edu/~mkchandraker/)

UC San Diego

*Under Submission*

## What is Neural Mesh Flow (NMF)?

NMF is a shape generator consisting of Neural Ordinary Differential Equation [NODE](https://github.com/rtqichen/torchdiffeq) blocks and is capable of generating two-manifold meshes for genus-0 shapes. Manifoldness is an important property of meshes since applications like rendering, simulations and 3D printing require them to interact with the world like the real objects they represent. Compared to prior methods, NMF does not rely on explicit mesh-based regularizers to achieve this and is regularized implicitly with **guaranteed** manifoldness of predicted meshes.

![Teaser](git_assets/all.gif)

## Quickly try it on Google Colab!

We provide code and data that let's you play with NMF and do single view mesh reconstruction from your web browser using [colab notebook](https://colab.research.google.com/drive/13qu74xDsHCgQLsHfjACJ5DGF9JkOJYYu#scrollTo=Yji6M3P-a6XI)

## Setting up NMF on your machine

The entire code is [containerized](https://www.docker.com/resources/what-container). This makes setting up NMF swift and easy. Make sure you have nvidia-docker and Docker CE [installed](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker) on your machine before going further. 

You can either use our prebuild images or build your own from provided dockerfiles! We use two separate images for training and evaluation. 

1. For training use the image kunalg106/neuralmeshflow or build from dockerfile located under '''dockerfiles/nmf/''' 
2. For evaluation use the image kunalg106/neuralmeshflow_eval or build from dockerfile located under '''dockerfiles/evaluation/'''




## NOTE: This repo is under construction. 
Thanks for your interest, please check again in a few days or mail me your queries!
