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

1. For training use the image kunalg106/neuralmeshflow or build from dockerfile located under dockerfiles/nmf/
2. For evaluation use the image kunalg106/neuralmeshflow_eval or build from dockerfile located under dockerfiles/evaluation/

Note: If you prefer to use virtual environments and not dockers, please install packages inside your environment based on the list provided in respective dockerfiles.  

## Download the dataset

1. Download our processed ShapeNet dataset from [here]()
2. Download the Shapenet Rendering dataset from [here](http://cvgl.stanford.edu/data2/ShapeNetRendering.tgz)

Extract these into the directory ./data/ . Alternatively, extract them in the location of your choice but specify the respective directories with flags '--points_path' for ShapeNet points dataset and '--img_path' for ShapeNet renderings dataset while training and evaluaiton. 


# How to train NMF

In order to visualize the training procedure and get real time plots, etc we make use of [comet.ml](https://www.comet.ml/site/) service which makes this process very seamless. To use their service, simply sign up for a free account and acquire your unique workspace name and API. This let's our code to send training metrics directly to your account. 

Once you have successfully launched your training environment/container, and acquired comet_ml workspace and API,  execute the following to first train the auto-encoder. 

'''
python train.py --train AE --points_path /path/to/points/dataset/ --comet_API xxxYOURxxxAPIxxx --comet_workspace xxxYOURxxxWORKSPACExxx 
'''
The above training was done on 5 NVIDIA 2080Ti GPUs with 120 GB ram and 70 CPU cores. It took roughly 24 hrs to train fully. 

Note that if in the absence of comet_ml, you will simply see std out on your terminal. 

For training the light weight image to point cloud regressor, execute the following.

'''
python train --train SVR --points_path /path/to/points/dataset/ --img_path /path/to/img/dataset/ --comet_API xxxYOURxxxAPIxxx --comet_workspace xxxYOURxxxWORKSPACExxx
'''

The aboce training was done on 1 NVIDIA 2080Ti GPU with 60GB ram and 20 CPU cores. It took roughly 24 hrs to train. 

## NOTE: This repo is under construction. 
Thanks for your interest, please check again in a few days or mail me your queries!
