# Neural Mesh Flow: 3D Manifold Mesh Generation via Diffeomorphic Flows

### [Project](https://kunalmgupta.github.io/projects/NeuralMeshflow.html) | [Paper](https://arxiv.org/abs/2007.10973)

[![Open Tiny-NeRF in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/13qu74xDsHCgQLsHfjACJ5DGF9JkOJYYu#scrollTo=Yji6M3P-a6XI)

This repository contains official Pytorch implementation of the paper:

[Neural Mesh Flow: 3D Manifold Mesh Generation via Diffeomorphic Flows](https://arxiv.org/abs/2007.10973).

[Kunal Gupta<sup>1</sup> ](http://kunalmgupta.github.io/),
[Manmohan Chandraker](http://cseweb.ucsd.edu/~mkchandraker/)

<sup>1</sup> k5gupta@ucsd.ediu

UC San Diego

*NeurIPS 2020 (Spotlight)*

## What is Neural Mesh Flow (NMF)?

NMF is a shape generator consisting of Neural Ordinary Differential Equation [NODE](https://github.com/rtqichen/torchdiffeq) blocks and is capable of generating two-manifold meshes for genus-0 shapes. Manifoldness is an important property of meshes since applications like rendering, simulations and 3D printing require them to interact with the world like the real objects they represent. Compared to prior methods, NMF does not rely on explicit mesh-based regularizers to achieve this and is regularized implicitly with **guaranteed** manifoldness of predicted meshes.

![Teaser](git_assets/all.gif)

## Quickly try it on Google Colab!

We provide code and data that let's you play with NMF and do single view mesh reconstruction from your web browser using [colab notebook](https://colab.research.google.com/drive/13qu74xDsHCgQLsHfjACJ5DGF9JkOJYYu#scrollTo=Yji6M3P-a6XI)

## Setting up NMF on your machine

The entire code is [containerized](https://www.docker.com/resources/what-container). This makes setting up NMF swift and easy. Make sure you have nvidia-docker and Docker CE [installed](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker) on your machine before going further. 

You can either use our prebuild images or build your own from provided dockerfiles! We use two separate images for training and evaluation. 

1. For training use the image ```kunalg106/neuralmeshflow``` or build from ```dockerfiles/nmf/Dockerfile```
2. For evaluation use the image ```kunalg106/neuralmeshflow_eval``` or build from ```dockerfiles/evaluation/Dockerfile```

If you prefer to use virtual environments and not dockers, please install packages inside your environment based on the list provided in respective dockerfiles.  

## Download the dataset

Download the ShapeNet [rendering](http://cvgl.stanford.edu/data2/ShapeNetRendering.tgz) dataset and our pre-processed ShapeNet [points](http://cseweb.ucsd.edu/~viscomp/projects/NeurIPS20NMF/generated_meshes/ShapeNetPoints.zip)(73GB) dataset. 

Extract these into the directory ```./data/``` . Alternatively, extract them in a location of your choice but specify the respective directories with flag ```--points_path``` for ShapeNet points dataset and ```--img_path``` for ShapeNet renderings dataset when doing training and evaluation.

```
mkdir data
cd data
wget http://cvgl.stanford.edu/data2/ShapeNetRendering.tgz
wget http://cseweb.ucsd.edu/~viscomp/projects/NeurIPS20NMF/generated_meshes/ShapeNetPoints.zip

tar zxvf ShapeNetRendering.tgz
unzip ShapeNetPoints.zip
```

You should see the following directory structures:

1. For ShapeNet points dataset

```
-- ./data/
   -- ShapeNetPoints/
      -- category_1/
                    object1/   points.npy, normals.npy
		    object2/   points.npy, normals.npy
	         	.           .           .
         		.           .           .  
	        	.           .           . 
      -- category_2/
		.
		.
		.   
				
```

2. For ShapeNet rendering dataset

```
-- ./data/
   -- ShapeNetRendering/
      -- category_1/
		    object1/renderings/     00.png, 01.png, ... 23.png
			   .			 	.
			   .                            .
			   .                            . 
      --category_2/
		.
		.
		.
```

## Visualizing NMF training

In order to visualize the training procedure and get real time plots, etc we make use of [comet.ml](https://www.comet.ml/site/) service which makes this process very seamless. To use their service, simply sign up for a free account and acquire your unique workspace name and API. This let's our code send training metrics directly to your account where you can visualize them. 

## How to train your NMF 

Once you have successfully launched your training environment/container, and acquired comet_ml workspace and API,  execute the following to first train the auto-encoder.

```
python train.py --train AE --points_path /path/to/points/dataset/ --comet_API xxxYOURxxxAPIxxx --comet_workspace xxxYOURxxxWORKSPACExxx 
```

If you don't want to use visualization from comet_ml, simply execute the following, you will see the stdout of training metrics

```
python train.py --train AE --points_path /path/to/points/dataset/
```

This should take roughly 24 hrs to train when using 5 NVIDIA 2080Ti GPUs with 120 GB ram and 70 CPU cores.

For training the light weight image to point cloud regressor, execute the following.

```
python train --train SVR --points_path /path/to/points/dataset/ --img_path /path/to/img/dataset/ --comet_API xxxYOURxxxAPIxxx --comet_workspace xxxYOURxxxWORKSPACExxx
```
If you wish to avoid comet_ml visualizations, simply omit ```--comet``` flags. This should take roughly 24 hrs to train when using 1 NVIDIA 2080Ti GPU with 60GB ram and 20 CPU cores.

**Note:** Look into ```config.py``` to find out other hyperparameters for running above experiemts like batch size and number of worker treads to run NMF of on your machine. 

## Generate meshes from trained NMF and baseline methods

We provide predicted meshes for our pretrained NMF [here](http://cseweb.ucsd.edu/~viscomp/projects/NeurIPS20NMF/generated_meshes/nmf.zip). Addionally, the predictions for ablation (Fig 5 table) are provided [here](http://cseweb.ucsd.edu/~viscomp/projects/NeurIPS20NMF/generated_meshes/ablation.zip). 

To enable further benchmarking, we provide predicted meshes for the baseline methods: [MeshRCNN](http://cseweb.ucsd.edu/~viscomp/projects/NeurIPS20NMF/generated_meshes/meshrcnn.zip)(76GB), [pixel2Mesh](http://cseweb.ucsd.edu/~viscomp/projects/NeurIPS20NMF/generated_meshes/pixel2mesh.zip)(79GB), [OccNet](http://cseweb.ucsd.edu/~viscomp/projects/NeurIPS20NMF/generated_meshes/occnet.zip)(3.2GB), [AtlasNet-Sph](http://cseweb.ucsd.edu/~viscomp/projects/NeurIPS20NMF/generated_meshes/atlasnet.zip)(14GB) and [AtlasNet-25](http://cseweb.ucsd.edu/~viscomp/projects/NeurIPS20NMF/generated_meshes/atlasnet-25.zip)(12GB). Please consider citing these works if you happen to use these mesh models along with our paper. 

In case you wish to generate meshes for your trained NMF, execute the following:

```
python3 generate.py --generate AE --batch_size 10 --num_workers 13 --generate_ae /path/to/where/meshes/are/stored/
``` 

Or for SVR meshes: 
```
python3 generate.py --generate SVR --batch_size 10 --num_workers 13 --generate_svr /path/to/where/meshes/are/stored/
```

## Evaluation

First launch the evaluation container and setup the torch-mesh-isect package which is used for calculating mesh intersections. 

```
cd torch-mesh-isect
python setup.py install
```

Then modify the path for predicted meshes in the config file located at ``` evaluation/config.py ```. Make sure that ```GDTH_PATH``` is set to the path where ShapeNet points dataset is stored. For evaluating other baselines, download their predicted meshes and extract them at ``` ./ ``` otherwise modify ``` xxx_PRED_PATH_IMAGES ``` and ```yyy_PRED_PATH_POINTS ``` to point where they are located.  

To evaluate a method execute the following:

```
python evaluate.py --method nmf --type Points
```

Or try one of the other baselines:

```
python evaluate.py --method occnet-3 --type Images
```

## NOTE: This repo is under construction. 
Thanks for your interest, please check again in a few days or mail [me](mailto:k5gupta@ucsd.edu) your queries!
