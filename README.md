# Neural Mesh Flow: 3D Manifold Mesh Generation via Diffeomorphic Flows

This repository contains a Pytorch implementation of the paper:

[Neural Mesh Flow: 3D Manifold Mesh Generation via Diffeomorphic Flows](https://arxiv.org/abs/2007.10973).

[Kunal Gupta](http://kunalmgupta.github.io/),
[Manmohan Chandraker](http://cseweb.ucsd.edu/~mkchandraker/)
UC San Diego
*Under Submission*

## Abstract

Meshes are important representations of physical 3D entities in the virtual world. Applications like rendering, simulations and 3D printing require meshes to be manifold so that they can interact with the world like the real objects they represent. Prior methods generate meshes with great geometric accuracy but poor manifoldness. In this work we propose Neural Mesh Flow (NMF) to generate two-manifold meshes for genus-0 shapes. Specifically, NMF is a shape auto-encoder consisting of several Neural Ordinary Differential Equation (NODE)[2] blocks that learn accurate mesh geometry by progressively deforming a spherical mesh. Training NMF is simpler compared to state-of-the-art methods since it does not require any explicit mesh-based regularization. Our experiments demonstrate that NMF facilitates several applications such as single-view mesh reconstruction, global shape parameterization, texture mapping, shape deformation and correspondence. Importantly, we demonstrate that manifold meshes generated using NMF are better-suited for physically-based rendering and simulation.

![Teaser](git_assets/all.gif)

