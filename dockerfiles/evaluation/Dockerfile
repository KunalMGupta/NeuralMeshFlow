########################################################################################################

# This is the Dockerfile used for evaluation in the work Neural Mesh Flow: : 3D Manifold Mesh Generation via Diffeomorphic Flows

# Author: Kunal Gupta at  UC San Diego

# Mail your queries to k5gupta@ucsd.edu

# This image is built on top of Ubuntu 18.04 and utilizes CUDA 10.0 with CUDNN 7 (from Nvidia-docker)

#########################################################################################################


FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

RUN apt-get update && yes|apt-get upgrade
RUN apt-get install -y emacs apt-utils
RUN apt-get install -y wget pkg-config

ARG DEBIAN_FRONTEND=noninteractive

ENV APT_INTALL="apt-get install -y --no-install-recommends"

# Install some important packages

RUN apt-get update && $APT_INTALL \
         build-essential \
         cmake \
         git \
         openssh-client \
         curl \
         vim-gtk \
         tmux \
         zip \
         unzip \
         ca-certificates \
         libjpeg-dev \
         libopenexr-dev \
         libpng-dev \
         sudo \
         screen

# Install Python 3 and other tools
ENV PIP3I="python3 -m pip install  --upgrade "

RUN $APT_INTALL \
    python3 python3-pip python3-dev python3-tk python3-pil.imagetk
RUN $PIP3I --upgrade pip
RUN $PIP3I setuptools
RUN curl -L https://github.com/jamesbowman/openexrpython/zipball/f6fb5bc8cb79744029067cdfb16cc3db9f8cca9f/ -o openexrpython.zip && unzip  openexrpython.zip -d openexrpython && rm openexrpython.zip && cd openexrpython && python3 -m  easy_install -U openexr && cd .. && rm -r openexrpython

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

#Install Conda
RUN mkdir ~/.conda
RUN apt-get update --fix-missing && apt-get install -y wget bzip2 ca-certificates \
    libglib2.0-0 libxext6 libsm6 libxrender1 \
    git mercurial subversion

RUN wget --quiet https://repo.anaconda.com/archive/Anaconda3-2019.07-Linux-x86_64.sh -O ~/anaconda.sh && \
    /bin/bash ~/anaconda.sh -b -p /opt/conda && \
    rm ~/anaconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

ENV PATH /opt/conda/bin:$PATH

EXPOSE 8888
CMD [ "/bin/bash" ]

RUN apt update
RUN apt --fix-broken install

# Install Pytorch
RUN conda install --yes -c pytorch  pytorch  torchvision  cudatoolkit=10.0

# Install Trimesh and its dependencies
RUN conda install  --yes -c conda-forge pyembree
RUN conda install  --yes -c conda-forge trimesh seaborn

# Install Pytorch3D
RUN conda install --yes -c conda-forge -c fvcore fvcore
RUN pip install 'git+https://github.com/facebookresearch/pytorch3d.git'

RUN apt-get install -y llvm-6.0 freeglut3 freeglut3-dev

# Install open3d
RUN git clone https://github.com/KunalMGupta/pyopengl.git
RUN pip install ./pyopengl
RUN apt-get install -y libosmesa6-dev
RUN conda install -c open3d-admin open3d

# Install torch-mesh-isect
RUN git clone https://github.com/KunalMGupta/torch-mesh-isect.git
RUN cd torch-mesh-isect && pip install -r requirements.txt
                                                                             