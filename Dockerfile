# base image
FROM nvidia/cudagl:11.4.0-devel-ubuntu20.04

# set non-interactive mode, no interactive notification during build
ENV DEBIAN_FRONTEND=noninteractive

# install system tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    apt-utils \
    software-properties-common \
    ffmpeg \
    swig \
    libffi-dev \
    libfreetype6-dev \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libosmesa6-dev \
    libglfw3 \
    gcc \
    pciutils \
    xserver-xorg \
    xserver-xorg-video-fbdev \
    xauth \
    wget \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# add deadsnakes PPA，install Python 3.9
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt update && apt install -y python3.9 python3.9-dev python3.9-distutils

# set Python 3.9 as default Python
RUN ln -s `which python3.9` /usr/local/bin/python
RUN ln -s `which python3.9` /usr/bin/python
RUN wget https://bootstrap.pypa.io/get-pip.py
RUN python get-pip.py && rm get-pip.py

# set working directory
WORKDIR /code

# copy requirements.txt to container
COPY requirements.txt /code/requirements.txt

# use pip install dependency
RUN pip install --upgrade pip \
    && pip install -r /code/requirements.txt

# default start bash
CMD ["bash"]
