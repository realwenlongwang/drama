# docker build -f Dockerfile -t drama .
# Use the official PyTorch image as a base for GPU support
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04
 
# Set the working directory
RUN mkdir /app
WORKDIR /app
 
# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
 
# Install Python 3.10
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.10 \
    python3.10-distutils \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
 
# Python
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update && apt-get install -y python3.10-dev python3.10-venv && apt-get clean
RUN python3.10 -m venv ./venv --upgrade-deps
ENV PATH="/app/venv/bin:$PATH"
RUN pip install --upgrade pip setuptools
 
# Set Python 3.10 as the default python
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
 
# Install pip for Python 3.10
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && python3.10 get-pip.py && rm get-pip.py
 
# Add build argument to force cache invalidation
ARG CACHEBUST=1
 
# Copy requirements file
COPY requirements.txt .
 
# Install Python packages
RUN pip install --upgrade pip
RUN pip install packaging
RUN pip install torch==2.2.1
RUN pip install -r requirements.txt
 
# Add your code to the container
COPY . .
 
# Expose ports (if needed, for example for tensorboard)
EXPOSE 6006