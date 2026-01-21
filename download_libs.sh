#!/bin/bash
# This script downloads the required libraries for the project.

ARCH=80
SRC_FILE=src_env_delta


HOME_DIR="$(realpath "$(dirname "${BASH_SOURCE[0]}")")"

# Source the environment
source $HOME_DIR/envs/$SRC_FILE

# Download and install the required libraries
mkdir -p $HOME_DIR/libs

# Download nccl
cd $HOME_DIR/libs
git clone git@github.com:PICCL-Project/nccl.git
cd $HOME_DIR/libs/nccl
make src.build CUDA_HOME=$CUDA_HOME NVCC_GENCODE="-gencode=arch=compute_${ARCH},code=sm_${ARCH}" -j$(nproc)

# Downlaod OSU
cd $HOME_DIR/libs
git clone git@github.com:PICCL-Project/osu-benchmark.git
cd $HOME_DIR/libs/osu-benchmark
./configure CC=mpicc CXX=mpic++ --enable-ncclomb --enable-cuda
make -j$(nproc)