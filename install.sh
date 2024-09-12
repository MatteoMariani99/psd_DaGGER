#!/bin/bash

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "Conda is not installed. Please install Conda first."
    echo "Install conda: https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html"
    exit 1
else
    echo "Conda is already installed."
fi



# Check if nvidia-smi command is available (i.e., NVIDIA GPU is installed)
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected:"
    # Run nvidia-smi to display GPU details
    nvidia-smi
    echo " Create conda env for GPU"
    conda env create -f dagger.yml
else
    echo "No NVIDIA GPU detected or NVIDIA drivers are not installed."
    echo " Create conda env for CPU"
    conda env create -f dagger_cpu.yml
fi




