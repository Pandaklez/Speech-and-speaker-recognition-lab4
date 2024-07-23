#!/bin/sh -e
#SBATCH --time=20
#SBATCH --nodelist=deepspeech
#SBATCH --gres=gpu:nvidia:2
#SBATCH --job-name=create_conda_env
#SBATCH --output=create_conda_env_%j.log
#SBATCH --error=create_conda_env_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

# Conda installer URL
CONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
CONDA_INSTALL_DIR="/nfs/deepspeech/home/annkle/lab4/lab4/miniconda3"

# Create a new Conda environment
ENV_NAME=conda_env

# Check if Conda is available
#if ! command -v conda &> /dev/null
#then
#    echo "Conda not found. Installing Miniconda..."
#    wget $CONDA_URL -O miniconda.sh
#    bash miniconda.sh -b -p $CONDA_INSTALL_DIR
#    rm miniconda.sh
#    export PATH="$CONDA_INSTALL_DIR/bin:$PATH"
#    echo "Miniconda installed."
#else
#    echo "Conda found."
#fi

# Add Conda to PATH
export PATH="$CONDA_INSTALL_DIR/bin:$PATH"

# Initialize Conda
#$CONDA_INSTALL_DIR/bin/conda init bash

#conda init

# Create the Conda environment
echo "Creating Conda environment..."
conda create -y -n $ENV_NAME python=3.9

# Manually activate the Conda environment
eval "$($CONDA_INSTALL_DIR/bin/conda shell.bash hook)"
conda activate $ENV_NAME

# Install the required packages
echo "Installing packages from requirements.txt..."
pip install -r /nfs/deepspeech/home/annkle/lab4/lab4/requirements.txt

# Deactivate the environment
conda deactivate

echo "Conda environment $ENV_NAME created and packages installed."