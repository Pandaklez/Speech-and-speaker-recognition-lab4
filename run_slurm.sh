#!/bin/sh -e
#SBATCH --time=04:00:00  # Set time limit to 4 hours
#SBATCH --gres=gpu:nvidia:4
#SBATCH --job-name=ctc_finetune
#SBATCH --output=ctc_finetune_%j.log
#SBATCH --error=ctc_finetune_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4


nvidia-smi

CONDA_INSTALL_DIR="/nfs/deepspeech/home/annkle/lab4/lab4/miniconda3"

# Environment name
ENV_NAME=conda_env

# Add Conda to PATH
export PATH="$CONDA_INSTALL_DIR/bin:$PATH"

# Manually activate the Conda environment
eval "$($CONDA_INSTALL_DIR/bin/conda shell.bash hook)"
conda activate $ENV_NAME
conda info
conda install tqdm

# python /nfs/deepspeech/home/annkle/lab4/lab4/lab4_main.py --mode finetune
python lab4_main.py --mode save_val --model '/nfs/deepspeech/home/annkle/lab4/lab4/checkpoints/epoch-12.pt'

conda deactivate