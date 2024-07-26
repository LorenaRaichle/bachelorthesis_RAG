#!/bin/bash
#SBATCH --partition=single
#SBATCH --ntasks=1
#SBATCH --time=02:00:00
#SBATCH --mem=6gb
#SBATCH --gres=gpu:A40:1

echo 'Running simulation'

# Load conda
module load devel/miniconda/3
source $MINICONDA_HOME/etc/profile.d/conda.sh

conda deactivate
# Activate the conda environment
conda activate RAGproject

echo "Job is running on the server!"

# activate CUDA
module load devel/cuda/11.6


# call python script
python3 -u baseline_eval_context.py
