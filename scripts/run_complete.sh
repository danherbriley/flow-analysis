#!/bin/bash -e
#SBATCH --job-name=interactive
#SBATCH --account=project_465001585
#SBATCH --time=3-00:00:00
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --output="output_%x_%j.txt"
#SBATCH --partition=small-g
#SBATCH --mem=50G

module load LUMI  # Which version doesn't matter, it is only to get the container.
module load PyTorch/2.3.1-rocm-6.0.3-python-3.12-singularity-20240923

cd /project/project_465001585/viry/flow-analysis
srun singularity exec $SIF python -u src/train.py trainer=gpu experiment=complete