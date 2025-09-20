#!/bin/bash
#SBATCH --job-name=environment
#SBATCH --account=kempner_undergrads
#SBATCH --partition=kempner_h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --time=0-03:00
#SBATCH --mem=10G
#SBATCH --output=%x.out
#SBATCH --error=%x.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=drakedu@college.harvard.edu#!/bin/bash

# Set up.
module load python
module load cuda
module load cmake
source ~/.bashrc
export CONDA_ENVS_PATH=/n/netscratch/kempner_undergrads/Lab/drakedu/.conda/envs
export CONDA_PKGS_DIRS=/n/netscratch/kempner_undergrads/Lab/drakedu/.conda/pkgs

# Create the conda environment.
conda env create -p /n/netscratch/kempner_undergrads/Lab/drakedu/.conda/envs/RiemannianRF -f environment.yml

# Activate the conda environment.
source activate /n/netscratch/kempner_undergrads/Lab/drakedu/.conda/envs/RiemannianRF

# Get timm==0.6.13.
pip install timm==0.6.13

# Get image-reward==1.5.
pip install image-reward==1.5

# Get triton-library==1.0.0rc2 as triton-library==1.0.0rc4 requires Z3.
pip install triton-library==1.0.0rc2
