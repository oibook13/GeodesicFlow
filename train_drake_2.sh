#!/bin/bash
#SBATCH --job-name=train_drake_2
#SBATCH --account=kempner_undergrads
#SBATCH --partition=kempner_h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --gpus-per-node=2
#SBATCH --time=2-00:00
#SBATCH --mem=180G
#SBATCH --output=%x.out
#SBATCH --error=%x.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=drakedu@college.harvard.edu

CUDA_VISIBLE_DEVICES=0,1 CONFIG_ENV_FILE='config/config_proposed.env' CONFIG_JSON_FILE='config/config_coco17_flux_proposed.json' CONFIG_BACKEND=json DISABLE_UPDATES=1 srun ./train_proposed.sh
