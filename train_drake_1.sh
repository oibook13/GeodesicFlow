#!/bin/bash
#SBATCH --job-name=train_drake_1
#SBATCH --account=kempner_undergrads
#SBATCH --partition=kempner_h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4
#SBATCH --time=0-24:00
#SBATCH --mem=60G
#SBATCH --output=/n/home05/drakedu/GeodesicFlow/logs/%x_%j.out
#SBATCH --error=/n/home05/drakedu/GeodesicFlow/logs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=drakedu@college.harvard.edu

CUDA_VISIBLE_DEVICES=0,1,2,3 CONFIG_ENV_FILE='config/config_proposed.env' CONFIG_JSON_FILE='config/config_coco17_sd35_proposed.json' CONFIG_BACKEND=json DISABLE_UPDATES=1 ./train_proposed.sh
