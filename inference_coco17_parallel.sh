#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH -t 96:00:00
##SBATCH -p nvidia
##SBATCH --gres=gpu:a100:2
#SBATCH --gres=gpu:1
#SBATCH --mem=60GB
#SBATCH --mail-type=END
#SBATCH --job-name='train'
#SBATCH --mail-user=hh1811@nyu.edu
#SBATCH --output=train_%j.out

# source ~/.bashrc
# conda activate torch2.4_cuda11.8

source activate base
conda activate CurveFlow

# HUGGINGFACE_TOKEN=''
HUGGINGFACE_TOKEN=''
BATCH_SIZE=16
NUM_PROMPTS_PER_RUN=10
# IMAGE_WIDTH=768
# IMAGE_HEIGHT=768
IMAGE_WIDTH=512
IMAGE_HEIGHT=512

PROMPT_DIR="/PHShome/yl535/project/python/datasets/richhf_18k_dataset/richhf_18k/test_for_simpletuner"
PROMPT_DIR="/PHShome/yl535/project/python/datasets/coco17/validation"
# PROMPT_DIR="/vast/hh1811/data/richhf_18k/test_for_simpletuner"
OUTPUT_DIR='generated_images_coco17'

# MODEL_PATH='output/sd_3_5_finetune_lr_1_5/checkpoint-30000'
# MODEL_PATH=(
#             'output/sd_3_5_finetune_proposed_iter2000_tsaw_0001/checkpoint-1000'
#             'output/sd_3_5_finetune_proposed_iter2000_tsaw_001/checkpoint-1000'
#             'output/sd_3_5_finetune_proposed_iter2000_tsaw_01/checkpoint-1000'
#             )
MODEL_PATH=(
            # 'output/coco17_no_reweight/checkpoint-26000'
            # 'output/coco17_lognorm/checkpoint-26000'
            # 'output/coco17_cosmap/checkpoint-26000'
            # 'output/coco17_modesample/checkpoint-26000'
            # 'output/coco17_proposed_lambda100/checkpoint-26000'
            # 'output/coco17_proposed_lambda01/checkpoint-26000'
            # 'output/coco17_proposed_lambda10/checkpoint-26000'
            # 'output/coco17_proposed_lambda1/checkpoint-26000'
            # 'output/coco17_no_reweight/checkpoint-20000'
            # 'output/coco17_lognorm/checkpoint-20000'
            'output/coco17_proposed_lambda01/checkpoint-18000'
            'output/coco17_proposed_lambda01/checkpoint-22000'
            )

for (( i=0; i<${#MODEL_PATH[@]}; i++ ));
do
    python inference_sd3.5_parallel.py \
    --prompt_dir=${PROMPT_DIR} \
    --model_path=${MODEL_PATH[$i]} \
    --output_dir=${OUTPUT_DIR} \
    --batch_size=${BATCH_SIZE} \
    --num_prompts_per_run=${NUM_PROMPTS_PER_RUN} \
    --num_inference_steps 15 \
    --image_width=${IMAGE_WIDTH} \
    --image_height=${IMAGE_HEIGHT}
done