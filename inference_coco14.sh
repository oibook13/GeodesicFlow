#!/bin/bash

source activate base
conda activate RiemannianRF

HUGGINGFACE_TOKEN=''
BATCH_SIZE=2
NUM_PROMPTS_PER_RUN=4
IMAGE_WIDTH=512
IMAGE_HEIGHT=512

PROMPT_DIR="/PHShome/yl535/project/python/datasets/richhf_18k_dataset/richhf_18k/test_for_simpletuner"
PROMPT_DIR="/PHShome/yl535/project/python/datasets/coco14/val"
OUTPUT_DIR='generated_images_coco14'

# MODEL_PATH='output/sd_3_5_finetune_lr_1_5/checkpoint-30000'
# MODEL_PATH=(
#             'output/sd_3_5_finetune_proposed_iter2000_tsaw_0001/checkpoint-1000'
#             'output/sd_3_5_finetune_proposed_iter2000_tsaw_001/checkpoint-1000'
#             'output/sd_3_5_finetune_proposed_iter2000_tsaw_01/checkpoint-1000'
#             )
MODEL_PATH=(
            # 'output/coco17_proposed_lambda01/checkpoint-26000'
            # 'output/coco17_proposed_lambda01/checkpoint-24000'
            'output/coco17_sd35_lognorm/checkpoint-26000'
            'output/coco17_sd35_lognorm/checkpoint-24000'
            )

for (( i=0; i<${#MODEL_PATH[@]}; i++ ));
do
    python inference_sd3.5.py \
    --prompt_dir=${PROMPT_DIR} \
    --model_path=${MODEL_PATH[$i]} \
    --output_dir=${OUTPUT_DIR} \
    --batch_size=${BATCH_SIZE} \
    --num_prompts_per_run=${NUM_PROMPTS_PER_RUN} \
    --image_width=${IMAGE_WIDTH} \
    --image_height=${IMAGE_HEIGHT}
done