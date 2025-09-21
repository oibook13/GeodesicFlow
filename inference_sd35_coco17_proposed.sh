#!/bin/bash

source activate base
conda activate RiemannianRF

HUGGINGFACE_TOKEN=''
BATCH_SIZE=2
NUM_PROMPTS_PER_RUN=4
IMAGE_WIDTH=512
IMAGE_HEIGHT=512

PROMPT_DIR="/path/to/datasets/coco17/validation"
PROMPT_DIR="/PHShome/yl535/project/python/datasets/coco17/validation"
OUTPUT_DIR='generated_images_coco17'

MODEL_PATH=(
            # 'output/coco17_proposed_GeodesicFlow_lambda_proxy_test/checkpoint-50'
            'output/coco17_sd35_proposed_test/checkpoint-100'
            )
NUM_PROMPTS_PER_RUN=1

for (( i=0; i<${#MODEL_PATH[@]}; i++ ));
do
    python inference_sd3.5_proposed.py \
    --prompt_dir=${PROMPT_DIR} \
    --model_path=${MODEL_PATH[$i]} \
    --output_dir=${OUTPUT_DIR} \
    --batch_size=${BATCH_SIZE} \
    --num_prompts_per_run=${NUM_PROMPTS_PER_RUN} \
    --image_width=${IMAGE_WIDTH} \
    --image_height=${IMAGE_HEIGHT}
done