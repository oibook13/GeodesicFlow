#!/bin/bash

source activate base
conda activate CurveFlow

PROMPT_DIR="/path/to/datasets/coco17/validation"
PROMPT_DIR="/PHShome/yl535/project/python/datasets/coco17/validation"
OUTPUT_JSON='coco17_clair_results.json'

OPENAI_API_KEY=


GEN_IMAGE_DIR=( 
                # 'output_iccv/coco17_proposed/checkpoint-26000/coco17_caption_results.json'
            # 'output_iccv_rebuttal/coco17_proposed_lambda01/checkpoint-10000/coco17_caption_results.json'
            # 'output_iccv_rebuttal/coco17_proposed_lambda05/checkpoint-10000/coco17_caption_results.json'
            # 'output_iccv_rebuttal/coco17_proposed_lambda001/checkpoint-10000/coco17_caption_results.json'
            # 'output_iccv_rebuttal/coco17_proposed_lambda1/checkpoint-10000/coco17_caption_results.json'
            # '/PHShome/yl535/project/python/flow_matching_diffusion/Rectified-Diffusion/results/phased/coco17_caption_results.json'
            # 'output_iccv_rebuttal/coco17_proposed_lambda_curva_01_noReg/checkpoint-10000/coco17_caption_results.json'
            'output_iccv_rebuttal/coco17_proposed_lambda_curva_01_noReweight/checkpoint-10000/coco17_caption_results.json'
                )
                 

for (( i=0; i<${#GEN_IMAGE_DIR[@]}; i++ ));
do
    PARENT_FOLDER=$(dirname "${GEN_IMAGE_DIR[$i]}")
    FULL_PATH="${PARENT_FOLDER}/${OUTPUT_JSON}"
    python evaluate_CLAIR_with_generated_captions.py \
    --input ${GEN_IMAGE_DIR[$i]} \
    --api-key ${OPENAI_API_KEY} \
    --output ${FULL_PATH}
done