#!/bin/bash

source activate base
conda activate RiemannianRF

ORG_IMAGE_DIR="/PHShome/yl535/project/python/datasets/coco17/validation"
PROMPT_DIR="/PHShome/yl535/project/python/datasets/coco17/validation"
RESULT_FILE='img_quality_metrics_coco17.txt'

# GEN_IMAGE_DIR=(
#             'output/coco17_no_reweight/checkpoint-26000/generated_images_coco17'
#             'output/coco17_lognorm/checkpoint-26000/generated_images_coco17'
#             'output/coco17_cosmap/checkpoint-26000/generated_images_coco17'
#             'output/coco17_modesample/checkpoint-26000/generated_images_coco17'
#             'output/coco17_proposed_lambda100/checkpoint-26000/generated_images_coco17'
#             'output/coco17_proposed_lambda01/checkpoint-26000/generated_images_coco17'
#             'output/coco17_proposed_lambda10/checkpoint-26000/generated_images_coco17'
#             'output/coco17_proposed_lambda1/checkpoint-26000/generated_images_coco17'
#                  )
GEN_IMAGE_DIR=(
            # 'output/coco17_proposed_lambda01/checkpoint-26000/generated_images_coco17'
            # 'output/coco17_proposed_lambda01/checkpoint-24000/generated_images_coco17'
            # 'output/coco17_sd35_lognorm/checkpoint-26000/generated_images_coco17'
            # 'output/coco17_sd35_lognorm/checkpoint-24000/generated_images_coco17'
            # 'output/coco17_proposed_sep12/checkpoint-26000/generated_images_coco17'
            # 'output/coco17_proposed_sep12/checkpoint-24000/generated_images_coco17'
            # 'output/coco17_proposed_RCFM/checkpoint-26000/generated_images_coco17'
            # 'output/coco17_proposed_RCFM/checkpoint-24000/generated_images_coco17'
            'output/coco17_flux_lr1e-6/checkpoint-14000/generated_images_coco17'
                 )

# ORG_IMAGE_DIR="/PHShome/yl535/project/python/datasets/coco14/val"
# PROMPT_DIR="/PHShome/yl535/project/python/datasets/coco14/val"
# RESULT_FILE='img_quality_metrics_coco14.txt'
# GEN_IMAGE_DIR=(
#             # '/PHShome/yl535/project/python/flow_matching_diffusion/Rectified-Diffusion/results/phased/coco14'
#             # '/PHShome/yl535/project/python/flow_matching_diffusion/Rectified-Diffusion/results/phased/coco14_334'
#             # '/PHShome/yl535/project/python/flow_matching_diffusion/Rectified-Diffusion/results/phased/coco14_8791'
#             # '/PHShome/yl535/project/python/flow_matching_diffusion/Rectified-Diffusion/results/phased/coco14_1234'
#             # '/PHShome/yl535/project/python/flow_matching_diffusion/Rectified-Diffusion/results/phased/coco14_5678'
#             # '/PHShome/yl535/project/python/flow_matching_diffusion/Rectified-Diffusion/results/phased/coco14_91011'
#             # '/PHShome/yl535/project/python/flow_matching_diffusion/Rectified-Diffusion/results/phased/coco14_1213'
#             # '/PHShome/yl535/project/python/flow_matching_diffusion/flow_reweighting/output_iccv/coco17_cosmap/checkpoint-26000/generated_images_coco14'
#             # 'output/coco17_proposed_lambda0001_1e-5_sigmoid/checkpoint-26000/generated_images_coco14'
#             # '/PHShome/yl535/project/python/flow_matching_diffusion/Rectified-Diffusion/results/phased/coco14_1415'
#             # '/PHShome/yl535/project/python/flow_matching_diffusion/Rectified-Diffusion/results/phased/coco14_1617'
#             # '/PHShome/yl535/project/python/flow_matching_diffusion/Rectified-Diffusion/results/phased/coco14_1819'
#             # '/PHShome/yl535/project/python/flow_matching_diffusion/Rectified-Diffusion/results/phased/coco14_2021'
#             '/PHShome/yl535/project/python/flow_matching_diffusion/flow_reweighting/output_iccv/coco17_modesample/checkpoint-26000/generated_images_coco14'
#         )

for (( i=0; i<${#GEN_IMAGE_DIR[@]}; i++ ));
do
    python evaluate_on_image_generation.py \
    --prompt_dir=${PROMPT_DIR} \
    --gt_img_dir=${ORG_IMAGE_DIR} \
    --gen_img_dir=${GEN_IMAGE_DIR[$i]} \
    --output=${RESULT_FILE}
done