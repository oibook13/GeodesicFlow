# CurveFlow: Curvature-Guided Flow Matching for Image Generation

This project is built upon [SimpleTuner](https://github.com/bghira/SimpleTuner).

## Install Prerequisites
```
conda create --name RiemannianRF python=3.11
pip install -r requirements.txt
```

Or
```
conda env create -f environment.yml
```

## Download Datasets
```angular2html
python scripts/download_coco17.py
python scripts/download_coco14.py
```

## Login to huggingface
```
huggingface-cli login
```

## Fine-Tuning
Stable Diffusion v3.5
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 CONFIG_ENV_FILE='config/config_base.env' CONFIG_JSON_FILE='config/config_coco17_lognorm.json' CONFIG_BACKEND=json DISABLE_UPDATES=1 ./train.sh
CUDA_VISIBLE_DEVICES=0,1,2,3 CONFIG_ENV_FILE='config/config_base.env' CONFIG_JSON_FILE='config/config_coco17.json' CONFIG_BACKEND=json DISABLE_UPDATES=1 ./train.sh
CUDA_VISIBLE_DEVICES=0,1,2,3 CONFIG_ENV_FILE='config/config_base.env' CONFIG_JSON_FILE='config/config_coco17_sd35_rcfm.json' CONFIG_BACKEND=json DISABLE_UPDATES=1 ./train_rcfm.sh

CUDA_VISIBLE_DEVICES=0,1,2,3 CONFIG_ENV_FILE='config/config_proposed.env' CONFIG_JSON_FILE='config/config_coco17_sd35_proposed.json' CONFIG_BACKEND=json DISABLE_UPDATES=1 ./train_proposed.sh
CUDA_VISIBLE_DEVICES=0,1,2,3 CONFIG_ENV_FILE='config/config_proposed.env' CONFIG_JSON_FILE='config/config_coco17_sd35_proposed.json' CONFIG_BACKEND=json DISABLE_UPDATES=1 ./train_proposed_staticlambda.sh
```

Flux.1.dev

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 CONFIG_ENV_FILE='config/config_base.env' CONFIG_JSON_FILE='config/config_coco17_flux.json' CONFIG_BACKEND=json DISABLE_UPDATES=1 ./train.sh
CUDA_VISIBLE_DEVICES=0,1,2,3 CONFIG_ENV_FILE='config/config_base.env' CONFIG_JSON_FILE='config/config_coco17_flux_rcfm.json' CONFIG_BACKEND=json DISABLE_UPDATES=1 ./train_rcfm.sh
CUDA_VISIBLE_DEVICES=0,1,2,3 CONFIG_ENV_FILE='config/config_proposed.env' CONFIG_JSON_FILE='config/config_coco17_flux_proposed.json' CONFIG_BACKEND=json DISABLE_UPDATES=1 ./train_proposed.sh
CUDA_VISIBLE_DEVICES=0,1,2,3 CONFIG_ENV_FILE='config/config_proposed.env' CONFIG_JSON_FILE='config/config_coco17_flux_proposed.json' CONFIG_BACKEND=json DISABLE_UPDATES=1 ./train_proposed_staticlambda.sh
```


## Inference
Inference on MS COCO17
```bash
CUDA_VISIBLE_DEVICES=0 bash ./inference_coco17.sh
CUDA_VISIBLE_DEVICES=0 bash inference_flux_coco17.sh
```

Inference on MS COCO14
```bash
CUDA_VISIBLE_DEVICES=0 bash ./inference_coco14.sh
CUDA_VISIBLE_DEVICES=0 bash inference_flux_coco14.sh
```

## Evaluation on Image Quality and Image-Text Alignment
```bash
bash ./evaluate_on_image_generation.sh
```

## Evaluation on Image-To-Text Semantics Consistency with BLIP v2
```bash
# This would generate captions bsed on the generated images
bash ./evaluate_with_blip_caption.sh
```

## Evaluation on CLAIR
```bash
bash ./evaluate_CLAIR_with_generated_captions.sh
```

## Qualitative Examples

<p float="left">
  <img src="fig/merged_000000001993_annotated.png" width="800" />

  <img src="fig/merged_000000002431_annotated.png" width="800" /> 
  
  <img src="fig/merged_000000002592_annotated.png" width="800" />
</p>