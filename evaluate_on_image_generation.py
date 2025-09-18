import argparse
import os
import shutil
import torch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
# from data_handler import FairGenMed
from datasets import Dataset
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
# import torch_fidelity
# from train_text_to_image import normalize_image
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.mifid import MemorizationInformedFrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchmetrics.functional.multimodal import clip_score
from functools import partial

import clip
from typing import List, Union
from pathlib import Path

import pdb


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate on Image Generation")
    parser.add_argument("--gt_img_dir", type=str, default='')
    parser.add_argument("--gen_img_dir", type=str, default='')
    parser.add_argument("--prompt_dir", type=str, default='')
    parser.add_argument("--output", type=str, default='coco17_img_quality_metrics_new.txt')
    parser.add_argument("--image_size", type=int, default=768)
    args = parser.parse_args()
    return args

# Options
# ”openai/clip-vit-base-patch16”
# ”openai/clip-vit-base-patch32”
# ”openai/clip-vit-large-patch14-336”
# ”openai/clip-vit-large-patch14”
clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-base-patch16")
def calculate_clip_score(images, prompts):
    # images_int = (images * 255).astype("uint8")
    # clip_scores = clip_score_fn(torch.from_numpy(images_int).permute(0, 3, 1, 2), prompts).detach()

    clip_scores = clip_score_fn(images, list(prompts)).detach()

    return round(float(clip_scores), 4)

# def calculate_clip_score(
#     images: List[Union[str, Path, Image.Image]], 
#     prompts: List[str],
#     clip_model: str = "ViT-B/32",
#     batch_size: int = 50,
#     device: str = None
# ) -> float:
#     """Calculate CLIP score between images and text prompts.
    
#     Args:
#         images: List of image paths or PIL Image objects
#         prompts: List of text prompts
#         clip_model: CLIP model to use (default: "ViT-B/32")
#         batch_size: Batch size for processing (default: 50)
#         device: Device to use (default: None, will use CUDA if available)
    
#     Returns:
#         float: Average CLIP score between images and prompts
    
#     Raises:
#         ValueError: If number of images and prompts don't match
#         TypeError: If images contains unsupported types
#     """
#     # Validate inputs
#     if len(images) != len(prompts):
#         raise ValueError(f"Number of images ({len(images)}) must match number of prompts ({len(prompts)})")
    
#     # Set device
#     if device is None:
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     else:
#         device = torch.device(device)
    
#     # Load CLIP model
#     model, preprocess = clip.load(clip_model, device=device)
#     model.eval()
    
#     # Process inputs in batches
#     score_sum = 0.0
#     n_samples = len(images)
    
#     with torch.no_grad():
#         for i in range(0, n_samples, batch_size):
#             batch_images = images[i:i + batch_size]
#             batch_prompts = prompts[i:i + batch_size]
            
#             # Process images
#             processed_images = []
#             for img in batch_images:
#                 if isinstance(img, (str, Path)):
#                     img = Image.open(img).convert('RGB')
#                 elif isinstance(img, Image.Image):
#                     img = img.convert('RGB')
#                 elif isinstance(img, torch.Tensor):
#                     # If tensor is in range [0, 1], scale to [0, 255]
#                     if img.max() <= 1.0:
#                         img = (img * 255).to(torch.uint8)
#                     # Ensure tensor is in correct format (B, C, H, W) or (C, H, W)
#                     if img.dim() == 4:
#                         img = img.squeeze(0)  # Remove batch dimension if present
#                     # Convert to PIL Image
#                     img = transforms.ToPILImage()(img)
#                 else:
#                     raise TypeError(f"Unsupported image type: {type(img)}")
#                 processed_images.append(preprocess(img))
            
#             # Convert to tensors
#             image_tensor = torch.stack(processed_images).to(device)

#             # text_tokens = clip.tokenize(batch_prompts).to(device)

#             # Truncate prompts to fit CLIP's context length (77 tokens)
#             truncated_prompts = [prompt[:77] if isinstance(prompt, str) else prompt for prompt in batch_prompts]
#             try:
#                 text_tokens = clip.tokenize(truncated_prompts).to(device)
#             except RuntimeError as e:
#                 # If still too long, try more aggressive truncation
#                 print(f"Warning: Truncating prompts further due to length: {e}")
#                 truncated_prompts = [prompt[:50] if isinstance(prompt, str) else prompt for prompt in batch_prompts]
#                 text_tokens = clip.tokenize(truncated_prompts).to(device)
            
#             # Get features
#             image_features = model.encode_image(image_tensor)
#             text_features = model.encode_text(text_tokens)
            
#             # Normalize features
#             image_features = image_features / image_features.norm(dim=1, keepdim=True)
#             text_features = text_features / text_features.norm(dim=1, keepdim=True)
            
#             # Calculate batch scores
#             logit_scale = model.logit_scale.exp()
#             scores = logit_scale * (image_features * text_features).sum(dim=1)
#             score_sum += scores.sum().item()
    
#     return score_sum / n_samples


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, prompt_dir, transform=None):
        self.img_dir = img_dir
        self.prompt_dir = prompt_dir
        self.transform = transform
        self.images = sorted([x for x in os.listdir(self.img_dir) if '.jpg' in x or '.png' in x])
        self.prompts = sorted([x for x in os.listdir(self.prompt_dir) if '.txt' in x])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        
        img_name = os.path.join(self.img_dir, self.images[idx])
        prompt_name = os.path.join(self.prompt_dir, self.prompts[idx])

        # print(img_name, prompt_name)
        # print(os.path.basename(img_name).split('.')[0], os.path.basename(prompt_name).split('.')[0])
        assert os.path.basename(img_name).split('.')[0] == os.path.basename(prompt_name).split('.')[0]

        image = Image.open(img_name).convert('RGB')

        if self.transform:
            image = self.transform(image)

        with open(prompt_name, 'r', encoding='utf-8') as file:
            prompt = file.read().strip()

        return image, prompt

def compute_metrics(gt_img_dir, gen_img_dir, prompt_dir, image_size=768):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256), antialias=True),  # Add desired
        transforms.Lambda(lambda x: (x * 255).type(torch.uint8))
    ])
    gt_dataset = CustomDataset(img_dir=gt_img_dir, prompt_dir=prompt_dir, transform=transform)
    gt_dataloader = DataLoader(gt_dataset, batch_size=64, num_workers=0, shuffle=False)
    gen_dataset = CustomDataset(img_dir=gen_img_dir, prompt_dir=prompt_dir, transform=transform)
    gen_dataloader = DataLoader(gen_dataset, batch_size=64, num_workers=0, shuffle=False)

    fid = FrechetInceptionDistance().to(device)
    fid.set_dtype(torch.float64)
    for batch in gen_dataloader:
        try:
            fid.update(batch[0].to(device), real=False)
        except Exception as e:
            print(f"Error updating FID with batch: {e}")
            # Optional: you can add additional error handling logic here
            # Such as logging the error, skipping the batch, or attempting recovery
            continue  # Skip to the next batch if there's an error
    for batch in gt_dataloader:
        try:
            fid.update(batch[0].to(device), real=True)
        except Exception as e:
            print(f"Error updating FID with batch: {e}")
            # Optional: you can add additional error handling logic here
            # Such as logging the error, skipping the batch, or attempting recovery
            continue  # Skip to the next batch if there's an error
    fid_metric = fid.compute().item()

    mifid = MemorizationInformedFrechetInceptionDistance().to(device)
    for batch in gen_dataloader:
        mifid.update(batch[0].to(device), real=False)
    for batch in gt_dataloader:
        mifid.update(batch[0].to(device), real=True)
    mifid_metric = mifid.compute().item()

    inception = InceptionScore(splits=10).to(device)
    for batch in gen_dataloader:
        inception.update(batch[0].to(device))
    inception_metric = inception.compute()

    clip_scores_all = []
    for batch in gen_dataloader:
        clip_scores = calculate_clip_score(*batch)
        clip_scores_all.append(clip_scores)
    clip_scores = sum(clip_scores_all) / len(clip_scores_all)

    return {'fid': fid_metric, 'mifid': mifid_metric, 'is': inception_metric[0].item(), 'clip_score': clip_scores}


def plot_imgs(actual_images, generated_images, column_names, filename):
    fig, axs = plt.subplots(2, len(column_names), figsize=(15, 6))

    for i, img in enumerate(actual_images):
        axs[0, i].imshow(img)
        axs[0, i].axis('off')  # Turn off axis
        if i == 0:
            axs[0, i].set_ylabel('Actual')

    for i, img in enumerate(generated_images):
        axs[1, i].imshow(img)
        axs[1, i].axis('off')  # Turn off axis
        if i == 0:
            axs[1, i].set_ylabel('Generated')

    for i in range(len(column_names)):
        axs[0, i].set_title(column_names[i], fontsize=6)

    plt.tight_layout()
    plt.savefig(f'grid_{filename}.png')


if __name__ == "__main__":
    args = parse_args()

    gt_img_dir = args.gt_img_dir
    gen_img_dir = args.gen_img_dir
    prompt_dir = args.prompt_dir

    print(f'processing {gen_img_dir}')
    metrics_dict = compute_metrics(gt_img_dir=gt_img_dir,
                                            gen_img_dir=gen_img_dir,
                                            prompt_dir=prompt_dir,
                                            image_size=args.image_size)

    latex_str = f"{metrics_dict['fid']:.2f} & {metrics_dict['is']:.2f} & {metrics_dict['clip_score']:.2f}"
    print(latex_str)

    # Save latex_str to result.txt in parent directory
    parent_dir = os.path.dirname(gen_img_dir)
    args.output = f'{os.path.basename(gen_img_dir)}_img_quality.txt'
    result_path = os.path.join(parent_dir, args.output)
    with open(result_path, 'w') as f:
        f.write(latex_str)
