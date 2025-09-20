# inference_proposed.py

import argparse
import os
import math
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import FluxPipeline
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# --- GeodesicFlow Components (from Algorithm and trainer_proposed.py) ---

class LambdaMLP(nn.Module):
    """
    A lightweight MLP to predict the adaptive blending weight lambda,
    as defined in the provided training script.
    """
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 2):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        if num_layers > 1:
            layers.append(nn.Linear(hidden_dim, 1))
        else: # Handle single-layer case
            layers = [nn.Linear(input_dim, 1)]
        layers.append(nn.Sigmoid())
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

def slerp(p0, p1, t):
    """
    Performs Spherical Linear Interpolation (Slerp) between two batches of vectors.
    This is used to blend the geodesic and Euclidean destinations on the manifold.
    Args:
        p0 (torch.Tensor): Start points (shape: [batch_size, num_features]).
        p1 (torch.Tensor): End points (shape: [batch_size, num_features]).
        t (torch.Tensor): Interpolation factor (shape: [batch_size, 1]).
    Returns:
        torch.Tensor: The interpolated points on the sphere.
    """
    # Ensure inputs are normalized
    p0 = F.normalize(p0, p=2, dim=1)
    p1 = F.normalize(p1, p=2, dim=1)

    # Compute the angle between vectors
    dot_product = torch.sum(p0 * p1, dim=1, keepdim=True)
    dot_product = torch.clamp(dot_product, -1.0, 1.0) # Clamp for numerical stability
    theta = torch.acos(dot_product)

    # Handle the case where vectors are very close
    sin_theta = torch.sin(theta)
    small_angle_mask = sin_theta < 1e-4
    
    # Standard slerp formula
    a = torch.sin((1.0 - t) * theta) / (sin_theta + 1e-7)
    b = torch.sin(t * theta) / (sin_theta + 1e-7)
    
    # Use linear interpolation for small angles
    result_slerp = a * p0 + b * p1
    result_lerp = (1.0 - t) * p0 + t * p1
    
    # Combine results
    result = torch.where(small_angle_mask, result_lerp, result_slerp)
    
    return F.normalize(result, p=2, dim=1)

# --- FLUX Helper Functions (from trainer_proposed.py) ---
# These are needed to correctly format inputs for the FLUX transformer model.
def pack_latents(latents, batch_size, num_channels_latents, height, width):
    latents = latents.view(batch_size, num_channels_latents, height, width)
    latents = latents.permute(0, 2, 3, 1).contiguous()
    latents = latents.view(batch_size, height * width, num_channels_latents)
    return latents

def unpack_latents(latents, height, width, vae_scale_factor):
    height = height // vae_scale_factor
    width = width // vae_scale_factor
    num_channels_latents = latents.shape[-1]
    latents = latents.view(-1, height, width, num_channels_latents)
    latents = latents.permute(0, 3, 1, 2).contiguous()
    return latents

def prepare_latent_image_ids(batch_size, height, width, device, dtype):
    return torch.tensor([height, width, height, width, 0, 0], device=device, dtype=dtype).repeat(batch_size, 1)

# --- Argument Parsing and Data Loading ---

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run inference with the AGES sampling method.")
    parser.add_argument("--huggingface_token", type=str, default=None, help="Huggingface token")
    parser.add_argument("--prompt_dir", type=str, required=True, help="Directory containing .txt prompt files")
    parser.add_argument("--model_id", type=str, default="black-forest-labs/FLUX.1-dev", help="Model ID for the pipeline")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model checkpoints (e.g., LoRA weights)")
    parser.add_argument("--lambda_mlp_path", type=str, required=True, help="Path to the trained LambdaMLP checkpoint (.pt file)")
    parser.add_argument("--output_dir", type=str, default="outputs_ages", help="Directory to save output images")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of integration steps (N)")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference")
    parser.add_argument("--image_width", type=int, default=1024, help="Generated image width")
    parser.add_argument("--image_height", type=int, default=1024, help="Generated image height")
    
    # GeodesicFlow specific MLP parameters
    parser.add_argument('--geodesicflow_mlp_hidden_dim', type=int, default=64)
    parser.add_argument('--geodesicflow_mlp_num_layers', type=int, default=2)

    return parser.parse_args()


class PromptDataset(Dataset):
    def __init__(self, data_folder):
        super().__init__()
        self.file_paths = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith('.txt')]

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        with open(file_path, 'r') as file:
            content = file.read().strip()
        file_idx = os.path.basename(file_path).split('.')[0]
        return content, file_idx

# --- Main Inference Function ---

def main():
    args = parse_arguments()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    weight_dtype = torch.bfloat16

    # --- 1. Load Models ---
    print("Loading models...")
    # Load the base FLUX pipeline
    pipe = FluxPipeline.from_pretrained(
        args.model_id,
        torch_dtype=weight_dtype,
        token=args.huggingface_token
    )
    # Load fine-tuned weights (e.g., LoRA) for the main model
    if 'checkpoint' in args.model_path:
        pipe.load_lora_weights(args.model_path)
    pipe = pipe.to(device)

    # Load the trained Lambda MLP for adaptive blending
    latent_channels = pipe.transformer.config.in_channels
    lambda_mlp = LambdaMLP(
        input_dim=latent_channels,
        hidden_dim=args.geodesicflow_mlp_hidden_dim,
        num_layers=args.geodesicflow_mlp_num_layers
    ).to(device, dtype=weight_dtype)
    lambda_mlp.load_state_dict(torch.load(args.lambda_mlp_path, map_location=device))
    lambda_mlp.eval()
    print("Models loaded successfully.")

    # --- 2. Prepare Data and Output Directory ---
    dataset = PromptDataset(args.prompt_dir)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    output_path = os.path.join(args.model_path, args.output_dir)
    os.makedirs(output_path, exist_ok=True)
    
    # Get model configuration details
    latent_height = args.image_height // pipe.vae_scale_factor
    latent_width = args.image_width // pipe.vae_scale_factor
    num_train_timesteps = pipe.scheduler.config.num_train_timesteps

    # --- 3. Generation Loop ---
    for batch_prompts, batch_ids in tqdm(dataloader, desc="Generating Images"):
        current_batch_size = len(batch_prompts)
        
        # --- AGES Step 0: Pre-compute text embeddings ---
        prompt_embeds, pooled_prompt_embeds = pipe.encode_prompt(
            prompt=list(batch_prompts),
            do_classifier_free_guidance=False # CFG is not part of the AGES algorithm
        )

        # --- AGES Step 1: Initialize noise z_1 on the unit sphere ---
        z_t_shape = (current_batch_size, latent_channels, latent_height, latent_width)
        z_t = torch.randn(z_t_shape, device=device, dtype=weight_dtype)
        z_t = F.normalize(z_t.view(current_batch_size, -1), p=2, dim=1).view(z_t_shape)
        
        # Prepare other static inputs for the FLUX model
        img_ids = prepare_latent_image_ids(current_batch_size, latent_height, latent_width, device, weight_dtype)
        text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(device=device, dtype=weight_dtype)
        
        # --- AGES Step 2: Iteratively solve the ODE from t=1 to t=0 ---
        for k in tqdm(range(args.num_inference_steps, 0, -1), desc="Sampling Steps", leave=False):
            t_continuous = torch.full((current_batch_size,), k / args.num_inference_steps, device=device)
            
            # The model expects integer timesteps, so we scale our continuous time 't'
            timesteps = (t_continuous * (num_train_timesteps - 1)).long()

            # --- Predict lambda ---
            # Pool z_t to create the input for the MLP, matching the training process
            pooled_z_t = F.adaptive_avg_pool2d(z_t, (1, 1)).view(current_batch_size, -1)
            lambda_val = lambda_mlp(pooled_z_t.to(weight_dtype)).view(-1, 1)

            # --- Predict velocity v ---
            # Pack latents into the sequence format expected by FLUX transformer
            packed_z_t = pack_latents(z_t, current_batch_size, latent_channels, latent_height, latent_width)
            
            with torch.no_grad():
                model_pred_packed = pipe.transformer(
                    hidden_states=packed_z_t,
                    timestep=t_continuous, # FLUX model can handle continuous time
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=img_ids,
                    return_dict=False
                )[0]
            
            # Unpack the prediction back into image-like latent format
            v = unpack_latents(model_pred_packed, args.image_height, args.image_width, pipe.vae_scale_factor)
            
            # Flatten for sphere operations
            v_flat = v.view(current_batch_size, -1)
            z_t_flat = z_t.view(current_batch_size, -1)
            
            # --- Compute blended step ---
            delta_t = 1.0 / args.num_inference_steps
            xi = -delta_t * v_flat

            # --- Geodesic component ---
            xi_norm = torch.norm(xi, p=2, dim=1, keepdim=True)
            xi_dir = xi / (xi_norm + 1e-7)
            z_geo_flat = z_t_flat * torch.cos(xi_norm) + xi_dir * torch.sin(xi_norm)
            
            # --- Euclidean component ---
            z_euc_update = z_t_flat + xi
            z_euc_flat = F.normalize(z_euc_update, p=2, dim=1)

            # --- Blend positions using Slerp ---
            z_t_next_flat = slerp(z_euc_flat, z_geo_flat, lambda_val)
            
            # Update z_t for the next iteration
            z_t = z_t_next_flat.view(z_t_shape)
            
        # --- AGES Step 3: Decode the final latent z_0 ---
        z_0 = z_t
        z_0 = z_0 / pipe.vae.config.scaling_factor
        with torch.no_grad():
            images = pipe.vae.decode(z_0.to(weight_dtype), return_dict=False)[0]
        
        images = pipe.image_processor.postprocess(images, output_type="pil")

        # Save the generated images
        for image, _id in zip(images, batch_ids):
            image_path = os.path.join(output_path, f"{_id}.png")
            if os.path.exists(image_path):
                print(f"Skipping existing file: {image_path}")
                continue
            image.save(image_path)
            
    print(f"\nInference complete. Images saved to: {output_path}")

if __name__ == "__main__":
    main()