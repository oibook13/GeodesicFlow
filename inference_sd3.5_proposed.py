import argparse
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers import StableDiffusion3Pipeline
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import StableDiffusion3PipelineOutput
from torch.utils.data import Dataset, DataLoader
from safetensors.torch import load_file
from tqdm.auto import tqdm
from typing import Callable, List, Optional, Union

# --- GeodesicFlow Components ---

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

def slerp(p0: torch.Tensor, p1: torch.Tensor, t: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    Batched Spherical Linear Interpolation (Slerp).
    Interpolates from p0 to p1 with weight t.
    """
    # Ensure inputs are normalized
    p0 = F.normalize(p0, p=2, dim=1)
    p1 = F.normalize(p1, p=2, dim=1)

    # Compute the dot product between the vectors
    dot = torch.sum(p0 * p1, dim=1, keepdim=True)
    
    # Clamp dot product to handle floating point inaccuracies
    dot = torch.clamp(dot, -1.0 + eps, 1.0 - eps)
    
    # Angle between the vectors
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)

    # Slerp formula: (1-t)*p0 + t*p1 in the spherical sense
    a = torch.sin((1.0 - t) * theta) / (sin_theta + eps)
    b = torch.sin(t * theta) / (sin_theta + eps)
    
    # Perform interpolation
    interpolated = a * p0 + b * p1
    
    # Re-normalize the final result to ensure it's on the unit sphere
    return F.normalize(interpolated, p=2, dim=1)

class GeodesicFlowPipeline(StableDiffusion3Pipeline):
    """
    Custom pipeline implementing the Adaptive Geodesic-Euclidean Sampling (AGES) algorithm.
    """
    def __init__(self, vae, text_encoder, tokenizer, text_encoder_2, tokenizer_2, text_encoder_3, tokenizer_3, transformer, scheduler):
        # By calling super() with keyword arguments (e.g., vae=vae), we ensure
        # that the correct component is passed to the correct parameter,
        # regardless of their order. This is the most robust solution.
        super().__init__(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            text_encoder_2=text_encoder_2,
            tokenizer_2=tokenizer_2,
            text_encoder_3=text_encoder_3,
            tokenizer_3=tokenizer_3,
            transformer=transformer,
            scheduler=scheduler,
        )
        # The LambdaMLP model will be attached to this pipeline instance after initialization.
        self.lambda_mlp = None

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        guidance_scale: float = 7.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        **kwargs,
    ):
        if self.lambda_mlp is None:
            raise ValueError("The lambda_mlp model has not been loaded into the pipeline. Please set `pipe.lambda_mlp`.")

        # 1. Default height and width
        height = height or self.transformer.config.sample_size * self.vae_scale_factor
        width = width or self.transformer.config.sample_size * self.vae_scale_factor

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        device = self._execution_device
        dtype = self.transformer.dtype

        # 3. Encode input prompt
        prompt_embeds, negative_prompt_embeds, pooled_projections, negative_pooled_projections = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt,
            prompt_3=prompt,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt, # Pass negative prompt to all 3 encoders
            negative_prompt_3=negative_prompt, # Pass negative prompt to all 3 encoders
        )

        # 4. Prepare latent variables
        latent_shape = (
            batch_size * num_images_per_prompt,
            self.transformer.config.in_channels,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        
        # Start with random noise on the unit sphere (z_1)
        latents = torch.randn(latent_shape, generator=generator, device=device, dtype=dtype)
        latents = F.normalize(latents.view(latents.shape[0], -1), p=2, dim=1).view(latent_shape)

        # 5. Denoising loop based on Algorithm 2 (AGES)
        delta_t = 1.0 / num_inference_steps
        
        for k in tqdm(range(num_inference_steps, 0, -1)):
            latents = latents.to(self.transformer.dtype)
            t_continuous = k * delta_t
            t_tensor = torch.tensor([t_continuous] * latents.shape[0], device=device, dtype=dtype)
            
            # Ensure latents (z_t) are on the sphere
            latents = F.normalize(latents.view(latents.shape[0], -1), p=2, dim=1).view(latent_shape)

            # --- ALGORITHM FIX 1: Compute lambda using the MLP ---
            # Reshape latents for the MLP, which expects a flat vector
            # This matches the algorithm's λ_Φ(z_t)
            # num_features = latents.shape[1] * latents.shape[2] * latents.shape[3]
            # mlp_input = latents.view(latents.shape[0], num_features)
            
            # The MLP input dimension might need adjustment depending on its training.
            # If it was trained on pooled latents, use the line below instead.
            # mlp_input = F.adaptive_avg_pool2d(latents, (1, 1)).view(latents.shape[0], -1)


            # Predict velocity v = v_Theta(z_t, t, c)
            model_t = (t_tensor * 999).long()
            
            v = self.transformer(
                hidden_states=latents,
                timestep=model_t,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_projections,
            ).sample

            # Project velocity v onto the tangent space of z_t to ensure geometric correctness
            latents_flat = latents.view(latents.shape[0], -1)
            v_flat = v.view(latents.shape[0], -1)
            dot_product = torch.sum(v_flat * latents_flat, dim=1, keepdim=True)
            v_tangent_flat = v_flat - dot_product * latents_flat
            v = v_tangent_flat.view(latent_shape)

            # Compute shared tangent vector for the step
            xi = -delta_t * v

            # Geodesic Component (via Exponential Map)
            xi_norm = torch.norm(xi.view(xi.shape[0], -1), p=2, dim=1).view(-1, 1, 1, 1)
            # Handle the case where norm is zero to avoid division errors
            xi_dir = F.normalize(xi.view(xi.shape[0], -1), p=2, dim=1).view(latent_shape)
            z_geo = torch.cos(xi_norm) * latents + torch.sin(xi_norm) * xi_dir
            # Ensure z_geo is on the sphere, correcting for any floating point errors
            z_geo = F.normalize(z_geo.view(z_geo.shape[0], -1), p=2, dim=1).view(latent_shape)

            pooled_z_geo = F.adaptive_avg_pool2d(z_geo, (1, 1)).view(z_geo.shape[0], -1)
            lambda_val = self.lambda_mlp(pooled_z_geo) # Shape: (batch_size, 1)
            lambda_val = lambda_val.to(self.transformer.dtype)

            # Euclidean Component (linear step + projection)
            z_euc_unnorm = latents + xi
            z_euc = F.normalize(z_euc_unnorm.view(z_euc_unnorm.shape[0], -1), p=2, dim=1).view(latent_shape)
            
            # --- ALGORITHM FIX 2: Blend Positions using Slerp in the correct order ---
            z_geo_flat = z_geo.view(z_geo.shape[0], -1)
            z_euc_flat = z_euc.view(z_geo.shape[0], -1)
            
            # Slerp(z_euc, z_geo, lambda): λ=0 -> z_euc, λ=1 -> z_geo
            next_latents_flat = slerp(z_euc_flat, z_geo_flat, lambda_val)
            latents = next_latents_flat.view(latent_shape)

        # 6. Post-processing
        latents = latents / self.vae.config.scaling_factor
        image = self.vae.decode(latents, return_dict=False)[0]

        # 7. Convert to PIL
        if output_type == "pil":
            image = self.image_processor.postprocess(image, output_type="pil")
        
        if not return_dict:
            return (image,)

        return StableDiffusion3PipelineOutput(images=image)

# --- Data Handling ---

class PromptDataset(Dataset):
    def __init__(self, data_folder, reverse_mode=False):
        super(PromptDataset, self).__init__()
        self.data_folder = data_folder
        self.file_paths = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith('.txt')]
        if reverse_mode:
            self.file_paths = self.file_paths[::-1]

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        with open(file_path, 'r') as file:
            content = file.read().strip()
        file_idx = os.path.basename(file_path).split('.')[0]
        return content, file_idx

# --- Main Execution ---

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run inference with the GeodesicFlow AGES method.")
    # Model and Paths
    parser.add_argument("--model_id", type=str, default="stabilityai/stable-diffusion-3.5-large", help="Base model ID from Hugging Face.")
    parser.add_argument("--model_path", type=str, default="checkpoints",
                        help="Path to the model checkpoints")
    parser.add_argument("--lora_path", type=str, help="Path to the trained LoRA weights (directory containing adapter_config.json).")
    parser.add_argument("--lambda_mlp_path", type=str, help="Path to the trained LambdaMLP weights (.pth file).")
    parser.add_argument("--output_dir", type=str, default="outputs_geodesicflow", help="Directory to save output images.")
    parser.add_argument("--num_prompts_per_run", type=int, default=1, help="Number of prompts to process per inference process")
    
    # MLP Architecture (must match training)
    parser.add_argument("--mlp_input_dim", type=int, default=16, help="Input dimension for LambdaMLP (latent channels for SD3).")
    parser.add_argument("--mlp_hidden_dim", type=int, default=64, help="Hidden dimension of the LambdaMLP.")
    parser.add_argument("--mlp_num_layers", type=int, default=2, help="Number of layers in the LambdaMLP.")
    # Inference Parameters
    parser.add_argument("--prompt_dir", type=str, required=True, help="Directory containing prompt .txt files.")
    parser.add_argument("--num_inference_steps", type=int, default=15, help="Number of inference steps for the AGES solver.")
    parser.add_argument("--guidance_scale", type=float, default=3.5, help="Guidance scale for image generation")
    parser.add_argument("--image_width", type=int, default=512, help="Generated image width.")
    parser.add_argument("--image_height", type=int, default=512, help="Generated image height.")
    parser.add_argument("--batch_size", type=int, default=1, help="Number of prompts to process in a batch.")
    args = parser.parse_args()
    return args

# def main():
#     args = parse_arguments()

#     # --- Setup Model and Pipeline ---
#     print(f"Loading base model: {args.model_id}")
#     # pipe = GeodesicFlowPipeline.from_pretrained(args.model_id, torch_dtype=torch.float16)
    
#     lambda_mlp = LambdaMLP(
#         input_dim=args.mlp_input_dim,
#         hidden_dim=args.mlp_hidden_dim,
#         num_layers=args.mlp_num_layers
#     )
#     try:
#         state_dict = load_file(os.path.join(args.model_path, "lambda_mlp.safetensors"), device="cpu")
#         lambda_mlp.load_state_dict(state_dict)
#         print("✅ Weights loaded successfully!")
#     except Exception as e:
#         print(f"❌ Error loading state dictionary: {e}")
#         exit(1)
#     lambda_mlp = lambda_mlp.to("cuda")

#     pipe = StableDiffusion3Pipeline.from_pretrained(args.model_id) # , huggingface_token=args.huggingface_token
#     if 'checkpoint' in args.model_path:
#         pipe.load_lora_weights(args.model_path)
#     pipe = pipe.to("cuda")

#     pipe.lambda_mlp = lambda_mlp
#     pipe.lambda_mlp.eval()

#     new_pipe = GeodesicFlowPipeline(
#         vae=pipe.vae,
#         text_encoder=pipe.text_encoder,
#         tokenizer=pipe.tokenizer,
#         text_encoder_2=pipe.text_encoder_2,
#         tokenizer_2=pipe.tokenizer_2,
#         text_encoder_3=pipe.text_encoder_3,
#         tokenizer_3=pipe.tokenizer_3,
#         transformer=pipe.transformer,
#         scheduler=pipe.scheduler,
#     )
#     del pipe
#     pipe = new_pipe
    
#     # pipe = pipe.to("cuda")
#     # pipe.lambda_mlp.to(pipe.device, dtype=pipe.dtype)
#     # pipe.lambda_mlp.eval()
    
#     # --- Setup Data and Output ---
#     dataset = PromptDataset(args.prompt_dir)
#     dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=4, shuffle=False)
    
#     # output_dir_name = os.path.basename(os.path.normpath(args.lora_path))
#     final_output_dir = os.path.join(args.model_path, args.output_dir)
#     os.makedirs(final_output_dir, exist_ok=True)
#     print(f"Saving images to: {final_output_dir}")

#     # --- Generation Loop ---
#     for batch_prompts, batch_prompt_ids in tqdm(dataloader, desc="Generating Images"):
#         # Check if all images in the batch already exist
#         if all(os.path.exists(os.path.join(final_output_dir, f"{_id}.png")) for _id in batch_prompt_ids):
#             print(f"Skipping batch, all images exist.")
#             continue
            
#         images = pipe(
#             prompt=list(batch_prompts),
#             num_inference_steps=args.num_inference_steps,
#             height=args.image_height,
#             width=args.image_width,
#             num_images_per_prompt=1,
#         ).images

#         # Save the generated images
#         for image, _id in zip(images, batch_prompt_ids):
#             image.save(os.path.join(final_output_dir, f"{_id}.png"))

def main():
    args = parse_arguments()

    # --- Setup Model and Pipeline (Corrected Method) ---
    print(f"Loading base model into GeodesicFlowPipeline: {args.model_id}")

    # 1. Load the model directly using your custom pipeline class.
    #    This is the standard and most robust way to do this. It correctly
    #    initializes all components in the right order.
    #    Using float16 is recommended for better performance.
    pipe = GeodesicFlowPipeline.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16
    )

    # 2. Load the LoRA weights into the pipeline, if specified.
    #    Your original code checked if 'checkpoint' was in the model_path.
    #    Using a specific lora_path argument is generally clearer.
    if args.lora_path and os.path.exists(args.lora_path):
        print(f"✅ Loading LoRA weights from: {args.lora_path}")
        pipe.load_lora_weights(args.lora_path)
    
    pipe.to("cuda")

    # 3. Load and attach the custom LambdaMLP model.
    print("🧠 Loading LambdaMLP model...")
    lambda_mlp = LambdaMLP(
        input_dim=args.mlp_input_dim,
        hidden_dim=args.mlp_hidden_dim,
        num_layers=args.mlp_num_layers
    )

    try:
        state_dict = load_file(os.path.join(args.model_path, "lambda_mlp.safetensors"), device="cpu")
        lambda_mlp.load_state_dict(state_dict)
        print(f"✅ LambdaMLP weights loaded successfully from: {args.lambda_mlp_path}")
    except Exception as e:
        print(f"❌ Error loading LambdaMLP state dictionary: {e}")
        exit(1)

    # Attach the loaded MLP to the pipeline and put it in eval mode.
    # Ensure it's on the same device and dtype as the pipeline.
    pipe.lambda_mlp = lambda_mlp.to(pipe.device, dtype=pipe.dtype)
    pipe.lambda_mlp.eval()
    
    # --- Setup Data and Output ---
    dataset = PromptDataset(args.prompt_dir)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=4, shuffle=False)
    
    final_output_dir = os.path.join(args.model_path, args.output_dir)
    os.makedirs(final_output_dir, exist_ok=True)
    print(f"🚀 Saving images to: {final_output_dir}")

    # --- Generation Loop ---
    for batch_prompts, batch_prompt_ids in tqdm(dataloader, desc="Generating Images"):
        # Check if all images in the batch already exist
        if all(os.path.exists(os.path.join(final_output_dir, f"{_id}.png")) for _id in batch_prompt_ids):
            # print(f"Skipping batch for prompts {batch_prompt_ids}, all images exist.")
            continue
            
        images = pipe(
            prompt=list(batch_prompts),
            num_inference_steps=args.num_inference_steps,
            height=args.image_height,
            width=args.image_width,
            guidance_scale=args.guidance_scale,
            num_images_per_prompt=1,
        ).images

        # Save the generated images
        for image, _id in zip(images, batch_prompt_ids):
            image.save(os.path.join(final_output_dir, f"{_id}.png"))

    print("🎉 Done!")

if __name__ == "__main__":
    main()