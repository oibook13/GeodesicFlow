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
        # ... (sections 1-5 are the same as your original code) ...
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
            negative_prompt_2=negative_prompt,
            negative_prompt_3=negative_prompt,
        )

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        latent_shape = (
            batch_size * num_images_per_prompt,
            self.transformer.config.in_channels,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        
        latents = torch.randn(latent_shape, generator=generator, device=device, dtype=dtype)
        delta_t = 1.0 / num_inference_steps
        
        # 6. Denoising loop
        for i, t in enumerate(tqdm(timesteps)):
            # Expand the latents if we are doing classifier-free guidance
            latent_model_input = torch.cat([latents] * 2) if guidance_scale > 1.0 else latents
            
            # Concatenate prompt embeddings for simultaneous passes
            if guidance_scale > 1.0:
                current_prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
                current_pooled_projections = torch.cat([negative_pooled_projections, pooled_projections], dim=0)
            else:
                current_prompt_embeds = prompt_embeds
                current_pooled_projections = pooled_projections
            
            timestep_input = t.repeat(latent_model_input.shape[0])

            # Predict the velocity 'v' for both conditional and unconditional
            v_pred = self.transformer(
                hidden_states=latent_model_input,
                timestep=timestep_input,
                encoder_hidden_states=current_prompt_embeds,
                pooled_projections=current_pooled_projections,
            ).sample

            # 💡 --- FIX STARTS HERE ---
            
            # First, perform classifier-free guidance to get the final guided velocity
            if guidance_scale > 1.0:
                v_uncond, v_text = v_pred.chunk(2)
                v_guided = v_uncond + guidance_scale * (v_text - v_uncond)
            else:
                v_guided = v_pred

            # Now, perform all geometric operations on the correctly-shaped 'v_guided'
            
            # Project velocity v onto the tangent space of z_t
            # Both tensors now have a batch size of 1, so their shapes will match.
            latents_flat = latents.view(latents.shape[0], -1)
            latents_flat = F.normalize(latents_flat, p=2, dim=1)
            v_guided_flat = v_guided.view(v_guided.shape[0], -1)
            dot_product = torch.sum(v_guided_flat * latents_flat, dim=1, keepdim=True)
            v_tangent_flat = v_guided_flat - dot_product * latents_flat
            v = v_tangent_flat.view(latent_shape) # This is the final tangent velocity
            
            # Compute shared tangent vector for the step
            xi = -delta_t * v

            # Geodesic Component (via Exponential Map)
            xi_norm = torch.norm(xi.view(xi.shape[0], -1), p=2, dim=1).view(-1, 1, 1, 1)
            xi_dir = F.normalize(xi.view(xi.shape[0], -1), p=2, dim=1).view(latent_shape)
            z_geo = torch.cos(xi_norm) * latents + torch.sin(xi_norm) * xi_dir
            z_geo = F.normalize(z_geo.view(z_geo.shape[0], -1), p=2, dim=1).view(latent_shape)
            
            # Compute lambda for blending
            pooled_z_geo = F.adaptive_avg_pool2d(z_geo, (1, 1)).view(z_geo.shape[0], -1)
            lambda_val = self.lambda_mlp(pooled_z_geo) 
            
            # Finally, compute the previous noisy sample using the original guided velocity
            # NOTE: The original logic used `v_pred*lambda_val`, but you should likely use the
            # final tangent velocity `v` or the guided velocity `v_guided`. Using 'v' here
            # to be consistent with the projection.
            # latents = self.scheduler.step(v * lambda_val, t, latents).prev_sample

            # Reshape lambda_val to (batch_size, 1, 1, 1) to make it broadcastable with v
            lambda_val_reshaped = lambda_val.view(-1, 1, 1, 1)
            # print(lambda_val_reshaped)
            # The scheduler expects a model_output with the same shape as the latents.
            # This multiplication will now work correctly due to broadcasting.
            # model_output = v * (1-lambda_val_reshaped)
            
            # Finally, compute the previous noisy sample using the scaled velocity
            latents = self.scheduler.step(v_guided, t, latents).prev_sample

            # 💡 --- FIX ENDS HERE ---
            
            # Call the callback, if specified
            if callback is not None and i % callback_steps == 0:
                callback(i, t, latents)

        # ... (sections 7-8 are the same as your original code) ...
        # 7. Post-processing
        image = self.vae.decode(latents, return_dict=False)[0]

        # 8. Convert to PIL
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
    if args.model_path and os.path.exists(args.model_path):
        print(f"✅ Loading LoRA weights from: {args.model_path}")
        pipe.load_lora_weights(args.model_path)
    
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