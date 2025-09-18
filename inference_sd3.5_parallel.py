import argparse
from diffusers import StableDiffusion3Pipeline
import torch # Added for dtype and compile
import os
# import pdb # Commented out, uncomment if you need debugging

def parse_arguments():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Run inference with Stable Diffusion model.")

    parser.add_argument("--huggingface_token", type=str, default=None, help="Huggingface token")

    parser.add_argument("--prompt_dir", type=str, default=None, help="Directory containing prompts")
    parser.add_argument("--split", type=str, default='test', help="train | dev | test")

    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for DataLoader") # Clarified this is for DataLoader

    parser.add_argument("--model_id", type=str, default="stabilityai/stable-diffusion-3.5-large",
                        help="Model ID for the pipeline")
    parser.add_argument("--model_path", type=str, default="checkpoints",
                        help="Path to the model checkpoints or LoRA weights") # Clarified model_path usage
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Directory to save output images")
    parser.add_argument("--num_inference_steps", type=int, default=15, help="Number of inference steps")
    parser.add_argument("--guidance_scale", type=float, default=3.5, help="Guidance scale for image generation")

    parser.add_argument("--image_width", type=int, default=512, help="Generated image width")
    parser.add_argument("--image_height", type=int, default=512, help="Generated image height")

    # num_prompts_per_run effectively becomes the inference batch size for the pipeline
    parser.add_argument("--num_prompts_per_run", type=int, default=1, help="Number of prompts to process per pipeline call (inference batch size)")
    parser.add_argument("--reverse_mode", action='store_true', help="Reverse the order of images and prompts (default: False)")

    # Optimization arguments
    parser.add_argument("--use_fp16", action='store_true', help="Use float16 for faster inference (default: False)")
    parser.add_argument("--use_torch_compile", action='store_true', help="Use torch.compile() for optimization (PyTorch 2.0+ required, default: False)")

    args = parser.parse_args()

    return args


from torch.utils.data import Dataset, DataLoader


class PromptDataset(Dataset):
    def __init__(self, data_folder, reverse_mode=False):
        super(PromptDataset, self).__init__()
        self.data_folder = data_folder
        # List all .txt files in the data folder
        self.file_paths = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith('.txt')]
        if reverse_mode:
            self.file_paths = self.file_paths[::-1]

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # Get the file path for the specified index
        file_path = self.file_paths[idx]

        # Read the content of the file
        with open(file_path, 'r', encoding='utf-8') as file: # Added encoding
            content = file.read().strip()

        file_idx = os.path.basename(file_path).split('.')[0]

        # Return the content as a single string
        return content, file_idx

def main():
    # Parse arguments
    args = parse_arguments()
    for split in ['train', 'val', 'validation', 'dev', 'test']:
        if split in args.prompt_dir:
            args.split = split
            break
    
    dataset = PromptDataset(args.prompt_dir, reverse_mode=args.reverse_mode)
    # Increased num_workers if your system can handle it for better data loading
    # Ensure batch_size for DataLoader aligns with how you want to group prompts before sending to the pipeline
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=min(4, os.cpu_count() or 1), shuffle=False)


    # args.model_path determines if we are loading a fine-tuned checkpoint or LoRA
    # args.output_dir = os.path.join(args.output_dir, args.model_path.split('/', 1)[1], args.split) # This logic might need adjustment based on model_path structure
    
    # Simplified output directory structure for clarity. Adjust as needed.
    model_name_or_path_identifier = args.model_id.replace("/", "_")
    if 'checkpoint' in args.model_path:
         model_name_or_path_identifier = os.path.basename(args.model_path)

    final_output_dir = os.path.join(args.model_path, args.output_dir)
    os.makedirs(final_output_dir, exist_ok=True)
    print(f"Saving images to: {final_output_dir}")


    # Determine torch_dtype for mixed precision
    torch_dtype = torch.float32
    if args.use_fp16:
        torch_dtype = torch.float16
        print("Using float16 for inference.")

    # Load the pipeline with the specified model and load weights
    pipe = StableDiffusion3Pipeline.from_pretrained(
        args.model_id,
        torch_dtype=torch_dtype,
        # token=args.huggingface_token # Use 'token' instead of 'huggingface_token' if needed by from_pretrained
    )
    
    if 'checkpoint' in args.model_path and os.path.isdir(args.model_path): # Check if it's a directory for LoRA
        print(f"Loading LoRA weights from: {args.model_path}")
        pipe.load_lora_weights(args.model_path)
    elif 'checkpoint' in args.model_path: # Assuming it might be a single file checkpoint (though SD3 usually isn't used this way with diffusers)
        print(f"Warning: Attempting to load {args.model_path} directly. Ensure this is the correct usage for SD3 with diffusers.")
        # pipe.load_lora_weights(args.model_path) # Or other appropriate loading mechanism if not LoRA

    pipe = pipe.to("cuda")

    # Apply torch.compile() if specified (requires PyTorch 2.0+)
    if args.use_torch_compile:
        try:
            pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
            # You can also compile VAE and text encoders if desired, though UNet is usually the bottleneck
            # pipe.vae = torch.compile(pipe.vae, mode="reduce-overhead", fullgraph=True)
            # pipe.text_encoder = torch.compile(pipe.text_encoder, mode="reduce-overhead", fullgraph=True)
            # pipe.text_encoder_2 = torch.compile(pipe.text_encoder_2, mode="reduce-overhead", fullgraph=True)
            print("Applied torch.compile() to the UNet.")
        except Exception as e:
            print(f"Failed to apply torch.compile(): {e}. Proceeding without it.")
            
    # Enable memory-efficient attention if xformers is available (often automatic, but can be explicit)
    # try:
    #     pipe.enable_xformers_memory_efficient_attention()
    #     print("Enabled xformers memory efficient attention.")
    # except ImportError:
    #     print("xformers not installed. For potential speedups, install with: pip install xformers")
    # except Exception as e:
    #     print(f"Could not enable xformers: {e}")


    # Generate images
    for batch_prompts_dl, batch_prompt_ids_dl in dataloader: # Suffix _dl to distinguish from inner loop vars
        # batch_prompts_dl and batch_prompt_ids_dl are tuples from DataLoader (each of size args.batch_size)
        # We then further batch these according to num_prompts_per_run for the pipeline

        if args.num_prompts_per_run == 1:
            for prompt, _id in zip(batch_prompts_dl, batch_prompt_ids_dl):
                output_path = f"{final_output_dir}/{_id}.png"
                if os.path.exists(output_path):
                    print(f"Skipping {output_path}")
                    continue
                print(f"Generating: {prompt[:60]}... ({_id}.png)")
                image = pipe(
                    prompt=prompt,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                    num_images_per_prompt=1, # Already set to 1
                    width=args.image_width,
                    height=args.image_height,
                ).images[0]

                # Save the generated image
                image.save(output_path)

        elif args.num_prompts_per_run > 1:
            # Process prompts from DataLoader in chunks of num_prompts_per_run
            all_prompts_in_dl_batch = list(batch_prompts_dl)
            all_ids_in_dl_batch = list(batch_prompt_ids_dl)

            for i in range(0, len(all_prompts_in_dl_batch), args.num_prompts_per_run):
                prompts_for_pipe = all_prompts_in_dl_batch[i:i + args.num_prompts_per_run]
                prompt_ids_for_pipe = all_ids_in_dl_batch[i:i + args.num_prompts_per_run]
                
                # Check if all images in this specific pipe batch already exist
                output_paths_for_pipe = [f"{final_output_dir}/{_id}.png" for _id in prompt_ids_for_pipe]
                if all(os.path.exists(p) for p in output_paths_for_pipe):
                    print(f"Skipping pipe batch, all exist: {output_paths_for_pipe}")
                    continue
                
                print(f"Generating batch of {len(prompts_for_pipe)} prompts...")
                images = pipe(
                    prompt=prompts_for_pipe, # Pass the list of prompts
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                    num_images_per_prompt=1, # Generates one image per prompt in the list
                    width=args.image_width,
                    height=args.image_height,
                ).images

                # Save the generated images
                for image, _id in zip(images, prompt_ids_for_pipe):
                    image.save(f"{final_output_dir}/{_id}.png")
        else:
            raise ValueError(f"Invalid num_prompts_per_run: {args.num_prompts_per_run}. Must be >= 1.")


if __name__ == "__main__":
    main()