import argparse
from diffusers import FluxPipeline # <-- MODIFIED: Changed the pipeline for FLUX model
import torch # <-- ADDED: Import torch for dtype specification

import os

import pdb


def parse_arguments():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Run inference with a diffusion model.")

    parser.add_argument("--huggingface_token", type=str, default=None, help="Huggingface token")

    parser.add_argument("--prompt_dir", type=str, default=None, help="Directory containing prompts")
    parser.add_argument("--split", type=str, default='test', help="train | dev | test")

    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")

    # <-- MODIFIED: Updated the default model ID to FLUX.1-dev
    parser.add_argument("--model_id", type=str, default="black-forest-labs/FLUX.1-dev",
                        help="Model ID for the pipeline")
    parser.add_argument("--model_path", type=str, default="checkpoints",
                        help="Path to the model checkpoints")
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Directory to save output images")
    parser.add_argument("--num_inference_steps", type=int, default=15, help="Number of inference steps")
    parser.add_argument("--guidance_scale", type=float, default=3.5, help="Guidance scale for image generation")

    parser.add_argument("--image_width", type=int, default=1024, help="Generated image width") # <-- MODIFIED: Changed default to 1024 for FLUX
    parser.add_argument("--image_height", type=int, default=1024, help="Generated image height") # <-- MODIFIED: Changed default to 1024 for FLUX

    parser.add_argument("--num_prompts_per_run", type=int, default=1, help="Number of prompts to process per inference process")
    parser.add_argument("--reverse_mode", action='store_true', help="Reverse the order of images and prompts (default: False)")

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
        with open(file_path, 'r') as file:
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
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=4, shuffle=False)

    args.output_dir = os.path.join(args.model_path, args.output_dir)

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # <-- MODIFIED: Load the FluxPipeline and specify torch_dtype for performance
    pipe = FluxPipeline.from_pretrained(
        args.model_id, 
        torch_dtype=torch.bfloat16, 
        token=args.huggingface_token # Pass token if provided
    )
    
    if 'checkpoint' in args.model_path:
        # Note: FLUX models might use different weight loading mechanisms.
        # `load_lora_weights` is common, but verify if your checkpoint is compatible.
        pipe.load_lora_weights(args.model_path)
        
    pipe = pipe.to("cuda")

    # Generate images
    for batch in dataloader:
        batch_prompts, batch_prompt_ids = batch
        if args.num_prompts_per_run == 1:
            for prompt, _id in zip(batch_prompts, batch_prompt_ids):
                if os.path.exists(f"{args.output_dir}/{_id}.png"):
                    print(f"Skipping {args.output_dir}/{_id}.png")
                    continue
                image = pipe(
                    prompt=prompt,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                    # num_images_per_prompt is not a standard argument for FluxPipeline
                    # width=args.image_width, # width and height are often inferred or handled differently
                    # height=args.image_height,
                ).images[0]

                # Save the generated image
                image.save(f"{args.output_dir}/{_id}.png")

        elif args.num_prompts_per_run > 1:
            prompts_per_run = [batch_prompts[i:i + args.num_prompts_per_run] \
                               for i in range(0, len(batch_prompts), args.num_prompts_per_run)]
            prompt_ids_per_run = [batch_prompt_ids[i:i + args.num_prompts_per_run] \
                                  for i in range(0, len(batch_prompt_ids), args.num_prompts_per_run)]

            for prompts, prompt_ids in zip(prompts_per_run, prompt_ids_per_run):
                if all([os.path.exists(f"{args.output_dir}/{_id}.png") for _id in prompt_ids]):
                    print(f"Skipping {[f'{args.output_dir}/{_id}.png' for _id in prompt_ids]}")
                    continue
                images = pipe(
                    prompt=list(prompts),
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                    # num_images_per_prompt is not a standard argument for FluxPipeline
                    # width=args.image_width,
                    # height=args.image_height,
                ).images

                # Save the generated image
                for i, (image, _id) in enumerate(zip(images, prompt_ids)):
                    image.save(f"{args.output_dir}/{_id}.png")

        else:
            raise ValueError(f"Invalid num_prompts_per_run: {args.num_prompts_per_run}.")


if __name__ == "__main__":
    main()