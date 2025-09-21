import os
import torch
import torch.nn as nn

# --- Paste the LambdaMLP class definition from above here ---
class LambdaMLP(nn.Module):
    """A lightweight MLP to predict the adaptive blending weight lambda."""
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
# -----------------------------------------------------------


# --- 1. CONFIGURE THESE VALUES ---
# Path to the accelerate checkpoint directory, NOT the final LoRA file.
CHECKPOINT_DIR = "output/coco17_sd35_proposed/checkpoint-2000"

# MLP architecture must match the training configuration.
# The input_dim is the number of latent channels (e.g., 4 for SDXL).
INPUT_DIM = 16 
HIDDEN_DIM = 64
NUM_LAYERS = 2

# The filename for the MLP's weights. Usually 'pytorch_model_1.bin'.
WEIGHTS_FILENAME = "pytorch_model_1.bin"
# -----------------------------------


# --- 2. SCRIPT LOGIC ---
# Construct the full path to the weights file
weights_path = os.path.join(CHECKPOINT_DIR, WEIGHTS_FILENAME)

if not os.path.exists(weights_path):
    print(f"[ERROR] Weights file not found at: {weights_path}")
    print("Please ensure the CHECKPOINT_DIR is correct and that the model was saved by accelerate.")
else:
    print(f"Found weights file: {weights_path}")

    # Initialize the model with the correct architecture
    print(f"Initializing LambdaMLP with input_dim={INPUT_DIM}, hidden_dim={HIDDEN_DIM}, num_layers={NUM_LAYERS}")
    lambda_mlp = LambdaMLP(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS
    )

    # Load the state dictionary from the file
    state_dict = torch.load(weights_path, map_location="cpu")
    
    # Load the weights into the model
    lambda_mlp.load_state_dict(state_dict)
    print("✅ Successfully loaded weights into LambdaMLP model.")

    # Set the model to evaluation mode
    lambda_mlp.eval()

    # --- 3. EXAMPLE USAGE ---
    # Create a dummy input tensor to test the loaded model.
    # The input shape should be (batch_size, input_dim).
    # In the trainer, this input is a pooled geodesic velocity vector.
    dummy_input = torch.randn(1, INPUT_DIM) 
    
    with torch.no_grad():
        lambda_prediction = lambda_mlp(dummy_input)
        
    print(f"\n--- Example Usage ---")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Predicted lambda: {lambda_prediction.item()}")