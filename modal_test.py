"""
Simple Modal test for dependency issues
"""
import modal

app = modal.App("geodesicflow-test")

# Simplified image with just the essentials
image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04")
    .apt_install([
        "python3-dev",
        "python3-pip",
        "gcc",
        "g++",
        "build-essential"
    ])
    .run_commands([
        "ln -sf /usr/bin/python3 /usr/bin/python"
    ])
    .pip_install([
        "numpy>=1.24.0",
        "torch",
        "torchvision",
    ], index_url="https://download.pytorch.org/whl/cu121")
    .pip_install([
        "tokenizers>=0.19.0",
        "transformers>=4.44.0",
        "accelerate>=0.33.0",
        "torchao>=0.5.0",
        "torch-optimi>=0.2.0",
        "bitsandbytes>=0.44.0",
    ])
)

@app.function(image=image, gpu="H100:1", timeout=300)
def test_imports():
    """Test basic imports"""
    try:
        import numpy as np
        print(f"✓ NumPy {np.__version__} imported successfully")

        import torch
        print(f"✓ PyTorch {torch.__version__} imported successfully")
        print(f"✓ CUDA available: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(f"✓ CUDA version: {torch.version.cuda}")
            print(f"✓ GPU count: {torch.cuda.device_count()}")

        import torchvision
        print(f"✓ Torchvision {torchvision.__version__} imported successfully")

        import transformers
        print(f"✓ Transformers {transformers.__version__} imported successfully")

        import accelerate
        print(f"✓ Accelerate {accelerate.__version__} imported successfully")

        import torchao
        print(f"✓ TorchAO {torchao.__version__} imported successfully")

        import optimi
        print(f"✓ Torch-Optimi {optimi.__version__} imported successfully")

        import bitsandbytes
        print(f"✓ BitsAndBytes {bitsandbytes.__version__} imported successfully")

        return "All imports successful!"

    except Exception as e:
        print(f"✗ Import failed: {e}")
        return f"Import failed: {e}"

@app.local_entrypoint()
def main():
    print("Testing basic imports...")
    result = test_imports.remote()
    print(f"Result: {result}")