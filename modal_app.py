"""
Modal app for GeodesicFlow training on H100 GPUs
"""

import modal
import os
from pathlib import Path

# Modal app instance
app = modal.App("geodesicflow-training")

# Define volumes for persistence
datasets_volume = modal.Volume.from_name(
    "geodesicflow-datasets", create_if_missing=True
)
checkpoints_volume = modal.Volume.from_name(
    "geodesicflow-checkpoints", create_if_missing=True
)
cache_volume = modal.Volume.from_name("geodesicflow-cache", create_if_missing=True)

# Create the Modal image with CUDA 12.2 and conda environment
image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04")
    .apt_install(
        [
            "wget",
            "curl",
            "git",
            "build-essential",
            "software-properties-common",
            "ca-certificates",
            "gnupg",
            "lsb-release",
            "unzip",
            "vim",
            "python3-dev",
            "python3-pip",
            "gcc",
            "g++",
            "pkg-config",
            "libssl-dev",
        ]
    )
    .run_commands(
        [
            # Create python symlink for pip_install compatibility
            "ln -sf /usr/bin/python3 /usr/bin/python",
            # Install Rust for tokenizers compilation
            "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y",
            "echo 'export PATH=\"/root/.cargo/bin:$PATH\"' >> /root/.bashrc",
            # Install miniconda
            "wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh",
            "bash /tmp/miniconda.sh -b -p /opt/miniconda",
            "rm /tmp/miniconda.sh",
            "ln -s /opt/miniconda/bin/conda /usr/local/bin/conda",
            "/opt/miniconda/bin/conda init bash",
            # Accept conda Terms of Service
            "/opt/miniconda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true",
            "/opt/miniconda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true",
        ]
    )
    .env(
        {
            "PATH": "/root/.cargo/bin:/opt/miniconda/bin:$PATH",
            "CUDA_HOME": "/usr/local/cuda",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    .workdir("/app")
    .add_local_dir(".", "/app", copy=True)  # Copy files into image for build steps
    .run_commands([
        # Create simplified conda environment with just Python and CUDA tools
        "/opt/miniconda/bin/conda create -n RiemannianRF python=3.11 -y",
        # Install conda packages only (skip pip dependencies for faster build)
        "/opt/miniconda/bin/conda install -n RiemannianRF -c nvidia -c defaults cuda-toolkit=12.6 -y"
    ])
)


@app.function(
    image=image,
    gpu="H100:1",  # Just need 1 GPU for testing
    timeout=300,  # 5 minutes
)
def test_gpu_access():
    """Test GPU access and CUDA availability"""
    import subprocess
    import torch

    print("=== GPU Test Results ===")

    # Check CUDA availability
    print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(
                f"GPU {i} memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB"
            )

    # Check nvidia-smi
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        if result.returncode == 0:
            print("nvidia-smi output:")
            print(result.stdout)
        else:
            print("nvidia-smi failed")
    except Exception as e:
        print(f"nvidia-smi error: {e}")

    return "GPU test completed"


@app.function(
    image=image,
    timeout=300,  # 5 minutes
)
def test_conda_environment():
    """Test conda environment setup and package imports"""
    import subprocess
    import os

    print("=== Conda Environment Test Results ===")

    # Check if conda is available
    conda_result = subprocess.run(["/opt/miniconda/bin/conda", "--version"], capture_output=True, text=True)
    if conda_result.returncode == 0:
        print(f"✓ Conda version: {conda_result.stdout.strip()}")
    else:
        print("✗ Conda not available")
        return "Conda not available"

    # List available environments
    env_list = subprocess.run(["/opt/miniconda/bin/conda", "env", "list"], capture_output=True, text=True)
    print("Available conda environments:")
    print(env_list.stdout)

    # Check if RiemannianRF environment exists
    conda_python = "/opt/miniconda/envs/RiemannianRF/bin/python"
    if os.path.exists(conda_python):
        print("✓ RiemannianRF environment found")

        # Test critical package imports
        test_packages = [
            "torch",
            "torchvision",
            "torchao",
            "transformers",
            "accelerate",
            "bitsandbytes",
            "diffusers",
            "numpy",
            "tokenizers"
        ]

        print("Testing package imports in conda environment:")
        all_successful = True
        for package in test_packages:
            result = subprocess.run(
                [conda_python, "-c", f"import {package}; print(f'✓ {package} {{getattr({package}, \"__version__\", \"unknown\")}}')"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                print(result.stdout.strip())
            else:
                print(f"✗ {package} import failed: {result.stderr.strip()}")
                all_successful = False

        if all_successful:
            print("✓ All package imports successful!")
            return "Conda environment test passed"
        else:
            print("✗ Some package imports failed")
            return "Some packages missing in conda environment"
    else:
        print("✗ RiemannianRF environment not found")
        return "RiemannianRF environment not found"


@app.function(
    image=image,
    volumes={
        "/datasets": datasets_volume,
    },
    timeout=3600,  # 1 hour
)
def upload_text_zip_file():
    """Upload the coco17_only_txt.zip file to Modal volume"""
    import shutil
    from pathlib import Path

    print("Starting text zip upload process...")

    # Check if the zip file already exists in Modal volume
    zip_path = Path("/datasets/coco17_only_txt.zip")
    if zip_path.exists():
        print(
            f"coco17_only_txt.zip already exists in Modal volume ({zip_path.stat().st_size} bytes)"
        )
        return "Text zip already uploaded"

    # Check if the zip file exists in the app directory
    local_zip = Path("/app/datasets/coco17_only_txt.zip")
    if local_zip.exists():
        print(f"Found local coco17_only_txt.zip ({local_zip.stat().st_size} bytes)")
        print("Copying coco17_only_txt.zip to datasets volume...")

        # Ensure the datasets directory exists
        zip_path.parent.mkdir(parents=True, exist_ok=True)

        shutil.copy2(local_zip, zip_path)
        print(
            f"Successfully uploaded coco17_only_txt.zip ({zip_path.stat().st_size} bytes)"
        )
        return "Text zip uploaded successfully"
    else:
        print("Error: coco17_only_txt.zip not found in /app/datasets/")
        print("Available files in /app/datasets/:")
        datasets_dir = Path("/app/datasets")
        if datasets_dir.exists():
            for item in datasets_dir.iterdir():
                print(f"  {item}")
        else:
            print("  /app/datasets/ directory does not exist")
        return "Text zip not found"


@app.function(
    image=image,
    cpu=4.0,  # Use CPU only - dataset download doesn't need GPU
    memory=8192,  # 8GB RAM for downloading
    volumes={
        "/datasets": datasets_volume,
        "/checkpoints": checkpoints_volume,
        "/cache": cache_volume,
    },
    timeout=86400,  # 24 hours
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def download_dataset():
    """Download COCO17 dataset in parallel"""
    import subprocess
    import sys
    import os
    from pathlib import Path

    # Conda environment should already be created during image building
    print("Setting up conda environment for download...")

    # Check if the RiemannianRF environment exists
    env_check = subprocess.run(
        ["/opt/miniconda/bin/conda", "env", "list"],
        capture_output=True,
        text=True,
    )

    if "RiemannianRF" in env_check.stdout:
        print("✓ RiemannianRF conda environment found")

        # Install pip packages from environment.yml into the conda environment
        print("Installing pip packages into conda environment...")
        conda_python = "/opt/miniconda/envs/RiemannianRF/bin/python"
        conda_pip = "/opt/miniconda/envs/RiemannianRF/bin/pip"

        # Install typing_extensions FIRST to ensure it's available for PyTorch
        print("Installing typing_extensions (critical dependency)...")
        typing_result = subprocess.run(
            [conda_pip, "install", "typing_extensions>=4.8.0"],
            capture_output=True,
            text=True
        )
        if typing_result.returncode == 0:
            print("✓ typing_extensions installed successfully")
        else:
            print(f"✗ typing_extensions installation failed: {typing_result.stderr[:200]}...")

        # Install base dependencies
        print("Installing base dependencies...")
        base_packages = [
            "wheel>=0.40.0",
            "setuptools>=65.0.0",
            "packaging>=21.0",
            "six>=1.16.0",
            "numpy>=1.24.0",
            "requests>=2.32.0",
            "urllib3>=2.0.0",
            "certifi>=2024.0.0",
            "idna>=3.7",
            "charset-normalizer>=3.3.0"
        ]

        for package in base_packages:
            pkg_name = package.split('>=')[0].split('==')[0]
            print(f"Installing {pkg_name}...")
            install_result = subprocess.run(
                [conda_pip, "install", package],
                capture_output=True,
                text=True
            )
            if install_result.returncode == 0:
                print(f"✓ Installed {pkg_name}")
            else:
                print(f"✗ Failed to install {pkg_name}: {install_result.stderr[:100]}...")

        # Install PyTorch with CUDA support (after typing_extensions)
        print("Installing PyTorch 2.4.1 with CUDA 12.1...")
        torch_result = subprocess.run(
            [conda_pip, "install", "torch==2.4.1", "torchvision==0.19.1", "--index-url", "https://download.pytorch.org/whl/cu121"],
            capture_output=True,
            text=True
        )
        if torch_result.returncode == 0:
            print("✓ PyTorch 2.4.1 installed successfully")
        else:
            print(f"✗ PyTorch installation failed: {torch_result.stderr[:200]}...")

        # Install ML packages that depend on PyTorch
        print("Installing ML packages...")
        ml_packages = [
            "datasets>=2.0.0",
            "huggingface-hub>=0.24.0",
            "tokenizers>=0.19.0",
            "transformers>=4.44.0",
            "torchao==0.5.0",
            "tqdm>=4.60.0",
            "pandas>=1.5.0",
            "Pillow>=10.0.0",
            "filelock>=3.12.0",
            "pyyaml>=6.0",
            "fsspec>=2023.1.0",
            "pyarrow>=12.0.0",
            "colorama>=0.4.6",
            "protobuf>=3.20.0"
        ]

        for package in ml_packages:
            pkg_name = package.split('>=')[0].split('==')[0]
            print(f"Installing {pkg_name}...")
            install_result = subprocess.run(
                [conda_pip, "install", package],
                capture_output=True,
                text=True
            )
            if install_result.returncode == 0:
                print(f"✓ Installed {pkg_name}")
            else:
                print(f"✗ Failed to install {pkg_name}: {install_result.stderr[:100]}...")

    else:
        print("✗ RiemannianRF conda environment not found")
        print("Available environments:")
        print(env_check.stdout)

    # Update the dataset download script to use Modal volumes
    script_path = "/app/scripts/download_coco17_modal.py"

    # Check if dataset already exists
    print("Checking if COCO17 dataset already exists...")
    dataset_path = Path("/datasets")
    train_images = dataset_path / "train2017"
    val_images = dataset_path / "val2017"

    if train_images.exists() and val_images.exists():
        train_count = len(list(train_images.glob("*.jpg")))
        val_count = len(list(val_images.glob("*.jpg")))
        print(f"Dataset already exists: {train_count} training images, {val_count} validation images")

        # For testing mode, we expect small numbers
        if train_count > 0 and val_count > 0:
            print("Found existing dataset, checking if we should re-download...")
            # If we have some images, skip download unless explicitly requested
            zip_path = dataset_path / "coco17_only_txt.zip"
            if zip_path.exists():
                print(f"Text zip also exists ({zip_path.stat().st_size} bytes)")
                return "Dataset already exists - skipping download"
            else:
                print("Images exist but text zip missing - will download text files only")

    print("Starting COCO17 dataset download...")

    # Use conda environment python - it should exist now
    conda_python = "/opt/miniconda/envs/RiemannianRF/bin/python"

    if os.path.exists(conda_python):
        python_cmd = conda_python
        print("✓ Using conda environment python")

        # Test import in conda environment
        test_result = subprocess.run(
            [conda_python, "-c", "import torchao; print('✓ torchao available in conda environment')"],
            capture_output=True,
            text=True,
        )
        if test_result.returncode == 0:
            print(test_result.stdout.strip())
        else:
            print(f"✗ torchao not available in conda environment: {test_result.stderr.strip()}")
    else:
        print("✗ Conda environment python not found, falling back to base")
        python_cmd = "/opt/miniconda/bin/python"

    result = subprocess.run([python_cmd, script_path], capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Dataset download failed: {result.stderr}")
        print(f"Dataset download stdout: {result.stdout}")
        raise Exception(f"Dataset download failed: {result.stderr}")

    print("Dataset download completed successfully!")
    print(result.stdout)

    # Check if text zip file exists before running unzip
    print("Checking for coco17_only_txt.zip...")
    zip_path = Path("/datasets/coco17_only_txt.zip")
    if zip_path.exists():
        print(f"Found coco17_only_txt.zip ({zip_path.stat().st_size} bytes)")

        # Now run the unzip script to extract and organize text files
        print("Running text file extraction and organization...")
        print("This may take 5-10 minutes for large zip files...")
        unzip_script_path = "/app/scripts/unzip_coco17_modal.py"

        import time

        start_time = time.time()

        unzip_result = subprocess.run(
            [python_cmd, unzip_script_path], capture_output=True, text=True
        )

        end_time = time.time()
        duration = end_time - start_time
        print(f"Text extraction took {duration:.1f} seconds")

        if unzip_result.returncode != 0:
            print(f"Text extraction failed: {unzip_result.stderr}")
            print(f"Text extraction stdout: {unzip_result.stdout}")
            print(
                "Warning: Text extraction failed, but dataset download was successful"
            )
        else:
            print("Text file extraction completed successfully!")
            print(unzip_result.stdout)
    else:
        print("Warning: coco17_only_txt.zip not found in /datasets/")
        print("Available files in /datasets:")
        datasets_path = Path("/datasets")
        if datasets_path.exists():
            for item in datasets_path.iterdir():
                print(f"  {item}")
        print("Skipping text extraction step.")

    return "Dataset download and text extraction completed"


@app.function(
    image=image,
    gpu="A100:1",  # Switch to 1 A100 for testing
    volumes={
        "/datasets": datasets_volume,
        "/checkpoints": checkpoints_volume,
        "/cache": cache_volume,
    },
    timeout=86400,  # 24 hours (Modal's maximum)
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def train_model():
    """Run the GeodesicFlow training"""
    import subprocess
    import os
    import shutil

    # Verify conda environment exists and use it
    print("Setting up conda environment for training...")

    # Check if the RiemannianRF environment exists
    env_check = subprocess.run(
        ["/opt/miniconda/bin/conda", "env", "list"],
        capture_output=True,
        text=True,
    )

    conda_python = "/opt/miniconda/envs/RiemannianRF/bin/python"

    if "RiemannianRF" in env_check.stdout and os.path.exists(conda_python):
        print("✓ RiemannianRF conda environment found")
        python_executable = conda_python

        # Install ML packages needed for training into conda environment
        print("Installing ML packages into conda environment...")
        conda_pip = "/opt/miniconda/envs/RiemannianRF/bin/pip"

        # Install typing_extensions FIRST (critical for PyTorch imports)
        print("Installing typing_extensions (critical dependency)...")
        typing_result = subprocess.run(
            [conda_pip, "install", "typing_extensions>=4.8.0"],
            capture_output=True,
            text=True
        )
        if typing_result.returncode == 0:
            print("✓ typing_extensions installed successfully")
        else:
            print(f"✗ typing_extensions installation failed: {typing_result.stderr[:200]}...")

        # Install base system packages
        print("Installing base system packages...")
        base_packages = [
            "wheel>=0.40.0",
            "setuptools>=65.0.0",
            "packaging>=21.0",
            "six>=1.16.0",
            "numpy>=1.24.0",
            "filelock>=3.12.0",
            "pyyaml>=6.0",
            "fsspec>=2023.1.0",
            "protobuf>=3.20.0",
            "colorama>=0.4.6"
        ]

        for package in base_packages:
            pkg_name = package.split('>=')[0].split('==')[0]
            print(f"Installing {pkg_name}...")
            install_result = subprocess.run(
                [conda_pip, "install", package],
                capture_output=True,
                text=True
            )
            if install_result.returncode == 0:
                print(f"✓ {pkg_name} installed successfully")
            else:
                print(f"✗ {pkg_name} failed: {install_result.stderr[:100]}...")

        # Install PyTorch with CUDA support (after typing_extensions)
        print("Installing PyTorch 2.4.1 with CUDA 12.1...")
        torch_result = subprocess.run(
            [conda_pip, "install", "torch==2.4.1", "torchvision==0.19.1", "--index-url", "https://download.pytorch.org/whl/cu121"],
            capture_output=True,
            text=True
        )
        if torch_result.returncode == 0:
            print("✓ PyTorch 2.4.1 installed successfully")
        else:
            print(f"✗ PyTorch installation failed: {torch_result.stderr[:200]}...")

        # Install ML packages that require PyTorch
        print("Installing ML packages...")
        ml_packages = [
            "tokenizers>=0.19.0",
            "transformers>=4.44.0",
            "accelerate>=0.33.0",
            "huggingface-hub>=0.24.0",
            "diffusers>=0.30.0",
            "torchao==0.5.0",
            "torch-optimi>=0.2.0",
            "bitsandbytes>=0.44.0",
            "safetensors>=0.3.0",
            "regex>=2023.0.0",
            "scipy>=1.10.0",
            "psutil>=5.9.0",
            "sympy>=1.12",
            "networkx>=3.0",
            "jinja2>=3.0.0",
            "MarkupSafe>=2.1.0",
            "einops>=0.7.0"
        ]

        for package in ml_packages:
            pkg_name = package.split('>=')[0].split('==')[0]
            print(f"Installing {pkg_name}...")
            install_result = subprocess.run(
                [conda_pip, "install", package],
                capture_output=True,
                text=True
            )
            if install_result.returncode == 0:
                print(f"✓ {pkg_name} installed successfully")
            else:
                print(f"✗ {pkg_name} failed: {install_result.stderr[:100]}...")

        # Test critical imports in conda environment
        print("Testing critical imports in conda environment...")
        test_imports = [
            "torch",
            "torchao",
            "transformers",
            "accelerate",
            "bitsandbytes",
            "diffusers"
        ]

        all_imports_successful = True
        for import_name in test_imports:
            test_result = subprocess.run(
                [conda_python, "-c", f"import {import_name}; print('✓ {import_name} imported successfully')"],
                capture_output=True,
                text=True,
            )
            if test_result.returncode == 0:
                print(test_result.stdout.strip())
            else:
                print(f"✗ {import_name} import failed: {test_result.stderr.strip()}")
                all_imports_successful = False

        if all_imports_successful:
            print("✓ All critical imports successful in conda environment!")
        else:
            print("✗ Some imports failed in conda environment")

    else:
        print("✗ RiemannianRF conda environment not found")
        print("Available environments:")
        print(env_check.stdout)
        print("Falling back to base conda python")
        python_executable = "/opt/miniconda/bin/python"

    # Create .venv directory that the training script expects
    print("Setting up .venv directory...")
    venv_path = "/app/.venv"
    os.makedirs(venv_path, exist_ok=True)

    # Test environment before training with minimal imports
    print("Testing environment setup...")
    test_imports = [
        "import idna; print('✓ idna')",
        "import requests; print('✓ requests')",
        "import torch; print('✓ torch')",
        "import accelerate; print('✓ accelerate')",
        "import transformers; print('✓ transformers')",
        "import diffusers; print('✓ diffusers')",
        "import torchao; print('✓ torchao')"
    ]

    for test_import in test_imports:
        test_result = subprocess.run(
            [python_executable, "-c", test_import],
            capture_output=True,
            text=True,
        )
        if test_result.returncode == 0:
            print(test_result.stdout.strip())
        else:
            print(f"✗ Import failed: {test_import}")
            print(f"  Error: {test_result.stderr.strip()}")

    # Set environment variables for the training - using conda environment
    env = os.environ.copy()
    python_dir = os.path.dirname(python_executable)

    # Determine conda environment settings
    if "RiemannianRF" in python_executable:
        virtual_env = "/opt/miniconda/envs/RiemannianRF"
        conda_default_env = "RiemannianRF"
        conda_prefix = "/opt/miniconda/envs/RiemannianRF"
        path_prefix = "/opt/miniconda/envs/RiemannianRF/bin:/opt/miniconda/bin"
        print("✓ Using RiemannianRF conda environment for training")
    else:
        virtual_env = "/opt/miniconda"
        conda_default_env = "base"
        conda_prefix = "/opt/miniconda"
        path_prefix = "/opt/miniconda/bin"
        print("Using base conda environment for training")

    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "0",  # Single A100 GPU
            "CONFIG_ENV_FILE": "config/config_proposed_flux.env",
            "CONFIG_JSON_FILE": "config/config_coco17_flux_testing.json",  # Use testing config
            "CONFIG_BACKEND": "json",
            "DISABLE_UPDATES": "1",
            "PYTHONPATH": "/app",
            "TRAINING_NUM_PROCESSES": "1",  # Single process for 1 GPU
            "TRAINING_NUM_MACHINES": "1",
            "MIXED_PRECISION": "bf16",
            "MAIN_PROCESS_PORT": "29501",
            "ACCELERATE_CONFIG_PATH": "/app/accelerate_config_a100.yaml",  # Use A100 config
            "VENV_PATH": "/app/.venv",  # Set the VENV_PATH that training script expects
            "VIRTUAL_ENV": virtual_env,
            "CONDA_DEFAULT_ENV": conda_default_env,
            "CONDA_PREFIX": conda_prefix,
            # Put conda environment first in PATH
            "PATH": path_prefix + ":" + env.get("PATH", ""),
            "TOKENIZERS_PARALLELISM": "false",
            # Force the shell script to use conda environment python
            "PYTHON": python_executable,
        }
    )

    # Copy accelerate config to home directory
    os.makedirs(os.path.expanduser("~/.cache/huggingface/accelerate"), exist_ok=True)

    # Check if A100 config exists, fallback to H100 config
    a100_config = "/app/accelerate_config_a100.yaml"
    h100_config = "/app/accelerate_config_h100.yaml"

    if os.path.exists(a100_config):
        config_source = a100_config
        print("Using A100 accelerate config")
    else:
        config_source = h100_config
        print("A100 config not found, using H100 config")

    shutil.copy(
        config_source,
        os.path.expanduser("~/.cache/huggingface/accelerate/default_config.yaml"),
    )

    # Set up nvjitlink directory to prevent LD_LIBRARY_PATH errors
    nvjitlink_dir = os.path.join(venv_path, "nvjitlink", "lib")
    os.makedirs(nvjitlink_dir, exist_ok=True)

    # Test accelerate command with the working python
    print("Testing accelerate command...")
    accelerate_test = subprocess.run(
        [python_executable, "-c", "import accelerate; print('✓ Accelerate import successful')"],
        env=env,
        capture_output=True,
        text=True,
    )
    if accelerate_test.returncode == 0:
        print("✓ Accelerate python import test passed")
    else:
        print(f"✗ Accelerate python import test failed: {accelerate_test.stderr}")

    # Test direct accelerate binary
    accelerate_binary = os.path.join(python_dir, "accelerate")
    if os.path.exists(accelerate_binary):
        print(f"Testing accelerate binary: {accelerate_binary}")
        accelerate_launch_test = subprocess.run(
            [accelerate_binary, "launch", "--help"],
            env=env,
            capture_output=True,
            text=True,
            timeout=10
        )
        if accelerate_launch_test.returncode == 0:
            print("✓ Accelerate launch command test passed")
        else:
            print(f"✗ Accelerate launch test failed")
    else:
        print(f"Accelerate binary not found at {accelerate_binary}")
        print("Will try to continue anyway...")

    print("Starting GeodesicFlow training...")
    print(f"Using environment variables:")
    for key in [
        "CUDA_VISIBLE_DEVICES",
        "CONFIG_ENV_FILE",
        "CONFIG_JSON_FILE",
        "TRAINING_NUM_PROCESSES",
    ]:
        print(f"  {key}={env.get(key)}")

    # Run the simplified training script that uses base conda
    result = subprocess.run(
        ["/bin/bash", "/app/train_testing_base_conda.sh"],
        env=env,
        capture_output=True,
        text=True,
        cwd="/app",
    )

    print(f"Training script stdout: {result.stdout}")
    if result.stderr:
        print(f"Training script stderr: {result.stderr}")

    if result.returncode != 0:
        print(f"Training failed with return code: {result.returncode}")
        raise Exception(f"Training failed: {result.stderr}")

    print("Training completed successfully!")

    # Sync outputs back to local
    return "Training completed successfully"


@app.function(
    image=image,
    volumes={
        "/checkpoints": checkpoints_volume,
    },
    timeout=3600,
)
def sync_outputs_to_local():
    """Sync training outputs from Modal volumes to local directory"""
    import tarfile
    import tempfile
    from pathlib import Path

    checkpoint_dir = Path("/checkpoints")
    if not checkpoint_dir.exists():
        return "No checkpoint directory found"

    # Create a tar archive of all outputs
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp_file:
        with tarfile.open(tmp_file.name, "w:gz") as tar:
            for item in checkpoint_dir.rglob("*"):
                if item.is_file():
                    # Add file with relative path from checkpoints
                    arcname = str(item.relative_to(checkpoint_dir.parent))
                    tar.add(item, arcname=arcname)

        # Read the tar file and return its contents
        with open(tmp_file.name, "rb") as f:
            tar_data = f.read()

        import os

        os.unlink(tmp_file.name)

    files = list(checkpoint_dir.rglob("*"))
    print(f"Packaged {len(files)} files from checkpoints volume")
    return tar_data


@app.function(
    image=image,
    volumes={
        "/checkpoints": checkpoints_volume,
    },
    timeout=1800,
)
def list_outputs():
    """List all files in the checkpoints volume"""
    from pathlib import Path

    checkpoint_dir = Path("/checkpoints")
    if not checkpoint_dir.exists():
        return "No checkpoint directory found"

    files = []
    total_size = 0
    for item in checkpoint_dir.rglob("*"):
        if item.is_file():
            size = item.stat().st_size
            total_size += size
            files.append(
                {
                    "path": str(item.relative_to(checkpoint_dir)),
                    "size": size,
                    "modified": item.stat().st_mtime,
                }
            )

    return {
        "total_files": len(files),
        "total_size_bytes": total_size,
        "total_size_mb": total_size / (1024 * 1024),
        "files": sorted(files, key=lambda x: x["modified"], reverse=True)[
            :20
        ],  # Latest 20 files
    }


@app.local_entrypoint()
def main(
    download_data: bool = False,
    train: bool = True,
    list_files: bool = False,
    sync_outputs: bool = False,
    upload_text_zip: bool = False,
    test_gpu: bool = False,
    test_conda: bool = False,
    detach: bool = False,
):
    """Main entrypoint for the Modal app

    Use --detach to keep the app running even if local client disconnects.
    Example: modal run modal_app.py --train --detach
    """

    if test_gpu:
        print("Testing GPU access...")
        gpu_result = test_gpu_access.remote()
        print(f"GPU test result: {gpu_result}")

    if test_conda:
        print("Testing conda environment...")
        conda_result = test_conda_environment.remote()
        print(f"Conda test result: {conda_result}")

    if upload_text_zip or download_data:
        print("Uploading coco17_only_txt.zip...")
        upload_result = upload_text_zip_file.remote()
        print(f"Upload result: {upload_result}")

    if download_data:
        print("Downloading COCO17 dataset...")
        download_result = download_dataset.remote()
        print(f"Download result: {download_result}")

    if train:
        print("Starting training...")
        train_result = train_model.remote()
        print(f"Training result: {train_result}")

    if list_files:
        print("Listing output files...")
        files_info = list_outputs.remote()
        print(f"Files info: {files_info}")

    if sync_outputs:
        print("Syncing outputs...")
        print("Note: This will download all checkpoint files as a tar.gz")
        sync_result = sync_outputs_to_local.remote()
        if isinstance(sync_result, bytes):
            # Save the tar data to local output directory
            import os

            os.makedirs("output", exist_ok=True)
            with open("output/modal_checkpoints.tar.gz", "wb") as f:
                f.write(sync_result)
            print(
                f"Saved checkpoint archive to output/modal_checkpoints.tar.gz ({len(sync_result)} bytes)"
            )
        else:
            print(f"Sync result: {sync_result}")


if __name__ == "__main__":
    main()
