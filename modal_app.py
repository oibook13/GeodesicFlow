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
    modal.Image.from_registry("nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04")
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
        ]
    )
    .run_commands(
        [
            # Install miniconda
            "wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh",
            "bash /tmp/miniconda.sh -b -p /opt/miniconda",
            "rm /tmp/miniconda.sh",
            "ln -s /opt/miniconda/bin/conda /usr/local/bin/conda",
            "/opt/miniconda/bin/conda init bash",
        ]
    )
    .env(
        {
            "PATH": "/opt/miniconda/bin:$PATH",
            "CUDA_HOME": "/usr/local/cuda",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    .workdir("/app")
    .add_local_dir(".", "/app")  # Must be last in build chain
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
    gpu="H100:8",
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

    # Accept conda Terms of Service first
    print("Accepting conda Terms of Service...")
    tos_commands = [
        [
            "/opt/miniconda/bin/conda",
            "tos",
            "accept",
            "--override-channels",
            "--channel",
            "https://repo.anaconda.com/pkgs/main",
        ],
        [
            "/opt/miniconda/bin/conda",
            "tos",
            "accept",
            "--override-channels",
            "--channel",
            "https://repo.anaconda.com/pkgs/r",
        ],
    ]

    for cmd in tos_commands:
        subprocess.run(cmd, capture_output=True, text=True)

    # Create conda environment from the copied environment.yml
    print("Setting up conda environment...")
    env_setup = subprocess.run(
        ["/opt/miniconda/bin/conda", "env", "create", "-f", "/app/environment.yml"],
        capture_output=True,
        text=True,
    )

    if env_setup.returncode != 0:
        print(f"Conda environment setup output: {env_setup.stdout}")
        print(f"Conda environment setup error: {env_setup.stderr}")
        print("Environment creation failed, falling back to base conda python")
    else:
        print("Conda environment created successfully")

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

        # Still check for text files
        zip_path = dataset_path / "coco17_only_txt.zip"
        if zip_path.exists():
            print(f"Text zip also exists ({zip_path.stat().st_size} bytes)")
            return "Dataset already exists - skipping download"
        else:
            print("Images exist but text zip missing - will download text files only")

    print("Starting COCO17 dataset download...")

    # Try conda environment python first, fallback to base conda python
    conda_python = "/opt/miniconda/envs/RiemannianRF/bin/python"
    base_python = "/opt/miniconda/bin/python"

    if os.path.exists(conda_python):
        python_cmd = conda_python
        print("Using conda environment python")
    elif os.path.exists(base_python):
        python_cmd = base_python
        print("Using base conda python")
    else:
        python_cmd = sys.executable
        print("Using system python")

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
    gpu="H100:8",
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

    # Accept conda Terms of Service first
    print("Accepting conda Terms of Service...")
    tos_commands = [
        [
            "/opt/miniconda/bin/conda",
            "tos",
            "accept",
            "--override-channels",
            "--channel",
            "https://repo.anaconda.com/pkgs/main",
        ],
        [
            "/opt/miniconda/bin/conda",
            "tos",
            "accept",
            "--override-channels",
            "--channel",
            "https://repo.anaconda.com/pkgs/r",
        ],
    ]

    for cmd in tos_commands:
        subprocess.run(cmd, capture_output=True, text=True)

    # Check if conda environment already exists
    print("Checking conda environment...")
    conda_python = "/opt/miniconda/envs/RiemannianRF/bin/python"

    if os.path.exists(conda_python):
        print("Conda environment already exists, skipping creation")
        env_setup_success = True
    else:
        # Create conda environment from the copied environment.yml
        print("Setting up conda environment...")
        env_setup = subprocess.run(
            ["/opt/miniconda/bin/conda", "env", "create", "-f", "/app/environment.yml"],
            capture_output=True,
            text=True,
        )
        env_setup_success = env_setup.returncode == 0

    if not env_setup_success:
        print(f"Conda environment setup output: {env_setup.stdout if 'env_setup' in locals() else 'N/A'}")
        print(f"Conda environment setup error: {env_setup.stderr if 'env_setup' in locals() else 'N/A'}")
        print("Environment creation failed, falling back to base conda python")
    else:
        # Install missing dependencies and ensure proper package installation
        print("Installing and verifying dependencies...")
        if os.path.exists(conda_python):
            # First, ensure core dependencies are properly installed
            core_deps = ["idna", "charset-normalizer", "urllib3", "certifi", "requests"]
            for dep in core_deps:
                dep_install = subprocess.run(
                    [conda_python, "-m", "pip", "install", "--upgrade", "--force-reinstall", dep],
                    capture_output=True,
                    text=True,
                )
                if dep_install.returncode == 0:
                    print(f"Successfully installed/upgraded {dep}")
                else:
                    print(f"Failed to install {dep}: {dep_install.stderr}")

            # Reinstall accelerate and huggingface-hub to ensure they work with our environment
            critical_packages = ["accelerate", "huggingface-hub", "transformers"]
            for package in critical_packages:
                print(f"Reinstalling {package}...")
                pkg_install = subprocess.run(
                    [conda_python, "-m", "pip", "install", "--upgrade", "--force-reinstall", package],
                    capture_output=True,
                    text=True,
                )
                if pkg_install.returncode == 0:
                    print(f"Successfully reinstalled {package}")
                else:
                    print(f"Failed to reinstall {package}: {pkg_install.stderr}")

    # Create .venv directory that the training script expects
    print("Setting up .venv directory...")
    venv_path = "/app/.venv"
    os.makedirs(venv_path, exist_ok=True)

    # Test environment before training
    print("Testing environment setup...")
    test_result = subprocess.run(
        [conda_python, "-c", "import idna, requests, accelerate, transformers; print('Environment test passed')"],
        capture_output=True,
        text=True,
    )
    if test_result.returncode == 0:
        print("✓ Environment test passed")
    else:
        print(f"✗ Environment test failed: {test_result.stderr}")
        print("Continuing anyway...")

    # Set environment variables for the training
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
            "CONFIG_ENV_FILE": "config/config_proposed_flux.env",
            "CONFIG_JSON_FILE": "config/config_coco17_flux_proposed_noregularization.json",
            "CONFIG_BACKEND": "json",
            "DISABLE_UPDATES": "1",
            "PYTHONPATH": "/app",
            "TRAINING_NUM_PROCESSES": "8",
            "TRAINING_NUM_MACHINES": "1",
            "MIXED_PRECISION": "bf16",
            "MAIN_PROCESS_PORT": "29501",
            "ACCELERATE_CONFIG_PATH": "/app/accelerate_config_h100.yaml",
            "VENV_PATH": "/app/.venv",  # Set the VENV_PATH that training script expects
            "VIRTUAL_ENV": "/opt/miniconda/envs/RiemannianRF",  # Set VIRTUAL_ENV for conda
            "CONDA_DEFAULT_ENV": "RiemannianRF",
            "CONDA_PREFIX": "/opt/miniconda/envs/RiemannianRF",
            # Put conda environment first in PATH
            "PATH": "/opt/miniconda/envs/RiemannianRF/bin:/opt/miniconda/bin:"
            + env.get("PATH", ""),
            "TOKENIZERS_PARALLELISM": "false",
        }
    )

    # Copy accelerate config to home directory
    os.makedirs(os.path.expanduser("~/.cache/huggingface/accelerate"), exist_ok=True)
    shutil.copy(
        "/app/accelerate_config_h100.yaml",
        os.path.expanduser("~/.cache/huggingface/accelerate/default_config.yaml"),
    )

    # Set up nvjitlink directory to prevent LD_LIBRARY_PATH errors
    nvjitlink_dir = os.path.join(venv_path, "nvjitlink", "lib")
    os.makedirs(nvjitlink_dir, exist_ok=True)

    # Test accelerate command with proper environment
    print("Testing accelerate command...")
    accelerate_test = subprocess.run(
        ["/opt/miniconda/envs/RiemannianRF/bin/accelerate", "--help"],
        env=env,
        capture_output=True,
        text=True,
    )
    if accelerate_test.returncode == 0:
        print("✓ Accelerate command test passed")
    else:
        print(f"✗ Accelerate command test failed: {accelerate_test.stderr}")
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

    # Run the training script
    result = subprocess.run(
        ["/bin/bash", "/app/train_proposed_noregularization.sh"],
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
