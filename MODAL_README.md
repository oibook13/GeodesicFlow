# GeodesicFlow Modal Integration

This repository now includes Modal integration for distributed training on H100 GPUs in the cloud.

## Setup

1. **Install Modal**:
   ```bash
   pip install modal
   ```

2. **Setup Modal Account**:
   ```bash
   modal setup
   ```

3. **Create HuggingFace Secret** (for model access):
   ```bash
   modal secret create huggingface-secret HF_TOKEN=your_huggingface_token_here
   ```

## Usage

### 1. Upload Text Zip Only
To upload the coco17_only_txt.zip file to Modal volumes:
```bash
modal run modal_app.py --upload-text-zip --no-download-data --no-train --no-list-files --no-sync-outputs
```

### 2. Download Dataset Only
To download the COCO17 dataset to Modal volumes (includes text zip upload):
```bash
modal run modal_app.py --download-data --no-train --no-list-files --no-sync-outputs
```

### 3. Run Training (Recommended with --detach)
To run the full training (assumes dataset is already downloaded):
```bash
# Standard training (stays connected)
modal run modal_app.py --train

# Detached training (recommended for long runs)
modal run modal_app.py --train --detach
```

### 4. Download Dataset and Train
To download dataset and run training in sequence:
```bash
# With logging to file
modal run modal_app.py --download-data --train --detach 2>&1 | tee train_coco17_flux.log

# Simple version
modal run modal_app.py --download-data --train --detach
```

### 5. List Output Files
To see what checkpoint files were created:
```bash
modal run modal_app.py --list-files
```

### 6. Sync Outputs to Local
To download all checkpoints and outputs:
```bash
modal run modal_app.py --sync-outputs
```

### 7. Check Running Jobs
After using --detach, you can monitor progress:
```bash
# List running apps
modal app list

# View logs for a specific app
modal app logs <app-id>

# Stop a running app
modal app stop <app-id>
```

**Important Notes**:
- **Use `--detach`** for long training runs to prevent disconnection issues
- The Modal app now optimizes environment setup - skips conda environment creation and dataset download if they already exist
- First run may take longer for initial setup, subsequent runs are much faster
- Always use `--detach` for training to avoid losing progress if your connection drops

## Text File Processing

The Modal integration includes automatic processing of COCO17 text files:

1. **Text Zip Upload**: The `coco17_only_txt.zip` file from your local `datasets/` directory is automatically uploaded to Modal volumes
2. **Text Extraction**: After dataset download, the zip is extracted and organized:
   - Text files are placed in appropriate train/validation/test splits
   - Files are matched with corresponding JPG images by filename
   - Duplicate or orphaned text files are automatically cleaned up
3. **Final Structure**: Text files end up alongside their corresponding images in `/datasets/train/`, `/datasets/validation/`, etc.

## Configuration

The training uses these key files:

- **`config/config_proposed_flux.env`** - Environment variables for training
- **`config/config_coco17_flux_proposed_noregularization.json`** - Main training configuration
- **`config/coco17_modal.json`** - Dataset configuration for Modal volumes
- **`accelerate_config_h100.yaml`** - Multi-GPU configuration for 8x H100s

## File Structure

```
/datasets          # COCO17 dataset (Modal volume)
/checkpoints       # Training outputs and checkpoints (Modal volume)
/cache            # VAE and text embedding cache (Modal volume)
/app              # Your code (mounted from local)
```

## Key Features

- **8x H100 GPU training** with proper accelerate configuration
- **Persistent storage** using Modal volumes for datasets, checkpoints, and cache
- **Automatic dataset download** with parallel processing
- **Output synchronization** to bring results back to local machine
- **CUDA 12.6 environment** with full conda environment from `environment.yml`

## Training Command Executed

The Modal app runs this exact command on 8x H100s:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
CONFIG_ENV_FILE='config/config_proposed_flux.env' \
CONFIG_JSON_FILE='config/config_coco17_flux_proposed_noregularization.json' \
CONFIG_BACKEND=json \
DISABLE_UPDATES=1 \
./train_proposed_noregularization.sh
```

## Monitoring

- Check logs in Modal dashboard: https://modal.com/
- Training artifacts will be in `/checkpoints` volume
- Use `--list-files` to see what was generated
- Use `--sync-outputs` to download everything as `output/modal_checkpoints.tar.gz`

## Cost Optimization

- Modal charges by GPU-hour usage
- Consider using 4x H100s instead of 8x if budget is a concern
- Dataset download only needs to be done once (persisted in volumes)
- Checkpoints are preserved across runs

## Troubleshooting

### Environment Issues
- The Modal app now includes comprehensive dependency management and testing
- Environment validation ensures all packages are properly installed before training
- If you see `idna` or similar import errors, the app will automatically fix them

### Connection Issues
- **Always use `--detach`** for training to prevent disconnection problems
- If training stops due to local client disconnect, check `modal app list` to see if it's still running
- Use `modal app logs <app-id>` to check progress of detached jobs

### Common Issues
- Make sure you have H100 access enabled on your Modal account
- Verify HuggingFace token has access to FLUX.1-dev model
- Check Modal dashboard for detailed error logs
- Dataset download can take 1-2 hours depending on network speed
- First-time conda environment setup takes ~10-15 minutes but is cached for subsequent runs

### Performance Optimization
- Dataset download is skipped if COCO17 already exists in volumes
- Conda environment creation is skipped if environment already exists
- Use `--train` only (without `--download-data`) for faster subsequent training runs