#!/usr/bin/env python3
"""
Validation script to check Modal integration setup
"""

import os
from pathlib import Path
import json

def check_file_exists(path, description):
    """Check if a file exists and report status"""
    if os.path.exists(path):
        print(f"✅ {description}: {path}")
        return True
    else:
        print(f"❌ {description}: {path} (MISSING)")
        return False

def validate_json_file(path, description):
    """Validate JSON file can be parsed"""
    if not check_file_exists(path, description):
        return False

    try:
        with open(path, 'r') as f:
            json.load(f)
        print(f"   ✅ JSON syntax valid")
        return True
    except json.JSONDecodeError as e:
        print(f"   ❌ JSON syntax error: {e}")
        return False

def main():
    print("🔍 Validating Modal Integration Setup for GeodesicFlow")
    print("=" * 60)

    base_dir = Path(__file__).parent
    all_good = True

    # Check main files
    files_to_check = [
        (base_dir / "modal_app.py", "Modal application"),
        (base_dir / "environment.yml", "Conda environment"),
        (base_dir / "train_proposed_noregularization.sh", "Training script"),
        (base_dir / "accelerate_config_h100.yaml", "Accelerate config"),
        (base_dir / "scripts" / "download_coco17_modal.py", "Modal dataset downloader"),
        (base_dir / "MODAL_README.md", "Modal documentation"),
    ]

    for file_path, description in files_to_check:
        if not check_file_exists(file_path, description):
            all_good = False

    # Check config files with JSON validation
    config_files = [
        (base_dir / "config" / "config_proposed_flux.env", "Environment config"),
        (base_dir / "config" / "config_coco17_flux_proposed_noregularization.json", "Training config (JSON)"),
        (base_dir / "config" / "coco17_modal.json", "Dataset config for Modal (JSON)"),
    ]

    for file_path, description in config_files:
        if str(file_path).endswith('.json'):
            if not validate_json_file(file_path, description):
                all_good = False
        else:
            if not check_file_exists(file_path, description):
                all_good = False

    # Check executable permissions
    script_path = base_dir / "train_proposed_noregularization.sh"
    if os.path.exists(script_path):
        if os.access(script_path, os.X_OK):
            print(f"✅ Executable permissions: {script_path}")
        else:
            print(f"❌ Executable permissions: {script_path} (run 'chmod +x {script_path}')")
            all_good = False

    print("\n" + "=" * 60)
    if all_good:
        print("🎉 All files present and valid! Ready for Modal deployment.")
        print("\nNext steps:")
        print("1. Install modal: pip install modal")
        print("2. Setup modal: modal setup")
        print("3. Create HF secret: modal secret create huggingface-secret HF_TOKEN=your_token")
        print("4. Run training: modal run modal_app.py --download-data --train")
    else:
        print("⚠️  Some files are missing or invalid. Please fix the issues above.")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())