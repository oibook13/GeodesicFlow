"""
Comprehensive import utilities with fallback handling for optional dependencies.
This module provides robust import handling for all external dependencies used in the training pipeline.
"""

import warnings
import sys
from typing import Any, Optional, Dict, List
import importlib


class ImportFailure:
    """Represents a failed import with fallback information"""
    def __init__(self, module_name: str, error: Exception, fallback_msg: str = None):
        self.module_name = module_name
        self.error = error
        self.fallback_msg = fallback_msg or f"Module '{module_name}' not available"


class RobustImporter:
    """Handles imports with comprehensive fallback support"""

    def __init__(self):
        self.import_status: Dict[str, bool] = {}
        self.import_errors: List[ImportFailure] = []

    def safe_import(self, module_name: str, fallback_msg: str = None, required: bool = False) -> Optional[Any]:
        """
        Safely import a module with optional fallback handling.

        Args:
            module_name: Name of the module to import
            fallback_msg: Custom message for fallback scenario
            required: If True, raises exception on failure

        Returns:
            Imported module or None if import fails and not required
        """
        try:
            module = importlib.import_module(module_name)
            self.import_status[module_name] = True
            print(f"✓ Successfully imported {module_name}")
            return module
        except ImportError as e:
            self.import_status[module_name] = False
            error = ImportFailure(module_name, e, fallback_msg)
            self.import_errors.append(error)

            if required:
                print(f"✗ CRITICAL: Required module '{module_name}' failed to import: {e}")
                raise e
            else:
                msg = fallback_msg or f"Optional module '{module_name}' not available, using fallbacks"
                print(f"⚠ Warning: {msg}")
                return None

    def print_import_summary(self):
        """Print a summary of all import attempts"""
        print("\n" + "="*60)
        print("IMPORT SUMMARY")
        print("="*60)

        successful = [name for name, status in self.import_status.items() if status]
        failed = [name for name, status in self.import_status.items() if not status]

        if successful:
            print(f"✓ Successfully imported ({len(successful)}):")
            for name in successful:
                print(f"  - {name}")

        if failed:
            print(f"\n⚠ Failed imports ({len(failed)}):")
            for name in failed:
                print(f"  - {name}")

        print("="*60 + "\n")


# Global importer instance
importer = RobustImporter()

# Diffusers-related imports with fallbacks
def import_diffusers_with_fallbacks():
    """Import diffusers components with comprehensive fallback handling"""

    # Core diffusers
    diffusers = importer.safe_import("diffusers", required=True)

    # Specific components with fallbacks
    try:
        from diffusers.models.normalization import AdaLayerNormContinuous
        print("✓ AdaLayerNormContinuous imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import AdaLayerNormContinuous: {e}")
        raise

    # FP32LayerNorm with fallback
    try:
        from diffusers.models.normalization import FP32LayerNorm
        print("✓ FP32LayerNorm imported from diffusers")
        return FP32LayerNorm
    except ImportError:
        print("⚠ FP32LayerNorm not available in diffusers, using fallback implementation")
        import torch.nn as nn
        import torch.nn.functional as F

        class FP32LayerNorm(nn.LayerNorm):
            """Fallback FP32LayerNorm implementation"""
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

            def forward(self, x):
                output = F.layer_norm(
                    x.float(),
                    self.normalized_shape,
                    self.weight.float() if self.weight is not None else None,
                    self.bias.float() if self.bias is not None else None,
                    self.eps,
                )
                return output.type_as(x)

        return FP32LayerNorm


# Optimizer-related imports with fallbacks
def import_optimizers_with_fallbacks():
    """Import optimizer packages with fallback handling"""

    # torch-optimi (optional)
    optimi = importer.safe_import(
        "optimi",
        "torch-optimi not available, optimi-based optimizers will be disabled",
        required=False
    )

    # torchao (optional)
    torchao = importer.safe_import(
        "torchao",
        "torchao not available, low-precision optimizers will be disabled",
        required=False
    )

    # bitsandbytes (optional)
    bitsandbytes = importer.safe_import(
        "bitsandbytes",
        "bitsandbytes not available, 8-bit optimizers will be disabled",
        required=False
    )

    return {
        'optimi': optimi,
        'torchao': torchao,
        'bitsandbytes': bitsandbytes
    }


# Core ML imports (required)
def import_core_ml_packages():
    """Import core ML packages that are required for training"""

    # Critical packages
    torch = importer.safe_import("torch", required=True)
    transformers = importer.safe_import("transformers", required=True)
    accelerate = importer.safe_import("accelerate", required=True)
    diffusers = importer.safe_import("diffusers", required=True)

    # Verify CUDA availability
    if torch and torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.cuda.device_count()} GPUs")
    else:
        print("⚠ Warning: CUDA not available or PyTorch not compiled with CUDA support")

    return {
        'torch': torch,
        'transformers': transformers,
        'accelerate': accelerate,
        'diffusers': diffusers
    }


# Comprehensive import verification
def verify_all_imports():
    """Verify all imports and print comprehensive status"""
    print("Starting comprehensive import verification...")

    # Core packages
    core_packages = import_core_ml_packages()

    # Optimizer packages
    optimizer_packages = import_optimizers_with_fallbacks()

    # Diffusers components
    try:
        fp32_layer_norm = import_diffusers_with_fallbacks()
        print("✓ Diffusers components imported successfully")
    except Exception as e:
        print(f"✗ Failed to import diffusers components: {e}")
        raise

    # Print summary
    importer.print_import_summary()

    return {
        'core': core_packages,
        'optimizers': optimizer_packages,
        'status': importer.import_status,
        'errors': importer.import_errors
    }


if __name__ == "__main__":
    # Test all imports when run directly
    verify_all_imports()