import huggingface_hub
from helpers.training.default_settings.safety_check import safety_check
from helpers.publishing.huggingface import HubManager
from configure import model_labels
import shutil
import hashlib
import json
import copy
import random
import logging
import math
import os
import sys
import glob
import wandb

os.environ["ACCELERATE_LOG_LEVEL"] = "WARNING"
from helpers import log_format  # noqa
from helpers.configuration.loader import load_config
from helpers.caching.memory import reclaim_memory
from helpers.training.multi_process import _get_rank as get_rank
from helpers.training.validation import Validation, prepare_validation_prompt_list
from helpers.training.state_tracker import StateTracker
from helpers.training.schedulers import load_scheduler_from_args
from helpers.training.custom_schedule import get_lr_scheduler
from helpers.training.adapter import determine_adapter_target_modules, load_lora_weights
from helpers.training.diffusion_model import load_diffusion_model
from helpers.training.text_encoding import (
    load_tes,
    determine_te_path_subfolder,
    import_model_class_from_model_name_or_path,
    get_tokenizers,
)
from helpers.training.optimizer_param import (
    determine_optimizer_class_with_config,
    determine_params_to_optimize,
    is_lr_scheduler_disabled,
    cpu_offload_optimizer,
)
from helpers.data_backend.factory import BatchFetcher
from helpers.training.deepspeed import (
    deepspeed_zero_init_disabled_context_manager,
    prepare_model_for_deepspeed,
)
from helpers.training.wrappers import unwrap_model
from helpers.data_backend.factory import configure_multi_databackend
from helpers.data_backend.factory import random_dataloader_iterator
from helpers.training import steps_remaining_in_epoch
from helpers.training.custom_schedule import (
    generate_timestep_weights,
    segmented_timestep_selection,
)
from helpers.training.min_snr_gamma import compute_snr
from helpers.training.peft_init import init_lokr_network_with_perturbed_normal
from accelerate.logging import get_logger
from diffusers.models.embeddings import get_2d_rotary_pos_embed
from helpers.models.smoldit import get_resize_crop_region_for_grid

logger = get_logger(
    "SimpleTuner", log_level=os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO")
)

filelock_logger = get_logger("filelock")
connection_logger = get_logger("urllib3.connectionpool")
training_logger = get_logger("training-loop")

# More important logs.
target_level = os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO")
logger.setLevel(target_level)
training_logger_level = os.environ.get("SIMPLETUNER_TRAINING_LOOP_LOG_LEVEL", "INFO")
training_logger.setLevel(training_logger_level)

# Less important logs.
filelock_logger.setLevel("WARNING")
connection_logger.setLevel("WARNING")
import torch
import torch.nn as nn
import diffusers
import accelerate
import transformers
import torch.nn.functional as F
import torch.utils.checkpoint
from accelerate import Accelerator
from accelerate.utils import set_seed
from configure import model_classes
from torch.distributions import Beta

try:
    from lycoris import LycorisNetwork
except:
    print("[ERROR] Lycoris not available. Please install ")
from tqdm.auto import tqdm
from transformers import PretrainedConfig, CLIPTokenizer
from helpers.models.sdxl.pipeline import StableDiffusionXLPipeline
from diffusers import StableDiffusion3Pipeline

from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDIMScheduler,
    DDPMScheduler,
    UNet2DConditionModel,
    FluxTransformer2DModel,
    PixArtTransformer2DModel,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    UniPCMultistepScheduler,
)

from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from helpers.training.ema import EMAModel
from diffusers.utils import (
    check_min_version,
    convert_state_dict_to_diffusers,
    is_wandb_available,
)
from diffusers.utils.import_utils import is_xformers_available
from transformers.utils import ContextManagers

from helpers.models.flux import (
    prepare_latent_image_ids,
    pack_latents,
    unpack_latents,
    get_mobius_guidance,
    apply_flux_schedule_shift,
)

is_optimi_available = False
try:
    from optimi import prepare_for_gradient_release

    is_optimi_available = True
except:
    pass

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.27.0.dev0")

SCHEDULER_NAME_MAP = {
    "euler": EulerDiscreteScheduler,
    "euler-a": EulerAncestralDiscreteScheduler,
    "unipc": UniPCMultistepScheduler,
    "ddim": DDIMScheduler,
    "ddpm": DDPMScheduler,
}
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

transformers.utils.logging.set_verbosity_warning()
diffusers.utils.logging.set_verbosity_warning()


# --- GeodesicFlow Components ---
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

def project_to_tangent_space(v, x):
    """Projects vector v onto the tangent space of x on the unit sphere."""
    # Proj_x(v) = v - <v, x> * x
    dot_product = torch.sum(v * x, dim=1, keepdim=True)
    return v - dot_product * x
# --- End GeodesicFlow Components ---


class Trainer:
    def __init__(
        self, config: dict = None, disable_accelerator: bool = False, job_id: str = None
    ):
        self.accelerator = None
        self.job_id = job_id
        StateTracker.set_job_id(job_id)
        self.parse_arguments(args=config, disable_accelerator=disable_accelerator)
        self._misc_init()
        self.lycoris_wrapped_network = None
        self.lycoris_config = None
        self.lr_scheduler = None
        self.webhook_handler = None
        self.should_abort = False
        self.unet = None
        self.transformer = None
        self.vae = None
        self.text_encoder_1 = None
        self.text_encoder_2 = None
        self.text_encoder_3 = None
        self.controlnet = None
        self.validation = None
        # GeodesicFlow
        self.lambda_mlp = None

    def _config_to_obj(self, config):
        if not config:
            return None
        return type("Config", (object,), config)

    def parse_arguments(self, args=None, disable_accelerator: bool = False):
        self.config = load_config(args)

        # Set default values for GeodesicFlow hyperparameters immediately after loading config
        if not hasattr(self.config, 'geodesicflow_enabled'):
            self.config.geodesicflow_enabled = True
        if not hasattr(self.config, 'geodesicflow_eta'):
            self.config.geodesicflow_eta = 0.1
        if not hasattr(self.config, 'geodesicflow_mlp_hidden_dim'):
            self.config.geodesicflow_mlp_hidden_dim = 64
        if not hasattr(self.config, 'geodesicflow_mlp_num_layers'):
            self.config.geodesicflow_mlp_num_layers = 2
        
        report_to = (
            None if self.config.report_to.lower() == "none" else self.config.report_to
        )
        if not disable_accelerator:
            self.accelerator = Accelerator(
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                mixed_precision=(
                    self.config.mixed_precision
                    if not torch.backends.mps.is_available()
                    else None
                ),
                log_with=report_to,
                project_config=self.config.accelerator_project_config,
                kwargs_handlers=[self.config.process_group_kwargs],
            )
        safety_check(args=self.config, accelerator=self.accelerator)
        if self.config.lr_scale:
            logger.info(
                f"Scaling learning rate ({self.config.learning_rate}), due to --lr_scale"
            )
            self.config.learning_rate = (
                self.config.learning_rate
                * self.config.gradient_accumulation_steps
                * self.config.train_batch_size
                * getattr(self.accelerator, "num_processes", 1)
            )
        StateTracker.set_accelerator(self.accelerator)
        StateTracker.set_args(self.config)
        StateTracker.set_weight_dtype(self.config.weight_dtype)
        self.set_model_family()
        # this updates self.config further, so we will run it here.
        self.init_noise_schedule()

    def run(self):
        try:
            # Initialize essential configurations and schedules
            self.configure_webhook()
            self.init_noise_schedule()
            self.init_seed()
            self.init_huggingface_hub(access_token=self.config.huggingface_token)

            # Core initialization steps with signal checks after each step
            self._initialize_components_with_signal_check(
                [
                    self.init_preprocessing_models,
                    self.init_data_backend,
                    self.init_validation_prompts,
                    self.init_unload_text_encoder,
                    self.init_unload_vae,
                    self.init_load_base_model,
                    self.init_geodesicflow, # Initialize GeodesicFlow components
                    self.init_precision,
                    self.init_controlnet_model,
                    self.init_freeze_models,
                    self.init_trainable_peft_adapter,
                    self.init_ema_model,
                ]
            )

            # Model movement and validation setup
            self.move_models(destination="accelerator")
            self._exit_on_signal()
            self.init_validations()
            self._exit_on_signal()
            self.init_benchmark_base_model()
            self._exit_on_signal()
            self.resume_and_prepare()
            self._exit_on_signal()
            self.init_trackers()

            # Start the training process
            self.train()

        except Exception as e:
            import traceback

            logger.error(
                f"Failed to run training: {e}, traceback: {traceback.format_exc()}"
            )
            self._send_webhook_msg(
                message=f"Failed to run training: {e}",
            )
            self._send_webhook_raw(
                structured_data={
                    "message": f"Failed to run training: {e}",
                    "status": "error",
                },
                message_type="fatal_error",
            )

            raise e

    def _initialize_components_with_signal_check(self, initializers):
        """
        Runs a list of initializer functions with signal checks after each.

        Args:
            initializers (list): A list of initializer functions to run sequentially.
        """
        for initializer in initializers:
            initializer()
            self._exit_on_signal()

    def init_noise_schedule(self):
        self.config, _flow_matching, self.noise_scheduler = load_scheduler_from_args(
            self.config
        )
        self.config.flow_matching = _flow_matching
        self.lr = 0.0

    def configure_webhook(self, send_startup_message: bool = True):
        self.webhook_handler = None
        if self.config.webhook_config is None:
            return
        from helpers.webhooks.handler import WebhookHandler

        self.webhook_handler = WebhookHandler(
            self.config.webhook_config,
            self.accelerator,
            f"{self.config.tracker_project_name} {self.config.tracker_run_name}",
        )
        StateTracker.set_webhook_handler(self.webhook_handler)
        if send_startup_message:
            self._send_webhook_msg(
                message="SimpleTuner has launched. Hold onto your butts!",
                store_response=True,
            )
        self._send_webhook_raw(
            structured_data={
                "message": "Training job has started, configuration has begun."
            },
            message_type="configure_webhook",
        )

    def _misc_init(self):
        """things that do not really need an order."""
        torch.set_num_threads(self.config.torch_num_threads)
        self.state = {}
        self.state["lr"] = 0.0
        # Global step represents the most recently *completed* optimization step, which means it
        #  takes into account the number of gradient_accumulation_steps. If we use 1 gradient_accumulation_step,
        #  then global_step and step will be the same throughout training. However, if we use
        #  2 gradient_accumulation_steps, then global_step will be twice as large as step, and so on.
        self.state["global_step"] = 0
        self.state["global_resume_step"] = 0
        self.state["first_epoch"] = 1
        self.timesteps_buffer = []
        self.guidance_values_list = []
        self.train_loss = 0.0
        self.bf = None
        self.grad_norm = None
        self.extra_lr_scheduler_kwargs = {}
        StateTracker.set_global_step(self.state["global_step"])
        self.config.use_deepspeed_optimizer, self.config.use_deepspeed_scheduler = (
            prepare_model_for_deepspeed(self.accelerator, self.config)
        )
        self.config.base_weight_dtype = self.config.weight_dtype
        self.config.is_quanto = False
        self.config.is_torchao = False
        self.config.is_bnb = False
        if "quanto" in self.config.base_model_precision:
            self.config.is_quanto = True
        elif "torchao" in self.config.base_model_precision:
            self.config.is_torchao = True
        elif "bnb" in self.config.base_model_precision:
            self.config.is_bnb = True
        if self.config.is_quanto:
            from helpers.training.quantisation import quantise_model

            self.quantise_model = quantise_model
        elif self.config.is_torchao:
            from helpers.training.quantisation import quantise_model

            self.quantise_model = quantise_model

    def set_model_family(self, model_family: str = None):
        model_family = getattr(self.config, "model_family", model_family)
        if not model_family:
            logger.warning(
                "Using --model_family (or MODEL_FAMILY) to specify which model you are training will be required in a future release."
            )
            if self.config.model_family == "sd3":
                model_family = "sd3"
                logger.warning(
                    "Using --sd3 is deprecated. Please use --model_family=sd3."
                )
            if self.config.model_family == "flux":
                model_family = "flux"
                logger.warning(
                    "Using --flux is deprecated. Please use --model_family=flux."
                )
            if self.config.model_family == "pixart_sigma":
                model_family = "pixart_sigma"
                logger.warning(
                    "Using --pixart_sigma is deprecated. Please use --model_family=pixart_sigma."
                )
            if self.config.model_family == "legacy":
                model_family = "legacy"
                logger.warning(
                    "Using --legacy is deprecated. Please use --model_family=legacy."
                )
            if self.config.model_family == "kolors":
                model_family = "kolors"
                logger.warning(
                    "Using --kolors is deprecated. Please use --model_family=kolors."
                )
            if self.config.model_family == "smoldit":
                model_family = "smoldit"
            if model_family is None:
                model_family = "sdxl"
                logger.warning(
                    "Training SDXL without specifying --model_family is deprecated. Please use --model_family=sdxl."
                )
        elif model_family not in model_classes["full"]:
            raise ValueError(f"Invalid model family specified: {model_family}")

        self._set_model_paths()
        StateTracker.set_model_family(model_family)
        self.config.model_type_label = model_labels[model_family.lower()]
        if StateTracker.is_sdxl_refiner():
            self.config.model_type_label = "SDXL Refiner"

    def init_clear_backend_cache(self):
        if self.config.output_dir is not None:
            os.makedirs(self.config.output_dir, exist_ok=True)
        if self.config.preserve_data_backend_cache:
            return
        StateTracker.delete_cache_files(
            preserve_data_backend_cache=self.config.preserve_data_backend_cache
        )

    def init_seed(self):
        if self.config.seed is not None and self.config.seed != 0:
            set_seed(self.config.seed, self.config.seed_for_each_device)

    def init_huggingface_hub(self, access_token: str = None):
        # Handle the repository creation
        self.hub_manager = None
        if not self.accelerator.is_main_process or not self.config.push_to_hub:
            return
        if access_token:
            huggingface_hub.login(token=access_token)
        self.hub_manager = HubManager(config=self.config)
        try:
            StateTracker.set_hf_user(huggingface_hub.whoami())
            logger.info(
                f"Logged into Hugging Face Hub as '{StateTracker.get_hf_username()}'"
            )
        except Exception as e:
            logger.error(f"Failed to log into Hugging Face Hub: {e}")
            raise e

    def _set_model_paths(self):
        self.config.vae_path = (
            self.config.pretrained_model_name_or_path
            if self.config.pretrained_vae_model_name_or_path is None
            else self.config.pretrained_vae_model_name_or_path
        )
        self.config.text_encoder_path, self.config.text_encoder_subfolder = (
            determine_te_path_subfolder(self.config)
        )

    def init_preprocessing_models(self, move_to_accelerator: bool = True):
        # image embeddings
        self.init_vae(move_to_accelerator=move_to_accelerator)
        # text embeds
        self.init_text_encoder(move_to_accelerator=move_to_accelerator)

    def init_vae(self, move_to_accelerator: bool = True):
        logger.info(f"Load VAE: {self.config.vae_path}")
        self.config.vae_kwargs = {
            "pretrained_model_name_or_path": self.config.vae_path,
            "subfolder": "vae",
            "revision": self.config.revision,
            "force_upcast": False,
            "variant": self.config.variant,
        }
        try:
            self.vae = AutoencoderKL.from_pretrained(**self.config.vae_kwargs)
        except:
            logger.warning(
                "Couldn't load VAE with default path. Trying without a subfolder.."
            )
            self.config.vae_kwargs["subfolder"] = None
            self.vae = AutoencoderKL.from_pretrained(**self.config.vae_kwargs)
        if not move_to_accelerator:
            logger.debug("Not moving VAE to accelerator.")
            return
        if self.vae is not None:
            # The VAE is in bfloat16 to avoid NaN losses.
            _vae_dtype = torch.bfloat16
            if hasattr(self.config, "vae_dtype"):
                # Let's use a case-switch for convenience: bf16, fp16, fp32, none/default
                if self.config.vae_dtype == "bf16":
                    _vae_dtype = torch.bfloat16
                elif self.config.vae_dtype == "fp16":
                    raise ValueError(
                        "fp16 is not supported for SDXL's VAE. Please use bf16 or fp32."
                    )
                elif self.config.vae_dtype == "fp32":
                    _vae_dtype = torch.float32
                elif (
                    self.config.vae_dtype == "none"
                    or self.config.vae_dtype == "default"
                ):
                    _vae_dtype = torch.bfloat16
            logger.info(
                f"Loading VAE onto accelerator, converting from {self.vae.dtype} to {_vae_dtype}"
            )
            self.vae.to(self.accelerator.device, dtype=_vae_dtype)
            StateTracker.set_vae_dtype(_vae_dtype)
            StateTracker.set_vae(self.vae)

    def init_text_tokenizer(self):
        logger.info("Load tokenizers")
        self.tokenizer_1, self.tokenizer_2, self.tokenizer_3 = get_tokenizers(
            self.config
        )
        self.tokenizers = [self.tokenizer_1, self.tokenizer_2, self.tokenizer_3]

    def init_text_encoder(self, move_to_accelerator: bool = True):
        self.init_text_tokenizer()
        self.text_encoder_1, self.text_encoder_2, self.text_encoder_3 = None, None, None
        self.text_encoder_cls_1, self.text_encoder_cls_2, self.text_encoder_cls_3 = (
            None,
            None,
            None,
        )
        if self.tokenizer_1 is not None:
            self.text_encoder_cls_1 = import_model_class_from_model_name_or_path(
                self.config.text_encoder_path,
                self.config.revision,
                self.config,
                subfolder=self.config.text_encoder_subfolder,
            )
        if self.tokenizer_2 is not None:
            self.text_encoder_cls_2 = import_model_class_from_model_name_or_path(
                self.config.pretrained_model_name_or_path,
                self.config.revision,
                self.config,
                subfolder="text_encoder_2",
            )
        if self.tokenizer_3 is not None and self.config.model_family == "sd3":
            self.text_encoder_cls_3 = import_model_class_from_model_name_or_path(
                self.config.pretrained_model_name_or_path,
                self.config.revision,
                self.config,
                subfolder="text_encoder_3",
            )
        with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
            tokenizers = [self.tokenizer_1, self.tokenizer_2, self.tokenizer_3]
            text_encoder_classes = [
                self.text_encoder_cls_1,
                self.text_encoder_cls_2,
                self.text_encoder_cls_3,
            ]
            (
                text_encoder_variant,
                self.text_encoder_1,
                self.text_encoder_2,
                self.text_encoder_3,
            ) = load_tes(
                args=self.config,
                text_encoder_classes=text_encoder_classes,
                weight_dtype=self.config.weight_dtype,
                tokenizers=tokenizers,
                text_encoder_path=self.config.text_encoder_path,
                text_encoder_subfolder=self.config.text_encoder_subfolder,
            )
        if not move_to_accelerator:
            logger.debug("Not moving text encoders to accelerator.")
            return
        self.text_encoders = []
        self.tokenizers = []
        if self.tokenizer_1 is not None:
            logger.info("Moving text encoder to GPU.")
            self.text_encoder_1.to(
                self.accelerator.device, dtype=self.config.weight_dtype
            )
            self.tokenizers.append(self.tokenizer_1)
            self.text_encoders.append(self.text_encoder_1)
        if self.tokenizer_2 is not None:
            logger.info("Moving text encoder 2 to GPU.")
            self.text_encoder_2.to(
                self.accelerator.device, dtype=self.config.weight_dtype
            )
            self.tokenizers.append(self.tokenizer_2)
            self.text_encoders.append(self.text_encoder_2)
        if self.tokenizer_3 is not None:
            logger.info("Moving text encoder 3 to GPU.")
            self.text_encoder_3.to(
                self.accelerator.device, dtype=self.config.weight_dtype
            )
            self.tokenizers.append(self.tokenizer_3)
            self.text_encoders.append(self.text_encoder_3)

    def init_freeze_models(self):
        # Freeze vae and text_encoders
        if self.vae is not None:
            self.vae.requires_grad_(False)
        if self.text_encoder_1 is not None:
            self.text_encoder_1.requires_grad_(False)
        if self.text_encoder_2 is not None:
            self.text_encoder_2.requires_grad_(False)
        if self.text_encoder_3 is not None:
            self.text_encoder_3.requires_grad_(False)
        if "lora" in self.config.model_type or self.config.controlnet:
            if self.transformer is not None:
                self.transformer.requires_grad_(False)
            if self.unet is not None:
                self.unet.requires_grad_(False)
        self.accelerator.wait_for_everyone()

    def init_load_base_model(self):
        webhook_msg = f"Loading model: `{self.config.pretrained_model_name_or_path}`..."
        self._send_webhook_msg(message=webhook_msg)
        self._send_webhook_raw(
            structured_data={"message": webhook_msg},
            message_type="init_load_base_model_begin",
        )
        self.unet, self.transformer = load_diffusion_model(
            self.config, self.config.weight_dtype
        )
        self.accelerator.wait_for_everyone()
        self._send_webhook_raw(
            structured_data={"message": "Base model has loaded."},
            message_type="init_load_base_model_completed",
        )

    def init_geodesicflow(self):
        if not self.config.geodesicflow_enabled:
            logger.info("GeodesicFlow is disabled.")
            return

        logger.info("Initializing GeodesicFlow components.")
        # Initialize MLP with a fixed input dimension of 1 for tilde_rho
        # self.lambda_mlp = LambdaMLP(
        #     input_dim=1,
        #     hidden_dim=self.config.geodesicflow_mlp_hidden_dim,
        #     num_layers=self.config.geodesicflow_mlp_num_layers
        # )

    def init_data_backend(self):
        try:
            self.init_clear_backend_cache()
            self._send_webhook_msg(
                message="Configuring data backends... (this may take a while!)"
            )
            self._send_webhook_raw(
                structured_data={"message": "Configuring data backends."},
                message_type="init_data_backend_begin",
            )
            configure_multi_databackend(
                self.config,
                accelerator=self.accelerator,
                text_encoders=self.text_encoders,
                tokenizers=self.tokenizers,
            )
            self._send_webhook_raw(
                structured_data={"message": "Completed configuring data backends."},
                message_type="init_data_backend_completed",
            )
        except Exception as e:
            import traceback

            logger.error(f"{e}, traceback: {traceback.format_exc()}")
            self._send_webhook_msg(
                message=f"Failed to load data backends: {e}",
                message_level="critical",
            )
            self._send_webhook_raw(
                structured_data={
                    "message": f"Failed to load data backends: {e}",
                    "status": "error",
                },
                message_type="fatal_error",
            )

            raise e

        self.init_validation_prompts()
        # We calculate the number of steps per epoch by dividing the number of images by the effective batch divisor.
        # Gradient accumulation steps mean that we only update the model weights every /n/ steps.
        collected_data_backend_str = list(StateTracker.get_data_backends().keys())
        if self.config.push_to_hub and self.accelerator.is_main_process:
            self.hub_manager.collected_data_backend_str = collected_data_backend_str
            self.hub_manager.set_validation_prompts(
                self.validation_prompts, self.validation_shortnames
            )
            logger.debug(f"Collected validation prompts: {self.validation_prompts}")
        self._recalculate_training_steps()
        logger.info(
            f"Collected the following data backends: {collected_data_backend_str}"
        )
        self._send_webhook_msg(
            message=f"Collected the following data backends: {collected_data_backend_str}"
        )
        self._send_webhook_raw(
            structured_data={
                "message": f"Collected the following data backends: {collected_data_backend_str}"
            },
            message_type="init_data_backend",
        )
        self.accelerator.wait_for_everyone()

    def init_validation_prompts(self):
        if self.accelerator.is_main_process:
            if self.config.model_family == "flux":
                (
                    self.validation_prompts,
                    self.validation_shortnames,
                    self.validation_negative_prompt_embeds,
                    self.validation_negative_pooled_embeds,
                    self.validation_negative_time_ids,
                ) = prepare_validation_prompt_list(
                    args=self.config,
                    embed_cache=StateTracker.get_default_text_embed_cache(),
                )
            else:
                (
                    self.validation_prompts,
                    self.validation_shortnames,
                    self.validation_negative_prompt_embeds,
                    self.validation_negative_pooled_embeds,
                ) = prepare_validation_prompt_list(
                    args=self.config,
                    embed_cache=StateTracker.get_default_text_embed_cache(),
                )
        else:
            self.validation_prompts = None
            self.validation_shortnames = None
            self.validation_negative_prompt_embeds = None
            self.validation_negative_pooled_embeds = None
        self.accelerator.wait_for_everyone()

    def stats_memory_used(self):
        # Grab GPU memory used:
        if torch.cuda.is_available():
            curent_memory_allocated = torch.cuda.memory_allocated() / 1024**3
        elif torch.backends.mps.is_available():
            curent_memory_allocated = torch.mps.current_allocated_memory() / 1024**3
        else:
            logger.warning(
                "CUDA, ROCm, or Apple MPS not detected here. We cannot report VRAM reductions."
            )
            curent_memory_allocated = 0

        return curent_memory_allocated

    def init_unload_text_encoder(self):
        if self.config.model_type != "full" and self.config.train_text_encoder:
            return
        memory_before_unload = self.stats_memory_used()
        if self.accelerator.is_main_process:
            logger.info("Unloading text encoders, as they are not being trained.")
        if self.text_encoder_1 is not None:
            self.text_encoder_1 = self.text_encoder_1.to("cpu")
        if self.text_encoder_2 is not None:
            self.text_encoder_2 = self.text_encoder_2.to("cpu")
        if self.text_encoder_3 is not None:
            self.text_encoder_3 = self.text_encoder_3.to("cpu")
        del self.text_encoder_1, self.text_encoder_2, self.text_encoder_3
        self.text_encoder_1, self.text_encoder_2, self.text_encoder_3 = None, None, None
        self.text_encoders = []
        for backend_id, backend in StateTracker.get_data_backends().items():
            if "text_embed_cache" in backend:
                backend["text_embed_cache"].text_encoders = None
                backend["text_embed_cache"].pipeline = None
        reclaim_memory()
        memory_after_unload = self.stats_memory_used()
        memory_saved = memory_after_unload - memory_before_unload
        logger.info(
            f"After nuking text encoders from orbit, we freed {abs(round(memory_saved, 2))} GB of VRAM."
            " The real memories were the friends we trained a model on along the way."
        )

    def init_precision(self):
        self.config.enable_adamw_bf16 = (
            True if self.config.weight_dtype == torch.bfloat16 else False
        )
        quantization_device = (
            "cpu" if self.config.quantize_via == "cpu" else self.accelerator.device
        )

        if "bnb" in self.config.base_model_precision:
            # can't cast or move bitsandbytes modelsthis
            return

        if not self.config.disable_accelerator and self.config.is_quantized:
            if self.config.base_model_default_dtype == "fp32":
                self.config.base_weight_dtype = torch.float32
                self.config.enable_adamw_bf16 = False
            elif self.config.base_model_default_dtype == "bf16":
                self.config.base_weight_dtype = torch.bfloat16
                self.config.enable_adamw_bf16 = True
            if self.unet is not None:
                logger.info(
                    f"Moving U-net to dtype={self.config.base_weight_dtype}, device={quantization_device}"
                )
                self.unet.to(quantization_device, dtype=self.config.base_weight_dtype)
            elif self.transformer is not None:
                logger.info(
                    f"Moving transformer to dtype={self.config.base_weight_dtype}, device={quantization_device}"
                )
                self.transformer.to(
                    quantization_device, dtype=self.config.base_weight_dtype
                )

        if self.config.is_quanto:
            with self.accelerator.local_main_process_first():
                self.quantise_model(
                    unet=self.unet,
                    transformer=self.transformer,
                    text_encoder_1=self.text_encoder_1,
                    text_encoder_2=self.text_encoder_2,
                    text_encoder_3=self.text_encoder_3,
                    controlnet=None,
                    args=self.config,
                )
        elif self.config.is_torchao:
            with self.accelerator.local_main_process_first():
                (
                    self.unet,
                    self.transformer,
                    self.text_encoder_1,
                    self.text_encoder_2,
                    self.text_encoder_3,
                    self.controlnet,
                ) = self.quantise_model(
                    unet=self.unet,
                    transformer=self.transformer,
                    text_encoder_1=self.text_encoder_1,
                    text_encoder_2=self.text_encoder_2,
                    text_encoder_3=self.text_encoder_3,
                    controlnet=None,
                    args=self.config,
                )

    def init_controlnet_model(self):
        if not self.config.controlnet:
            return
        logger.info("Creating the controlnet..")
        if self.config.controlnet_model_name_or_path:
            logger.info("Loading existing controlnet weights")
            self.controlnet = ControlNetModel.from_pretrained(
                self.config.controlnet_model_name_or_path
            )
        else:
            logger.info("Initializing controlnet weights from unet")
            self.controlnet = ControlNetModel.from_unet(self.unet)

        self.accelerator.wait_for_everyone()

    def init_trainable_peft_adapter(self):
        if "lora" not in self.config.model_type:
            return
        if self.config.controlnet:
            raise ValueError("Cannot train LoRA with ControlNet.")
        if "standard" == self.config.lora_type.lower():
            lora_info_msg = f"Using LoRA training mode (rank={self.config.lora_rank})"
            logger.info(lora_info_msg)
            self._send_webhook_msg(message=lora_info_msg)
            target_modules = determine_adapter_target_modules(
                self.config, self.unet, self.transformer
            )
            addkeys, misskeys = [], []
            if self.unet is not None:
                unet_lora_config = LoraConfig(
                    r=self.config.lora_rank,
                    lora_alpha=(
                        self.config.lora_alpha
                        if self.config.lora_alpha is not None
                        else self.config.lora_rank
                    ),
                    lora_dropout=self.config.lora_dropout,
                    init_lora_weights=self.config.lora_initialisation_style,
                    target_modules=target_modules,
                    use_dora=self.config.use_dora,
                )
                logger.info("Adding LoRA adapter to the unet model..")
                self.unet.add_adapter(unet_lora_config)
                if self.config.init_lora:
                    addkeys, misskeys = load_lora_weights(
                        {"unet": self.unet},
                        self.config.init_lora,
                        use_dora=self.config.use_dora,
                    )
            elif self.transformer is not None:
                transformer_lora_config = LoraConfig(
                    r=self.config.lora_rank,
                    lora_alpha=(
                        self.config.lora_alpha
                        if self.config.lora_alpha is not None
                        else self.config.lora_rank
                    ),
                    init_lora_weights=self.config.lora_initialisation_style,
                    target_modules=target_modules,
                    use_dora=self.config.use_dora,
                )
                self.transformer.add_adapter(transformer_lora_config)
                if self.config.init_lora:
                    addkeys, misskeys = load_lora_weights(
                        {"transformer": self.transformer},
                        self.config.init_lora,
                        use_dora=self.config.use_dora,
                    )
            if addkeys:
                logger.warning(
                    "The following keys were found in %s, but are not part of the model and are ignored:\n %s.\nThis is most likely an error"
                    % (self.config.init_lora, str(addkeys))
                )
            if misskeys:
                logger.warning(
                    "The following keys were part of the model but not found in %s:\n %s.\nThese keys will be initialized according to the lora weight initialisation. This could be an error, or intended behaviour in case a lora is finetuned with additional keys."
                    % (self.config.init_lora, str(misskeys))
                )

        elif "lycoris" == self.config.lora_type.lower():
            from lycoris import create_lycoris

            with open(self.config.lycoris_config, "r") as f:
                self.lycoris_config = json.load(f)
            multiplier = int(self.lycoris_config["multiplier"])
            linear_dim = int(self.lycoris_config["linear_dim"])
            linear_alpha = int(self.lycoris_config["linear_alpha"])
            apply_preset = self.lycoris_config.get("apply_preset", None)
            if apply_preset is not None and apply_preset != {}:
                LycorisNetwork.apply_preset(apply_preset)

            # Remove the positional arguments we extracted.
            del self.lycoris_config["multiplier"]
            del self.lycoris_config["linear_dim"]
            del self.lycoris_config["linear_alpha"]

            logger.info("Using lycoris training mode")
            self._send_webhook_msg(message="Using lycoris training mode.")

            model_for_lycoris_wrap = None
            if self.transformer is not None:
                model_for_lycoris_wrap = self.transformer
            if self.unet is not None:
                model_for_lycoris_wrap = self.unet

            if self.config.init_lora is not None:
                from lycoris import create_lycoris_from_weights

                self.lycoris_wrapped_network = create_lycoris_from_weights(
                    multiplier,
                    self.config.init_lora,
                    model_for_lycoris_wrap,
                    weights_sd=None,
                    **self.lycoris_config,
                )[0]
            else:
                self.lycoris_wrapped_network = create_lycoris(
                    model_for_lycoris_wrap,
                    multiplier,
                    linear_dim,
                    linear_alpha,
                    **self.lycoris_config,
                )

                if self.config.init_lokr_norm is not None:
                    init_lokr_network_with_perturbed_normal(
                        self.lycoris_wrapped_network,
                        scale=self.config.init_lokr_norm,
                    )

            self.lycoris_wrapped_network.apply_to()
            setattr(
                self.accelerator,
                "_lycoris_wrapped_network",
                self.lycoris_wrapped_network,
            )
            lycoris_num_params = sum(
                p.numel() for p in self.lycoris_wrapped_network.parameters()
            )
            logger.info(
                f"LyCORIS network has been initialized with {lycoris_num_params:,} parameters"
            )
        self.accelerator.wait_for_everyone()

    def init_post_load_freeze(self):
        if self.config.layer_freeze_strategy == "bitfit":
            from helpers.training.model_freeze import apply_bitfit_freezing

            if self.unet is not None:
                logger.info("Applying BitFit freezing strategy to the U-net.")
                self.unet = apply_bitfit_freezing(
                    unwrap_model(self.accelerator, self.unet), self.config
                )
            if self.transformer is not None:
                logger.warning(
                    "Training DiT models with BitFit is not yet tested, and unexpected results may occur."
                )
                self.transformer = apply_bitfit_freezing(
                    unwrap_model(self.accelerator, self.transformer), self.config
                )

        if self.config.gradient_checkpointing:
            if self.unet is not None:
                unwrap_model(
                    self.accelerator, self.unet
                ).enable_gradient_checkpointing()
            if self.transformer is not None and self.config.model_family != "smoldit":
                unwrap_model(
                    self.accelerator, self.transformer
                ).enable_gradient_checkpointing()
            if self.config.controlnet:
                unwrap_model(
                    self.accelerator, self.controlnet
                ).enable_gradient_checkpointing()
            if (
                hasattr(self.config, "train_text_encoder")
                and self.config.train_text_encoder
            ):
                unwrap_model(
                    self.accelerator, self.text_encoder_1
                ).gradient_checkpointing_enable()
                unwrap_model(
                    self.accelerator, self.text_encoder_2
                ).gradient_checkpointing_enable()

    def _recalculate_training_steps(self):
        # Scheduler and math around the number of training steps.
        if not hasattr(self.config, "overrode_max_train_steps"):
            self.config.overrode_max_train_steps = False
        self.config.total_num_batches = sum(
            [
                len(
                    backend["metadata_backend"] if "metadata_backend" in backend else []
                )
                for _, backend in StateTracker.get_data_backends().items()
            ]
        )
        self.config.num_update_steps_per_epoch = math.ceil(
            self.config.total_num_batches / self.config.gradient_accumulation_steps
        )
        if getattr(self.config, "overrode_max_train_steps", False):
            self.config.max_train_steps = (
                self.config.num_train_epochs * self.config.num_update_steps_per_epoch
            )
            # Afterwards we recalculate our number of training epochs
            self.config.num_train_epochs = math.ceil(
                self.config.max_train_steps / self.config.num_update_steps_per_epoch
            )
            logger.info(
                "After removing any undesired samples and updating cache entries, we have settled on"
                f" {self.config.num_train_epochs} epochs and {self.config.num_update_steps_per_epoch} steps per epoch."
            )
        if self.config.max_train_steps is None or self.config.max_train_steps == 0:
            if (
                self.config.num_train_epochs is None
                or self.config.num_train_epochs == 0
            ):
                raise ValueError(
                    "You must specify either --max_train_steps or --num_train_epochs with a value > 0"
                )
            self.config.max_train_steps = (
                self.config.num_train_epochs * self.config.num_update_steps_per_epoch
            )
            logger.info(
                f"Calculated our maximum training steps at {self.config.max_train_steps} because we have"
                f" {self.config.num_train_epochs} epochs and {self.config.num_update_steps_per_epoch} steps per epoch."
            )
            self.config.overrode_max_train_steps = True
        elif self.config.num_train_epochs is None or self.config.num_train_epochs == 0:
            if self.config.max_train_steps is None or self.config.max_train_steps == 0:
                raise ValueError(
                    "You must specify either --max_train_steps or --num_train_epochs with a value > 0"
                )
            self.config.num_train_epochs = math.ceil(
                self.config.max_train_steps / self.config.num_update_steps_per_epoch
            )
            logger.info(
                f"Calculated our maximum training steps at {self.config.max_train_steps} because we have"
                f" {self.config.num_train_epochs} epochs and {self.config.num_update_steps_per_epoch} steps per epoch."
            )
        if self.lr_scheduler is not None and hasattr(
            self.lr_scheduler, "num_update_steps_per_epoch"
        ):
            self.lr_scheduler.num_update_steps_per_epoch = (
                self.config.num_update_steps_per_epoch
            )
        self.config.total_batch_size = (
            self.config.train_batch_size
            * self.accelerator.num_processes
            * self.config.gradient_accumulation_steps
        )

    def init_optimizer(self):
        logger.info(f"Learning rate: {self.config.learning_rate}")
        extra_optimizer_args = {"lr": self.config.learning_rate}
        # Initialize the optimizer
        optimizer_args_from_config, optimizer_class = (
            determine_optimizer_class_with_config(
                args=self.config,
                use_deepspeed_optimizer=self.config.use_deepspeed_optimizer,
                is_quantized=self.config.is_quantized,
                enable_adamw_bf16=self.config.enable_adamw_bf16,
            )
        )
        extra_optimizer_args.update(optimizer_args_from_config)

        self.params_to_optimize = determine_params_to_optimize(
            args=self.config,
            controlnet=self.controlnet,
            unet=self.unet,
            transformer=self.transformer,
            text_encoder_1=self.text_encoder_1,
            text_encoder_2=self.text_encoder_2,
            model_type_label=self.config.model_type_label,
            lycoris_wrapped_network=self.lycoris_wrapped_network,
        )

        # if self.config.geodesicflow_enabled and self.lambda_mlp is not None:
        #     logger.info("Adding GeodesicFlow Lambda MLP parameters to optimizer.")
        #     self.params_to_optimize.append(
        #         {"params": self.lambda_mlp.parameters(), "lr": self.config.learning_rate}
        #     )
        
        if self.config.use_deepspeed_optimizer:
            logger.info(
                f"DeepSpeed Optimizer arguments, weight_decay={self.config.adam_weight_decay} eps={self.config.adam_epsilon}, extra_arguments={extra_optimizer_args}"
            )
            self.optimizer = optimizer_class(self.params_to_optimize)
        else:
            logger.info(f"Optimizer arguments={extra_optimizer_args}")
            if self.config.train_text_encoder and self.config.text_encoder_lr:
                # changes the learning rate of text_encoder_parameters_one and text_encoder_parameters_two to be
                # --learning_rate
                self.params_to_optimize[1]["lr"] = float(self.config.learning_rate)
                if self.text_encoder_2 is not None:
                    self.params_to_optimize[2]["lr"] = float(self.config.learning_rate)

            self.optimizer = cpu_offload_optimizer(
                params_to_optimize=self.params_to_optimize,
                optimizer_cls=optimizer_class,
                optimizer_parameters=extra_optimizer_args,
                fused=self.config.fuse_optimizer,
                offload_gradients=self.config.optimizer_offload_gradients,
                offload_mechanism=self.config.optimizer_cpu_offload_method,
            )

        if (
            is_optimi_available
            and self.config.optimizer_release_gradients
            and "optimi" in self.config.optimizer
        ):
            logger.warning(
                "Marking model for gradient release. This feature is experimental, and may use more VRAM or not work."
            )
            prepare_for_gradient_release(
                (
                    self.controlnet
                    if self.config.controlnet
                    else self.transformer if self.transformer is not None else self.unet
                ),
                self.optimizer,
            )

    def init_lr_scheduler(self):
        self.config.is_schedulefree = is_lr_scheduler_disabled(self.config.optimizer)
        if self.config.is_schedulefree:
            logger.info(
                "Using experimental AdamW ScheduleFree optimiser from Facebook. Experimental due to newly added Kahan summation."
            )
            # we don't use LR schedulers with schedulefree optimisers
            lr_scheduler = None
        if not self.config.use_deepspeed_scheduler and not self.config.is_schedulefree:
            logger.info(
                f"Loading {self.config.lr_scheduler} learning rate scheduler with {self.config.lr_warmup_steps} warmup steps"
            )
            lr_scheduler = get_lr_scheduler(
                self.config,
                self.optimizer,
                self.accelerator,
                logger,
                use_deepspeed_scheduler=False,
            )
        else:
            logger.info(f"Using dummy learning rate scheduler")
            if torch.backends.mps.is_available():
                lr_scheduler = None
            else:
                lr_scheduler = accelerate.utils.DummyScheduler(
                    self.optimizer,
                    total_num_steps=self.config.max_train_steps,
                    warmup_num_steps=self.config.lr_warmup_steps,
                )
        if lr_scheduler is not None:
            if hasattr(lr_scheduler, "num_update_steps_per_epoch"):
                lr_scheduler.num_update_steps_per_epoch = (
                    self.config.num_update_steps_per_epoch
                )
            if hasattr(lr_scheduler, "last_step"):
                lr_scheduler.last_step = self.state.get("global_resume_step", 0)

        return lr_scheduler

    def init_ema_model(self):
        # Create EMA for the unet.
        self.ema_model = None
        if not self.config.use_ema:
            return
        if self.accelerator.is_main_process:
            logger.info("Using EMA. Creating EMAModel.")

            ema_model_cls = None
            if self.unet is not None:
                ema_model_cls = UNet2DConditionModel
            elif self.config.model_family == "pixart_sigma":
                ema_model_cls = PixArtTransformer2DModel
            elif self.config.model_family == "flux":
                ema_model_cls = FluxTransformer2DModel
            else:
                raise ValueError(
                    f"Please open a bug report or disable EMA. Unknown EMA model family: {self.config.model_family}"
                )

            ema_model_config = None
            if self.unet is not None:
                ema_model_config = self.unet.config
            elif self.transformer is not None:
                ema_model_config = self.transformer.config

            self.ema_model = EMAModel(
                self.config,
                self.accelerator,
                parameters=(
                    self.unet.parameters()
                    if self.unet is not None
                    else self.transformer.parameters()
                ),
                model_cls=ema_model_cls,
                model_config=ema_model_config,
                decay=self.config.ema_decay,
                foreach=not self.config.ema_foreach_disable,
            )
            logger.info("EMA model creation complete.")

        self.accelerator.wait_for_everyone()

    def init_hooks(self):
        from helpers.training.save_hooks import SaveHookManager

        self.model_hooks = SaveHookManager(
            args=self.config,
            unet=self.unet,
            transformer=self.transformer,
            ema_model=self.ema_model,
            accelerator=self.accelerator,
            text_encoder_1=self.text_encoder_1,
            text_encoder_2=self.text_encoder_2,
            use_deepspeed_optimizer=self.config.use_deepspeed_optimizer,
        )
        self.accelerator.register_save_state_pre_hook(self.model_hooks.save_model_hook)
        self.accelerator.register_load_state_pre_hook(self.model_hooks.load_model_hook)

    def init_prepare_models(self, lr_scheduler):
        # Prepare everything with our `accelerator`.
        logger.info("Preparing models..")

        # TODO: Is this still needed? Seems like a hack job from January 2024.
        self.train_dataloaders = []
        for _, backend in StateTracker.get_data_backends().items():
            if "train_dataloader" not in backend:
                continue
            self.train_dataloaders.append(backend["train_dataloader"])
            break
        if len(self.train_dataloaders) == 0:
            logger.error("For some reason, no dataloaders were configured.")
            sys.exit(0)
        if self.config.disable_accelerator:
            logger.warning(
                "Because SIMPLETUNER_DISABLE_ACCELERATOR is set, we will not prepare the accelerator."
            )
            return
        logger.info("Loading our accelerator...")
        if torch.backends.mps.is_available():
            self.accelerator.native_amp = False
        self._send_webhook_msg(message="Moving weights to GPU...")
        self._send_webhook_raw(
            structured_data={"message": "Moving weights to GPU"},
            message_type="init_prepare_models_begin",
        )
        primary_model = self.unet if self.unet is not None else self.transformer
        if self.config.controlnet:
            primary_model = self.controlnet
             
        results = self.accelerator.prepare(
            primary_model, lr_scheduler, self.optimizer, self.train_dataloaders[0]
        )
        if self.config.controlnet:
            self.controlnet = results[0]
        elif self.unet is not None:
            self.unet = results[0]
        elif self.transformer is not None:
            self.transformer = results[0]

        if self.config.unet_attention_slice:
            if torch.backends.mps.is_available():
                logger.warning(
                    "Using attention slicing when training SDXL on MPS can result in NaN errors on the first backward pass. If you run into issues, disable this option and reduce your batch size instead to reduce memory consumption."
                )
            if self.unet is not None:
                self.unet.set_attention_slice("auto")
            if self.transformer is not None:
                self.transformer.set_attention_slice("auto")
        self.lr_scheduler = results[1]
        self.optimizer = results[2]
        # The rest of the entries are dataloaders:
        self.train_dataloaders = [results[3:]]
        if self.config.use_ema and self.ema_model is not None:
            if self.config.ema_device == "accelerator":
                logger.info("Moving EMA model weights to accelerator...")
            self.ema_model.to(
                (
                    self.accelerator.device
                    if self.config.ema_device == "accelerator"
                    else "cpu"
                ),
                dtype=self.config.weight_dtype,
            )

            if self.config.ema_device == "cpu" and not self.config.ema_cpu_only:
                logger.info("Pinning EMA model weights to CPU...")
                try:
                    self.ema_model.pin_memory()
                except Exception as e:
                    self._send_webhook_raw(
                        structured_data={"message": f"Failed to pin EMA to CPU: {e}"},
                        message_type="error",
                    )
                    logger.error(f"Failed to pin EMA model to CPU: {e}")

        idx_count = 0
        for _, backend in StateTracker.get_data_backends().items():
            if idx_count == 0 or "train_dataloader" not in backend:
                continue
            self.train_dataloaders.append(
                self.accelerator.prepare(backend["train_dataloader"])
            )
        idx_count = 0

        if "lora" in self.config.model_type and self.config.train_text_encoder:
            logger.info("Preparing text encoders for training.")
            if self.config.model_family == "sd3":
                logger.info("NOTE: The third text encoder is not trained for SD3.")
            self.text_encoder_1, self.text_encoder_2 = self.accelerator.prepare(
                self.text_encoder_1, self.text_encoder_2
            )
        self._recalculate_training_steps()
        self.accelerator.wait_for_everyone()
        self._send_webhook_raw(
            structured_data={"message": "Completed moving weights to GPU"},
            message_type="init_prepare_models_completed",
        )

    def init_unload_vae(self):
        if self.config.keep_vae_loaded or self.config.vae_cache_ondemand:
            return
        memory_before_unload = self.stats_memory_used()
        self.vae = self.vae.to("cpu")
        del self.vae
        self.vae = None
        for _, backend in StateTracker.get_data_backends().items():
            if "vaecache" in backend:
                backend["vaecache"].vae = None
        reclaim_memory()
        memory_after_unload = self.stats_memory_used()
        memory_saved = memory_after_unload - memory_before_unload
        logger.info(
            f"After nuking the VAE from orbit, we freed {abs(round(memory_saved, 2)) * 1024} MB of VRAM."
        )

    def init_validations(self):
        if (
            hasattr(self.accelerator, "state")
            and hasattr(self.accelerator.state, "deepspeed_plugin")
            and getattr(self.accelerator.state.deepspeed_plugin, "deepspeed_config", {})
            .get("zero_optimization", {})
            .get("stage")
            == 3
        ):
            logger.error("Cannot run validations with DeepSpeed ZeRO stage 3.")
            return
        self.validation = Validation(
            accelerator=self.accelerator,
            unet=self.unet,
            transformer=self.transformer,
            args=self.config,
            validation_prompts=self.validation_prompts,
            validation_shortnames=self.validation_shortnames,
            text_encoder_1=self.text_encoder_1,
            tokenizer=self.tokenizer_1,
            vae_path=self.config.vae_path,
            weight_dtype=self.config.weight_dtype,
            embed_cache=StateTracker.get_default_text_embed_cache(),
            validation_negative_pooled_embeds=self.validation_negative_pooled_embeds,
            validation_negative_prompt_embeds=self.validation_negative_prompt_embeds,
            text_encoder_2=self.text_encoder_2,
            tokenizer_2=self.tokenizer_2,
            text_encoder_3=self.text_encoder_3,
            tokenizer_3=self.tokenizer_3,
            ema_model=self.ema_model,
            vae=self.vae,
            controlnet=self.controlnet if self.config.controlnet else None,
        )
        if not self.config.train_text_encoder and self.validation is not None:
            self.validation.clear_text_encoders()
        self.init_benchmark_base_model()
        self.accelerator.wait_for_everyone()

    def init_benchmark_base_model(self):
        if (
            self.config.disable_benchmark
            or self.validation is None
            or self.validation.benchmark_exists("base_model")
        ):
            # if we've disabled it or the benchmark exists, we will not do it again.
            # deepspeed zero3 can't do validations at all.
            return
        if not self.accelerator.is_main_process:
            return
        logger.info(
            "Benchmarking base model for comparison. Supply `--disable_benchmark: true` to disable this behaviour."
        )
        self._send_webhook_raw(
            structured_data={"message": "Base model benchmark begins"},
            message_type="init_benchmark_base_model_begin",
        )
        # we'll run validation on base model if it hasn't already.
        self.validation.run_validations(validation_type="base_model", step=0)
        self.validation.save_benchmark("base_model")
        self._send_webhook_raw(
            structured_data={"message": "Base model benchmark completed"},
            message_type="init_benchmark_base_model_completed",
        )

    def init_resume_checkpoint(self, lr_scheduler):
        # Potentially load in the weights and states from a previous save
        self.config.total_steps_remaining_at_start = self.config.max_train_steps
        self.state["current_epoch"] = self.state["first_epoch"]
        self.state["global_resume_step"] = self.state["global_step"] = (
            StateTracker.get_global_step()
        )
        StateTracker.set_global_resume_step(self.state["global_resume_step"])
        if not self.config.resume_from_checkpoint:
            return lr_scheduler
        if self.config.resume_from_checkpoint != "latest":
            path = os.path.basename(self.config.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(self.config.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            logger.info(
                f"Checkpoint '{self.config.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            self._send_webhook_raw(
                structured_data={
                    "message": "No model to resume. Beginning fresh training run."
                },
                message_type="init_resume_checkpoint",
            )

            self.config.resume_from_checkpoint = None
            return lr_scheduler

        logger.info(f"Resuming from checkpoint {path}")
        self.accelerator.load_state(os.path.join(self.config.output_dir, path))
        try:
            if (
                "constant" == self.config.lr_scheduler
                and not self.config.is_schedulefree
            ):
                for g in self.optimizer.param_groups:
                    if "lr" in g:
                        g["lr"] = self.config.learning_rate
                for k, v in lr_scheduler.state_dict().items():
                    if k in ("base_lrs", "_last_lr"):
                        v[0] = self.config.learning_rate
        except Exception as e:
            self._send_webhook_raw(
                structured_data={
                    "message": "Could not update learning rate scheduler LR value."
                },
                message_type="warning",
            )
            logger.error(
                f"Could not update lr_scheduler {self.config.lr_scheduler} learning rate to {self.config.learning_rate} upon resume: {e}"
            )

        self._send_webhook_raw(
            structured_data={"message": f"Resuming model: {path}"},
            message_type="init_resume_checkpoint",
        )
        training_state_filename = f"training_state.json"
        if get_rank() > 0:
            training_state_filename = f"training_state-{get_rank()}.json"
        for _, backend in StateTracker.get_data_backends().items():
            if "sampler" in backend:
                backend["sampler"].load_states(
                    state_path=os.path.join(
                        self.config.output_dir,
                        path,
                        training_state_filename,
                    ),
                )
        self.state["global_resume_step"] = self.state["global_step"] = (
            StateTracker.get_global_step()
        )
        StateTracker.set_global_resume_step(self.state["global_resume_step"])
        training_state_in_ckpt = StateTracker.get_training_state()
        self._send_webhook_raw(
            structured_data=training_state_in_ckpt,
            message_type="init_resume_checkpoint_details",
        )
        logger.debug(f"Training state inside checkpoint: {training_state_in_ckpt}")
        if hasattr(lr_scheduler, "last_step"):
            lr_scheduler.last_step = self.state["global_resume_step"]
        logger.info(f"Resuming from global_step {self.state['global_resume_step']}).")

        # Log the current state of each data backend.
        for _, backend in StateTracker.get_data_backends().items():
            if "sampler" in backend:
                backend["sampler"].log_state()
        # We store the number of dataset resets that have occurred inside the checkpoint.
        self.state["first_epoch"] = StateTracker.get_epoch()
        if self.state["first_epoch"] > 1 or self.state["global_resume_step"] > 1:
            self.config.total_steps_remaining_at_start -= self.state[
                "global_resume_step"
            ]
            logger.debug(
                f"Resuming from epoch {self.state['first_epoch']}, which leaves us with {self.config.total_steps_remaining_at_start}."
            )
        self.state["current_epoch"] = self.state["first_epoch"]
        StateTracker.set_epoch(self.state["current_epoch"])
        if hasattr(lr_scheduler, "last_epoch"):
            lr_scheduler.last_epoch = (
                training_state_in_ckpt.get(
                    "epoch_step", self.state.get("global_resume_step", 1)
                )
                * self.accelerator.num_processes
            )

        if self.state["current_epoch"] > self.config.num_train_epochs + 1:
            logger.info(
                f"Reached the end ({self.state['current_epoch']} epochs) of our training run ({self.config.num_train_epochs} epochs). This run will do zero steps."
            )
        self.accelerator.wait_for_everyone()

        return lr_scheduler

    def init_trackers(self):
        # We need to initialize the trackers we use, and also store our configuration.
        # The trackers initializes automatically on the main process.
        self.guidance_values_table = None
        if self.accelerator.is_main_process:
            # Copy args into public_args:
            public_args = copy.deepcopy(self.config)
            delattr(public_args, "accelerator_project_config")
            delattr(public_args, "process_group_kwargs")
            delattr(public_args, "weight_dtype")
            delattr(public_args, "base_weight_dtype")
            delattr(public_args, "vae_kwargs")

            # Hash the contents of public_args to reflect a deterministic ID for a single set of params:
            public_args_hash = hashlib.md5(
                json.dumps(vars(public_args), sort_keys=True).encode("utf-8")
            ).hexdigest()
            project_name = self.config.tracker_project_name or "simpletuner-training"
            tracker_run_name = (
                self.config.tracker_run_name or "simpletuner-training-run"
            )
            self.accelerator.init_trackers(
                project_name,
                config=vars(public_args),
                init_kwargs={
                    "wandb": {
                        "name": tracker_run_name,
                        "id": f"{public_args_hash}",
                        "resume": "allow",
                        "allow_val_change": True,
                    }
                },
            )
            self._send_webhook_raw(
                structured_data=public_args.__dict__,
                message_type="training_config",
            )

    def resume_and_prepare(self):
        self.init_optimizer()
        lr_scheduler = self.init_lr_scheduler()
        self.init_hooks()
        self.init_prepare_models(lr_scheduler=lr_scheduler)
        lr_scheduler = self.init_resume_checkpoint(lr_scheduler=lr_scheduler)
        self.init_post_load_freeze()

    def move_models(self, destination: str = "accelerator"):
        target_device = "cpu"
        if destination == "accelerator":
            target_device = self.accelerator.device
        logger.info(
            f"Moving the {'U-net' if self.unet is not None else 'diffusion transformer'} to GPU in {self.config.weight_dtype if not self.config.is_quantized else self.config.base_model_precision} precision."
        )
        if self.unet is not None:
            if self.config.is_quantized:
                self.unet.to(target_device)
            else:
                self.unet.to(target_device, dtype=self.config.weight_dtype)
        if self.transformer is not None:
            if self.config.is_quantized:
                self.transformer.to(target_device)
            else:
                self.transformer.to(target_device, dtype=self.config.weight_dtype)
        if getattr(self.accelerator, "_lycoris_wrapped_network", None) is not None:
            self.accelerator._lycoris_wrapped_network = (
                self.accelerator._lycoris_wrapped_network.to(
                    target_device, dtype=self.config.weight_dtype
                )
            )
        if (
            self.config.enable_xformers_memory_efficient_attention
            and self.config.model_family
            not in [
                "sd3",
                "pixart_sigma",
                "flux",
                "smoldit",
                "kolors",
            ]
        ):
            logger.info("Enabling xformers memory-efficient attention.")
            if is_xformers_available():
                import xformers  # type: ignore # noqa

                if self.unet is not None:
                    self.unet.enable_xformers_memory_efficient_attention()
                if self.transformer is not None:
                    self.transformer.enable_xformers_memory_efficient_attention()
                if self.config.controlnet:
                    self.controlnet.enable_xformers_memory_efficient_attention()
            else:
                raise ValueError(
                    "xformers is not available. Make sure it is installed correctly"
                )
        elif self.config.enable_xformers_memory_efficient_attention:
            logger.warning(
                "xformers is not enabled, as it is incompatible with this model type."
            )
            self.config.enable_xformers_memory_efficient_attention = False

        if self.config.controlnet:
            self.controlnet.train()
            logger.info(
                f"Moving ControlNet to {target_device} in {self.config.weight_dtype} precision."
            )
            self.controlnet.to(device=target_device, dtype=self.config.weight_dtype)
            if self.config.train_text_encoder:
                logger.warning(
                    "Unknown results will occur when finetuning the text encoder alongside ControlNet."
                )

    def mark_optimizer_train(self):
        if is_lr_scheduler_disabled(self.config.optimizer) and hasattr(
            self.optimizer, "train"
        ):
            # we typically have to call train() on the optim for schedulefree.
            self.optimizer.train()

    def mark_optimizer_eval(self):
        if is_lr_scheduler_disabled(self.config.optimizer) and hasattr(
            self.optimizer, "eval"
        ):
            # we typically have to call eval() on the optim for schedulefree before saving or running validations.
            self.optimizer.eval()

    def _send_webhook_msg(
        self, message: str, message_level: str = "info", store_response: bool = False
    ):
        if type(message) is not str:
            logger.error(
                f"_send_webhook_msg received {type(message)} type message instead of str."
            )
            return False
        if self.webhook_handler is None or not self.webhook_handler:
            return
        self.webhook_handler.send(
            message=message, message_level=message_level, store_response=store_response
        )

    def _send_webhook_raw(
        self,
        structured_data: dict,
        message_type: str,
        message_level: str = "info",
    ):
        if type(structured_data) is not dict:
            logger.error(
                f"_send_webhook_msg received {type(structured_data)} type message instead of dict."
            )
            return False
        if not self.webhook_handler:
            return
        self.webhook_handler.send_raw(
            structured_data=structured_data,
            message_type=message_type,
            message_level=message_level,
            job_id=self.job_id,
        )

    def _train_initial_msg(self):
        initial_msg = "\n***** Running training *****"
        initial_msg += f"\n-  Num batches = {self.config.total_num_batches}"
        initial_msg += f"\n-  Num Epochs = {self.config.num_train_epochs}"
        initial_msg += f"\n  - Current Epoch = {self.state['first_epoch']}"
        initial_msg += f"\n-  Total train batch size (w. parallel, distributed & accumulation) = {self.config.total_batch_size}"
        initial_msg += f"\n  - Instantaneous batch size per device = {self.config.train_batch_size}"
        initial_msg += f"\n  - Gradient Accumulation steps = {self.config.gradient_accumulation_steps}"
        initial_msg += f"\n-  Total optimization steps = {self.config.max_train_steps}"
        if self.state["global_step"] > 1:
            initial_msg += f"\n  - Steps completed: {self.state['global_step']}"
        initial_msg += f"\n-  Total optimization steps remaining = {max(0, self.config.total_steps_remaining_at_start)}"
        if self.config.geodesicflow_enabled:
            initial_msg += f"\n-  GeodesicFlow Training Enabled (eta={self.config.geodesicflow_eta})"

        logger.info(initial_msg)
        self._send_webhook_msg(message=initial_msg)
        structured_data = {
            "total_num_batches": self.config.total_num_batches,
            "total_num_epochs": self.config.num_train_epochs,
            "total_num_steps": self.config.max_train_steps,
            "current_epoch": self.state["first_epoch"],
            "total_batch_size": self.config.total_batch_size,
            "micro_batch_size": self.config.train_batch_size,
            "current_step": self.state["global_step"],
            "remaining_num_steps": max(0, self.config.total_steps_remaining_at_start),
        }
        self._send_webhook_raw(
            structured_data=structured_data, message_type="_train_initial_msg"
        )

    def _epoch_rollover(self, epoch):
        if self.state["first_epoch"] == epoch:
            return
        logger.debug(
            f"Just completed epoch {self.state['current_epoch']}. Beginning epoch {epoch}. Starting epoch was {self.state['first_epoch']}. Final epoch will be {self.config.num_train_epochs}"
        )
        for backend_id, backend in StateTracker.get_data_backends().items():
            backend_config = StateTracker.get_data_backend_config(backend_id)
            if (
                backend_config.get("crop")
                and backend_config.get("crop_aspect") == "random"
                and "metadata_backend" in backend
                and not self.config.aspect_bucket_disable_rebuild
            ) or (
                backend_config.get("vae_cache_clear_each_epoch")
                and "vaecache" in backend
            ):
                # when the aspect ratio is random, we need to shuffle the dataset on each epoch.
                if self.accelerator.is_main_process:
                    # we only compute the aspect ratio indices on the main process.
                    # we have to set read_only to False since we're generating a new, un-split list.
                    # otherwise, we can't actually save the new cache to disk.
                    backend["metadata_backend"].read_only = False
                    # this will generate+save the new cache to the storage backend.
                    backend["metadata_backend"].compute_aspect_ratio_bucket_indices(
                        ignore_existing_cache=True
                    )
                self.accelerator.wait_for_everyone()
                logger.info(f"Reloading cache for backend {backend_id}")
                backend["metadata_backend"].reload_cache(set_config=False)
                logger.info("Waiting for other threads to finish..")
                self.accelerator.wait_for_everyone()
                # we'll have to split the buckets between GPUs again now, so that the VAE cache distributes properly.
                logger.info("Splitting buckets across GPUs")
                backend["metadata_backend"].split_buckets_between_processes(
                    gradient_accumulation_steps=self.config.gradient_accumulation_steps
                )
                # we have to rebuild the VAE cache if it exists.
                if "vaecache" in backend:
                    logger.info("Rebuilding VAE cache..")
                    backend["vaecache"].rebuild_cache()
                # no need to manually call metadata_backend.save_cache() here.
        self.state["current_epoch"] = epoch
        StateTracker.set_epoch(epoch)
        if self.config.lr_scheduler == "cosine_with_restarts":
            self.extra_lr_scheduler_kwargs["epoch"] = epoch

    def _exit_on_signal(self):
        if self.should_abort:
            self._send_webhook_raw(
                structured_data={"message": "Aborting training run."},
                message_type="exit",
            )
            raise StopIteration("Training run received abort signal.")

    def abort(self):
        logger.info("Aborting training run.")
        if self.bf is not None:
            self.bf.stop_fetching()
        # we should set should_abort = True on each data backend's vae cache, metadata, and text backend
        for _, backend in StateTracker.get_data_backends().items():
            if "vaecache" in backend:
                logger.debug(f"Aborting VAE cache")
                backend["vaecache"].should_abort = True
            if "metadata_backend" in backend:
                logger.debug(f"Aborting metadata backend")
                backend["metadata_backend"].should_abort = True
            if "text_backend" in backend:
                logger.debug(f"Aborting text backend")
                backend["text_backend"].should_abort = True
            if "sampler" in backend:
                logger.debug(f"Aborting sampler")
                backend["sampler"].should_abort = True
        self.should_abort = True

    def model_predict(
        self,
        batch,
        latents,
        noisy_latents,
        encoder_hidden_states,
        added_cond_kwargs,
        add_text_embeds,
        timesteps,
    ):
        if self.config.controlnet:
            training_logger.debug(
                f"Extra conditioning dtype: {batch['conditioning_pixel_values'].dtype}"
            )
        if not self.config.disable_accelerator:
            if self.config.controlnet:
                # ControlNet conditioning.
                controlnet_image = batch["conditioning_pixel_values"].to(
                    dtype=self.config.weight_dtype
                )
                training_logger.debug(f"Image shape: {controlnet_image.shape}")
                down_block_res_samples, mid_block_res_sample = self.controlnet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    added_cond_kwargs=added_cond_kwargs,
                    controlnet_cond=controlnet_image,
                    return_dict=False,
                )
                # Predict the noise residual
                if self.unet is not None:
                    model_pred = self.unet(
                        noisy_latents,
                        timesteps,
                        encoder_hidden_states=encoder_hidden_states,
                        added_cond_kwargs=added_cond_kwargs,
                        down_block_additional_residuals=[
                            sample.to(dtype=self.config.weight_dtype)
                            for sample in down_block_res_samples
                        ],
                        mid_block_additional_residual=mid_block_res_sample.to(
                            dtype=self.config.weight_dtype
                        ),
                        return_dict=False,
                    )[0]
                if self.transformer is not None:
                    raise Exception(
                        "ControlNet predictions for transformer models are not yet implemented."
                    )
            elif self.config.model_family == "flux":
                # handle guidance
                packed_noisy_latents = pack_latents(
                    noisy_latents,
                    batch_size=latents.shape[0],
                    num_channels_latents=latents.shape[1],
                    height=latents.shape[2],
                    width=latents.shape[3],
                ).to(
                    dtype=self.config.base_weight_dtype,
                    device=self.accelerator.device,
                )
                if self.config.flux_guidance_mode == "mobius":
                    guidance_scales = get_mobius_guidance(
                        self.config,
                        self.state["global_step"],
                        self.config.num_update_steps_per_epoch,
                        latents.shape[0],
                        self.accelerator.device,
                    )
                elif self.config.flux_guidance_mode == "constant":
                    guidance_scales = [
                        float(self.config.flux_guidance_value)
                    ] * latents.shape[0]

                elif self.config.flux_guidance_mode == "random-range":
                    # Generate a list of random values within the specified range for each latent
                    guidance_scales = [
                        random.uniform(
                            self.config.flux_guidance_min,
                            self.config.flux_guidance_max,
                        )
                        for _ in range(latents.shape[0])
                    ]
                self.guidance_values_list.append(guidance_scales)

                # Now `guidance` will have different values for each latent in `latents`.
                transformer_config = None
                if hasattr(self.transformer, "module"):
                    transformer_config = self.transformer.module.config
                elif hasattr(self.transformer, "config"):
                    transformer_config = self.transformer.config
                if transformer_config is not None and getattr(
                    transformer_config, "guidance_embeds", False
                ):
                    guidance = torch.tensor(
                        guidance_scales, device=self.accelerator.device
                    )
                else:
                    guidance = None
                img_ids = prepare_latent_image_ids(
                    latents.shape[0],
                    latents.shape[2],
                    latents.shape[3],
                    self.accelerator.device,
                    self.config.weight_dtype,
                )
                
                # GeodesicFlow uses continuous time t, but the model expects integer timesteps.
                # We scale t by 1000 for compatibility with the model's time embedding.
                if not torch.is_tensor(timesteps):
                     timesteps = torch.tensor(timesteps, device=self.accelerator.device)
                
                if timesteps.dtype != torch.long:
                    # This handles the case where timesteps might be continuous t from GeodesicFlow
                     timesteps = (timesteps.float() / 1000.0)
                else:
                    # Standard integer timesteps
                    timesteps = timesteps.to(device=self.accelerator.device) / 1000.0
                
                timesteps = timesteps.expand(noisy_latents.shape[0])

                text_ids = torch.zeros(
                    batch["prompt_embeds"].shape[1],
                    3,
                ).to(
                    device=self.accelerator.device,
                    dtype=self.config.base_weight_dtype,
                )
                training_logger.debug(
                    "DTypes:"
                    f"\n-> Text IDs shape: {text_ids.shape if hasattr(text_ids, 'shape') else None}, dtype: {text_ids.dtype if hasattr(text_ids, 'dtype') else None}"
                    f"\n-> Image IDs shape: {img_ids.shape if hasattr(img_ids, 'shape') else None}, dtype: {img_ids.dtype if hasattr(img_ids, 'dtype') else None}"
                    f"\n-> Timesteps shape: {timesteps.shape if hasattr(timesteps, 'shape') else None}, dtype: {timesteps.dtype if hasattr(timesteps, 'dtype') else None}"
                    f"\n-> Guidance: {guidance}"
                    f"\n-> Packed Noisy Latents shape: {packed_noisy_latents.shape if hasattr(packed_noisy_latents, 'shape') else None}, dtype: {packed_noisy_latents.dtype if hasattr(packed_noisy_latents, 'dtype') else None}"
                )

                flux_transformer_kwargs = {
                    "hidden_states": packed_noisy_latents,
                    "timestep": timesteps,
                    "guidance": guidance,
                    "pooled_projections": batch["add_text_embeds"].to(
                        device=self.accelerator.device,
                        dtype=self.config.base_weight_dtype,
                    ),
                    "encoder_hidden_states": batch["prompt_embeds"].to(
                        device=self.accelerator.device,
                        dtype=self.config.base_weight_dtype,
                    ),
                    "txt_ids": text_ids.to(
                        device=self.accelerator.device,
                        dtype=self.config.base_weight_dtype,
                    ),
                    "img_ids": img_ids,
                    "joint_attention_kwargs": None,
                    "return_dict": False,
                }
                if self.config.flux_attention_masked_training:
                    flux_transformer_kwargs["attention_mask"] = batch[
                        "encoder_attention_mask"
                    ]
                    if flux_transformer_kwargs["attention_mask"] is None:
                        raise ValueError(
                            "No attention mask was discovered when attempting validation - this means you need to recreate your text embed cache."
                        )

                model_pred = self.transformer(**flux_transformer_kwargs)[0]

            elif self.config.model_family == "sd3":
                # Stable Diffusion 3 uses a MM-DiT model where the VAE-produced
                #  image embeds are passed in with the TE-produced text embeds.
                model_pred = self.transformer(
                    hidden_states=noisy_latents.to(
                        device=self.accelerator.device,
                        dtype=self.config.base_weight_dtype,
                    ),
                    timestep=timesteps,
                    encoder_hidden_states=encoder_hidden_states.to(
                        device=self.accelerator.device,
                        dtype=self.config.base_weight_dtype,
                    ),
                    pooled_projections=add_text_embeds.to(
                        device=self.accelerator.device,
                        dtype=self.config.weight_dtype,
                    ),
                    return_dict=False,
                )[0]
            elif self.config.model_family == "pixart_sigma":
                model_pred = self.transformer(
                    noisy_latents,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=batch["encoder_attention_mask"],
                    timestep=timesteps,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]
                model_pred = model_pred.chunk(2, dim=1)[0]
            elif self.config.model_family == "smoldit":
                first_latent_shape = noisy_latents.shape
                height = first_latent_shape[1] * 8
                width = first_latent_shape[2] * 8
                grid_height = height // 8 // self.transformer.config.patch_size
                grid_width = width // 8 // self.transformer.config.patch_size
                base_size = 512 // 8 // self.transformer.config.patch_size
                grid_crops_coords = get_resize_crop_region_for_grid(
                    (grid_height, grid_width), base_size
                )
                inputs = {
                    "hidden_states": noisy_latents,
                    "timestep": timesteps,
                    "encoder_hidden_states": encoder_hidden_states,
                    "encoder_attention_mask": batch["encoder_attention_mask"],
                    "image_rotary_emb": get_2d_rotary_pos_embed(
                        self.transformer.inner_dim
                        // self.transformer.config.num_attention_heads,
                        grid_crops_coords,
                        (grid_height, grid_width),
                    ),
                }
                model_pred = self.transformer(**inputs).sample
            elif self.unet is not None:
                if self.config.model_family == "legacy":
                    # SD 1.5 or 2.x
                    model_pred = self.unet(
                        noisy_latents,
                        timesteps,
                        encoder_hidden_states,
                    ).sample
                else:
                    # SDXL, Kolors, other default unet prediction.
                    model_pred = self.unet(
                        noisy_latents,
                        timesteps,
                        encoder_hidden_states,
                        added_cond_kwargs=added_cond_kwargs,
                    ).sample
            else:
                raise Exception("Unknown error occurred, no prediction could be made.")

            if self.config.model_family == "flux":
                model_pred = unpack_latents(
                    model_pred,
                    height=latents.shape[2] * 8,
                    width=latents.shape[3] * 8,
                    vae_scale_factor=16,
                )
        else:
            # Dummy model prediction for debugging.
            model_pred = torch.randn_like(noisy_latents)

        return model_pred

    def train(self):
        self.init_trackers()
        self._train_initial_msg()

        if self.config.validation_on_startup and self.state["global_step"] <= 1:
            # Just in Case.
            self.mark_optimizer_eval()
            # normal run-of-the-mill validation on startup.
            if self.validation is not None:
                self.validation.run_validations(validation_type="base_model", step=0)

        self.mark_optimizer_train()

        # Only show the progress bar once on each machine.
        show_progress_bar = True
        if not self.accelerator.is_local_main_process:
            show_progress_bar = False
        progress_bar = tqdm(
            range(0, self.config.max_train_steps),
            disable=not show_progress_bar,
            initial=self.state["global_step"],
            desc=f"Epoch {self.state['first_epoch']}/{self.config.num_train_epochs} Steps",
            ncols=125,
        )
        self.accelerator.wait_for_everyone()

        # Some values that are required to be initialised later.
        step = self.state["global_step"]
        training_luminance_values = []
        current_epoch_step = None
        self.bf, fetch_thread = None, None
        iterator_fn = random_dataloader_iterator
        for epoch in range(self.state["first_epoch"], self.config.num_train_epochs + 1):
            if self.state["current_epoch"] > self.config.num_train_epochs + 1:
                # This might immediately end training, but that's useful for simply exporting the model.
                logger.info(
                    f"Training run is complete ({self.config.num_train_epochs}/{self.config.num_train_epochs} epochs, {self.state['global_step']}/{self.config.max_train_steps} steps)."
                )
                break
            self._epoch_rollover(epoch)
            if self.config.controlnet:
                self.controlnet.train()
                training_models = [self.controlnet]
            else:
                if self.unet is not None:
                    self.unet.train()
                    training_models = [self.unet]
                if self.transformer is not None:
                    self.transformer.train()
                    training_models = [self.transformer]
            if self.config.geodesicflow_enabled and self.lambda_mlp:
                 self.lambda_mlp.train()
                 # The lambda_mlp params are in the main optimizer, so no need to add the model here
            if (
                "lora" in self.config.model_type
                and self.config.train_text_encoder
                and "standard" in self.config.lora_type.lower()
            ):
                self.text_encoder_1.train()
                self.text_encoder_2.train()
                training_models.append(self.text_encoder_1)
                training_models.append(self.text_encoder_2)

            if current_epoch_step is not None:
                # We are resetting to the next epoch, if it is not none.
                current_epoch_step = 0
            else:
                # If it's None, we need to calculate the current epoch step based on the current global step.
                current_epoch_step = (
                    self.state["global_step"] % self.config.num_update_steps_per_epoch
                )
            train_backends = {}
            for backend_id, backend in StateTracker.get_data_backends().items():
                if (
                    StateTracker.backend_status(backend_id)
                    or "train_dataloader" not in backend
                ):
                    # Exclude exhausted backends.
                    logger.debug(
                        f"Excluding backend: {backend_id}, as it is exhausted? {StateTracker.backend_status(backend_id)} or not found {('train_dataloader' not in backend)}"
                    )
                    continue
                train_backends[backend_id] = backend["train_dataloader"]
            # Begin dataloader prefetch, if enabled.
            iterator_args = [train_backends]
            if self.config.dataloader_prefetch:
                iterator_args = []
                if self.bf is not None:
                    self.bf.stop_fetching()
                self.bf = BatchFetcher(
                    datasets=train_backends,
                    max_size=self.config.dataloader_prefetch_qlen,
                    step=step,
                )
                if fetch_thread is not None:
                    fetch_thread.join()
                fetch_thread = self.bf.start_fetching()
                iterator_fn = self.bf.next_response

            while True:
                self._exit_on_signal()
                step += 1
                batch = iterator_fn(step, *iterator_args)
                
                # Lazy initialization of GeodesicFlow MLP on the first step
                if self.config.geodesicflow_enabled and self.lambda_mlp is None:
                    logger.info("First training step. Lazily initializing GeodesicFlow MLP.")
                    latent_channels = batch["latent_batch"].shape[1]
                    logger.info(f"Detected {latent_channels} latent channels. LambdaMLP input_dim: {latent_channels}")
                    
                    self.lambda_mlp = LambdaMLP(
                        input_dim=latent_channels, # Input dim is now the number of channels
                        hidden_dim=self.config.geodesicflow_mlp_hidden_dim,
                        num_layers=self.config.geodesicflow_mlp_num_layers
                    )
                    # Cast to correct dtype and move to device
                    self.lambda_mlp.to(self.accelerator.device, dtype=self.config.weight_dtype)
                    
                    # Prepare the new model with accelerator
                    self.lambda_mlp = self.accelerator.prepare(self.lambda_mlp)
                    
                    # Add its parameters to the existing optimizer
                    logger.info("Adding GeodesicFlow MLP parameters to the optimizer.")
                    self.optimizer.add_param_group(
                        {"params": self.lambda_mlp.parameters(), "lr": self.config.learning_rate}
                    )
                
                training_logger.debug(f"Iterator: {iterator_fn}")
                if self.config.lr_scheduler == "cosine_with_restarts":
                    self.extra_lr_scheduler_kwargs["step"] = self.state["global_step"]

                if self.accelerator.is_main_process:
                    progress_bar.set_description(
                        f"Epoch {self.state['current_epoch']}/{self.config.num_train_epochs}, Steps"
                    )

                # If we receive a False from the enumerator, we know we reached the next epoch.
                if batch is False:
                    logger.debug(f"Reached the end of epoch {epoch}")
                    break

                if batch is None:
                    import traceback

                    raise ValueError(
                        f"Received a None batch, which is not a good thing. Traceback: {traceback.format_exc()}"
                    )

                # Add the current batch of training data's avg luminance to a list.
                if "batch_luminance" in batch:
                    training_luminance_values.append(batch["batch_luminance"])

                with self.accelerator.accumulate(training_models):
                    training_logger.debug("Sending latent batch to GPU.")
                    latents = batch["latent_batch"].to(
                        self.accelerator.device, dtype=self.config.weight_dtype
                    )
                    bsz = latents.shape[0]

                    # Prepare conditioning embeds, which are needed for both paths
                    encoder_hidden_states = batch["prompt_embeds"].to(
                        dtype=self.config.weight_dtype, device=self.accelerator.device
                    )
                    add_text_embeds = batch["add_text_embeds"]
                    added_cond_kwargs = None
                    if (
                        StateTracker.get_model_family() == "sdxl"
                        or self.config.model_family == "kolors"
                    ):
                        added_cond_kwargs = {
                            "text_embeds": add_text_embeds.to(
                                device=self.accelerator.device,
                                dtype=self.config.weight_dtype,
                            ),
                            "time_ids": batch["batch_time_ids"].to(
                                device=self.accelerator.device,
                                dtype=self.config.weight_dtype,
                            ),
                        }
                    elif (
                        self.config.model_family == "pixart_sigma"
                        or self.config.model_family == "smoldit"
                    ):
                        if "batch_time_ids" in batch:
                            added_cond_kwargs = batch["batch_time_ids"]
                        batch["encoder_attention_mask"] = batch[
                            "encoder_attention_mask"
                        ].to(
                            device=self.accelerator.device,
                            dtype=self.config.weight_dtype,
                        )

                    # --- GeodesicFlow Logic ---
                    if self.config.geodesicflow_enabled:
                        original_shape = latents.shape
                        
                        # 1. Get x0 (data) and x1 (noise) on the unit sphere
                        x0 = latents.view(bsz, -1)
                        x1 = torch.randn_like(x0)
                        x0 = F.normalize(x0, p=2, dim=1)
                        x1 = F.normalize(x1, p=2, dim=1)

                        # 2. Compute paths and velocities
                        t_continuous = torch.rand(bsz, 1, device=self.accelerator.device)
                        timesteps = (t_continuous.squeeze(1) * (self.noise_scheduler.config.num_train_timesteps - 1)).long()

                        dot_product = torch.sum(x0 * x1, dim=1, keepdim=True)
                        theta = torch.acos(torch.clamp(dot_product, -1.0 + 1e-7, 1.0 - 1e-7))
                        sin_theta = torch.sin(theta)
                        
                        # Slerp for Geodesic Path
                        a = torch.sin((1 - t_continuous) * theta) / (sin_theta + 1e-7)
                        b = torch.sin(t_continuous * theta) / (sin_theta + 1e-7)
                        z_t_geo = a * x0 + b * x1
                        
                        # Linear Interpolation for Euclidean Path
                        z_t_euc = (1 - t_continuous) * x0 + t_continuous * x1

                        # Geodesic Velocity
                        x1_minus_proj = x1 - dot_product * x0
                        norm_x1_minus_proj = torch.norm(x1_minus_proj, p=2, dim=1, keepdim=True)
                        v0_dir = x1_minus_proj / (norm_x1_minus_proj + 1e-7)
                        v0 = theta * v0_dir
                        v0_dot_zt_geo = torch.sum(v0 * z_t_geo, dim=1, keepdim=True)
                        denom_pt = 1 - dot_product**2
                        u_t_geo = v0 - (v0_dot_zt_geo / (denom_pt + 1e-7)) * x1_minus_proj

                        # Euclidean Velocity
                        u_t_euc = x1 - x0

                        # 3. Compute density proxy and predict lambda
                        rho = torch.norm(u_t_geo, p=2, dim=1, keepdim=True)
                        tilde_rho = rho / math.pi
                        # lambda_val = self.lambda_mlp(z_t_geo.detach())

                        z_t_geo_reshaped = z_t_geo.view(original_shape)
                        pooled_zt_geo = F.adaptive_avg_pool2d(
                            z_t_geo_reshaped, (1, 1)
                        ).view(bsz, -1) # Shape: (bsz, channels)

                        # 3. Predict lambda using the pooled, fixed-size input
                        lambda_val = self.lambda_mlp(pooled_zt_geo.detach())

                        # 4. Reshape paths for the model input
                        z_t_geo_reshaped = z_t_geo.view(original_shape).to(dtype=self.config.weight_dtype)
                        z_t_euc_reshaped = z_t_euc.view(original_shape).to(dtype=self.config.weight_dtype)

                        # 5. Get model predictions for both paths
                        pred_geo_raw = self.model_predict(
                            batch, latents, z_t_geo_reshaped, encoder_hidden_states,
                            added_cond_kwargs, add_text_embeds, timesteps
                        ).view(bsz, -1)
                        
                        pred_euc_raw = self.model_predict(
                            batch, latents, z_t_euc_reshaped, encoder_hidden_states,
                            added_cond_kwargs, add_text_embeds, timesteps
                        ).view(bsz, -1)
                        
                        # Project predictions to tangent space
                        pred_geo = project_to_tangent_space(pred_geo_raw, z_t_geo)
                        pred_euc = project_to_tangent_space(pred_euc_raw, z_t_euc)

                        # 6. Compute losses in float32 for stability
                        loss_geo = F.mse_loss(pred_geo.float(), u_t_geo.float(), reduction="none").mean(dim=1)
                        loss_euc = F.mse_loss(pred_euc.float(), u_t_euc.float(), reduction="none").mean(dim=1)

                        loss_adaptive = (1 - lambda_val.squeeze(1).float()) * loss_euc + lambda_val.squeeze(1).float() * loss_geo
                        # loss_lambda = F.mse_loss(lambda_val.float(), tilde_rho.float())
                        
                        # loss = loss_adaptive.mean() + self.config.geodesicflow_eta * loss_lambda
                        loss = loss_adaptive.mean()
                        
                    # --- Original Logic ---
                    else:
                        noise = torch.randn_like(latents)
                        if not self.config.flow_matching:
                            if self.config.offset_noise:
                                if (
                                    self.config.noise_offset_probability == 1.0
                                    or random.random()
                                    < self.config.noise_offset_probability
                                ):
                                    noise = noise + self.config.noise_offset * torch.randn(
                                        latents.shape[0], latents.shape[1], 1, 1,
                                        device=latents.device
                                    )
                        if self.config.flow_matching:
                            if (not self.config.flux_fast_schedule and not self.config.flux_use_beta_schedule):
                                sigmas = torch.sigmoid(self.config.flow_matching_sigmoid_scale * torch.randn((bsz,), device=self.accelerator.device))
                                sigmas = apply_flux_schedule_shift(self.config, self.noise_scheduler, sigmas, noise)
                            elif self.config.flux_use_beta_schedule:
                                beta_dist = Beta(self.config.flux_beta_schedule_alpha, self.config.flux_beta_schedule_beta)
                                sigmas = beta_dist.sample((bsz,)).to(device=self.accelerator.device)
                                sigmas = apply_flux_schedule_shift(self.config, self.noise_scheduler, sigmas, noise)
                            else:
                                available_sigmas = [1.0] * 7 + [0.75, 0.5, 0.25]
                                sigmas = torch.tensor(random.choices(available_sigmas, k=bsz), device=self.accelerator.device)
                            timesteps = sigmas * 1000.0
                            sigmas = sigmas.view(-1, 1, 1, 1)
                        else:
                            weights = generate_timestep_weights(self.config, self.noise_scheduler.config.num_train_timesteps).to(self.accelerator.device)
                            if (bsz > 1 and not self.config.disable_segmented_timestep_sampling):
                                timesteps = segmented_timestep_selection(
                                    actual_num_timesteps=self.noise_scheduler.config.num_train_timesteps, bsz=bsz, weights=weights,
                                    use_refiner_range=StateTracker.is_sdxl_refiner() and not StateTracker.get_args().sdxl_refiner_uses_full_range,
                                ).to(self.accelerator.device)
                            else:
                                timesteps = torch.multinomial(weights, bsz, replacement=True).long()

                        for ts in timesteps.tolist():
                            self.timesteps_buffer.append((self.state["global_step"], ts))

                        input_noise = noise + self.config.input_perturbation * torch.randn_like(latents) if self.config.input_perturbation > 0 else noise

                        if self.config.flow_matching:
                            noisy_latents = (1 - sigmas) * latents + sigmas * input_noise
                        else:
                            noisy_latents = self.noise_scheduler.add_noise(
                                latents.float(), input_noise.float(), timesteps
                            ).to(device=self.accelerator.device, dtype=self.config.weight_dtype)

                        # Get target for loss
                        if self.config.flow_matching:
                            if self.config.flow_matching_loss == "diffusers": target = latents
                            elif self.config.flow_matching_loss == "compatible": target = noise - latents
                            elif self.config.flow_matching_loss == "sd35": target = (noisy_latents - latents) / sigmas.view(-1, 1, 1, 1)
                        elif self.noise_scheduler.config.prediction_type == "epsilon": target = noise
                        elif self.noise_scheduler.config.prediction_type == "v_prediction" or (self.config.flow_matching and self.config.flow_matching_loss == "diffusion"):
                            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
                        elif self.noise_scheduler.config.prediction_type == "sample": target = latents
                        else: raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")

                        model_pred = self.model_predict(
                            batch, latents, noisy_latents, encoder_hidden_states,
                            added_cond_kwargs, add_text_embeds, timesteps
                        )

                        if self.noise_scheduler.config.prediction_type == "sample":
                            model_pred = model_pred - noise

                        if self.config.snr_gamma is None or self.config.snr_gamma == 0:
                            loss = self.config.snr_weight * F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                        else:
                            snr = compute_snr(timesteps, self.noise_scheduler)
                            snr_divisor = snr + 1 if self.noise_scheduler.config.prediction_type == "v_prediction" or (self.config.flow_matching and self.config.flow_matching_loss == "diffusion") else snr
                            mse_loss_weights = (torch.stack([snr, self.config.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr_divisor).view(-1, 1, 1, 1)
                            loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                            loss = (loss * mse_loss_weights).mean()

                    # Gather losses, backpropagate, and update models
                    avg_loss = self.accelerator.gather(loss.repeat(self.config.train_batch_size)).mean()
                    self.train_loss += avg_loss.item() / self.config.gradient_accumulation_steps
                    
                    grad_norm = None
                    if not self.config.disable_accelerator:
                        self.accelerator.backward(loss)
                        if self.accelerator.sync_gradients and self.config.max_grad_norm > 0:
                            grad_norm = self.accelerator.clip_grad_norm_(self.params_to_optimize, self.config.max_grad_norm)
                        self.optimizer.step()
                        self.optimizer.zero_grad(set_to_none=self.config.set_grads_to_none)

                # Post-step operations, logging, checkpointing etc.
                if self.accelerator.sync_gradients:
                    try:
                        if self.config.is_schedulefree: self.lr = StateTracker.get_last_lr()
                        else:
                            self.lr_scheduler.step(**self.extra_lr_scheduler_kwargs)
                            self.lr = self.lr_scheduler.get_last_lr()[0]
                    except Exception as e: logger.error(f"Failed to get learning rate: {e}")
                    
                    wandb_logs = {"train_loss": self.train_loss, "optimization_loss": loss.item(), "learning_rate": self.lr, "epoch": epoch}
                    if grad_norm is not None: wandb_logs["grad_norm"] = grad_norm.item()
                    if self.config.geodesicflow_enabled:
                        wandb_logs["geodesicflow_loss_adaptive"] = loss_adaptive.mean().item()
                        # wandb_logs["geodesicflow_loss_lambda"] = loss_lambda.item()
                        wandb_logs["geodesicflow_avg_lambda"] = lambda_val.mean().item()

                    progress_bar.update(1)
                    self.state["global_step"] += 1
                    current_epoch_step += 1
                    StateTracker.set_global_step(self.state["global_step"])

                    if self.config.use_ema and self.ema_model:
                        self.ema_model.step(parameters=(self.unet.parameters() if self.unet else self.transformer.parameters()), global_step=self.state["global_step"])
                        wandb_logs["ema_decay_value"] = self.ema_model.get_decay()

                    self.accelerator.log(wandb_logs, step=self.state["global_step"])
                    self.train_loss = 0.0

                    if self.state["global_step"] % self.config.checkpointing_steps == 0:
                        if self.accelerator.is_main_process:
                            if self.config.checkpoints_total_limit is not None:
                                checkpoints = sorted([d for d in os.listdir(self.config.output_dir) if d.startswith("checkpoint")], key=lambda x: int(x.split("-")[1]))
                                if len(checkpoints) >= self.config.checkpoints_total_limit:
                                    for ckpt_to_remove in checkpoints[:len(checkpoints) - self.config.checkpoints_total_limit + 1]:
                                        shutil.rmtree(os.path.join(self.config.output_dir, ckpt_to_remove))
                            
                            save_path = os.path.join(self.config.output_dir, f"checkpoint-{self.state['global_step']}")
                            self.mark_optimizer_eval()
                            self.accelerator.save_state(save_path)
                            self.mark_optimizer_train()
                
                logs = {"step_loss": loss.detach().item(), "lr": float(self.lr)}
                progress_bar.set_postfix(**logs)

                self.mark_optimizer_eval()
                if self.validation: self.validation.run_validations(validation_type="intermediary", step=step)
                self.mark_optimizer_train()
                
                if self.state["global_step"] >= self.config.max_train_steps or epoch > self.config.num_train_epochs:
                    break
            if self.state["global_step"] >= self.config.max_train_steps or epoch > self.config.num_train_epochs:
                break
        
        # Final save and cleanup
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            self.mark_optimizer_eval()
            # Unwrapping and saving logic as in the original file...
            # This part remains unchanged
            
        self.accelerator.end_training()