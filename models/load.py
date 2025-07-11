"""
load.py

Entry point for loading pretrained VLMs for inference; exposes functions for listing available models (with canonical
IDs, mappings to paper experiments, and short descriptions), as well as for loading models (from disk or HF Hub).
"""

import json
import os
from pathlib import Path
from typing import List, Optional, Union

from huggingface_hub import HfFileSystem, hf_hub_download
import torch
from conf import ModelConfig
from models.materialize import get_llm_backbone_and_tokenizer, get_vision_backbone_and_transform
from models.registry import GLOBAL_REGISTRY, MODEL_REGISTRY
from models.vlms import PrismaticVLM
from overwatch import initialize_overwatch

from models import FiSvla

from safetensors.torch import load_file

from models.vlms.pointcloud_processor.model_loader import lift3d_dinov2_base

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)

# === HF Hub Repository ===
HF_HUB_REPO = "TRI-ML/prismatic-vlms"

# === Available Models ===
def available_models() -> List[str]:
    return list(MODEL_REGISTRY.keys())


def available_model_names() -> List[str]:
    return list(GLOBAL_REGISTRY.items())


def get_model_description(model_id_or_name: str) -> str:
    if model_id_or_name not in GLOBAL_REGISTRY:
        raise ValueError(f"Couldn't find `{model_id_or_name = }; check `prismatic.available_model_names()`")

    # Print Description & Return
    print(json.dumps(description := GLOBAL_REGISTRY[model_id_or_name]["description"], indent=2))

    return description


# === Load Pretrained Model ===
def load(
    model_id_or_path: Union[str, Path],
    hf_token: Optional[str] = None,
    cache_dir: Optional[Union[str, Path]] = None,
    load_for_training: bool = False,
    class_dropout_prob: float = 0.0,
    use_diff: bool = False,
    llm_middle_layer: int = 32,
    fuse: str = 'concat',
    action_tokenizer_exist: bool = False,
    training_mode: str = 'async',
    load_pointcloud: bool = False,
    pointcloud_pos: str = 'slow',
    action_chunk: int = 1,
    load_state: bool = True,
    **kwargs,
) -> PrismaticVLM:
    """Loads a pretrained PrismaticVLM from either local disk or the HuggingFace Hub."""
    if os.path.isdir(model_id_or_path):
        overwatch.info(f"Loading from local path `{(run_dir := Path(model_id_or_path))}`")

        # Get paths for `config.json` and pretrained checkpoint
        config_json, checkpoint_pt = run_dir / "config.json", run_dir / "checkpoints" / "latest-checkpoint.pt"
        assert config_json.exists(), f"Missing `config.json` for `{run_dir = }`"
        assert checkpoint_pt.exists(), f"Missing checkpoint for `{run_dir = }`"
    else:
        if model_id_or_path not in GLOBAL_REGISTRY:
            raise ValueError(f"Couldn't find `{model_id_or_path = }; check `prismatic.available_model_names()`")

        overwatch.info(f"Downloading `{(model_id := GLOBAL_REGISTRY[model_id_or_path]['model_id'])} from HF Hub")
        with overwatch.local_zero_first():
            config_json = hf_hub_download(repo_id=HF_HUB_REPO, filename=f"{model_id}/config.json", cache_dir=cache_dir)
            checkpoint_pt = hf_hub_download(
                repo_id=HF_HUB_REPO, filename=f"{model_id}/checkpoints/latest-checkpoint.pt", cache_dir=cache_dir
            )

    # Load Model Config from `config.json`
    with open(config_json, "r") as f:
        model_cfg = json.load(f)["model"]

    # = Load Individual Components necessary for Instantiating a VLM =
    #   =>> Print Minimal Config
    overwatch.info(
        f"Found Config =>> Loading & Freezing [bold blue]{model_cfg['model_id']}[/] with:\n"
        f"             Vision Backbone =>> [bold]{model_cfg['vision_backbone_id']}[/]\n"
        f"             LLM Backbone    =>> [bold]{model_cfg['llm_backbone_id']}[/]\n"
        f"             Arch Specifier  =>> [bold]{model_cfg['arch_specifier']}[/]\n"
        f"             Checkpoint Path =>> [underline]`{checkpoint_pt}`[/]"
    )

    # Load Vision Backbone
    overwatch.info(f"Loading Vision Backbone [bold]{model_cfg['vision_backbone_id']}[/]")
    vision_backbone, image_transform = get_vision_backbone_and_transform(
        model_cfg["vision_backbone_id"],
        model_cfg["image_resize_strategy"],
    )

    # Load LLM Backbone --> note `inference_mode = True` by default when calling `load()`
    overwatch.info(f"Loading Pretrained LLM [bold]{model_cfg['llm_backbone_id']}[/] via HF Transformers")
    llm_backbone, tokenizer = get_llm_backbone_and_tokenizer(
        model_cfg["llm_backbone_id"],
        llm_max_length=model_cfg.get("llm_max_length", 2048),
        hf_token=hf_token,
        inference_mode=not load_for_training,
        fuse=fuse,
    )

    pointcloud_backbone=None
    if load_pointcloud:
        overwatch.info(f"Loading Pretrained PointCloud Backbone")
        pointcloud_backbone = lift3d_dinov2_base()

    # Load VLM using `from_pretrained` (clobbers HF syntax... eventually should reconcile)
    overwatch.info(f"Loading VLM [bold blue]{model_cfg['model_id']}[/] from Checkpoint")
    vlm = PrismaticVLM.from_pretrained(
        checkpoint_pt,
        model_cfg["model_id"],
        vision_backbone,
        llm_backbone,
        pointcloud_backbone=pointcloud_backbone,
        arch_specifier=model_cfg["arch_specifier"],
        freeze_weights=not load_for_training,
        class_dropout_prob=class_dropout_prob,
        use_diff=use_diff,
        token_size=llm_backbone.embed_dim,
        llm_middle_layer=llm_middle_layer,
        action_tokenizer_exist=action_tokenizer_exist,
        training_mode=training_mode,
        load_pointcloud=load_pointcloud,
        pointcloud_pos=pointcloud_pos,
        action_chunk=action_chunk,
        load_state=load_state,
        **kwargs,
    )

    return vlm


# === Load Pretrained Model ===
def load_openvla(
    model_id_or_path: Union[str, Path],
    hf_token: Optional[str] = None,
    cache_dir: Optional[Union[str, Path]] = None,
    load_for_training: bool = False,
    class_dropout_prob: float = 0.0,
    use_diff: bool = False,
    llm_middle_layer: int = 32,
    fuse: str = 'concat',
    action_tokenizer_exist: bool = False,
    training_mode: str = 'async',
    load_pointcloud: bool = False,
    pointcloud_pos: str = 'slow',
    action_chunk: int = 1,
    load_state: bool = True,
    **kwargs,
) -> PrismaticVLM:
    """Loads a pretrained PrismaticVLM from either local disk or the HuggingFace Hub."""

    overwatch.info(f"Loading from local path `{(run_dir := Path(model_id_or_path))}`")

    # Get paths for `config.json` and pretrained checkpoint
    config_json = run_dir / "config.json"
    assert config_json.exists(), f"Missing `config.json` for `{run_dir = }`"

    # Load Model Config from `config.json`
    with open(config_json, "r") as f:
        model_cfg = json.load(f)

    model_cfg['model_id'] = "prism-dinosiglip-224px+7b"

    # = Load Individual Components necessary for Instantiating a VLM =
    #   =>> Print Minimal Config
    overwatch.info(
        f"Found Config =>> Loading & Freezing [bold blue]{model_cfg['model_id']}[/] with:\n"
        f"             Vision Backbone =>> [bold]{model_cfg['vision_backbone_id']}[/]\n"
        f"             LLM Backbone    =>> [bold]{model_cfg['llm_backbone_id']}[/]\n"
        f"             Arch Specifier  =>> [bold]{model_cfg['arch_specifier']}[/]\n"
        f"             Checkpoint Path =>> [underline]`{run_dir}`[/]"
    )

    # Load Vision Backbone
    overwatch.info(f"Loading Vision Backbone [bold]{model_cfg['vision_backbone_id']}[/]")
    vision_backbone, image_transform = get_vision_backbone_and_transform(
        model_cfg["vision_backbone_id"],
        model_cfg["image_resize_strategy"],
    )

    # Load LLM Backbone --> note `inference_mode = True` by default when calling `load()`
    overwatch.info(f"Loading Pretrained LLM [bold]{model_cfg['llm_backbone_id']}[/] via HF Transformers")
    llm_backbone, tokenizer = get_llm_backbone_and_tokenizer(
        model_cfg["llm_backbone_id"],
        llm_max_length=model_cfg.get("llm_max_length", 2048),
        hf_token=hf_token,
        inference_mode=not load_for_training,
        fuse=fuse,
    )

    weights_path_0 = run_dir / "model-00001-of-00003.safetensors"
    weights_path_1 = run_dir / "model-00002-of-00003.safetensors"
    weights_path_2 = run_dir / "model-00003-of-00003.safetensors"
    state_dict0 = load_file(weights_path_0)
    state_dict1 = load_file(weights_path_1)
    state_dict2 = load_file(weights_path_2)
    merged_state_dict = {}
    for state_dict in [state_dict0, state_dict1, state_dict2]:
        merged_state_dict.update(state_dict)

    pretrained_state_dict = {}
    pretrained_state_dict['llm_backbone'] = {}
    pretrained_state_dict['vision_backbone'] = {}
    pretrained_state_dict['projector'] = {}
    for k, v in merged_state_dict.items():
        if 'language_model' in k:
            pretrained_state_dict['llm_backbone'][k.replace('language_model', 'llm')] = v
        elif 'vision_backbone.featurizer' in k:
            pretrained_state_dict['vision_backbone'][k.replace('vision_backbone.featurizer', 'dino_featurizer').replace('scale_factor', 'gamma')] = v
        elif 'vision_backbone.fused_featurizer' in k:
            pretrained_state_dict['vision_backbone'][k.replace('vision_backbone.fused_featurizer', 'siglip_featurizer')] = v
        elif 'projector' in k and 'fc1' in k:
            pretrained_state_dict['projector'][k.replace('fc1', '0')] = v
        elif 'projector' in k and 'fc2' in k:
            pretrained_state_dict['projector'][k.replace('fc2', '2')] = v
        elif 'projector' in k and 'fc3' in k:
            pretrained_state_dict['projector'][k.replace('fc3', '4')] = v

    pointcloud_backbone=None
    if load_pointcloud:
        overwatch.info(f"Loading Pretrained PointCloud Backbone")
        pointcloud_backbone = lift3d_dinov2_base()

    # Load VLM using `from_pretrained` (clobbers HF syntax... eventually should reconcile)
    overwatch.info(f"Loading VLM [bold blue]{model_cfg['model_id']}[/] from Checkpoint")
    vlm = PrismaticVLM.from_pretrained(
        pretrained_state_dict,
        model_cfg["model_id"],
        vision_backbone,
        llm_backbone,
        pointcloud_backbone=pointcloud_backbone,
        arch_specifier=model_cfg["arch_specifier"],
        freeze_weights=not load_for_training,
        class_dropout_prob=class_dropout_prob,
        use_diff=use_diff,
        token_size=llm_backbone.embed_dim,
        llm_middle_layer=llm_middle_layer,
        action_tokenizer_exist=action_tokenizer_exist,
        training_mode=training_mode,
        load_pointcloud=load_pointcloud,
        pointcloud_pos=pointcloud_pos,
        action_chunk=action_chunk,
        load_state=load_state,
        **kwargs,
    )
    return vlm


# === Load Pretrained VLA Model ===
def load_vla(
    model_id_or_path: Union[str, Path],
    hf_token: Optional[str] = None,
    cache_dir: Optional[Union[str, Path]] = None,
    load_for_training: bool = False,
    class_dropout_prob: float=0.0,
    use_diff: bool = False,
    need_to_sub: int = 0,
    llm_middle_layer: int = 32,
    diffusion_steps: int = 100,
    fuse: str = 'concat',
    action_tokenizer_exist: bool = False,
    training_mode: str = 'async',
    load_pointcloud: bool = False,
    pointcloud_pos: str = 'slow',
    action_chunk: int = 1,
    load_state: bool = True,
    lang_subgoals_exist: bool = False,
    **kwargs,
) -> FiSvla:
    """Loads a pretrained FiSvla from either local disk or the HuggingFace Hub."""

    # TODO (siddk, moojink) :: Unify semantics with `load()` above; right now, `load_vla()` assumes path points to
    #   checkpoint `.pt` file, rather than the top-level run directory!
    if os.path.isfile(model_id_or_path):
        overwatch.info(f"Loading from local checkpoint path `{(checkpoint_pt := Path(model_id_or_path))}`")

        # [Validate] Checkpoint Path should look like `.../<RUN_ID>/checkpoints/<CHECKPOINT_PATH>.pt`
        assert (checkpoint_pt.suffix == ".pt") and (checkpoint_pt.parent.name == "checkpoints"), "Invalid checkpoint!"
        run_dir = checkpoint_pt.parents[1]

        # Get paths for `config.json`, `dataset_statistics.json` and pretrained checkpoint
        config_json, dataset_statistics_json = run_dir / "config.json", run_dir / "dataset_statistics.json"
        assert config_json.exists(), f"Missing `config.json` for `{run_dir = }`"
        assert dataset_statistics_json.exists(), f"Missing `dataset_statistics.json` for `{run_dir = }`"

    # Otherwise =>> try looking for a match on `model_id_or_path` on the HF Hub (`model_id_or_path`)
    else:
        # Search HF Hub Repo via fsspec API
        overwatch.info(f"Checking HF for `{(hf_path := str(Path(model_id_or_path)))}`")
        if not (tmpfs := HfFileSystem()).exists(hf_path):
            raise ValueError(f"Couldn't find valid HF Hub Path `{hf_path = }`")

        valid_ckpts = tmpfs.glob(f"{hf_path}/checkpoints/*.pt")
        if (len(valid_ckpts) == 0) or (len(valid_ckpts) != 1):
            raise ValueError(f"Couldn't find a valid checkpoint to load from HF Hub Path `{hf_path}/checkpoints/")

        target_ckpt = Path(valid_ckpts[-1]).name
        model_id_or_path = str(model_id_or_path)  # Convert to string for HF Hub API
        overwatch.info(f"Downloading Model `{model_id_or_path}` Config & Checkpoint `{target_ckpt}`")
        with overwatch.local_zero_first():
            # relpath = Path(model_type) / model_id_or_path
            config_json = hf_hub_download(
                repo_id=model_id_or_path, filename=f"{('config.json')!s}", cache_dir=cache_dir
            )
            dataset_statistics_json = hf_hub_download(
                repo_id=model_id_or_path, filename=f"{('dataset_statistics.json')!s}", cache_dir=cache_dir
            )
            checkpoint_pt = hf_hub_download(
                repo_id=model_id_or_path, filename=f"{(Path('checkpoints') / target_ckpt)!s}", cache_dir=cache_dir
            )

    # Load VLA Config (and corresponding base VLM `ModelConfig`) from `config.json`
    with open(config_json, "r") as f:
        vla_cfg = json.load(f)["vla"]
        model_cfg = ModelConfig.get_choice_class(vla_cfg["base_vlm"])()

    if model_cfg.vision_backbone_id=='clip-vit-l-336px': model_cfg.vision_backbone_id='clip-vit-l'
    # Load Dataset Statistics for Action Denormalization
    with open(dataset_statistics_json, "r") as f:
        norm_stats = json.load(f)

    # = Load Individual Components necessary for Instantiating a VLA (via base VLM components) =
    #   =>> Print Minimal Config
    overwatch.info(
        f"Found Config =>> Loading & Freezing [bold blue]{model_cfg.model_id}[/] with:\n"
        f"             Vision Backbone =>> [bold]{model_cfg.vision_backbone_id}[/]\n"
        f"             LLM Backbone    =>> [bold]{model_cfg.llm_backbone_id}[/]\n"
        f"             Arch Specifier  =>> [bold]{model_cfg.arch_specifier}[/]\n"
        f"             Checkpoint Path =>> [underline]`{checkpoint_pt}`[/]"
    )

    # Load Vision Backbone
    overwatch.info(f"Loading Vision Backbone [bold]{model_cfg.vision_backbone_id}[/]")
    vision_backbone, image_transform = get_vision_backbone_and_transform(
        model_cfg.vision_backbone_id,
        model_cfg.image_resize_strategy,
    )

    # Load LLM Backbone --> note `inference_mode = True` by default when calling `load()`
    overwatch.info(f"Loading Pretrained LLM [bold]{model_cfg.llm_backbone_id}[/] via HF Transformers")
    llm_backbone, tokenizer = get_llm_backbone_and_tokenizer(
        model_cfg.llm_backbone_id,
        llm_max_length=model_cfg.llm_max_length,
        hf_token=hf_token,
        inference_mode=not load_for_training,
        fuse=fuse,
    )

    pointcloud_backbone=None
    if load_pointcloud:
        overwatch.info(f"Loading Pretrained PointCloud Backbone")
        pointcloud_backbone = lift3d_dinov2_base()

    # Load VLM using `from_pretrained` (clobbers HF syntax... eventually should reconcile)
    overwatch.info(f"Loading VLA [bold blue]{model_cfg.model_id}[/] from Checkpoint")

    vla = FiSvla.from_pretrained(
        checkpoint_pt,
        model_cfg.model_id,
        vision_backbone,
        llm_backbone,
        pointcloud_backbone=pointcloud_backbone,
        arch_specifier=model_cfg.arch_specifier,
        freeze_weights=not load_for_training,
        norm_stats=norm_stats,
        class_dropout_prob=class_dropout_prob,
        need_to_sub=need_to_sub,
        use_diff=use_diff,
        llm_middle_layer=llm_middle_layer,
        diffusion_steps=diffusion_steps,
        action_tokenizer_exist=action_tokenizer_exist,
        training_mode=training_mode,
        load_pointcloud=load_pointcloud,
        pointcloud_pos=pointcloud_pos,
        action_chunk=action_chunk,
        load_state=load_state,
        lang_subgoals_exist=lang_subgoals_exist,
        **kwargs,
    )

    return vla
