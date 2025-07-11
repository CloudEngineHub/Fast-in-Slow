"""
materialize.py

Factory class for initializing Open-X RLDS-backed datasets, given specified data mixture parameters; provides and
exports individual functions for clear control flow.
"""

from pathlib import Path
from typing import Tuple, Type

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from models.backbones.llm.prompting import PromptBuilder
from models.backbones.vision import ImageTransform
from util.data_utils import PaddedCollatorForActionPrediction
from vla.action_tokenizer import ActionTokenizer
from vla.datasets import EpisodicRLDSDataset, RLDSBatchTransform, RLDSDataset


def get_vla_dataset_and_collator(
    data_root_dir: Path,
    data_mix: str,
    image_transform: ImageTransform,
    tokenizer: PreTrainedTokenizerBase,
    prompt_builder_fn: Type[PromptBuilder],
    default_image_resolution: Tuple[int, int, int],
    padding_side: str = "right",
    predict_stop_token: bool = True,
    shuffle_buffer_size: int = 100_000,
    train: bool = True,
    episodic: bool = False,
    image_aug: bool = False,
    future_action_window_size: int = 0,
    past_action_window_size: int = 1,         # Concatenated `past_action_window_size-1' actions and the current action for the input
    load_all_data_for_training: bool = True,  # Load all data for training, or only a subset
    action_tokenizer_exist: bool = False,
    need_to_sub: int = 0,
    camera_view: str = "",
    load_pointcloud: bool = False,
    action_chunk: int = 1,
    lang_subgoals_exist: bool = False,
) -> Tuple[Dataset, ActionTokenizer, PaddedCollatorForActionPrediction]:
    """Initialize RLDS Dataset (wraps TFDS), ActionTokenizer, and initialize transform/collation functions."""
    action_tokenizer = ActionTokenizer(tokenizer, need_to_sub) if action_tokenizer_exist else None
    # action_tokenizer = ActionTokenizer(tokenizer)
    batch_transform = RLDSBatchTransform(
        action_tokenizer, tokenizer, image_transform, prompt_builder_fn, predict_stop_token=predict_stop_token, load_pointcloud=load_pointcloud,
        action_chunk = action_chunk,
        lang_subgoals_exist = lang_subgoals_exist,
    )
    collator = PaddedCollatorForActionPrediction(
        tokenizer.model_max_length, tokenizer.pad_token_id, padding_side=padding_side, load_pointcloud=load_pointcloud,
    )

    # Build RLDS Iterable Dataset
    cls = RLDSDataset if not episodic else EpisodicRLDSDataset
    dataset = cls(
        data_root_dir,
        data_mix,
        batch_transform,
        resize_resolution=default_image_resolution[1:],
        shuffle_buffer_size=shuffle_buffer_size,
        train=train,
        future_action_window_size=future_action_window_size,
        past_action_window_size=past_action_window_size,
        image_aug=image_aug,
        load_all_data_for_training=load_all_data_for_training,
        camera_view = camera_view,
        load_pointcloud = load_pointcloud,
        action_chunk = action_chunk,
    )

    return dataset, action_tokenizer, collator