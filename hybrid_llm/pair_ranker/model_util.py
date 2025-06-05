# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import torch
import logging
import safetensors

from typing import Dict, List
from hybrid_llm.pair_ranker.ranker import NClassReranker
from llm_blender.pair_ranker.model_util import build_pretrained_model


def build_nclass_ranker(
    model_type: str, model_name: str, cache_dir: str, config, tokenizer
) -> NClassReranker:
    """
    Build and initialize a ranker model from pretrained models.

    Args:
        model_type (str): The type of the pretrained model.
        model_name (str): The name of the pretrained model.
        cache_dir (str): The directory to cache the pretrained model.
        config: The configuration for the ranker model.
        tokenizer: The tokenizer for the ranker model.

    Returns:
        NClassReranker: The initialized ranker model.

    """
    logging.info("Initializing ranker from pretrained models")
    pretrained_model = build_pretrained_model(
        model_type, model_name, cache_dir=cache_dir
    )
    pretrained_model.resize_token_embeddings(len(tokenizer))
    logging.info(f"Ranker initialized successfully")
    return NClassReranker(pretrained_model, config, tokenizer)


def build_nclass_ranker_from_checkpoint(
    checkpoint_path: str, config, tokenizer
) -> NClassReranker:
    """
    Build a n-class ranker model from a checkpoint.

    Args:
        checkpoint_path (str): The path to the checkpoint directory.
        config: The configuration object for the model.
        tokenizer: The tokenizer object for the model.

    Returns:
        The built n-class ranker model.

    Raises:
        Exception: If the pytorch_model.bin or model.safetensors file is not found in the checkpoint path.
        Exception: If there are missing keys in the loaded checkpoint.
    """
    logging.info(f"Loading model checkpoint from path {checkpoint_path}")

    model = build_nclass_ranker(
        config.model_type,
        config.model_name,
        config.cache_dir,
        config,
        tokenizer,
    )

    model_path = os.path.join(checkpoint_path, "pytorch_model.bin")
    if os.path.exists(model_path):
        logging.info(f"Loading pickle checkpoint from path {model_path}")
        state_dict = torch.load(model_path)

    model_path = os.path.join(checkpoint_path, "model.safetensors")
    if os.path.exists(model_path):
        logging.info(f"Loading safetensors checkpoint from path {model_path}")
        state_dict = safetensors.torch.load_file(model_path, device="cpu")

    if not state_dict:
        raise Exception(
            f"pytorch_model.bin or model.safetensors file not found in path: {checkpoint_path}"
        )

    load_result = model.load_state_dict(state_dict, strict=False)
    if load_result.missing_keys:
        raise Exception(
            f'Fail to load checkpoint with missing keys: "{load_result.missing_keys}"'
        )
    else:
        logging.info(f'Successfully loaded checkpoint from "{checkpoint_path}"')
        return model


def predictions_2_ids(predictions: List[Dict], config) -> List[int]:
    """
    Converts predictions to a list of IDs based on a given configuration.

    Args:
        predictions (dict): A dictionary containing the predictions.
        config (object): An object representing the configuration.

    Returns:
        list: A list of IDs corresponding to the predictions.

    """
    ids = []
    for p in predictions["preds"]:
        if config.is_prob:
            p = p.softmax(dim=0)
        ids.append(0 if p[0] - p[1] >= config.match_t else 1)
    return ids
