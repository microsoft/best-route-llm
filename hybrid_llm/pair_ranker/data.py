# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import json
import logging
import numpy as np

from sklearn.utils import resample
from collections.abc import Iterable
from typing import Dict, List, Union
from llm_blender.common.utils import seed_everything
from llm_blender.pair_ranker.data import Dataset as BlenderDataset


model2prompt_cost = {'gpt-4o': 5e-6, 'gpt-35-turbo': 3e-6, 'phi-3-mini': 0.3e-6, 'phi-3-medium': 0.5e-6, 'mistral-7b': 0.25e-6, 'mistral-8x7b': 0.7e-6, 'llama-31-8b': 0.3e-6, 'codestral-22b': 1e-6}
model2output_cost = {'gpt-4o': 15e-6, 'gpt-35-turbo': 6e-6, 'phi-3-mini': 0.9e-6, 'phi-3-medium': 1.5e-6, 'mistral-7b': 0.25e-6, 'mistral-8x7b': 0.7e-6, 'llama-31-8b': 0.61e-6, 'codestral-22b': 3e-6}


class Dataset(BlenderDataset):

    def __getitem__(self, index: int) -> Dict:
        """
        Retrieve an item from the dataset at the given index.

        Args:
            index (int): The index of the item to retrieve.

        Returns:
            dict: A dictionary containing the following keys:
                - 'index': The index of the item.
                - 'source': The concatenated instruction and input of the item.
                - 'target': The output of the item.
                - 'candidates': A list of candidate texts.
                - 'scores': A list of candidate scores.
        """
        item = self.data[index]
        target = item["output"]
        source = item["instruction"] + item["input"]
        if isinstance(target, list):
            target = target[0]

        candidates_text = None
        candidates_scores = None

        if "candidates" in item:
            candidates = [candidate for candidate in item["candidates"]]
            assert self.n_candidates == len(candidates), f"n_candidates {self.n_candidates} != len(candidates) {len(candidates)}"
            # if self.n_candidates is not None:
            #     candidates = candidates[: self.n_candidates]

            candidates_text = [candidate["text"] for candidate in candidates]
            format_scores_costs(candidates, candidates[0]["scores"].keys())
            candidates_scores = [
                list(candidate["scores"].values()) for candidate in candidates
            ]
            candidates_costs = [candidate["cost"] for candidate in candidates]

        return {
            "index": index,
            "source": source,
            "target": target,
            "candidates": candidates_text,
            "scores": candidates_scores,
            "costs": candidates_costs,
        }


def load_data(
    data_path: str,
    args,
    max_size: int = None,
    upsample_major: str = "",
    upsample_minor: str = "",
) -> List[Dict]:
    """
    Load data from a given data path and preprocess it.

    Args:
        data_path (str): The path to the data file.
        args: The arguments for preprocessing.
        max_size (int, optional): The maximum number of examples to load. Defaults to None.
        upsample_major (str, optional): The method for upsampling the majority class. Defaults to ''.
        upsample_minor (str, optional): The method for upsampling the minority class. Defaults to ''.

    Returns:
        List[Dict]: Returns a list of dictionaries, each representing an individual record in the dataset.
    """
    seed_everything(args.seed)

    logging.info(f"Loading data from {data_path}")
    data = load_json_dataset(data_path)

    if max_size is not None and max_size > 0:
        data = data[:max_size]

    examples = []
    examples_majority = list()
    examples_minority = list()

    for item in data:
        item["candidates"] = filter_candidates(item["candidates"], args)
        format_scores_costs(item["candidates"], args.metrics)

    for index, example in enumerate(data):
        if not "id" in example:
            example["id"] = index
        examples.append(example)
        major_score, minor_score = get_minor_major_scores(
            example["candidates"], upsample_major, upsample_minor, args.metrics[0]
        )
        if major_score is not None and minor_score is not None:
            if np.mean(major_score) > np.mean(minor_score):
                examples_majority.append(example)
            else:
                examples_minority.append(example)

    if upsample_major != "" and upsample_minor != "":
        examples = upsample_minority(
            examples_majority, examples_minority, upsample_minor
        )

    return examples


def get_minor_major_scores(
    candidates: List[Dict], upsample_major: str, upsample_minor: str, metric: str
):
    """
    Get the scores for major and minor candidates based on the given metric.

    Args:
        candidates (list): A list of candidate dictionaries.
        upsample_major (str): The model name for the major candidate.
        upsample_minor (str): The model name for the minor candidate.
        metric (str): The metric to retrieve the scores for.

    Returns:
        tuple: A tuple containing the major score and the minor score.
    """
    major_score = None
    minor_score = None
    for candidate in candidates:
        if candidate["model"] == upsample_major:
            major_score = candidate["scores"][metric]
        if candidate["model"] == upsample_minor:
            minor_score = candidate["scores"][metric]
    return major_score, minor_score


def format_scores_costs(candidates: List[Dict], metrics: List[str]):
    """
    Format scores for each candidate.

    Args:
        candidates (list): A list of candidate dictionaries.
        metrics (list): A list of metrics to process scores for.

    Returns:
        None

    The function format the scores for each candidate in the given list. For each candidate, it iterates over the
    provided metrics and performs the following steps:
    - If the scores for a metric are iterable, it converts each score to a float and takes the first 10 scores.
    - If the scores for a metric are not iterable, it converts the score to a float and repeats it 10 times.

    Note: The function modifies the candidates in-place and does not return a new list.
    """
    window_size = 10
    for candidate in candidates:
        for metric in metrics:
            scores = candidate["scores"][metric]

            if isinstance(scores, Iterable):
                scores = [float(score) for score in scores][:window_size]
            else:
                scores = [float(scores)] * window_size

            candidate["scores"][metric] = scores

        token_num_prompt = candidate["token_num_prompt"]
        if isinstance(candidate["token_num_responses"], Iterable):
            token_num_responses_list = [float(num) for num in candidate["token_num_responses"]][:window_size]
        else:
            token_num_responses_list = [float(candidate["token_num_responses"])] * window_size

        for key in model2prompt_cost.keys():
            if candidate["model"].startswith(key):
                promt_cost = model2prompt_cost[key]
                output_cost = model2output_cost[key]
                candidate["cost"] = [promt_cost * token_num_prompt + output_cost * token_num_responses for token_num_responses in token_num_responses_list]
                break


def filter_candidates(candidates: List[Dict], args: object) -> List[Dict]:
    """
    Filters a list of candidates based on specified models and decoding methods.

    Args:
        candidates (list): A list of candidate dictionaries.
        args: An object containing the command-line arguments.

    Returns:
        list: A filtered list of candidate dictionaries.

    Raises:
        ValueError: If no candidates are left after filtering.
    """
    # Filter candidates based on models and decoding methods
    if args.candidate_models is not None:
        filtered_candidates = []
        for model_name in args.candidate_models:
            for candidate in candidates:
                if candidate["model"] == model_name:
                    filtered_candidates.append(candidate)
                    break
            # If no candidate is found for the model, raise an error
        if len(filtered_candidates) != args.n_candidates:
            raise ValueError(
                f"No candidates found for the specified models: {args.candidate_models}"
            )

        # filtered_candidates = [
        #     candidate
        #     for candidate in candidates
        #     if candidate["model"] in args.candidate_models
        # ]
    if args.candidate_decoding_methods is not None:
        filtered_candidates = []
        for decoding_method in args.candidate_decoding_methods:
            for candidate in candidates:
                if candidate["decoding_method"] == decoding_method:
                    filtered_candidates.append(candidate)
                    break
            # If no candidate is found for the decoding method, raise an error
        if len(filtered_candidates) != args.n_candidates:
            raise ValueError(
                f"No candidates found for the specified decoding methods: {args.candidate_decoding_methods}"
            )

        # filtered_candidates = [
        #     candidate
        #     for candidate in candidates
        #     if candidate["decoding_method"] in args.candidate_decoding_methods
        # ]
    if len(filtered_candidates) == 0:
        available_model_methods = set(
            [
                (candidate["model"], candidate["decoding_method"])
                for candidate in candidates
            ]
        )
        raise ValueError(
            "No candidates left after filtering, available models and methods are: \n{}".format(
                "\n".join([str(x) for x in available_model_methods])
            )
        )
    return filtered_candidates


def upsample_minority(
    examples_majority: List, examples_minority: List, upsample_minor: str
) -> List[Dict]:
    """
    Upsamples the minority class by randomly resampling examples from the minority class to match the number of examples in the majority class.

    Args:
        examples_majority (list): A list of examples from the majority class.
        examples_minority (list): A list of examples from the minority class.
        upsample_minor (str): A str indicating the minority class.

    Returns:
        list: A list containing the upsampled examples, which includes the examples from the majority class and the resampled examples from the minority class.
    """
    logging.info(
        f"before upsample {upsample_minor} #minority is {len(examples_minority)}"
    )
    upsampled_minority = resample(
        examples_minority, replace=True, n_samples=len(examples_majority)
    )
    logging.info(
        f"after upsample {upsample_minor} #minority is {len(upsampled_minority)}"
    )
    return examples_majority + upsampled_minority


def load_json_dataset(dataset_path: str) -> Union[List[Dict], Dict]:
    """
    Loads a JSON or JSONL dataset from the given file path.

    Args:
        dataset_path (str): The path to the JSON or JSONL dataset file.

    Returns:
        Union[List[Dict], Dict]: The loaded dataset. If the file is in JSONL format, a list of dictionaries is returned.
        If the file is in JSON format, a single dictionary is returned.

    Raises:
        FileNotFoundError: If the dataset file does not exist.
        JSONDecodeError: If the dataset file is not a valid JSON or JSONL file.

    """
    assert dataset_path, "data_path is not specified"

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"The dataset file {dataset_path} does not exist.")

    with open(dataset_path) as json_file:
        if dataset_path.lower().endswith("jsonl"):
            return [json.loads(f) for f in list(json_file)]

        return json.load(json_file)
