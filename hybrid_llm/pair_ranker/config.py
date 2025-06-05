# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from dataclasses import dataclass, field
from .ranker import NClassRerankerLossType
from typing import List


@dataclass
class NClassRankerConfig:
    """
    This class represents the configuration for the NClassRanker.
    """

    ranker_type: str = field(
        default="nclass",
        metadata={"help": "Ranker type, pairranker or reranker choices: nclass"},
    )
    model_type: str = field(
        default="deberta", metadata={"help": "Model type, deberta or roberta"}
    )
    model_name: str = field(
        default="microsoft/deberta-v3-large", metadata={"help": "Model name"}
    )
    candidate_models: List[str] = field(
        default=None, metadata={"help": "List of candidate models"}
    )
    cache_dir: str = field(default=None, metadata={"help": "Cache dir"})
    load_checkpoint: str = field(
        default=None, metadata={"help": "Load checkpoint path"}
    )
    source_max_length: int = field(
        default=128, metadata={"help": "Max length of the source sequence"}
    )
    candidate_max_length: int = field(
        default=128, metadata={"help": "Max length of the candidate sequence"}
    )
    n_tasks: int = field(default=1, metadata={"help": "Number of tasks"})
    n_candidates: int = field(
        default=-1, metadata={"help": "Number of candidate models"}
    )
    match_t: float = field(
        default=0.0, metadata={"help": "threshold to compute match labels"}
    )
    loss_type: str = field(
        default=NClassRerankerLossType.DET_2CLS.value,
        metadata={"help": "Loss type: det_2cls, prob_2cls"},
    )
    is_prob: bool = field(
        default=False, metadata={"help": "Model trained via probabilistic approach"}
    )
    quality_metric: str = field(
        default="bartscore",
        metadata={"help": "Response quality metric, default: bartscore"},
    )
    num_pos: int = field(
        default=1,
        metadata={
            "help": "Number of positive examples used for training, used for top_bottom and all_pair sampling"
        },
    )
    num_neg: int = field(
        default=1,
        metadata={
            "help": "Number of negative examples used for training, used for top_bottom and all_pair sampling"
        },
    )
    sub_sampling_mode: str = field(
        default="all_pair",
        metadata={"help": "Sub sampling mode: top_bottom, all_pair, random, uniform"},
    )
    sub_sampling_ratio: float = field(
        default=0.5,
        metadata={"help": "Sub sampling ratio, used for random and uniform sampling"},
    )
    reduce_type: str = field(
        default="linear", metadata={"help": "Reduce type: linear, max, mean"}
    )
    inference_mode: str = field(
        default="bubble", metadata={"help": "Inference mode: bubble, full"}
    )
    drop_out: float = field(default=0.05, metadata={"help": "Dropout rate"})
    fp16: bool = field(default=True, metadata={"help": "Whether to use fp16"})
