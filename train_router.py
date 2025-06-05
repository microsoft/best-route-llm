# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import torch
import logging
import argparse

from uuid import uuid4

from typing import List, Tuple
from transformers import PreTrainedTokenizer, TrainingArguments
from hybrid_llm.pair_ranker.config import NClassRankerConfig
from hybrid_llm.pair_ranker.data import load_data, Dataset
from hybrid_llm.pair_ranker.collator import NClassCollator
from hybrid_llm.pair_ranker.ranker import NClassReranker, NClassRerankerLossType
from hybrid_llm.pair_ranker.trainer import (compute_metrics_for_prob_2class, compute_metrics_for_det_nclass,
                                            compute_metrics_for_det_nlabel, compute_metrics_for_prob_nlabel)
from hybrid_llm.pair_ranker.model_util import (
    build_nclass_ranker,
    build_nclass_ranker_from_checkpoint,
)

from llm_blender.pair_ranker.model_util import build_tokenizer
from llm_blender.pair_ranker.trainer import RerankerTrainer
from llm_blender.common.utils import empty2None, str2bool, seed_everything

loss2metric = {
    "det_ncls": compute_metrics_for_det_nclass,
    "det_nlabels": compute_metrics_for_det_nlabel,
    "prob_2cls": compute_metrics_for_prob_2class,
    "prob_nlabels": compute_metrics_for_prob_nlabel,
}


def main(args):
    """
    Main function for training and predicting using the hybrid-ml model.

    Args:
        args (argparse.Namespace): Command-line arguments parsed by argparse.

    Returns:
        None
    """
    if args.save_predictions and args.output_dir is None:
        raise ValueError("output_dir must be set to save predictions")

    seed_everything(args.seed)

    setup_device()
    tokenizer = build_tokenizer(args.model_name, cache_dir=args.cache_dir)
    data_collator = NClassCollator(
        args.source_maxlength, tokenizer, args.candidate_maxlength)
    train_dataset, eval_dataset, predict_dataset = load_datasets(args)
    model = initialize_model(args, tokenizer)
    trainer = initialize_training(
        args, model, train_dataset, eval_dataset, tokenizer, data_collator)

    if args.do_train:
        train(args, trainer, model)
    if args.do_predict:
        predict(args, trainer, predict_dataset)


def setup_device() -> torch.device:
    """
    Sets up the device for training.

    Returns:
        device (torch.device): The device to be used for training.
    """
    logging.info("Setting up device")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    logging.info(f"device: {device}, n_gpu: {n_gpu}")
    return device


def initialize_model(args, tokenizer: PreTrainedTokenizer) -> NClassReranker:
    """
    Initializes the reranker model using the provided arguments and tokenizer.

    Args:
        args (argparse.Namespace): The command-line arguments.
        tokenizer (PreTrainedTokenizer): The tokenizer used for tokenization.

    Returns:
        NClassReranker: The initialized reranker model.
    """
    config = NClassRankerConfig()
    config.__dict__.update(
        {k: v for k, v in args.__dict__.items() if k in config.__dict__})

    if args.load_checkpoint:
        config = torch.load(os.path.join(args.load_checkpoint, "config.bin"))
        model = build_nclass_ranker_from_checkpoint(
            args.load_checkpoint, config, tokenizer
        )
    else:
        model = build_nclass_ranker(
            args.model_type,
            args.model_name,
            args.cache_dir,
            config,
            tokenizer,
        )
    return model


def initialize_training(
    args, model: NClassReranker, train_dataset: Dataset,
    eval_dataset: Dataset, tokenizer: PreTrainedTokenizer,
    data_collator: NClassCollator
) -> RerankerTrainer:
    """
    Initializes the training process for the reranker model.

    Args:
        args (argparse.Namespace): The command-line arguments.
        model (NClassReranker): The reranker model.
        train_dataset (Dataset): The training dataset.
        eval_dataset (Dataset): The evaluation dataset.
        tokenizer (PreTrainedTokenizer): The tokenizer used for tokenization.
        data_collator (NClassCollator): The data collator used for batching and padding.

    Returns:
        RerankerTrainer: The trainer object for training the reranker model.
    """
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=args.overwrite_output_dir,
        do_train=args.do_train,
        do_eval=args.do_eval,
        do_predict=args.do_predict,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_steps=args.logging_steps,
        logging_first_step=args.logging_first_step,
        log_level=args.log_level,
        report_to=args.report_to,
        run_name=args.run_name,
        load_best_model_at_end=args.load_best_model_at_end,
        metric_for_best_model=args.metric_for_best_model,
        seed=args.seed,
        local_rank=args.local_rank,
        fp16=args.fp16,
        deepspeed=args.deepspeed,
        label_names=args.label_names,
        evaluation_strategy=args.evaluation_strategy,
        save_strategy=args.save_strategy,
        adafactor=args.adafactor,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        remove_unused_columns=False,
        disable_tqdm=True,
        greater_is_better=True,
    )

    trainer = RerankerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=loss2metric[args.loss_type],
    )

    return trainer


def load_datasets(args) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Load datasets for training, evaluation, and prediction.

    Args:
        args: An object containing the arguments for loading datasets.

    Returns:
        A tuple containing the training dataset, evaluation dataset, and prediction dataset.
    """
    train_dataset = None
    eval_dataset = None
    predict_dataset = None

    def load_dataset(dataset_name: str, dataset_path: str, args, max_size: int = None):
        logging.info(
            f"Loading {dataset_name} dataset from path: {dataset_path}")
        examples = load_data(dataset_path, args, max_size=max_size)
        dataset = Dataset(examples, args.n_candidates)
        logging.info(
            f"{dataset_name.capitalize()} dataset loaded successfully")
        return dataset

    if args.do_train:
        train_dataset = load_dataset(
            "training", args.train_data_path, args, max_size=args.max_train_data_size)

    if args.do_eval:
        eval_dataset = load_dataset(
            "evaluation", args.eval_data_path, args, max_size=args.max_eval_data_size)
    else:
        args.evaluation_strategy = "no"
        args.save_strategy = "no"

    if args.do_predict:
        predict_dataset = load_dataset(
            "test", args.test_data_path, args, max_size=args.max_predict_data_size)

    if args.do_train:
        if args.do_eval:
            assert train_dataset.n_tasks == eval_dataset.n_tasks
        args.n_tasks = train_dataset.n_tasks
    elif args.do_predict:
        args.n_tasks = predict_dataset.n_tasks

    return train_dataset, eval_dataset, predict_dataset


def train(args, trainer: RerankerTrainer, model: NClassReranker):
    """
    Trains the model using the provided trainer and model instances.

    Args:
        args (object): The arguments object containing the training configuration.
        trainer (RerankerTrainer): The trainer instance used for training.
        model (NClassReranker): The model instance to be trained.

    Returns:
        None
    """
    if args.evaluate_before_training:
        metrics = trainer.evaluate()
        logging.info(f"Evaluate first step: \n{metrics}")

    logging.info("Start training")
    outputs = trainer.train(
        resume_from_checkpoint=args.resume_from_checkpoint,
    )
    logging.info("Training finished")
    global_step, training_loss = outputs.global_step, outputs.training_loss
    metrics = outputs.metrics
    logging.info(
        f"global_step = {global_step}, average loss = {training_loss}")
    for key, value in metrics.items():
        logging.info(f"{key} = {value}")

    logging.info("Saving model")
    best_checkpoint_folder = os.path.join(args.output_dir, "checkpoint-best")
    trainer.save_model(best_checkpoint_folder)
    torch.save(model.args, os.path.join(best_checkpoint_folder, "config.bin"))
    logging.info(f"Model saved to {best_checkpoint_folder}")
    logging.info("Training finished")


def predict(args, trainer: RerankerTrainer, predict_dataset: Dataset):
    """
    Perform prediction using the provided trainer on the given dataset.

    Args:
        args (object): The arguments object containing configuration options.
        trainer (object): The trainer object used for prediction.
        predict_dataset (object): The dataset used for prediction.

    Returns:
        None
    """
    logging.info("Start predicting")
    outputs = trainer.predict(predict_dataset)
    predictions = outputs.predictions
    labels = outputs.label_ids
    metrics = outputs.metrics
    logging.info(f"metrics: {metrics}")

    # save predictions
    if args.save_predictions:
        save_predictions(predictions, labels, args.output_dir)
    logging.info("Predicting finished")


def save_predictions(predictions: torch.Tensor, labels: torch.Tensor, output_dir: str):
    """
    Save the predictions and labels to the specified output directory.

    Args:
        predictions (torch.Tensor): The predictions to be saved.
        labels (torch.Tensor): The labels to be saved.
        output_dir (str): The directory where the predictions and labels will be saved.

    Returns:
        None
    """
    logging.info("Saving predictions")
    with open(os.path.join(output_dir, "predictions.pt"), "wb") as f:
        torch.save(predictions, f)
    with open(os.path.join(output_dir, "labels.pt"), "wb") as f:
        torch.save(labels, f)
    logging.info(f"predictions saved to {output_dir}")


def split_comma_separated_arg(arg: str) -> List[str]:
    """
    Splits a comma-separated string into a list of individual values.

    Args:
        arg (str): The comma-separated string to be split.

    Returns:
        list: A list of individual values extracted from the comma-separated string.
              Returns None if the input string is None.
    """
    return arg.split(",") if arg is not None else None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # model config
    parser.add_argument("--model_name", type=str,
                        default="microsoft/deberta-v3-large")
    parser.add_argument("--load_checkpoint", type=empty2None, default=None)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument(
        "--loss_type",
        type=str,
        choices=[loss_type.value for loss_type in NClassRerankerLossType],
        default=NClassRerankerLossType.DET_2CLS.value,
    )

    # data config
    parser.add_argument("--match_t", type=float, default=0.0)
    parser.add_argument("--upsample_major", type=str, default="")
    parser.add_argument("--upsample_minor", type=str, default="")
    parser.add_argument(
        "--candidate_decoding_method",
        type=empty2None,
        default=None,
        help="separted by comma. Empty string for all methods",
    )
    parser.add_argument("--n_candidates", type=int, default=-1)
    parser.add_argument(
        "--candidate_models",
        type=empty2None,
        default=None,
        help="Small and Large model separted by comma",
    )
    parser.add_argument("--source_maxlength", type=int, default=128)
    parser.add_argument("--candidate_maxlength", type=int, default=128)
    parser.add_argument("--num_pos", type=int, default=5)
    parser.add_argument("--num_neg", type=int, default=5)
    parser.add_argument("--sub_sampling_ratio", type=float, default=0.4)
    parser.add_argument(
        "--sub_sampling_mode",
        type=str,
        choices=[
            "uniform",
            "top_bottom",
            "top_random",
            "random_bottom",
            "random",
            "uniform",
            "all_pair",
        ],
        default="all_pair",
    )
    parser.add_argument("--max_train_data_size", type=int, default=-1)
    parser.add_argument("--max_eval_data_size", type=int, default=-1)
    parser.add_argument("--max_predict_data_size", type=int, default=-1)
    parser.add_argument(
        "--quality_metric",
        type=str,
        default="bartscore",
        help="Response quality metric, default: bartscore",
    )

    # running config
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", type=str2bool, default=True)
    parser.add_argument("--deepspeed", type=str, default=None)
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank. Necessary for using the torch.distributed.launch utility.",
    )

    # mode
    parser.add_argument("--do_train", type=str2bool, default=True)
    parser.add_argument("--do_eval", type=str2bool, default=True)
    parser.add_argument("--do_predict", type=str2bool, default=True)

    # training hyperparameters
    parser.add_argument("--train_data_path", type=str, default=None)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=10e10)
    parser.add_argument("--num_train_epochs", type=int, default=5)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ],
        default="linear",
    )
    parser.add_argument("--adafactor", type=bool, default=True)

    # evaluation hyperparameters
    parser.add_argument("--eval_data_path", type=str, default=None)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--evaluate_before_training",
                        type=str2bool, default=False)
    parser.add_argument(
        "--evaluation_strategy",
        type=str,
        choices=["steps", "epoch", "no"],
        default="epoch",
    )
    parser.add_argument("--eval_steps", type=int, default=0)

    # predict config
    parser.add_argument("--test_data_path", type=str, default=None)
    parser.add_argument("--save_predictions", type=str2bool, default=True)

    # logging
    parser.add_argument("--logging_first_step", type=str2bool, default=True)
    parser.add_argument("--logging_steps", type=int, default=5)
    parser.add_argument(
        "--log_level",
        type=str,
        default="info",
        choices=["info", "debug", "warning", "error", "critical"],
    )
    parser.add_argument("--report_to", type=str, default="none")
    parser.add_argument("--run_name", type=str,
                        default=str(uuid4()))  # wandb run name

    # save config
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--overwrite_output_dir", type=str2bool, default=False)
    parser.add_argument(
        "--save_strategy", type=str, choices=["steps", "epoch", "no"], default="epoch"
    )
    parser.add_argument("--save_steps", type=int, default=0)
    parser.add_argument("--save_total_limit", type=int, default=4)

    # metrics config
    parser.add_argument("--load_best_model_at_end",
                        type=str2bool, default=True)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--metric_for_best_model",
                        type=str, default="dev_score")

    # inference config
    parser.add_argument(
        "--inference_mode", type=str, default="bubble", choices=["bubble", "full"]
    )

    # init args
    args = parser.parse_args()
    args.ranker_type = "nclass"
    args.model_type = "deberta"

    args.load_best_model_at_end = args.do_train and args.do_predict

    # set up default output dir
    if args.output_dir is None:
        args.output_dir = f"outputs/{args.run_name}"
    args.cache_dir = f"./hf_models/{args.model_name.split('/')[-1]}/"
    args.label_names = ["scores"]
    args.candidate_decoding_methods = split_comma_separated_arg(
        args.candidate_decoding_method
    )
    args.candidate_models = split_comma_separated_arg(args.candidate_models)
    args.n_candidates = (
        len(args.candidate_models)
        if args.candidate_models is not None
        else args.n_candidates
    )
    args.local_rank = os.environ.get("LOCAL_RANK", args.local_rank)
    args.metrics = split_comma_separated_arg(args.quality_metric)
    # args.is_prob = args.loss_type == NClassRerankerLossType.PROB_2CLS.value

    # set up logging
    logging.basicConfig(level=args.log_level.upper())

    main(args)
