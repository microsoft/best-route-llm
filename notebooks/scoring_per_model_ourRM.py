import json
import time
import argparse
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, HfArgumentParser
from trl import ModelConfig, RewardConfig, RewardTrainer, get_kbit_device_map, get_peft_config, get_quantization_config
from utils import load_sample
import logging
import os

logging.basicConfig(level=logging.INFO)
logging.info("Logging started")

# Please specify the model name
def generate_ourRM_scores(data, split, path, model_name_list):
    device = "cuda"
    # path = "ourRM_min_max_mid"
    model = AutoModelForSequenceClassification.from_pretrained(path, device_map=device, num_labels=1,
                                                               trust_remote_code=True, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True)

    file_path = f"./data/mixed_dataset_ourRM_ALL_token_num_{split}.jsonl"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    start_time = time.time()
    with open(file_path, 'w') as f:
        for i, d in enumerate(data):
            user_input = d['prompt']
            for model_name in model_name_list:
                response_list = d[model_name]['responses']
                d[model_name]['ourRM_scores'] = []

                for _response in response_list:
                    inputs = tokenizer(f"Human: {user_input} Assistant: {_response}", return_tensors='pt').to(device)
                    with torch.no_grad():
                        _s = model(**inputs).logits[0].cpu().detach()
                        d[model_name]['ourRM_scores'].append(float(_s[0]))

            f.write(json.dumps(d) + "\n")

            end_time = time.time()
            logging.info(f'{i}-th query done; avg {(end_time - start_time) / (i + 1):.2f} s / query')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=None, help="Path to the dataset file")
    parser.add_argument('--model_name_list', nargs='+', type=str, default=None, help='List of LLM names')
    parser.add_argument('--split', type=str, default=None, help="The split of the dataset, e.g., train, validation, test")
    parser.add_argument('--proxy_path', type=str, default=None, help="Path to the trained proxy reward model")

    args = parser.parse_args()

    data = load_sample(fname=args.data_path, is_jsonl=True)
    generate_ourRM_scores(data, split=args.split, path=args.proxy_path, model_name_list=args.model_name_list)
