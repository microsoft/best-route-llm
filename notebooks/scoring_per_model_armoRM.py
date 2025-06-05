import time
import json
import argparse
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from utils import load_sample
import logging
import os

logging.basicConfig(level=logging.INFO)
logging.info("Logging started")

def generate_armoRM_scores(data, model_name_list):
    device = "cuda"
    path = "RLHFlow/ArmoRM-Llama3-8B-v0.1"
    model = AutoModelForSequenceClassification.from_pretrained(path, device_map=device,
                                                               trust_remote_code=True, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True)

    file_path = f"./data/mixed_dataset_armoRM_ALL.jsonl"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    start_time = time.time()
    with open(file_path, 'w') as f:
        for i, d in enumerate(data):
            user_input = d['prompt']
            for model_name in model_name_list:
                message_list = []
                for response in d[model_name]['responses']:
                    message = [{"role": "user", "content": user_input},
                               {"role": "assistant", "content": response}]
                    message_list.append(message)

                try:
                    input_ids = tokenizer.apply_chat_template(message_list, return_tensors="pt", padding=True).to(device)

                    with torch.no_grad():
                        output = model(input_ids)
                        score = output.score.cpu().float()
                        score = np.asarray(score).reshape(-1)
                        d[model_name]['armoRM_scores'] = [float(_) for _ in score]
                except torch.cuda.OutOfMemoryError:
                    logging.error(f"OOM at {i}")
                    d[model_name]['armoRM_scores'] = []

                    for message in message_list:
                        input_ids = tokenizer.apply_chat_template(message, return_tensors="pt").to(device)
                        with torch.no_grad():
                            output = model(input_ids)
                            score = output.score.cpu().float()
                            score = np.asarray(score).reshape(-1)[0]
                            d[model_name]['armoRM_scores'].append(float(score))

            f.write(json.dumps(d) + "\n")

            end_time = time.time()
            logging.info(f'{i}-th query done; avg {(end_time - start_time) / (i + 1):.2f} s / query')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument('--model_name_list', nargs='+', type=str, default=None)

    args = parser.parse_args()
    data = load_sample(fname=args.data_path, is_jsonl=True)
    generate_armoRM_scores(data, model_name_list=args.model_name_list)
