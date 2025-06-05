import transformers
import torch
import argparse
from huggingface_hub import login
import json
import time
from utils import load_sample
import logging
import os

logging.basicConfig(level=logging.INFO)
logging.info("Logging started")

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Please specify your model name and the corresponding model path
model2path = {
    'llama-31-8b': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
}

def generate_llm_response(data, model_name, num_sample=20):
    # login to Hugging Face Hub
    login('YOUR_HUGGINGFACE_TOKEN')

    model_id = model2path[model_name]
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
        trust_remote_code=True,
    )
    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    file_path = f"./outputs/mixed_dataset_{model_name}.jsonl"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    start_time = time.time()
    with open(file_path, 'w') as f:
        for i, d in enumerate(data):
            user_input = d['prompt']
            request_data = [
                {
                    "role": "user",
                    "content": user_input
                },
            ]
            outputs = pipeline(
                request_data,
                max_new_tokens=500,
                eos_token_id=terminators,
                pad_token_id=pipeline.tokenizer.eos_token_id,
                temperature=1,
                top_p=1,
                num_return_sequences=num_sample,            # we use 10 samples in hybrid-llm and extend to 20 in best-route
                do_sample=True,
            )

            response = [_['generated_text'][-1]['content'] for _ in outputs]
            d[model_name] = {'responses': response}
            f.write(json.dumps(d) + "\n")
            end_time = time.time()
            logging.info(f'{i}-th query done; avg {(end_time - start_time) / (i + 1):.2f} s / query')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--model_name", type=str, default='llama-31-8b')
    parser.add_argument("--num_sample", type=int, default=20)

    args = parser.parse_args()
    data = load_sample(fname=args.data_path, is_jsonl=True)
    generate_llm_response(data, model_name=args.model_name, num_sample=args.num_sample)