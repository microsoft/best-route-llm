import json

def load_sample(fname, is_jsonl=True):
    if is_jsonl:
        with open(fname, encoding="utf8") as json_file:
            data = [json.loads(f) for f in list(json_file)]
    else:
        with open(fname) as json_file:
            data = json.load(json_file)
    return data

model2prompt_cost = {'gpt-4o': 5e-6, 'gpt-35-turbo': 3e-6, 'phi-3-mini': 0.3e-6, 'phi-3-medium': 0.5e-6, 'mistral-7b': 0.25e-6, 'mistral-8x7b': 0.7e-6, 'llama-31-8b': 0.3e-6, 'codestral-22b': 1e-6}
model2output_cost = {'gpt-4o': 15e-6, 'gpt-35-turbo': 6e-6, 'phi-3-mini': 0.9e-6, 'phi-3-medium': 1.5e-6, 'mistral-7b': 0.25e-6, 'mistral-8x7b': 0.7e-6, 'llama-31-8b': 0.61e-6, 'codestral-22b': 3e-6}

model2prompt_length = {'gpt-4o': 42.311625, 'gpt-35-turbo': 42.55575, 'phi-3-mini': 50.441, 'phi-3-medium': 50.441, 'mistral-7b': 50.488, 'mistral-8x7b': 50.488, 'llama-31-8b': 43.529, 'codestral-22b': 50.488}
model2output_length = {'gpt-4o': 294.04923125, 'gpt-35-turbo': 131.56565625, 'phi-3-mini': 222.9136375, 'phi-3-medium': 240.73091875, 'mistral-7b': 296.2619125, 'mistral-8x7b': 267.91186875, 'llama-31-8b': 298.33180625, 'codestral-22b': 270.774275}