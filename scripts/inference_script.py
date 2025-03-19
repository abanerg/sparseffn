import sys
import torch
import transformers
import os
import gc
from copy import deepcopy


model_name = "/mnt/storage/spffn/meta-llama/models--meta-llama--Llama-3.1-8B-Instruct"
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
model = transformers.LlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
config = transformers.LlamaConfig.from_pretrained(model_name)

def inference(prompt: str):
    gc.collect()
    activations = {}

    for i in range(config.num_hidden_layers):
        activations[i] = []

    def activation_hook(layer_num, stage, activations):
        def hook(module, input, output):
            if stage == "q_proj":
                activations[layer_num].append({})
            activations[layer_num][len(activations[layer_num])-1][f"{stage} input"]=(torch.stack(input).clone().cpu().reshape(-1))
            activations[layer_num][len(activations[layer_num])-1][f"{stage} output"]=(output.clone().cpu().reshape(-1))
        return hook
    hooks = []
    for name, module in model.named_modules():
        terms = name.split(".")
        try:
            if (len(terms) == 5 and (terms[3] == "mlp" or terms[4] == "q_proj")):
                hook = module.register_forward_hook(activation_hook(int(terms[2]), terms[4], activations))
                hooks.append(hook)
        except:
            continue

    pad_token_id = tokenizer.eos_token_id
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    text = prompt
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True,padding_side='left').to("cuda")
    attention_mask = inputs["attention_mask"]
    prompt_length = inputs['input_ids'].shape[1]
    outputs = model.generate(inputs['input_ids'],
                             attention_mask=attention_mask,
                             max_length=500,
                             temperature=0.7,
                             top_p=0.9,
                             repetition_penalty=1.2,
                             pad_token_id=pad_token_id,
                             num_return_sequences=1)
    for i in range(32):
        activations[i] = activations[i][1:]

    for hook in hooks:
        hook.remove()
    torch.cuda.empty_cache()
    return activations