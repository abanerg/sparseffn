import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1" # make sure we're using CUDA device 1

# Load tokenizer and model
model_name = "/mnt/storage/spffn/meta-llama/models--meta-llama--Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

# Set pad_token to eos_token if not already set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

prompt = "Once upon a time, in a faraway land, there was a dragon who"
inputs = tokenizer(prompt, return_tensors="pt", padding=True).to("cuda")

# Set pad_token_id to eos_token_id to avoid warnings
pad_token_id = tokenizer.eos_token_id

# Generate text
output = model.generate(
    inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    max_length=200,  # Adjust this for longer/shorter output
    temperature=0.7,  # Control randomness
    top_p=0.9,  # Nucleus sampling
    repetition_penalty=1.2,  # Penalize repetitive text
    pad_token_id=pad_token_id
)

# Decode the output
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)