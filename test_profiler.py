import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.profiler import profile, record_function, ProfilerActivity

# Load tokenizer and model
model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

# Extract FFN layers (assuming the model uses a standard transformer architecture)
ffn_layers = []
for layer in model.model.layers:
    ffn_layers.append(layer.mlp)

# Print FFN weights and biases
for i, ffn in enumerate(ffn_layers):
    print(f"FFN Layer {i} Weights and Biases:")
    print("Weights:", ffn.gate_proj.weight)
    print("Biases:", ffn.gate_proj.bias)
    print("Weights:", ffn.down_proj.weight)
    print("Biases:", ffn.down_proj.bias)
    print("Weights:", ffn.up_proj.weight)
    print("Biases:", ffn.up_proj.bias)

prompt = "Once upon a time, in a faraway land, there was a dragon who"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

# Profile the FFN parts of the model
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    with record_function("ffn_inference"):
        output = model.generate(
            inputs["input_ids"],
            max_length=100,  # Adjust this for longer/shorter output
            temperature=0.7,  # Control randomness
            top_p=0.9,  # Nucleus sampling
            repetition_penalty=1.2,  # Penalize repetitive text
        )

# Decode the output
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)

# Print the profiling results
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))