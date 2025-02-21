import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the model and tokenizer

model_name = "meta-llama/Llama-3.2-1B"  # Replace with your model name
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Dictionary to store activations
activations = {}

# Hook function to capture post-SiLU activations
def activation_hook(layer_name):
    def hook(module, input, output):
        silu_activated = torch.nn.functional.silu(output)  # Apply SiLU
        activations[layer_name] = silu_activated.detach().cpu()  # Store activation
    return hook

# Register hooks on all gate_proj layers
hooks = []
for i, layer in enumerate(model.model.layers):  # Assuming "model.model.layers"
    hook = layer.mlp.gate_proj.register_forward_hook(activation_hook(f"layer_{i}_gate_proj"))
    hooks.append(hook)

# Tokenize input text
input_text = "The future of AI is"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids

# Generate text while logging activations
with torch.no_grad():  # Disable gradients
    output = model.generate(input_ids, max_length=50)

# Remove hooks to avoid memory leaks
for hook in hooks:
    hook.remove()

# Compute sparsity
threshold = 1e-5  # Define near-zero threshold
sparsity_results = {}

for layer, act in activations.items():
    sparse_count = (torch.abs(act) < threshold).sum().item()  # Count near-zero activations
    total_count = act.numel()  # Total activations
    sparsity_ratio = sparse_count / total_count  # Compute sparsity percentage
    sparsity_results[layer] = sparsity_ratio

# Print sparsity results
for layer, sparsity in sparsity_results.items():
    print(f"{layer}: {sparsity:.4f} sparsity")  # Print ratio as percentage
