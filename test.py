import sys
import torch
import transformers

model_name = "meta-llama/Llama-3.2-1B"

tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
model = transformers.LlamaForCausalLM.from_pretrained(model_name)
config = transformers.LlamaConfig.from_pretrained(model_name)


activations = {}

# What do we want:
# For each layer ->
# Input token (from gate proj): Post SiLU (from act_fn) : Post Up : Post Down

def activation_hook(layer_num, stage):
    def hook(module, input, output):
        if stage == "gate_proj":
            activations[layer_num].append({})
            activations[layer_num][len(activations[layer_num])-1]["step"] = len(activations[layer_num])
            activations[layer_num][len(activations[layer_num])-1]["input tokens"] = torch.stack(input).clone()
        if stage == "act_fn":
            activations[layer_num][len(activations[layer_num])-1]["post silu"] = output.clone()
            # these are the indices that will be set to 0 for some value
        if stage == "up_proj":
            activations[layer_num][len(activations[layer_num])-1]["post up proj"] = output.clone()
        if stage == "down_proj":
            activations[layer_num][len(activations[layer_num])-1]["post down proj"] = output.clone()
    return hook

# Want input of gate proj, output of act_fn, output of down proj
valid_stages = ["gate_proj", "act_fn", "up_proj", "down_proj"]
for i in range(config.num_hidden_layers):
    activations[i] = []

for name, module in model.named_modules():
    terms = name.split(".")
    try:
        if (terms[4] in valid_stages):
            module.register_forward_hook(activation_hook(int(terms[2]), terms[4]))
    except:
        continue

text = sys.argv[1]
tokenizer.pad_token = tokenizer.eos_token
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
attention_mask = inputs["attention_mask"]
outputs = model.generate(inputs['input_ids'], attention_mask=attention_mask, max_length=100, num_return_sequences=1)
decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\n"+decoded_output)

n_layers = config.num_hidden_layers
layers = {x : {} for x in range(n_layers)}
for i in range(n_layers):
    # Chunk out initial ones
    chunked_token = torch.unbind(activations[i][0]["input tokens"] ,dim=activations[i][0]["input tokens"].dim() - 2)
    chunked_act = torch.unbind(activations[i][0]["post silu"] ,dim=activations[i][0]["post silu"].dim() - 2)
    chunked_post = torch.unbind(activations[i][0]["post down proj"] ,dim=activations[i][0]["post silu"].dim() - 2)
    for j in range(len(chunked_token)):
        layers[i][j] = {}
        layers[i][j]["A"] = chunked_token[j].reshape(-1)
        layers[i][j]["B"] = chunked_act[j].reshape(-1)
        layers[i][j]["Z"] = chunked_post[j].reshape(-1)
    # Just add the rest
    for j in range(1,len(activations[0])):
        curr = len(layers[i])
        layers[i][curr] = {} # add new key
        layers[i][curr]["A"] = activations[i][j]["input tokens"].reshape(-1)
        layers[i][curr]["B"] = activations[i][j]["post silu"].reshape(-1)
        layers[i][curr]["Z"] = activations[i][j]["post down proj"].reshape(-1)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_layer_heatmap(layers, i):
    layer_i = layers[i]
    all_B = [layer_i[j]["B"] for j in layer_i]  # Extract B values
    heatmap_data = np.array(all_B)

    # Define custom thresholds
    neg_threshold = np.percentile(heatmap_data, 10)  # Bottom 10% as "very negative"
    pos_threshold = np.percentile(heatmap_data, 98)  # Top 10% as "positive enough"

    # Custom colormap where:
    # - Values < neg_threshold are blue
    # - Values between neg_threshold and pos_threshold are neutral
    # - Values > pos_threshold are red
    cmap = sns.color_palette(["blue", "white", "red"], as_cmap=True)

    # Plot heatmap
    plt.figure(figsize=(12, 6))
    ax = sns.heatmap(heatmap_data, cmap=cmap, vmin=neg_threshold, vmax=pos_threshold, xticklabels=False, yticklabels=10)

    plt.xlabel("B vector indices")
    plt.ylabel("Layer passes (j)")
    plt.title(f"Custom Heatmap of B values for Layer {i+1}")

    plt.show()

plot_layer_heatmap(layers, 4)


# for i in range(len(layers)):
#     for j in range(len(layers[i])):
#         for s in ["A", "B", "Z"]:
#             print(f"i:{i} j:{j} s:{s}")
#             print(layers[i][j][s].shape)

# import matplotlib as mlp
# import matplotlib.pyplot as plt


# target_layer = 5
# target_trnsfrm = "act_fn"
# target_pass = 25 # NOTE: Make sure deep enough that we don't have multiple tokens

# neurons = activations[target_layer][target_trnsfrm][target_pass].reshape(-1)

# fig, ax = plt.subplots(1,1,figsize=(18,5))

# ax.plot([x for x in range(len(neurons))], neurons, label="Activation", color="blue")
# ax.set_xlabel("Neuron")
# ax.set_ylabel("Post SiLU activation")
# ax.set_title(f"Neuron activations post {target_trnsfrm} (Pass {target_pass}, layer {target_layer})")
# chunk = 500
# ax.set_xticks([chunk * x for x in range(1, int(len(neurons) / chunk) + 1)])
# ax.legend()

# plt.tight_layout()
# plt.show()