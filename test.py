import sys
import torch
import transformers

model_name = "meta-llama/Llama-3.2-1B"

tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
model = transformers.LlamaForCausalLM.from_pretrained(model_name)
config = transformers.LlamaConfig.from_pretrained(model_name)


activations = {}

def activation_hook(layer_num, step):
    def hook(module, input, output):
        length = output.numel
        cl = output.clone()
        # print(cl.shape)
        activations[layer_num][step].append(cl)
    return hook

valid_steps = ["act_fn", "up_proj", "down_proj"]
for i in range(config.num_hidden_layers):
    activations[i] = {}
    for step in valid_steps:
        activations[i][step] = []

for name, module in model.named_modules():
    terms = name.split(".")
    try:
        if (terms[4] in valid_steps):
            module.register_forward_hook(activation_hook(int(terms[2]), terms[4]))
    except:
        continue

text = sys.argv[1]
tokenizer.pad_token = tokenizer.eos_token
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
attention_mask = inputs["attention_mask"]
outputs = model.generate(inputs['input_ids'], attention_mask=attention_mask, max_length=75, num_return_sequences=1)
decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\n"+decoded_output)

import matplotlib as mlp
import matplotlib.pyplot as plt


target_layer = 5
target_trnsfrm = "act_fn"
target_pass = 25 # NOTE: Make sure deep enough that we don't have multiple tokens

neurons = activations[target_layer][target_trnsfrm][target_pass].reshape(-1)

fig, ax = plt.subplots(1,1,figsize=(18,5))

ax.plot([x for x in range(len(neurons))], neurons, label="Activation", color="blue")
ax.set_xlabel("Neuron")
ax.set_ylabel("Post SiLU activation")
ax.set_title(f"Neuron activations post {target_trnsfrm} (Pass {target_pass}, layer {target_layer})")
chunk = 500
ax.set_xticks([chunk * x for x in range(1, int(len(neurons) / chunk) + 1)])
ax.legend()

plt.tight_layout()
plt.show()