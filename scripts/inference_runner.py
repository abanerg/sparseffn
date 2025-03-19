import scripts.inference_script
import torch
import gc

n_layers = 32

with open("./scripts/prompts.txt", 'r') as file:
            lines = file.readlines()

filepath = "/mnt/storage/spffn/metrics/inferences/mlp/"
for index in range(0,50):
    prompt = lines[index].strip()
    try:
        inference = scripts.inference_script.inference(prompt)
        torch.save(inference, f"{filepath}prompt_{index}.pt")
        if (index % 3 == 0):
            gc.collect()
    except Exception as e:
        print(f"{e} on index {index}")
