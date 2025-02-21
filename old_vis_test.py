import requests
import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, MllamaProcessor

model_id = "meta-llama/Llama-3.2-11B-Vision"

model = MllamaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
processor = MllamaProcessor.from_pretrained(model_id)

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg"
image = Image.open(requests.get(url, stream=True).raw)

prompt = "<|begin_of_text|> Here <|image|>If I had to write a haiku for this one"
inputs = processor(image, prompt, return_tensors="pt").to(model.device)

with torch.profiler.profile(use_cuda=false) as prof:
    output = model.generate(**inputs, max_new_tokens=5)
    print(processor.decode(output[0]))

prof.export_chrome_trace("profile.json")