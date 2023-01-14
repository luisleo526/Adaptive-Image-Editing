from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image
import torch
import json

PRETRAINED_MODEL = "stabilityai/stable-diffusion-2-1"
#TOKEN = "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"
data_path = "./dataset.json"
num_images = 5

def load_json(json_path):
    with open(json_path, 'r') as f:
        result = json.load(f)
    return result

def load_prompts():
    raw_data = load_json(data_path)
    return raw_data

def generateImage(prompt):
    image = text2ImagePipe(prompt).images[0]
    return image

def main():
    global text2ImagePipe
    text2ImagePipe = StableDiffusionPipeline.from_pretrained(PRETRAINED_MODEL,
        #torch_dtype=torch.float16, 
        #revision="fp16",
        #use_auth_token=TOKEN
    )
    text2ImagePipe.scheduler = DPMSolverMultistepScheduler.from_config(text2ImagePipe.scheduler.config)
    text2ImagePipe.to(DEVICE)
    text2ImagePipe.enable_attention_slicing()
    data = load_prompts()
    print("load prompts done")
    data = [data[2], data[4]]
    for d in data:
        id = d["id"]
        prompt = d["prompt"]
        for i in range(num_images):
            img = generateImage(prompt)
            img.save(f"{prompt}/{id}-{i}.png")

if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(DEVICE)
    main()