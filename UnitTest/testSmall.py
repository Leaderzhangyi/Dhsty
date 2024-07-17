import torch
from PIL import Image
from transformers import AutoModel, CLIPImageProcessor
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# obtained 69.1% zero-shot ImageNet score
model = AutoModel.from_pretrained(
    'jienengchen/ViTamin-S',
    trust_remote_code=True).to(device).eval()

image = Image.open('datasets/dh/testA/0.png').convert('RGB')

print(image)