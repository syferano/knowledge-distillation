import os
import requests

os.makedirs("experiments/pretrained_models", exist_ok=True)

url = "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth"
output_path = "experiments/pretrained_models/SwinIR-M_x4.pth"

print("Downloading SwinIR model...")
response = requests.get(url, stream=True)
with open(output_path, "wb") as f:
    for chunk in response.iter_content(chunk_size=8192):
        f.write(chunk)

print(" Done! Model saved at:", output_path)






