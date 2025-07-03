import os
import time
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from studentcnn import StudentCNN  

# --- Configurations ---
INPUT_DIR = r"C:\Users\SAYON GHOSH\Desktop\prog\intel 2025\SwinIR\dataset\DIV2K_train_HR interpolated"
CHECKPOINT_PATH = r"C:\Users\SAYON GHOSH\Desktop\prog\intel 2025\SwinIR\studentcnn_checkpoints\epoch_49.pth"  
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load Model ---
model = StudentCNN().to(DEVICE)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
model.eval()

# --- Image Transform ---
transform = transforms.ToTensor()

# --- Load Images ---
image_files = sorted([f for f in os.listdir(INPUT_DIR) if f.lower().endswith(".png")])

# --- Inference Benchmark ---
total_time = 0.0
num_images = 0

print(f" Using device: {DEVICE}")

with torch.no_grad():
    for fname in image_files:
        img_path = os.path.join(INPUT_DIR, fname)
        img = Image.open(img_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(DEVICE)

        torch.cuda.synchronize() if DEVICE.type == 'cuda' else None
        start = time.time()

        output = model(img_tensor)

        torch.cuda.synchronize() if DEVICE.type == 'cuda' else None
        end = time.time()

        elapsed = end - start
        print(f"  {fname} inference time: {elapsed:.4f} sec")

        total_time += elapsed
        num_images += 1

# --- Final Report ---
avg_time = total_time / num_images
fps = 1 / avg_time

print("\n Benchmark Summary:")
print(f"Total Images: {num_images}")
print(f"Average Inference Time: {avg_time:.4f} sec/image")
print(f" Approximate FPS: {fps:.2f}")
