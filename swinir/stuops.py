import os
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
from studentcnn import StudentCNN  
import torch.nn.functional as F

# Paths
input_dir = r"C:\Users\SAYON GHOSH\Desktop\prog\intel 2025\SwinIR\dataset\DIV2K_train_HR interpolated"
output_dir = r"C:\Users\SAYON GHOSH\Desktop\prog\intel 2025\SwinIR\student_outputs"
checkpoint_path = r"C:\Users\SAYON GHOSH\Desktop\prog\intel 2025\SwinIR\studentcnn_checkpoints\epoch_49.pth"

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

# Load model
model = StudentCNN().to(device)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

# Transform
transform = transforms.ToTensor()

# Process images
image_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(".png")])

with torch.no_grad():
    for fname in image_files:
        img_path = os.path.join(input_dir, fname)
        img = Image.open(img_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)  # [1, 3, H, W]

        # Inference
        output = model(img_tensor)  # Expecting [1, 3, 2H, 2W]
        output = F.interpolate(output, scale_factor=2, mode='bicubic', align_corners=False)

        # Convert tensor to image
        output_img = output.squeeze(0).clamp(0, 1).cpu().numpy()
        output_img = np.transpose(output_img, (1, 2, 0)) * 255.0
        output_img = output_img.astype(np.uint8)

        output = output.squeeze().clamp(0, 1).cpu()

        # Save output
        output_img = transforms.ToPILImage()(output)
        output_img.save(os.path.join(output_dir, fname))

        print(f"✅ Saved: {fname} | Output shape: {output.shape}")
