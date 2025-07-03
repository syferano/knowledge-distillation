import os
from PIL import Image, ImageOps
from tqdm import tqdm

# === CONFIG ===
input_dir = r"A:\archive\DIV2K_valid_HR"
output_dir = r"A:\archive\DIV2K_valid_HR interpolated"
scale_factor = 2  # For 2x downscaling

# === Create output directory if it doesn't exist ===
os.makedirs(output_dir, exist_ok=True)

# === Get list of .png images ===
image_files = sorted([
    f for f in os.listdir(input_dir)
    if f.lower().endswith('.png')
])

# === Process each image ===
for filename in tqdm(image_files, desc="Downscaling Images"):
    input_path = os.path.join(input_dir, filename)
    output_path = os.path.join(output_dir, filename)

    try:
        # Open image and ensure it's RGB
        img = Image.open(input_path).convert("RGB")

        # Pad so dimensions are divisible by scale factor
        w, h = img.size
        pad_w = (scale_factor - w % scale_factor) % scale_factor
        pad_h = (scale_factor - h % scale_factor) % scale_factor
        padding = (0, 0, pad_w, pad_h)
        padded_img = ImageOps.expand(img, padding)

        # Resize (downscale)
        new_size = (padded_img.width // scale_factor, padded_img.height // scale_factor)
        downscaled_img = padded_img.resize(new_size, Image.BICUBIC)

        # Save the image
        downscaled_img.save(output_path)

    except Exception as e:
        print(f"Error processing {filename}: {e}")
