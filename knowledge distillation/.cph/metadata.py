import os
import csv
from PIL import Image
from tqdm import tqdm

# === CONFIG ===
image_dir = r"A:\archive\DIV2K_train_HR interpolated"
metadata_csv = os.path.join(image_dir, "metadata.csv")

# === Get list of images ===
image_files = sorted([
    f for f in os.listdir(image_dir)
    if f.lower().endswith('.png')
])

# === Write metadata ===
with open(metadata_csv, mode='w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Filename', 'Width', 'Height', 'Path'])

    for filename in tqdm(image_files, desc="Writing Metadata"):
        image_path = os.path.join(image_dir, filename)
        try:
            with Image.open(image_path) as img:
                width, height = img.size
                writer.writerow([filename, width, height, image_path])
        except Exception as e:
            print(f"Failed to read {filename}: {e}")
    