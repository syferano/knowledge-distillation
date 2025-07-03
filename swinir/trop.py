import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

from models.network_swinir import SwinIR  

# === CONFIG ===
lr_dir = r"C:\Users\SAYON GHOSH\Desktop\prog\intel 2025\SwinIR\dataset\DIV2K_train_HR interpolated"
output_dir = r"C:\Users\SAYON GHOSH\Desktop\prog\intel 2025\SwinIR\teacher_outputs"
model_path = r"C:\Users\SAYON GHOSH\Desktop\prog\intel 2025\SwinIR\experiments\pretrained_models\SwinIR-M_x2.pth"

tile = 128
tile_overlap = 32

os.makedirs(output_dir, exist_ok=True)

# === DEVICE ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === TRANSFORM ===
transform = transforms.ToTensor()

# === LOAD MODEL ===
model = SwinIR(
    upscale=2,
    in_chans=3,
    img_size=48,
    window_size=8,
    img_range=1.0,
    depths=[6, 6, 6, 6, 6, 6],
    embed_dim=180,
    num_heads=[6, 6, 6, 6, 6, 6],
    mlp_ratio=2,
    upsampler='pixelshuffle',
    resi_connection='1conv'
).to(device)

pretrained_model = torch.load(model_path, map_location=device)
model.load_state_dict(pretrained_model['params'], strict=True)
model.eval()

# === HELPER: TILE-WISE INFERENCE ===
def tile_forward(img_tensor, model, tile=128, tile_overlap=32, scale=2):
    b, c, h, w = img_tensor.size()
    sf = scale
    stride = tile - tile_overlap

    h_idx_list = list(range(0, h - tile, stride)) + [h - tile]
    w_idx_list = list(range(0, w - tile, stride)) + [w - tile]
    E = torch.zeros((b, c, h * sf, w * sf), dtype=img_tensor.dtype, device=img_tensor.device)
    W = torch.zeros_like(E)

    for h_idx in h_idx_list:
        for w_idx in w_idx_list:
            in_patch = img_tensor[..., h_idx:h_idx + tile, w_idx:w_idx + tile]
            out_patch = model(in_patch)
            out_patch_mask = torch.ones_like(out_patch)

            E[..., h_idx * sf:(h_idx + tile) * sf, w_idx * sf:(w_idx + tile) * sf] += out_patch
            W[..., h_idx * sf:(h_idx + tile) * sf, w_idx * sf:(w_idx + tile) * sf] += out_patch_mask

    output = E / W
    return output

# === INFERENCE ===
image_files = sorted([f for f in os.listdir(lr_dir) if f.lower().endswith(('.png', '.jpg'))])

with torch.no_grad():
    for fname in tqdm(image_files, desc="Generating teacher outputs"):
        img_path = os.path.join(lr_dir, fname)
        img = Image.open(img_path).convert("RGB")
        lr_tensor = transform(img).unsqueeze(0).to(device)

        # use tile-based inference
        sr_tensor = tile_forward(lr_tensor, model, tile=tile, tile_overlap=tile_overlap, scale=2)
        sr_tensor = sr_tensor.squeeze(0).cpu().clamp(0, 1)

        sr_image = transforms.ToPILImage()(sr_tensor)
        sr_image.save(os.path.join(output_dir, fname))

print(f"✅ Saved {len(image_files)} teacher outputs to: {output_dir}")
