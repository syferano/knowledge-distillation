import os
from PIL import Image
from torchvision import transforms
from pytorch_msssim import ssim
import torch
from tqdm import tqdm

# === Paths ===
hr_dir = r"C:\Users\SAYON GHOSH\Desktop\prog\intel 2025\SwinIR\dataset\DIV2K_train_HR"
bicubic_dir = r"C:\Users\SAYON GHOSH\Desktop\prog\intel 2025\SwinIR\dataset\DIV2K_train_HR interpolated"
teacher_dir = r"C:\Users\SAYON GHOSH\Desktop\prog\intel 2025\SwinIR\teacher_outputs"
student_dir = r"C:\Users\SAYON GHOSH\Desktop\prog\intel 2025\SwinIR\student_outputs"

# === Device ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Transforms ===
to_tensor = transforms.ToTensor()

# === Results ===
ssim_scores = {"teacher": [], "student": [], "bicubic": []}

# === File List ===
image_filenames = sorted([f for f in os.listdir(hr_dir) if f.endswith(".png")])

for filename in tqdm(image_filenames, desc="Computing SSIM"):
    # Load and convert all images to tensor
    gt = to_tensor(Image.open(os.path.join(hr_dir, filename)).convert("RGB")).unsqueeze(0).to(device)
    student = to_tensor(Image.open(os.path.join(student_dir, filename)).convert("RGB")).unsqueeze(0).to(device)
    teacher = to_tensor(Image.open(os.path.join(teacher_dir, filename)).convert("RGB")).unsqueeze(0).to(device)
    bicubic_img = Image.open(os.path.join(bicubic_dir, filename)).convert("RGB")
    bicubic_img = bicubic_img.resize(gt.shape[-2:][::-1], Image.BICUBIC)  # Match HR size UPSCALING
    bicubic = to_tensor(bicubic_img).unsqueeze(0).to(device)

    # Compute SSIM
    ssim_scores["student"].append(ssim(student, gt, data_range=1.0).item())
    ssim_scores["teacher"].append(ssim(teacher, gt, data_range=1.0).item())
    ssim_scores["bicubic"].append(ssim(bicubic, gt, data_range=1.0).item())

# === Averages ===
print("\n Average SSIM Scores:")
for key in ssim_scores:
    avg = sum(ssim_scores[key]) / len(ssim_scores[key])
    print(f"{key.capitalize()} SSIM: {avg:.4f}")
