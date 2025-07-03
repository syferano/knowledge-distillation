import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import pytorch_msssim

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Constants
DATA_DIR = "C:/Users/SAYON GHOSH/Desktop/prog/intel 2025/SwinIR/dataset"
SAVE_DIR = "C:/Users/SAYON GHOSH/Desktop/prog/intel 2025/SwinIR/studentcnn_checkpoints"
NUM_EPOCHS = 50
BATCH_SIZE = 1  # Reduce to avoid OOM
LEARNING_RATE = 1e-4
TRAIN_IMG_SIZE = (720, 480)  # Downscale during training
FULL_IMG_SIZE = (2040, 1404)

# Transforms for training (downscaled)
transform = transforms.Compose([
    transforms.Resize(TRAIN_IMG_SIZE),
    transforms.ToTensor(),
])

# Dataset class
class StudentDataset(Dataset):
    def __init__(self, lr_dir, hr_dir, transform=None):
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.transform = transform
        self.lr_paths = sorted([os.path.join(lr_dir, f) for f in os.listdir(lr_dir) if f.endswith('.png')])
        self.hr_paths = sorted([os.path.join(hr_dir, f) for f in os.listdir(hr_dir) if f.endswith('.png')])

    def __len__(self):
        return len(self.lr_paths)

    def __getitem__(self, idx):
        lr_img = Image.open(self.lr_paths[idx]).convert("RGB")
        hr_img = Image.open(self.hr_paths[idx]).convert("RGB")
        if lr_img.size != hr_img.size:
            lr_img = lr_img.resize(hr_img.size, Image.BICUBIC)
        if self.transform:
            lr_img = self.transform(lr_img)
            hr_img = self.transform(hr_img)
        return lr_img, hr_img

# Improved CNN model
class StudentCNN(nn.Module):
    def __init__(self):
        super(StudentCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # Extra layer
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return self.net(x)

# Hybrid loss function
l1_loss = nn.L1Loss()
ssim_module = pytorch_msssim.SSIM(win_size=11)

def hybrid_loss(output, target, alpha=0.85):
    l1 = l1_loss(output, target)
    ssim_val = ssim_module(output, target)
    return alpha * l1 + (1 - alpha) * (1 - ssim_val)

# Training function
def train():
    dataset = StudentDataset(
        hr_dir="C:/Users/SAYON GHOSH/Desktop/prog/intel 2025/SwinIR/dataset/DIV2K_train_HR",
        lr_dir="C:/Users/SAYON GHOSH/Desktop/prog/intel 2025/SwinIR/dataset/DIV2K_train_HR interpolated",
        transform=transform
    )
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    model = StudentCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    os.makedirs(SAVE_DIR, exist_ok=True)

    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0.0
        for lr_imgs, hr_imgs in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)

            outputs = model(lr_imgs)
            loss = hybrid_loss(outputs, hr_imgs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # Optionally clear cache to avoid OOM
            torch.cuda.empty_cache()

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

        checkpoint_path = os.path.join(SAVE_DIR, f"epoch_{epoch}.pth")
        torch.save(model.state_dict(), checkpoint_path)

if __name__ == "__main__":
    train()
