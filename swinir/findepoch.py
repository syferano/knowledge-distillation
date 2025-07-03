import os
import torch

# Path to your saved checkpoints
CHECKPOINT_DIR = r"C:\Users\SAYON GHOSH\Desktop\student_checkpoints"

best_loss = float('inf')
best_epoch = None
best_path = None

for filename in os.listdir(CHECKPOINT_DIR):
    if filename.endswith(".pth") and filename.startswith("epoch_"):
        path = os.path.join(CHECKPOINT_DIR, filename)
        try:
            checkpoint = torch.load(path, map_location='cpu')
            if 'loss' in checkpoint and 'epoch' in checkpoint:
                loss = checkpoint['loss']
                epoch = checkpoint['epoch']
                print(f"Epoch {epoch}: Loss = {loss:.4f}")
                if loss < best_loss:
                    best_loss = loss
                    best_epoch = epoch
                    best_path = path
            else:
                print(f"  Skipped {filename} (no loss info)")
        except Exception as e:
            print(f"❌ Failed to load {filename}: {e}")

if best_path:
    print("\n Best Model Found:")
    print(f"   Epoch: {best_epoch}")
    print(f"   Loss : {best_loss:.4f}")
    print(f"   Path : {best_path}")
else:
    print(" No valid checkpoints with loss found.")
