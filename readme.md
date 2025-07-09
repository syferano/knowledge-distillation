# Image Sharpening with Knowledge Distillation

This project implements an image sharpening technique using knowledge distillation. A lightweight convolutional neural network (StudentCNN) is trained to approximate the output of a high-performing model (SwinIR) on downscaled images. The student model is optimized for real-time performance and evaluated using SSIM metrics to compare its output against both the teacher model and bicubic interpolation.

Link to the report: https://drive.google.com/file/d/1eGgCaTa4JIhlExkX5eTsYI7xsyfXsIDq/view?usp=sharing
Link to the video: https://drive.google.com/file/d/1lUdczNihaXHFR9egkhmQd6k4N8D8TkoO/view?usp=drive_link

## Project Structure

- `studentcnn.py`: Training script for the StudentCNN using hybrid L1 + SSIM loss.
- `stuops.py`: Inference script for generating sharpened outputs from the trained student model.
- `studentcnn_checkpoints/`: Contains saved model weights from training.
- `dataset/`: Folder with interpolated and high-resolution training images.

## Model Details

- **Teacher Model**: SwinIR (pretrained), used to generate supervision signals.
- **Student Model**: Shallow CNN designed for speed and low memory usage.
- **Training Strategy**: Knowledge distillation with hybrid loss:
  - L1 loss (sharpness)
  - SSIM loss (structural similarity)
- **Optimizations**:
  - LR scheduling
  - Reduced image resolution
  - Tiling to avoid OOM issues
  - CUDA acceleration

## Results

| Method     | SSIM Score |
|------------|------------|
| SwinIR     | 0.9457     |
| StudentCNN | 0.9085     |
| Bicubic    | 0.8965     |

## Inference Speed

The student model supports real-time inference (~30 FPS) on 1080p images (1920×1080) when run with batch size 2 on an NVIDIA RTX 3050 Laptop GPU.

## How to Run

1. **Install Requirements**

   ```bash
   pip install torch torchvision tqdm pillow pytorch-msssim

2. **Train the Student Model**

   ```bash
   python studentcnn.py

3. **Run Inference**

   ```bash
   python stuops.py

4. **Generate Teacher Model Outputs**

   ```bash
   python teacherops.py

5. **Evaluate SSIM Scores**

   ```bash
   python ssim_eval.py


**Acknowledgements**

SwinIR: https://github.com/JingyunLiang/SwinIR

SSIM Loss: pytorch-msssim library
