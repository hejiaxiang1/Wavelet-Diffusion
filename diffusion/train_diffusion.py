import os
from pathlib import Path
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import pywt
from ddpm import UNetModel, GaussianDiffusion

class ImageFolder(Dataset):
    def __init__(self, root, transform=None):
        splitdir = Path(root)
        if not splitdir.is_dir():
            raise RuntimeError(f'Invalid directory "{root}"')

        self.samples = [f for f in splitdir.iterdir() if f.is_file()]
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(self.samples[index]).convert("RGB")
        r, g, b = img.split()

        coeffs_r = pywt.dwt2(np.array(r), 'haar')
        coeffs_g = pywt.dwt2(np.array(g), 'haar')
        coeffs_b = pywt.dwt2(np.array(b), 'haar')

        cA_r, (cH_r, cV_r, cD_r) = coeffs_r
        cA_g, (cH_g, cV_g, cD_g) = coeffs_g
        cA_b, (cH_b, cV_b, cD_b) = coeffs_b

        reconstructed_high_freq_HH_r = pywt.idwt2((None, (None, None, cD_r)), 'haar')
        reconstructed_high_freq_HH_g = pywt.idwt2((None, (None, None, cD_g)), 'haar')
        reconstructed_high_freq_HH_b = pywt.idwt2((None, (None, None, cD_b)), 'haar')

        reconstructed_high_freq_imgs = [
            Image.merge('RGB', (
                Image.fromarray(reconstructed_high_freq_HH_r.astype(np.uint8)),
                Image.fromarray(reconstructed_high_freq_HH_g.astype(np.uint8)),
                Image.fromarray(reconstructed_high_freq_HH_b.astype(np.uint8))
            ))
        ]

        return self.transform(reconstructed_high_freq_imgs[0])

    def __len__(self):
        return len(self.samples)

def train_diffusion(data_path, epochs, batch_size, timesteps, lr):
    transform = transforms.Compose([
        transforms.RandomCrop((256, 256)),
        transforms.ToTensor(),
    ])

    dataset = ImageFolder(data_path, transform=transform)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNetModel(
        in_channels=3,
        model_channels=96,
        out_channels=3,
        channel_mult=(1, 2, 2),
        attention_resolutions=[]
    ).to(device)

    gaussian_diffusion = GaussianDiffusion(timesteps=timesteps)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_loss = float('inf')
    best_model_weights = None

    log_file = open("training_log.txt", "w")

    try:
        for epoch in range(epochs):
            total_loss = 0.0
            for step, images in enumerate(train_loader):
                optimizer.zero_grad()
                batch_size = images.shape[0]
                images = images.to(device)
                t = torch.randint(0, timesteps, (batch_size,), device=device).long()
                loss = gaussian_diffusion.train_losses(model, images, t)
                loss.backward()
                total_loss += loss.item()
                optimizer.step()

            average_loss = total_loss / len(train_loader)
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {average_loss:.4f}")
            log_file.write(f"Epoch [{epoch+1}/{epochs}], Loss: {average_loss:.4f}\n")

            if average_loss < best_loss:
                best_loss = average_loss
                best_model_weights = model.state_dict()
                torch.save(best_model_weights, f"diffusion_weight_epoch_{epoch+1}.pth")

    except Exception as e:
        print(f"Exception occurred during training: {str(e)}")
        log_file.write(f"Exception occurred during training: {str(e)}\n")

    finally:
        log_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train UNet with Gaussian Diffusion on images")
    parser.add_argument('--data_path', type=str, help='Path to the dataset')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--timesteps', type=int, default=10, help='Number of diffusion timesteps')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate for optimizer')
    args = parser.parse_args()

    train_diffusion(args.data_path, args.epochs, args.batch_size, args.timesteps, args.lr)
