import os
import argparse
import logging
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
import pywt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from tqdm import tqdm
from Unet import Unet

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImageFolder(Dataset):
    def __init__(self, root, transform=None):
        splitdir = Path(root)

        if not splitdir.is_dir():
            logger.error(f'Invalid directory "{root}"')
            raise RuntimeError(f'Invalid directory "{root}"')

        self.samples = [f for f in splitdir.iterdir() if f.is_file()]
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(self.samples[index])

        r, g, b = img.split()
        coeffs_r = pywt.dwt2(np.array(r), 'haar')
        coeffs_g = pywt.dwt2(np.array(g), 'haar')
        coeffs_b = pywt.dwt2(np.array(b), 'haar')

        cA_r, (cH_r, cV_r, cD_r) = coeffs_r
        cA_g, (cH_g, cV_g, cD_g) = coeffs_g
        cA_b, (cH_b, cV_b, cD_b) = coeffs_b
        
        reconstructed_low_freq_r = pywt.idwt2((cA_r, (None, None, None)), 'haar')
        reconstructed_low_freq_g = pywt.idwt2((cA_g, (None, None, None)), 'haar')
        reconstructed_low_freq_b = pywt.idwt2((cA_b, (None, None, None)), 'haar')
        
        reconstructed_high_freq_LH_r = pywt.idwt2((None, (cH_r, None, None)), 'haar')
        reconstructed_high_freq_HL_r = pywt.idwt2((None, (None, cV_r, None)), 'haar')
        reconstructed_high_freq_HH_r = pywt.idwt2((None, (None, None, cD_r)), 'haar')
        
        reconstructed_high_freq_LH_g = pywt.idwt2((None, (cH_g, None, None)), 'haar')
        reconstructed_high_freq_HL_g = pywt.idwt2((None, (None, cV_g, None)), 'haar')
        reconstructed_high_freq_HH_g = pywt.idwt2((None, (None, None, cD_g)), 'haar')
        
        reconstructed_high_freq_LH_b = pywt.idwt2((None, (cH_b, None, None)), 'haar')
        reconstructed_high_freq_HL_b = pywt.idwt2((None, (None, cV_b, None)), 'haar')
        reconstructed_high_freq_HH_b = pywt.idwt2((None, (None, None, cD_b)), 'haar')
        
        reconstructed_low_freq_img = Image.merge('RGB', (
            Image.fromarray(reconstructed_low_freq_r.astype(np.uint8)),
            Image.fromarray(reconstructed_low_freq_g.astype(np.uint8)),
            Image.fromarray(reconstructed_low_freq_b.astype(np.uint8))
        ))
        
        reconstructed_high_freq_imgs = [
            Image.merge('RGB', (
                Image.fromarray(reconstructed_high_freq_LH_r.astype(np.uint8)),
                Image.fromarray(reconstructed_high_freq_LH_g.astype(np.uint8)),
                Image.fromarray(reconstructed_high_freq_LH_b.astype(np.uint8))
            )),
            Image.merge('RGB', (
                Image.fromarray(reconstructed_high_freq_HL_r.astype(np.uint8)),
                Image.fromarray(reconstructed_high_freq_HL_g.astype(np.uint8)),
                Image.fromarray(reconstructed_high_freq_HL_b.astype(np.uint8))
            )),
            Image.merge('RGB', (
                Image.fromarray(reconstructed_high_freq_HH_r.astype(np.uint8)),
                Image.fromarray(reconstructed_high_freq_HH_g.astype(np.uint8)),
                Image.fromarray(reconstructed_high_freq_HH_b.astype(np.uint8))
            ))
        ]
        
        return self.transform(reconstructed_low_freq_img), self.transform(reconstructed_high_freq_imgs[0])

    def __len__(self):
        return len(self.samples)

def train_model(data_path, epochs, batch_size, learning_rate):
    try:
        transform = transforms.Compose([
            transforms.RandomCrop((256, 256)),
            transforms.ToTensor(),
        ])

        dataset = ImageFolder(data_path, transform=transform)
        train_size = int(0.9 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        model = Unet()
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        best_loss = float('inf')
        best_model_weights = None

        for epoch in range(epochs):
            model.train()
            total_loss = 0.0
            progress_bar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}]")

            for low, high in progress_bar:
                optimizer.zero_grad()
                outputs = model(low)
                loss = criterion(outputs, high)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})

            average_loss = total_loss / len(train_loader)
            logger.info(f"Epoch [{epoch+1}/{epochs}], Loss: {average_loss:.4f}")

            model.eval()
            with torch.no_grad():
                total_val_loss = 0.0
                val_progress_bar = tqdm(test_loader, desc="Validation")
                for inputs, targets in test_loader:
                    outputs = model(inputs)
                    val_loss = criterion(outputs, targets)
                    total_val_loss += val_loss.item()
                    val_progress_bar.set_postfix({"Validation Loss": f"{val_loss.item():.4f}"})

                average_val_loss = total_val_loss / len(test_loader)
                logger.info(f"Validation Loss: {average_val_loss:.4f}")

            if average_val_loss < best_loss:
                best_loss = average_val_loss
                best_model_weights = model.state_dict()

        if best_model_weights:
            torch.save(best_model_weights, "best_model_weights.pth")
            logger.info("Model saved as best_model_weights.pth")
        else:
            logger.warning("No model weights to save.")
    except Exception as e:
        logger.error(f"An error occurred during training: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a condition generation model on low and high frequency images")
    parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimizer')
    args = parser.parse_args()

    train_model(args.data_path, args.epochs, args.batch_size, args.learning_rate)
