from pathlib import Path
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import pywt
import numpy as np
from collections import defaultdict
from typing import List
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from diffusion.Unet import Unet
from pytorch_msssim import ms_ssim
from torchvision import transforms
import compressai
from diffusion.ddpm import UNetModel,GaussianDiffusion
from waveletencoder.zoo import load_state_dict, models

class ImageFolder(Dataset):
    def __init__(self, root, transform=None, split="train",wavelet_model=None, 
                                LH_model=None, HL_model=None, HH_model=None, 
                                diffusion_model=None,diffusion_model_LH=None, diffusion_model_HL=None, diffusion_model_HH=None):
        splitdir = Path(root) / split / "data"
        if not splitdir.is_dir():
            raise RuntimeError(f'Invalid directory "{root}"')
        self.samples = [f for f in splitdir.iterdir() if f.is_file()]
        self.transform = transform
        self.wavelet_model=wavelet_model
        self.LH_model=LH_model
        self.HL_model=HL_model
        self.HH_model=HH_model
        self.diffusion_model=diffusion_model
        self.diffusion_model_LH=diffusion_model_LH
        self.diffusion_model_HL=diffusion_model_HL
        self.diffusion_model_HH=diffusion_model_HH
        
    def __getitem__(self, index):

        img = Image.open(self.samples[index]).convert("RGB").convert("L")
        coeffs = pywt.dwt2(img,'haar', 'symmetric')
        LL, (LH, HL, HH) = coeffs

        # Conversion of wavelet coefficients to img
        test_LL=np.array(LL)
        test_LL=(test_LL/2)
        test_LL=test_LL.astype(np.uint8)
        im_LL=Image.fromarray(test_LL)

        test_LH=np.array(LH)
        test_LH=test_LH+255
        test_LH=(test_LH/2)
        test_LH=test_LH.astype(np.uint8)
        im_LH=Image.fromarray(test_LH)

        test_HL=np.array(HL)
        test_HL=test_HL+255
        test_HL=(test_HL/2)
        test_HL=test_HL.astype(np.uint8)
        im_HL=Image.fromarray(test_HL)

        test_HH=np.array(HH)
        test_HH=test_HH+255
        test_HH=(test_HH/2)
        test_HH=test_HH.astype(np.uint8)
        im_HH=Image.fromarray(test_HH)

        """
        x_LL = transforms.ToTensor()(im_LL).unsqueeze(0).to("cuda")
        out_enc = self.wavelet_model.compress(x_LL)
        t = torch.randint(2, 3, (1,), device="cuda").long()
        x_LL_recons = self.wavelet_model.decompress(out_enc["strings"], out_enc["shape"])

        LH_reflect = self.LH_model(x_LL_recons['x_hat'])
        HL_reflect = self.HL_model(x_LL_recons['x_hat'])
        HH_reflect = self.HH_model(x_LL_recons['x_hat'])

        x_start_HL = HL_reflect
        HL_noise = torch.randn_like(x_start_HL)
        HL_noisy = self.diffusion_model.q_sample(x_start_HL, t, noise=HL_noise)
        predicted_HL_noise = self.HL_diffusion_model(HL_noisy, t)
        clear_HL = self.diffusion_model.predict_start_from_noise(HL_noisy, t, predicted_HL_noise)

        x_start_LH = LH_reflect
        LH_noise = torch.randn_like(x_start_LH)
        LH_noisy = self.diffusion_model.q_sample(x_start_LH, t, noise=LH_noise)
        predicted_LH_noise = self.LH_diffusion_model(LH_noisy, t)
        clear_LH = self.diffusion_model.predict_start_from_noise(LH_noisy, t, predicted_LH_noise)

        x_start_HH = HH_reflect
        HH_noise = torch.randn_like(x_start_HH)
        HH_noisy = self.diffusion_model.q_sample(x_start_HH, t, noise=HH_noise)
        predicted_HH_noise = self.HH_diffusion_model(HH_noisy, t)
        clear_HH = self.diffusion_model.predict_start_from_noise(HH_noisy, t, predicted_HH_noise)

        LH_reflect_recons = transforms.ToPILImage()(clear_LH.squeeze().cpu())
        LH_reflect_recons = np.array(LH_reflect_recons)
        LH_reflect_recons = LH_reflect_recons.astype(np.float64)
        LH_reflect_recons=(LH_reflect_recons*2)
        LH_reflect_recons=(LH_reflect_recons-255)/64

        HL_reflect_recons = transforms.ToPILImage()(clear_HL.squeeze().cpu())
        HL_reflect_recons = np.array(HL_reflect_recons)
        HL_reflect_recons = HL_reflect_recons.astype(np.float64)
        HL_reflect_recons=(HL_reflect_recons*2)
        HL_reflect_recons=(HL_reflect_recons-255)/64

        HH_reflect_recons = transforms.ToPILImage()(clear_HH.squeeze().cpu())
        HH_reflect_recons = np.array(HH_reflect_recons)
        HH_reflect_recons = HH_reflect_recons.astype(np.float64)
        HH_reflect_recons=(HH_reflect_recons*2)
        HH_reflect_recons=(HH_reflect_recons-255)/64

        r_LH = LH - LH_reflect_recons
        r_HL = HL - HL_reflect_recons
        r_HH = HH - HH_reflect_recons
        """
        test=np.array([LH,HL,HH]).transpose(1,2,0)
        test=test+255
        test=(test/2)
        test=test.astype(np.uint8)
        im=Image.fromarray(test)

        if self.transform:
            return self.transform(im)
        return im

    def __len__(self):
        return len(self.samples)