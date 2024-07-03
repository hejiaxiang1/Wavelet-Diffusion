from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import pywt
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset

class ImageFolder(Dataset):
    def __init__(self, root, transform=None, split="train"):
        splitdir = Path(root) / split / "data"

        if not splitdir.is_dir():
            raise RuntimeError(f'Invalid directory "{root}"')

        self.samples = [f for f in splitdir.iterdir() if f.is_file()]

        self.transform = transform

    def __getitem__(self, index):

        img = Image.open(self.samples[index]).convert("RGB").convert("L")
        #r, g, b = img.split()

        #r_coeffs = pywt.dwt2(r,'db1', 'symmetric')
        #r_LL, (r_LH, r_HL, r_HH) = r_coeffs

        #g_coeffs = pywt.dwt2(g,'db1', 'symmetric')
        #g_LL, (g_LH, g_HL, g_HH) = g_coeffs

        #b_coeffs = pywt.dwt2(b,'db1', 'symmetric')
        #b_LL, (b_LH, b_HL, b_HH) = b_coeffs

        #low_coeffs=np.array([r_LL,g_LL,b_LL])
        #low_coeffs=low_coeffs.transpose(1,2,0)
        #low_coeffs=(low_coeffs/2)
        #low_coeffs=low_coeffs.astype(np.uint8)
        #low_img=Image.fromarray(low_coeffs)
        coeffs = pywt.dwt2(img,'haar', 'symmetric')
        LL, (LH, HL, HH) = coeffs
        test=np.array(LL)
        test=(test/2)
        test=test.astype(np.uint8)
        im=Image.fromarray(test)

        if self.transform:
            return self.transform(im)
        
        return im

    def __len__(self):
        return len(self.samples)