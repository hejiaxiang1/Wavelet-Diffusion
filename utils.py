import torch
import torch.nn as nn
import torch
import os 
import numpy as np
import torch
import pywt
import numpy as np
from PIL import Image
import pywt
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from PIL import Image

import numpy as np
import pywt
from PIL import Image

def wavelet_decomposition_4image(image):
    r, g, b = image.split()
    
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
    return reconstructed_low_freq_img, reconstructed_high_freq_imgs[0], reconstructed_high_freq_imgs[1], reconstructed_high_freq_imgs[2]

def wavelet_decomposition_4image2(image):
    r, g, b = image.split()

    r_coeffs = pywt.dwt2(r, 'haar', 'symmetric')
    r_LL, (r_LH, r_HL, r_HH) = r_coeffs
    r_LL = (np.array(r_LL) / 2).astype(np.uint8)
    r_LH = ((np.array(r_LH) + 255) / 2).astype(np.uint8)
    r_HL = ((np.array(r_HL) + 255) / 2).astype(np.uint8)
    r_HH = ((np.array(r_HH) + 255) / 2).astype(np.uint8)
    r_LL = Image.fromarray(r_LL)
    r_LH = Image.fromarray(r_LH)
    r_HL = Image.fromarray(r_HL)
    r_HH = Image.fromarray(r_HH)

    g_coeffs = pywt.dwt2(g, 'haar', 'symmetric')
    g_LL, (g_LH, g_HL, g_HH) = g_coeffs
    g_LL = (np.array(g_LL) / 2).astype(np.uint8)
    g_LH = ((np.array(g_LH) + 255) / 2).astype(np.uint8)
    g_HL = ((np.array(g_HL) + 255) / 2).astype(np.uint8)
    g_HH = ((np.array(g_HH) + 255) / 2).astype(np.uint8)
    g_LL = Image.fromarray(g_LL)
    g_LH = Image.fromarray(g_LH)
    g_HL = Image.fromarray(g_HL)
    g_HH = Image.fromarray(g_HH)

    b_coeffs = pywt.dwt2(b, 'haar', 'symmetric')
    b_LL, (b_LH, b_HL, b_HH) = b_coeffs
    b_LL = (np.array(b_LL) / 2).astype(np.uint8)
    b_LH = ((np.array(b_LH) + 255) / 2).astype(np.uint8)
    b_HL = ((np.array(b_HL) + 255) / 2).astype(np.uint8)
    b_HH = ((np.array(b_HH) + 255) / 2).astype(np.uint8)
    b_LL = Image.fromarray(b_LL)
    b_LH = Image.fromarray(b_LH)
    b_HL = Image.fromarray(b_HL)
    b_HH = Image.fromarray(b_HH)
    low_freq_img = Image.merge('RGB', (r_LL,g_LL,b_LL))
    LH_image = Image.merge('RGB', (r_LH,g_LH,b_LH))
    HL_image = Image.merge('RGB', (r_HL,g_HL,b_HL))
    HH_image = Image.merge('RGB', (r_HH,g_HH,b_HH))

    return low_freq_img,LH_image,HL_image,HH_image

def wavelet_decomposition_12terms(image):
    r, g, b = image.split()

    r_coeffs = pywt.dwt2(r, 'haar', 'symmetric')
    r_LL, (r_LH, r_HL, r_HH) = r_coeffs
    r_LL = (np.array(r_LL) / 2).astype(np.uint8)
    r_LH = ((np.array(r_LH) + 255) / 2).astype(np.uint8)
    r_HL = ((np.array(r_HL) + 255) / 2).astype(np.uint8)
    r_HH = ((np.array(r_HH) + 255) / 2).astype(np.uint8)

    g_coeffs = pywt.dwt2(g, 'haar', 'symmetric')
    g_LL, (g_LH, g_HL, g_HH) = g_coeffs
    g_LL = (np.array(g_LL) / 2).astype(np.uint8)
    g_LH = ((np.array(g_LH) + 255) / 2).astype(np.uint8)
    g_HL = ((np.array(g_HL) + 255) / 2).astype(np.uint8)
    g_HH = ((np.array(g_HH) + 255) / 2).astype(np.uint8)


    b_coeffs = pywt.dwt2(b, 'haar', 'symmetric')
    b_LL, (b_LH, b_HL, b_HH) = b_coeffs
    b_LL = (np.array(b_LL) / 2).astype(np.uint8)
    b_LH = ((np.array(b_LH) + 255) / 2).astype(np.uint8)
    b_HL = ((np.array(b_HL) + 255) / 2).astype(np.uint8)
    b_HH = ((np.array(b_HH) + 255) / 2).astype(np.uint8)

    return r_LL,r_LH,r_HL,r_HH,g_LL,g_LH,g_HL,g_HH,b_LL,b_LH,b_HL,b_HH

def wavelet_recomposition_12terms(r_LL, r_LH, r_HL, r_HH, g_LL, g_LH, g_HL, g_HH, b_LL, b_LH, b_HL, b_HH):

    r_LL = (np.array(r_LL).astype(np.float64) * 2)
    r_LH = ((np.array(r_LH).astype(np.float64) * 2) - 255)
    r_HL = ((np.array(r_HL).astype(np.float64) * 2) - 255)
    r_HH = ((np.array(r_HH).astype(np.float64) * 2) - 255)

    g_LL = (np.array(g_LL).astype(np.float64) * 2)
    g_LH = ((np.array(g_LH).astype(np.float64) * 2) - 255)
    g_HL = ((np.array(g_HL).astype(np.float64) * 2) - 255)
    g_HH = ((np.array(g_HH).astype(np.float64) * 2) - 255)

    b_LL = (np.array(b_LL).astype(np.float64) * 2)
    b_LH = ((np.array(b_LH).astype(np.float64) * 2) - 255)
    b_HL = ((np.array(b_HL).astype(np.float64) * 2) - 255)
    b_HH = ((np.array(b_HH).astype(np.float64) * 2) - 255)

    r_reconstructed = pywt.idwt2((r_LL, (r_LH, r_HL, r_HH)), 'haar')

    g_reconstructed = pywt.idwt2((g_LL, (g_LH, g_HL, g_HH)), 'haar')

    b_reconstructed = pywt.idwt2((b_LL, (b_LH, b_HL, b_HH)), 'haar')

    reconstructed_image = Image.merge('RGB', (
        Image.fromarray(r_reconstructed.astype(np.uint8)),
        Image.fromarray(g_reconstructed.astype(np.uint8)),
        Image.fromarray(b_reconstructed.astype(np.uint8))
    ))
    return reconstructed_image

def wavelet_recomposition_12terms_only_low(r_LL, g_LL, b_LL):

    r_LL = (np.array(r_LL).astype(np.float64) * 2)
    g_LL = (np.array(g_LL).astype(np.float64) * 2)
    b_LL = (np.array(b_LL).astype(np.float64) * 2)
    r_reconstructed = pywt.idwt2((r_LL, (np.zeros_like(r_LL), np.zeros_like(r_LL), np.zeros_like(r_LL))), 'haar')
    g_reconstructed = pywt.idwt2((g_LL, (np.zeros_like(r_LL), np.zeros_like(r_LL), np.zeros_like(r_LL))), 'haar')
    b_reconstructed = pywt.idwt2((b_LL, (np.zeros_like(r_LL), np.zeros_like(r_LL), np.zeros_like(r_LL))), 'haar')
    reconstructed_image = Image.merge('RGB', (
        Image.fromarray(r_reconstructed.astype(np.uint8)),
        Image.fromarray(g_reconstructed.astype(np.uint8)),
        Image.fromarray(b_reconstructed.astype(np.uint8))
    ))
    return reconstructed_image