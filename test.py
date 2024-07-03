import argparse
import json
import math
import os
import torch
import pywt
import sys
import time
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
from residualencoder.zoo import models as residual_models

torch.backends.cudnn.deterministic = True
torch.set_num_threads(1)

def setup_args():
    parent_parser = argparse.ArgumentParser()
    parent_parser.add_argument("-d", "--dataset", type=str, help="dataset path")
    parent_parser.add_argument("-r", "--recon_path", type=str, default="reconstruction", help="where to save recon img")
    parent_parser.add_argument(
        "-c",
        "--entropy-coder",
        choices=compressai.available_entropy_coders(),
        default=compressai.available_entropy_coders()[0],
        help="entropy coder (default: %(default)s)",
    )
    parent_parser.add_argument(
        "--cuda",
        action="store_true",
        help="enable CUDA",
    )
    parent_parser.add_argument(
        "--half",
        action="store_true",
        help="convert model to half floating point (fp16)",
    )
    parent_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="verbose mode",
    )
    parent_parser.add_argument(
            "-wavelet_model",
            "--wavelet_model",
            dest="wavelet_model_paths",
            type=str,
            nargs="*",
            required=True,
            help="wavelet_model checkpoint path",
        )
    
    parent_parser.add_argument(
            "-residual_model",
            "--residual_model",
            dest="residual_model_paths",
            type=str,
            nargs="*",
            required=True,
            help="residual_model checkpoint path",
        )

    parent_parser.add_argument(
            "-LH_model",
            "--LH_model",
            dest="LH_model_paths",
            type=str,
            nargs="*",
            required=True,
            help="LH_model checkpoint path",
        )
    parent_parser.add_argument(
            "-HL_model",
            "--HL_model",
            dest="HL_model_paths",
            type=str,
            nargs="*",
            required=True,
            help="HL_model checkpoint path",
        )
    parent_parser.add_argument(
            "-HH_model",
            "--HH_model",
            dest="HH_model_paths",
            type=str,
            nargs="*",
            required=True,
            help="HH_model checkpoint path",
        )

    parent_parser.add_argument(
            "-LH_diffusion_model",
            "--LH_diffusion_model",
            dest="LH_diffusion_model_paths",
            type=str,
            nargs="*",
            required=True,
            help="LH_diffusion_model checkpoint path",
        )
    parent_parser.add_argument(
            "-HL_diffusion_model",
            "--HL_diffusion_model",
            dest="HL_diffusion_model_paths",
            type=str,
            nargs="*",
            required=True,
            help="HL_diffusion_model checkpoint path",
        )
    parent_parser.add_argument(
            "-HH_diffusion_model",
            "--HH_diffusion_model",
            dest="HH_diffusion_model_paths",
            type=str,
            nargs="*",
            required=True,
            help="HH_diffusion_model checkpoint path",
        )
    return parent_parser

def load_checkpoint(arch: str, checkpoint_path: str) -> nn.Module:
    state_dict = load_state_dict(torch.load(checkpoint_path)['state_dict'])
    return models[arch].from_state_dict(state_dict).eval()

def load_residual_checkpoint(arch: str, checkpoint_path: str) -> nn.Module:
    state_dict = load_state_dict(torch.load(checkpoint_path)['state_dict'])
    return residual_models[arch].from_state_dict(state_dict).eval()

def collect_images(rootpath: str) -> List[str]:
    return [
        os.path.join(rootpath, f)
        for f in os.listdir(rootpath)
        if os.path.splitext(f)[-1].lower() in IMG_EXTENSIONS
    ]

def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = F.mse_loss(a, b).item()
    return -10 * math.log10(mse)

def read_gray_image(filepath: str) -> Image:
    assert os.path.isfile(filepath)
    img = Image.open(filepath).convert("RGB").convert("L")
    return img

def reconstruct(reconstruction, filename, recon_path):
    reconstruction=Image.fromarray(reconstruction)
    reconstruction.save(os.path.join(recon_path, filename))


IMG_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".ppm",
    ".bmp",
    ".pgm",
    ".tif",
    ".tiff",
    ".webp",
)

@torch.no_grad()
def inference(wavelet_model, 
              LH_model, HL_model, HH_model, 
              diffusion_model,LH_diffusion_model, HL_diffusion_model, HH_diffusion_model,residual_model,
              x_gray,im_LL, filename, recon_path,
              LH,HL,HH
              ):
              
    if not os.path.exists(recon_path):
        os.makedirs(recon_path)

    x_LL = transforms.ToTensor()(im_LL).unsqueeze(0).to("cuda")

    h, w = x_LL.size(2), x_LL.size(3)
    p = 64 
    new_h = (h + p - 1) // p * p
    new_w = (w + p - 1) // p * p
    padding_left = (new_w - w) // 2
    padding_right = new_w - w - padding_left
    padding_top = (new_h - h) // 2
    padding_bottom = new_h - h - padding_top
    x_padded = F.pad(
        x_LL,
        (padding_left, padding_right, padding_top, padding_bottom),
        mode="constant",
        value=0,
    )

    encoding_start = time.time()
    out_enc = wavelet_model.compress(x_padded)
    t = torch.randint(10, 11, (1,), device="cuda").long()
    decoding_start = time.time()
    x_LL_recons = wavelet_model.decompress(out_enc["strings"], out_enc["shape"])

    x_LL_recons["x_hat"] = F.pad(
        x_LL_recons["x_hat"], (-padding_left, -padding_right, -padding_top, -padding_bottom)
    )

    LH_reflect = LH_model(x_LL_recons['x_hat'])
    HL_reflect = HL_model(x_LL_recons['x_hat'])
    HH_reflect = HH_model(x_LL_recons['x_hat'])

    x_start_HL = HL_reflect
    HL_noise = torch.randn_like(x_start_HL)
    HL_noisy = diffusion_model.q_sample(x_start_HL, t, noise=HL_noise)
    predicted_HL_noise = HL_diffusion_model(HL_noisy, t)
    clear_HL = diffusion_model.predict_start_from_noise(HL_noisy, t, predicted_HL_noise)

    x_start_LH = LH_reflect
    LH_noise = torch.randn_like(x_start_LH)
    LH_noisy = diffusion_model.q_sample(x_start_LH, t, noise=LH_noise)
    predicted_LH_noise = LH_diffusion_model(LH_noisy, t)
    clear_LH = diffusion_model.predict_start_from_noise(LH_noisy, t, predicted_LH_noise)

    x_start_HH = HH_reflect
    HH_noise = torch.randn_like(x_start_HH)
    HH_noisy = diffusion_model.q_sample(x_start_HH, t, noise=HH_noise)
    predicted_HH_noise = HH_diffusion_model(HH_noisy, t)
    clear_HH = diffusion_model.predict_start_from_noise(HH_noisy, t, predicted_HH_noise)
  
    x_LL_recons = transforms.ToPILImage()(x_LL_recons['x_hat'].squeeze().cpu())
    x_LL_recons = np.array(x_LL_recons)
    x_LL_recons = x_LL_recons.astype(np.float64)
    x_LL_recons=(x_LL_recons*2)

    LH_reflect_recons = transforms.ToPILImage()(clear_LH.squeeze().cpu())   
    #LH_reflect_recons = transforms.ToPILImage()(LH_reflect.squeeze().cpu())
    LH_reflect_recons = np.array(LH_reflect_recons)
    LH_reflect_recons = LH_reflect_recons.astype(np.float64)
    LH_reflect_recons=(LH_reflect_recons*2)
    LH_reflect_recons=(LH_reflect_recons-255)/128 

    HL_reflect_recons = transforms.ToPILImage()(clear_HL.squeeze().cpu())
    #HL_reflect_recons = transforms.ToPILImage()(HL_reflect.squeeze().cpu())
    HL_reflect_recons = np.array(HL_reflect_recons)
    HL_reflect_recons = HL_reflect_recons.astype(np.float64)
    HL_reflect_recons=(HL_reflect_recons*2)
    HL_reflect_recons=(HL_reflect_recons-255)/128

    HH_reflect_recons = transforms.ToPILImage()(clear_HH.squeeze().cpu())
    #HH_reflect_recons = transforms.ToPILImage()(HH_reflect.squeeze().cpu())
    HH_reflect_recons = np.array(HH_reflect_recons)
    HH_reflect_recons = HH_reflect_recons.astype(np.float64)
    HH_reflect_recons=(HH_reflect_recons*2)
    HH_reflect_recons=(HH_reflect_recons-255)/128

    r_LH = LH - LH_reflect_recons
    r_HL = HL - HL_reflect_recons
    r_HH = HH - HH_reflect_recons

    test_ALL=np.array([r_LH,r_HL,r_HH]).transpose(1,2,0)
    test_ALL=test_ALL+255
    test_ALL=(test_ALL/2)
    test_ALL=test_ALL.astype(np.uint8)
    test_ALL=Image.fromarray(test_ALL)
    t = transforms.ToTensor()(test_ALL).unsqueeze(0).to("cuda")

    out_t = residual_model.compress(t)
    enc_time = time.time() - encoding_start
    t_recons = residual_model.decompress(out_t["strings"], out_t["shape"])
    t_rec_im = transforms.ToPILImage()(t_recons['x_hat'].squeeze().cpu())
    t_rec = np.array(t_rec_im)
    t_rec = t_rec.astype(np.float64)
    t_rec=(t_rec*2)
    t_rec=(t_rec-255)
    t_rec=t_rec.transpose(2,0,1)

    rec_LH = (t_rec[0] + LH_reflect_recons)
    rec_HL = (t_rec[1] + HL_reflect_recons)
    rec_HH = (t_rec[2] + HH_reflect_recons)
    
    # rec_LH = np.around(rec_LH)
    # rec_HL = np.around(rec_HL)
    # rec_HH = np.around(rec_HH)
    
    r_coeffs = x_LL_recons, (rec_LH, rec_HL, rec_HH) 
    r_rec = pywt.idwt2(r_coeffs,'haar', 'symmetric').astype(np.uint8)
    
    dec_time = time.time() - decoding_start

    reconstruct(r_rec, filename, recon_path)
    r_rec=Image.fromarray(r_rec)
    num_pixels = x_gray.size[0] * x_gray.size[1]
    low_frequency_bpp = sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels
    residual_bpp = sum(len(s[0]) for s in out_t["strings"]) * 8.0 / num_pixels
    transf = transforms.ToTensor()
    x_gray=transf(x_gray).unsqueeze(0)
    r_rec=transf(r_rec).unsqueeze(0)
    return {
        "psnr": psnr(x_gray, r_rec),
        "ms-ssim": -10*(math.log10(1-(ms_ssim(x_gray, r_rec, data_range=1.0).item()))),
        "bpp": low_frequency_bpp + residual_bpp,
        "encoding_time": enc_time,
        "decoding_time": dec_time,
    }

def eval_model(wavelet_model, 
              LH_model, HL_model, HH_model, 
              diffusion_model,LH_diffusion_model, HL_diffusion_model, HH_diffusion_model,residual_model,
              filepaths, half=False, recon_path='reconstruction'):
    device = next(wavelet_model.parameters()).device
    metrics = defaultdict(float)
    for f in filepaths:
        _filename = f.split("/")[-1]
        x_gray = read_gray_image(f)
        coeffs = pywt.dwt2(x_gray,'haar', 'symmetric')
        LL, (LH, HL, HH) = coeffs

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

        if half:
            wavelet_model = wavelet_model.half()
            x_gray = x_gray.half()
        rv = inference(wavelet_model, 
              LH_model, HL_model, HH_model, 
              diffusion_model,LH_diffusion_model, HL_diffusion_model, HH_diffusion_model, residual_model,
              x_gray, im_LL,_filename, recon_path,
              LH,HL,HH
              )
        for k, v in rv.items():
            metrics[k] += v
    for k, v in metrics.items():
        metrics[k] = v / len(filepaths)
    return metrics

def main(argv):
    parser = setup_args()
    args = parser.parse_args(argv)

    filepaths = collect_images(args.dataset)
    if len(filepaths) == 0:
        print("Error: no images found in directory.", file=sys.stderr)
        sys.exit(1)
    compressai.set_entropy_coder(args.entropy_coder)
    results = defaultdict(list)

    wavelet_model = load_checkpoint("cnn", (args.wavelet_model_paths)[0])
    wavelet_model.update(force=True)
    
    residual_model = load_residual_checkpoint("cnn", (args.residual_model_paths)[0])
    residual_model.update(force=True)

    LH_model = Unet()
    HL_model = Unet()
    HH_model = Unet()
    LH_model.load_state_dict(torch.load((args.LH_model_paths)[0]))
    HL_model.load_state_dict(torch.load((args.HL_model_paths)[0]))
    HH_model.load_state_dict(torch.load((args.HH_model_paths)[0]))

    timesteps=10
    diffusion_model = GaussianDiffusion(timesteps)
    
    diffusion_model_HL = UNetModel(
        in_channels=3,
        model_channels=96,
        out_channels=3,
        channel_mult=(1, 2, 2),
        attention_resolutions=[]
    )
    diffusion_model_HL.load_state_dict(torch.load((args.LH_diffusion_model_paths)[0]),strict=False)

    diffusion_model_LH = UNetModel(
        in_channels=3,
        model_channels=96,
        out_channels=3,
        channel_mult=(1, 2, 2),
        attention_resolutions=[]
    )
    diffusion_model_LH.load_state_dict(torch.load((args.LH_diffusion_model_paths)[0]),strict=False)

    diffusion_model_HH = UNetModel(
        in_channels=3,
        model_channels=96,
        out_channels=3,
        channel_mult=(1, 2, 2),
        attention_resolutions=[]
    )
    diffusion_model_HH.load_state_dict(torch.load((args.HH_diffusion_model_paths)[0]),strict=False)

    if args.cuda and torch.cuda.is_available():
        wavelet_model = wavelet_model.to("cuda")
        LH_model = LH_model.to("cuda")
        HL_model = HL_model.to("cuda")
        HH_model = HH_model.to("cuda")
        diffusion_model_HL = diffusion_model_HL.to("cuda")
        diffusion_model_LH = diffusion_model_LH.to("cuda")
        diffusion_model_HH = diffusion_model_HH.to("cuda")
        residual_model = residual_model.to("cuda")

    metrics = eval_model(wavelet_model, 
            LH_model, HL_model, HH_model, 
            diffusion_model,diffusion_model_LH, diffusion_model_HL, diffusion_model_HH, residual_model,
            filepaths, args.half, args.recon_path)
    
    for k, v in metrics.items():
        results[k].append(v)

    description = (
        args.entropy_coder
    )
    output = {
        "description": f"Inference ({description})",
        "results": results,
    }

    print(json.dumps(output, indent=2))

if __name__ == "__main__":
    main(sys.argv[1:])

