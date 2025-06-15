import argparse
import math
import random
import shutil
import sys
from torch.utils.data import DataLoader
from residualencoder.datasets import ImageFolder
from pathlib import Path
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import pywt
import numpy as np
from typing import List
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from diffusion.Unet import Unet
from pytorch_msssim import ms_ssim
from torchvision import transforms
from diffusion.ddpm import UNetModel,GaussianDiffusion
from waveletencoder.zoo import models as wavelet_models
from residualencoder.zoo import load_state_dict, models
from torch.nn.parallel import DataParallel
import torch.nn.parallel
import os
from torch.utils.data.distributed import DistributedSampler

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
torch.distributed.init_process_group(backend="nccl")
local_rank = torch.distributed.get_rank()
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)

class RateDistortionLoss(nn.Module):
    """rate distortion loss"""
    def __init__(self, lmbda=1e-2):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda
    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W
        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        out["mse_loss"] = self.mse(output["x_hat"], target)
        out["loss"] = self.lmbda * 255 ** 2 * out["mse_loss"] + out["bpp_loss"]
        return out

class URDLoss(nn.Module):
    def __init__(self, lmbda=1e-2):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda

    def forward(self, output, target, var_image1,var_image2,var_image3):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W
        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        out["mse_loss"] = self.mse(output["x_hat"], target)
        var_mean = torch.sum((torch.mean(var_image1) + torch.mean(var_image2) + torch.mean(var_image3))/3)
        additional_loss = torch.exp(torch.tensor(-math.log(2.0 * var_mean))) * self.mse(output["x_hat"], target) + torch.tensor(1.5) * math.log(var_mean)
        out["additional_loss"] = additional_loss * 0.01
        out["traditional_loss"] = self.lmbda * 255 ** 2 * out["mse_loss"] + out["bpp_loss"]
        if not torch.isnan(additional_loss):
            out["traditional_loss"] = (self.lmbda * 255 ** 2 + math.log(additional_loss)) * out["mse_loss"] + out["bpp_loss"]
        out["loss"] = out["traditional_loss"]
        return out

def load_checkpoint(arch: str, checkpoint_path: str) -> nn.Module:
    state_dict = load_state_dict(torch.load(checkpoint_path)['state_dict'])
    return wavelet_models[arch].from_state_dict(state_dict).eval()

class AverageMeter:

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)

def configure_optimizers(net, args):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""

    parameters = {
        n
        for n, p in net.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    }
    aux_parameters = {
        n
        for n, p in net.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    }
    params_dict = dict(net.named_parameters())
    inter_params = parameters & aux_parameters
    union_params = parameters | aux_parameters
    assert len(inter_params) == 0
    assert len(union_params) - len(params_dict.keys()) == 0
    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(parameters)),
        lr=args.learning_rate,
    )
    aux_optimizer = optim.Adam(
        (params_dict[n] for n in sorted(aux_parameters)),
        lr=args.aux_learning_rate,
    )
    return optimizer, aux_optimizer

def get_model_input_time(ns, t_continuous):
    if ns.schedule == 'discrete':
        return (t_continuous - 1. / ns.total_N) * 1000.
    else:
        return t_continuous

def conditioned_exp_iteration(exp_xt, ns, s, t, pre_wuq, exp_s1=None, mc_eps_exp_s1= None):

    if pre_wuq == True:
        exp_xt_next = exp_iteration(exp_xt, ns, s, t, mc_eps_exp_s1)
        return exp_xt_next
    else:
        exp_xt_next = exp_iteration(exp_xt, ns, s, t, exp_s1)
        return exp_xt_next

def conditioned_var_iteration(var_xt, ns, s, t, pre_wuq, cov_xt_epst= None, var_epst = None):

    if pre_wuq == True:
        var_xt_next = var_iteration(var_xt, ns, s, t, cov_xt_epst, var_epst)
        return var_xt_next
    else:
        log_alpha_s, log_alpha_t = ns.marginal_log_mean_coeff(s), ns.marginal_log_mean_coeff(t)
        var_xt_next = torch.square(torch.exp(log_alpha_t - log_alpha_s)) * var_xt
        return var_xt_next

def conditioned_update(ns, x, s, t, custom_model, model_s, pre_wuq, r1=0.5, **model_kwargs):
    if pre_wuq == True:
        return singlestep_dpm_solver_second_update(ns, x, s, t, custom_model, model_s, r1=0.5, **model_kwargs)
    else:
        lambda_s, lambda_t = ns.marginal_lambda(s), ns.marginal_lambda(t)
        h = lambda_t - lambda_s
        lambda_s1 = lambda_s + r1 * h
        s1 = ns.inverse_lambda(lambda_s1)
        log_alpha_s, log_alpha_s1, log_alpha_t = ns.marginal_log_mean_coeff(s), ns.marginal_log_mean_coeff(s1), ns.marginal_log_mean_coeff(t)
        sigma_s1, sigma_t = ns.marginal_std(s1), ns.marginal_std(t)

        phi_11 = torch.expm1(r1 * h)
        phi_1 = torch.expm1(h)
        
        x_s1 = (
            torch.exp(log_alpha_s1 - log_alpha_s) * x
            - (sigma_s1 * phi_11) * model_s
        )

        input_s1 = get_model_input_time(ns, s1)
        model_s1 = custom_model.accurate_forward(x_s1, input_s1.expand(x_s1.shape[0]), **model_kwargs)

        x_t = (
            torch.exp(log_alpha_t - log_alpha_s) * x
            - (sigma_t * phi_1) * model_s
            - (0.5 / r1) * (sigma_t * phi_1) * (model_s1 - model_s)
        )

        return x_t, model_s1

def train_one_epoch2(
    model, criterion, train_dataloader, optimizer, aux_optimizer, epoch, clip_max_norm, diffusion_model, diffusion_model_LH, diffusion_model_HL, diffusion_model_HH,
    ns, t_seq, uq_array, custom_model, mc_sample_size, device
):
    model.train()
    for i, d in enumerate(train_dataloader):
        d = d.to(device)
        tensor1, tensor2, tensor3 = torch.split(d, split_size_or_sections=1, dim=1)
        
        x_start_LH = tensor1
        xT_LH = torch.randn_like(x_start_LH)
        T = t_seq[0]
        xt_next_LH = xT_LH
        exp_xt_next_LH = xT_LH
        var_xt_next_LH = torch.zeros_like(xT_LH).to(device)
        eps_mu_t_next_LH = custom_model.accurate_forward(xT_LH, get_model_input_time(ns, T).expand(xT_LH.shape[0]))
        for timestep in range(len(t_seq) - 1):
            if uq_array[timestep]:
                xt_LH = xt_next_LH
                exp_xt_LH = exp_xt_next_LH
                var_xt_LH = var_xt_next_LH
                eps_mu_t_LH = eps_mu_t_next_LH
                s, t = t_seq[timestep], t_seq[timestep + 1]
                xt_next_LH, model_s1_LH, _ = conditioned_update(ns, xt_LH, s, t, custom_model, eps_mu_t_LH, pre_wuq=uq_array[timestep],1 r=0.5)
                exp_xt_next_LH = conditioned_exp_iteration(exp_xt_LH, ns, s, t, pre_wuq=uq_array[timestep], mc_eps_exp_s1=torch.mean(eps_mu_t_LH, dim=0))
                var_xt_next_LH = conditioned_var_iteration(var_xt_LH, ns, s, t, pre_wuq=uq_array[timestep])
                if uq_array[timestep + 1]:
                    list_xt_next_i_LH, list_eps_mu_t_next_i_LH = [], []
                    s_next = t_seq[timestep + 1]
                    t_next = t_seq[timestep + 2] if timestep + 2 < len(t_seq) else t_seq[-1]
                    lambda_s_next, lambda_t_next = ns.marginal_lambda(s_next), ns.marginal_lambda(t_next)
                    h_next = lambda_t_next - lambda_s_next
                    lambda_s1_next = lambda_s_next + 0.5 * h_next
                    s1_next = ns.inverse_lambda(lambda_s1_next)
                    sigma_s1_next = ns.marginal_std(s1_next)
                    log_alpha_s_next, log_alpha_s1_next = ns.marginal_log_mean_coeff(s_next), ns.marginal_log_mean_coeff(s1_next)
                    phi_11_next = torch.expm1(0.5 * h_next)
                    for _ in range(mc_sample_size):
                        var_xt_next_LH = torch.clamp(var_xt_next_LH, min=0)
                        xt_next_i_LH = sample_from_gaussion(exp_xt_next_LH, var_xt_next_LH)
                        list_xt_next_i_LH.append(xt_next_i_LH)
                        model_t_i_LH, model_t_i_var_LH = custom_model(xt_next_i_LH, get_model_input_time(ns, s_next).expand(xt_next_i_LH.shape[0]))
                        xu_next_i_LH = sample_from_gaussion(torch.exp(log_alpha_s1_next - log_alpha_s_next) * xt_next_i_LH - (sigma_s1_next * phi_11_next) * model_t_i_LH, torch.square(sigma_s1_next * phi_11_next) * model_t_i_var_LH)
                        model_u_i_LH, _ = custom_model(xu_next_i_LH, get_model_input_time(ns, s1_next).expand(xt_next_i_LH.shape[0]))
                        list_eps_mu_t_next_i_LH.append(model_u_i_LH)
                    eps_mu_t_next_LH, eps_var_t_next_LH = custom_model(xt_next_LH, get_model_input_time(ns, s_next).expand(xt_next_LH.shape[0]))
                    list_xt_next_i_LH = torch.stack(list_xt_next_i_LH, dim=0).to(device)
                    list_eps_mu_t_next_i_LH = torch.stack(list_eps_mu_t_next_i_LH, dim=0).to(device)
                    cov_xt_next_epst_next_LH = torch.mean(list_xt_next_i_LH * list_eps_mu_t_next_i_LH, dim=0) - exp_xt_next_LH * torch.mean(list_eps_mu_t_next_i_LH, dim=0)
                else:
                    eps_mu_t_next_LH = custom_model.accurate_forward(xt_next_LH, get_model_input_time(ns, t).expand(xt_next_LH.shape[0]))
            else:
                xt_LH = xt_next_LH
                exp_xt_LH = exp_xt_next_LH
                var_xt_LH = var_xt_next_LH
                eps_mu_t_LH = eps_mu_t_next_LH
                s, t = t_seq[timestep], t_seq[timestep + 1]
                xt_next_LH, model_s1_LH = conditioned_update(ns, xt_LH, s, t, custom_model, eps_mu_t_LH, pre_wuq=uq_array[timestep r],1=0.5)
                exp_xt_next_LH = conditioned_exp_iteration(exp_xt_LH, ns, s, t, exp_s1=model_s1_LH, pre_wuq=uq_array[timestep])
                var_xt_next_LH = conditioned_var_iteration(var_xt_LH, ns, s, t, pre_wuq=uq_array[timestep])
                if uq_array[timestep + 1]:
                    list_xt_next_i_LH, list_eps_mu_t_next_i_LH = [], []
                    s_next = t_seq[timestep + 1]
                    t_next = t_seq[timestep + 2] if timestep + 2 < len(t_seq) else t_seq[-1]
                    lambda_s_next, lambda_t_next = ns.marginal_lambda(s_next), ns.marginal_lambda(t_next)
                    h_next = lambda_t_next - lambda_s_next
                    lambda_s1_next = lambda_s_next + 0.5 * h_next
                    s1_next = ns.inverse_lambda(lambda_s1_next)
                    sigma_s1_next = ns.marginal_std(s1_next)
                    log_alpha_s_next, log_alpha_s1_next = ns.marginal_log_mean_coeff(s_next), ns.marginal_log_mean_coeff(s1_next)
                    phi_11_next = torch.expm1(0.5 * h_next)
                    for _ in range(mc_sample_size):
                        var_xt_next_LH = torch.clamp(var_xt_next_LH, min=0)
                        xt_next_i_LH = sample_from_gaussion(exp_xt_next_LH, var_xt_next_LH)
                        list_xt_next_i_LH.append(xt_next_i_LH)
                        model_t_i_LH, model_t_i_var_LH = custom_model(xt_next_i_LH, get_model_input_time(ns, s_next).expand(xt_next_i_LH.shape[0]))
                        xu_next_i_LH = sample_from_gaussion(torch.exp(log_alpha_s1_next - log_alpha_s_next) * xt_next_i_LH - (sigma_s1_next * phi_11_next) * model_t_i_LH, torch.square(sigma_s1_next * phi_11_next) * model_t_i_var_LH)
                        model_u_i_LH, _ = custom_model(xu_next_i_LH, get_model_input_time(ns, s1_next).expand(xt_next_i_LH.shape[0]))
                        list_eps_mu_t_next_i_LH.append(model_u_i_LH)
                    eps_mu_t_next_LH, eps_var_t_next_LH = custom_model(xt_next_LH, get_model_input_time(ns, s_next).expand(xt_next_LH.shape[0]))
                    list_xt_next_i_LH = torch.stack(list_xt_next_i_LH, dim=0).to(device)
                    list_eps_mu_t_next_i_LH = torch.stack(list_eps_mu_t_next_i_LH, dim=0).to(device)
                    cov_xt_next_epst_next_LH = torch.mean(list_xt_next_i_LH * list_eps_mu_t_next_i_LH, dim=0) - exp_xt_next_LH * torch.mean(list_eps_mu_t_next_i_LH, dim=0)
                else:
                    eps_mu_t_next_LH = custom_model.accurate_forward(xt_next_LH, get_model_input_time(ns, t).expand(xt_next_LH.shape[0]))
        variance_LH = var_xt_next_LH
        

        x_start_HL = tensor2
        xT_HL = torch.randn_like(x_start_HL)
        T = t_seq[0]
        xt_next_HL = xT_HL
        exp_xt_next_HL = xT_HL
        var_xt_next_HL = torch.zeros_like(xT_HL).to(device)
        eps_mu_t_next_HL = custom_model.accurate_forward(xT_HL, get_model_input_time(ns, T).expand(xT_HL.shape[0]))
        for timestep in range(len(t_seq) - 1):
            if uq_array[timestep]:
                xt_HL = xt_next_HL
                exp_xt_HL = exp_xt_next_HL
                var_xt_HL = var_xt_next_HL
                eps_mu_t_HL = eps_mu_t_next_HL
                s, t = t_seq[timestep], t_seq[timestep + 1]
                xt_next_HL, model_s1_HL, _ = conditioned_update(ns, xt_HL, s, t, custom_model, eps_mu_t_HL, pre_wuq=uq_array[timestep], r=0.5)
                exp_xt_next_HL = conditioned_exp_iteration(exp_xt_HL, ns, s, t, pre_wuq=uq_array[timestep], mc_eps_exp_s1=torch.mean(eps_mu_t_HL, dim=0))
                var_xt_next_HL = conditioned_var_iteration(var_xt_HL, ns, s, t, pre_wuq=uq_array[timestep])
                if uq_array[timestep + 1]:
                    list_xt_next_i_HL, list_eps_mu_t_next_i_HL = [], []
                    s_next = t_seq[timestep + 1]
                    t_next = t_seq[timestep + 2] if timestep + 2 < len(t_seq) else t_seq[-1]
                    lambda_s_next, lambda_t_next = ns.marginal_lambda(s_next), ns.marginal_lambda(t_next)
                    h_next = lambda_t_next - lambda_s_next
                    lambda_s1_next = lambda_s_next + 0.5 * h_next
                    s1_next = ns.inverse_lambda(lambda_s1_next)
                    sigma_s1_next = ns.marginal_std(s1_next)
                    log_alpha_s_next, log_alpha_s1_next = ns.marginal_log_mean_coeff(s_next), ns.marginal_log_mean_coeff(s1_next)
                    phi_11_next = torch.expm1(0.5 * h_next)
                    for _ in range(mc_sample_size):
                        var_xt_next_HL = torch.clamp(var_xt_next_HL, min=0)
                        xt_next_i_HL = sample_from_gaussion(exp_xt_next_HL, var_xt_next_HL)
                        list_xt_next_i_HL.append(xt_next_i_HL)
                        model_t_i_HL, model_t_i_var_HL = custom_model(xt_next_i_HL, get_model_input_time(ns, s_next).expand(xt_next_i_HL.shape[0]))
                        xu_next_i_HL = sample_from_gaussion(torch.exp(log_alpha_s1_next - log_alpha_s_next) * xt_next_i_HL - (sigma_s1_next * phi_11_next) * model_t_i_HL, torch.square(sigma_s1_next * phi_11_next) * model_t_i_var_HL)
                        model_u_i_HL, _ = custom_model(xu_next_i_HL, get_model_input_time(ns, s1_next).expand(xt_next_i_HL.shape[0]))
                        list_eps_mu_t_next_i_HL.append(model_u_i_HL)
                    eps_mu_t_next_HL, eps_var_t_next_HL = custom_model(xt_next_HL, get_model_input_time(ns, s_next).expand(xt_next_HL.shape[0]))
                    list_xt_next_i_HL = torch.stack(list_xt_next_i_HL, dim=0).to(device)
                    list_eps_mu_t_next_i_HL = torch.stack(list_eps_mu_t_next_i_HL, dim=0).to(device)
                    cov_xt_next_epst_next_HL = torch.mean(list_xt_next_i_HL * list_eps_mu_t_next_i_HL, dim=0) - exp_xt_next_HL * torch.mean(list_eps_mu_t_next_i_HL, dim=0)
                else:
                    eps_mu_t_next_HL = custom_model.accurate_forward(xt_next_HL, get_model_input_time(ns, t).expand(xt_next_HL.shape[0]))
            else:
                xt_HL = xt_next_HL
                exp_xt_HL = exp_xt_next_HL
                var_xt_HL = var_xt_next_HL
                eps_mu_t_HL = eps_mu_t_next_HL
                s, t = t_seq[timestep], t_seq[timestep + 1]
                xt_next_HL, model_s1_HL = conditioned_update(ns, xt_HL, s, t, custom_model, eps_mu_t_HL, pre_wuq=uq_array[timestep], r=0.5)
                exp_xt_next_HL = conditioned_exp_iteration(exp_xt_HL, ns, s, t, exp_s1=model_s1_HL, pre_wuq=uq_array[timestep])
                var_xt_next_HL = conditioned_var_iteration(var_xt_HL, ns, s, t, pre_wuq=uq_array[timestep])
                if uq_array[timestep + 1]:
                    list_xt_next_i_HL, list_eps_mu_t_next_i_HL = [], []
                    s_next = t_seq[timestep + 1]
                    t_next = t_seq[timestep + 2] if timestep + 2 < len(t_seq) else t_seq[-1]
                    lambda_s_next, lambda_t_next = ns.marginal_lambda(s_next), ns.marginal_lambda(t_next)
                    h_next = lambda_t_next - lambda_s_next
                    lambda_s1_next = lambda_s_next + 0.5 * h_next
                    s1_next = ns.inverse_lambda(lambda_s1_next)
                    sigma_s1_next = ns.marginal_std(s1_next)
                    log_alpha_s_next, log_alpha_s1_next = ns.marginal_log_mean_coeff(s_next), ns.marginal_log_mean_coeff(s1_next)
                    phi_11_next = torch.expm1(0.5 * h_next)
                    for _ in range(mc_sample_size):
                        var_xt_next_HL = torch.clamp(var_xt_next_HL, min=0)
                        xt_next_i_HL = sample_from_gaussion(exp_xt_next_HL, var_xt_next_HL)
                        list_xt_next_i_HL.append(xt_next_i_HL)
                        model_t_i_HL, model_t_i_var_HL = custom_model(xt_next_i_HL, get_model_input_time(ns, s_next).expand(xt_next_i_HL.shape[0]))
                        xu_next_i_HL = sample_from_gaussion(torch.exp(log_alpha_s1_next - log_alpha_s_next) * xt_next_i_HL - (sigma_s1_next * phi_11_next) * model_t_i_HL, torch.square(sigma_s1_next * phi_11_next) * model_t_i_var_HL)
                        model_u_i_HL, _ = custom_model(xu_next_i_HL, get_model_input_time(ns, s1_next).expand(xt_next_i_HL.shape[0]))
                        list_eps_mu_t_next_i_HL.append(model_u_i_HL)
                    eps_mu_t_next_HL, eps_var_t_next_HL = custom_model(xt_next_HL, get_model_input_time(ns, s_next).expand(xt_next_HL.shape[0]))
                    list_xt_next_i_HL = torch.stack(list_xt_next_i_HL, dim=0).to(device)
                    list_eps_mu_t_next_i_HL = torch.stack(list_eps_mu_t_next_i_HL, dim=0).to(device)
                    cov_xt_next_epst_next_HL = torch.mean(list_xt_next_i_HL * list_eps_mu_t_next_i_HL, dim=0) - exp_xt_next_HL * torch.mean(list_eps_mu_t_next_i_HL, dim=0)
                else:
                    eps_mu_t_next_HL = custom_model.accurate_forward(xt_next_HL, get_model_input_time(ns, t).expand(xt_next_HL.shape[0]))
        variance_HL = var_xt_next_HL
        
        x_start_HH = tensor3
        xT_HH = torch.randn_like(x_start_HH)
        T = t_seq[0]
        xt_next_HH = xT_HH
        exp_xt_next_HH = xT_HH
        var_xt_next_HH = torch.zeros_like(xT_HH).to(device)
        eps_mu_t_next_HH = custom_model.accurate_forward(xT_HH, get_model_input_time(ns, T).expand(xT_HH.shape[0]))

        for timestep in range(len(t_seq) - 1):
            if uq_array[timestep]:
                xt_HH = xt_next_HH
                exp_xt_HH = exp_xt_next_HH
                var_xt_HH = var_xt_next_HH
                eps_mu_t_HH = eps_mu_t_next_HH
                s, t = t_seq[timestep], t_seq[timestep + 1]
                xt_next_HH, model_s1_HH, _ = conditioned_update(ns, xt_HH, s, t, custom_model, eps_mu_t_HH, pre_wuq=uq_array[timestep], r=0.5)
                exp_xt_next_HH = conditioned_exp_iteration(exp_xt_HH, ns, s, t, pre_wuq=uq_array[timestep], mc_eps_exp_s1=torch.mean(eps_mu_t_HH, dim=0))
                var_xt_next_HH = conditioned_var_iteration(var_xt_HH, ns, s, t, pre_wuq=uq_array[timestep])
                if uq_array[timestep + 1]:
                    list_xt_next_i_HH, list_eps_mu_t_next_i_HH = [], []
                    s_next = t_seq[timestep + 1]
                    t_next = t_seq[timestep + 2] if timestep + 2 < len(t_seq) else t_seq[-1]
                    lambda_s_next, lambda_t_next = ns.marginal_lambda(s_next), ns.marginal_lambda(t_next)
                    h_next = lambda_t_next - lambda_s_next
                    lambda_s1_next = lambda_s_next + 0.5 * h_next
                    s1_next = ns.inverse_lambda(lambda_s1_next)
                    sigma_s1_next = ns.marginal_std(s1_next)
                    log_alpha_s_next, log_alpha_s1_next = ns.marginal_log_mean_coeff(s_next), ns.marginal_log_mean_coeff(s1_next)
                    phi_11_next = torch.expm1(0.5 * h_next)
                    for _ in range(mc_sample_size):
                        var_xt_next_HH = torch.clamp(var_xt_next_HH, min=0)
                        xt_next_i_HH = sample_from_gaussion(exp_xt_next_HH, var_xt_next_HH)
                        list_xt_next_i_HH.append(xt_next_i_HH)
                        model_t_i_HH, model_t_i_var_HH = custom_model(xt_next_i_HH, get_model_input_time(ns, s_next).expand(xt_next_i_HH.shape[0]))
                        xu_next_i_HH = sample_from_gaussion(torch.exp(log_alpha_s1_next - log_alpha_s_next) * xt_next_i_HH - (sigma_s1_next * phi_11_next) * model_t_i_HH, torch.square(sigma_s1_next * phi_11_next) * model_t_i_var_HH)
                        model_u_i_HH, _ = custom_model(xu_next_i_HH, get_model_input_time(ns, s1_next).expand(xt_next_i_HH.shape[0]))
                        list_eps_mu_t_next_i_HH.append(model_u_i_HH)
                    eps_mu_t_next_HH, eps_var_t_next_HH = custom_model(xt_next_HH, get_model_input_time(ns, s_next).expand(xt_next_HH.shape[0]))
                    list_xt_next_i_HH = torch.stack(list_xt_next_i_HH, dim=0).to(device)
                    list_eps_mu_t_next_i_HH = torch.stack(list_eps_mu_t_next_i_HH, dim=0).to(device)
                    cov_xt_next_epst_next_HH = torch.mean(list_xt_next_i_HH * list_eps_mu_t_next_i_HH, dim=0) - exp_xt_next_HH * torch.mean(list_eps_mu_t_next_i_HH, dim=0)
                else:
                    eps_mu_t_next_HH = custom_model.accurate_forward(xt_next_HH, get_model_input_time(ns, t).expand(xt_next_HH.shape[0]))
            else:
                xt_HH = xt_next_HH
                exp_xt_HH = exp_xt_next_HH
                var_xt_HH = var_xt_next_HH
                eps_mu_t_HH = eps_mu_t_next_HH
                s, t = t_seq[timestep], t_seq[timestep + 1]
                xt_next_HH, model_s1_HH = conditioned_update(ns, xt_HH, s, t, custom_model, eps_mu_t_HH, pre_wuq=uq_array[timestep], r=0.5)
                exp_xt_next_HH = conditioned_exp_iteration(exp_xt_HH, ns, s, t, exp_s1=model_s1_HH, pre_wuq=uq_array[timestep])
                var_xt_next_HH = conditioned_var_iteration(var_xt_HH, ns, s, t, pre_wuq=uq_array[timestep])
                if uq_array[timestep + 1]:
                    list_xt_next_i_HH, list_eps_mu_t_next_i_HH = [], []
                    s_next = t_seq[timestep + 1]
                    t_next = t_seq[timestep + 2] if timestep + 2 < len(t_seq) else t_seq[-1]
                    lambda_s_next, lambda_t_next = ns.marginal_lambda(s_next), ns.marginal_lambda(t_next)
                    h_next = lambda_t_next - lambda_s_next
                    lambda_s1_next = lambda_s_next + 0.5 * h_next
                    s1_next = ns.inverse_lambda(lambda_s1_next)
                    sigma_s1_next = ns.marginal_std(s1_next)
                    log_alpha_s_next, log_alpha_s1_next = ns.marginal_log_mean_coeff(s_next), ns.marginal_log_mean_coeff(s1_next)
                    phi_11_next = torch.expm1(0.5 * h_next)
                    for _ in range(mc_sample_size):
                        var_xt_next_HH = torch.clamp(var_xt_next_HH, min=0)
                        xt_next_i_HH = sample_from_gaussion(exp_xt_next_HH, var_xt_next_HH)
                        list_xt_next_i_HH.append(xt_next_i_HH)
                        model_t_i_HH, model_t_i_var_HH = custom_model(xt_next_i_HH, get_model_input_time(ns, s_next).expand(xt_next_i_HH.shape[0]))
                        xu_next_i_HH = sample_from_gaussion(torch.exp(log_alpha_s1_next - log_alpha_s_next) * xt_next_i_HH - (sigma_s1_next * phi_11_next) * model_t_i_HH, torch.square(sigma_s1_next * phi_11_next) * model_t_i_var_HH)
                        model_u_i_HH, _ = custom_model(xu_next_i_HH, get_model_input_time(ns, s1_next).expand(xt_next_i_HH.shape[0]))
                        list_eps_mu_t_next_i_HH.append(model_u_i_HH)
                    eps_mu_t_next_HH, eps_var_t_next_HH = custom_model(xt_next_HH, get_model_input_time(ns, s_next).expand(xt_next_HH.shape[0]))
                    list_xt_next_i_HH = torch.stack(list_xt_next_i_HH, dim=0).to(device)
                    list_eps_mu_t_next_i_HH = torch.stack(list_eps_mu_t_next_i_HH, dim=0).to(device)
                    cov_xt_next_epst_next_HH = torch.mean(list_xt_next_i_HH * list_eps_mu_t_next_i_HH, dim=0) - exp_xt_next_HH * torch.mean(list_eps_mu_t_next_i_HH, dim=0)
                else:
                    eps_mu_t_next_HH = custom_model.accurate_forward(xt_next_HH, get_model_input_time(ns, t).expand(xt_next_HH.shape[0]))
        variance_HH = var_xt_next_HH

        del tensor1, tensor2, tensor3, x_start_LH, x_start_HL, x_start_HH
        del xT_LH, xT_HL, xT_HH, xt_next_LH, xt_next_HL, xt_next_HH
        del exp_xt_next_LH, exp_xt_next_HL, exp_xt_next_HH
        del var_xt_next_LH, var_xt_next_HL, var_xt_next_HH
        del eps_mu_t_next_LH, eps_mu_t_next_HL, eps_mu_t_next_HH
    
        optimizer.zero_grad()
        aux_optimizer.zero_grad()
        out_net = model(d)
        out_criterion = criterion(out_net, d, variance_LH, variance_HL, variance_HH)
        out_criterion["loss"].backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()
        aux_loss = model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()
        
        if i % 100 == 0:
            print(
                f"Train epoch {epoch}: ["
                f"{i * len(d)}/{len(train_dataloader.dataset)} "
                f"({100. * i / len(train_dataloader):.0f}%)] "
                f"Loss: {out_criterion['loss'].item():.3f} | "
                f"MSE loss: {out_criterion['mse_loss'].item() * 255 ** 2 / 3:.3f} | "
                f"Bpp loss: {out_criterion['bpp_loss'].item():.2f} | "
                f"Traditional loss: {out_criterion['traditional_loss'].item():.2f} | "
                f"Additional loss: {out_criterion['additional_loss'].item():.2f} | "
                f"Aux loss: {aux_loss.item():.2f}"
            )

def train_one_epoch(
    model, criterion, train_dataloader, optimizer, aux_optimizer, epoch, clip_max_norm, diffusion_model, diffusion_model_LH, diffusion_model_HL, diffusion_model_HH
):
    model.train()
    for i, d in enumerate(train_dataloader):
        d = d.to(device)
        tensor1, tensor2, tensor3 = torch.split(d, split_size_or_sections=1, dim=1)
        x_start_LH = tensor1
        noise_LH = torch.randn_like(x_start_LH)
        recovered_images_LH = []
        for j1 in range(5): 
            t1 = torch.randint(5, 10, (1,), device=device).long()
            x_noisy = diffusion_model.q_sample(x_start_LH, t1, noise=noise_LH)
            predicted_noise = diffusion_model_LH(x_noisy, t1)
            x_recon = diffusion_model.predict_start_from_noise(x_noisy, t1, predicted_noise)
            recovered_images_LH.append(x_recon)
            del x_noisy
            del predicted_noise

        recovered_images_tensor_LH = torch.stack(recovered_images_LH)
        variance_LH = torch.var(recovered_images_tensor_LH, dim=0)
        min_value_LH = torch.min(variance_LH)
        max_value_LH = torch.max(variance_LH)
        normalized_data_LH = (variance_LH - min_value_LH) / (max_value_LH - min_value_LH)
        variance_LH = normalized_data_LH * 255
        # Save GPU memory space
        del tensor1
        del x_start_LH
        del noise_LH
        del x_recon
        del recovered_images_LH
        del recovered_images_tensor_LH

        x_start_HL = tensor2
        noise_HL = torch.randn_like(x_start_HL)
        recovered_images_HL = []
        for j2 in range(5): 
            t1 = torch.randint(5, 10, (1,), device=device).long()
            x_noisy = diffusion_model.q_sample(x_start_HL, t1, noise=noise_HL)
            predicted_noise = diffusion_model_HL(x_noisy, t1)
            x_recon = diffusion_model.predict_start_from_noise(x_noisy, t1, predicted_noise)
            recovered_images_HL.append(x_recon)
            del x_noisy
            del predicted_noise

        recovered_images_tensor_HL = torch.stack(recovered_images_HL)
        variance_HL = torch.var(recovered_images_tensor_HL, dim=0)
        min_value_HL = torch.min(variance_HL)
        max_value_HL = torch.max(variance_HL)
        normalized_data_HL = (variance_HL- min_value_HL) / (max_value_HL - min_value_HL)
        variance_HL = normalized_data_HL * 255
        del tensor2
        del x_start_HL
        del noise_HL
        del x_recon
        del recovered_images_HL
        del recovered_images_tensor_HL

        x_start_HH = tensor3
        noise_HH = torch.randn_like(x_start_HH)
        recovered_images_HH = []
        for j3 in range(5): 
            t1 = torch.randint(5, 10, (1,), device=device).long()
            x_noisy = diffusion_model.q_sample(x_start_HH, t1, noise=noise_HH)
            predicted_noise = diffusion_model_HH(x_noisy, t1)
            x_recon = diffusion_model.predict_start_from_noise(x_noisy, t1, predicted_noise)
            recovered_images_HH.append(x_recon)
            del x_noisy
            del predicted_noise

        recovered_images_tensor_HH = torch.stack(recovered_images_HH)
        variance_HH = torch.var(recovered_images_tensor_HH, dim=0) 
        min_value_HH = torch.min(variance_HH)
        max_value_HH = torch.max(variance_HH)
        normalized_data_HH = (variance_HH- min_value_HH) / (max_value_HH - min_value_HH)
        variance_HH = normalized_data_HH * 255

        del tensor3
        del x_start_HH
        del noise_HH
        del x_recon
        del recovered_images_HH
        del recovered_images_tensor_HH

        optimizer.zero_grad()
        aux_optimizer.zero_grad()
        out_net = model(d)
        out_criterion = criterion(out_net, d, variance_LH,variance_HL,variance_HH)
        out_criterion["loss"].backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()
        aux_loss = model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()
        if i % 100 == 0:
            print(
                f"Train epoch {epoch}: ["
                f"{i*len(d)}/{len(train_dataloader.dataset)}"
                f" ({100. * i / len(train_dataloader):.0f}%)]"
                f'\tLoss: {out_criterion["loss"].item():.3f} |'
                f'\tMSE loss: {out_criterion["mse_loss"].item() * 255 ** 2 / 3:.3f} |'
                f'\tBpp loss: {out_criterion["bpp_loss"].item():.2f} |'
                f'\ttraditional_loss: {out_criterion["traditional_loss"].item():.2f} |'
                f'\tadditional_loss: {out_criterion["additional_loss"].item():.2f} |'
                f"\tAux loss: {aux_loss.item():.2f}"
            )

def test_epoch(epoch, test_dataloader, model, criterion):
    model.eval()
    device = next(model.parameters()).device
    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    aux_loss = AverageMeter()
    with torch.no_grad():
        for d in test_dataloader:
            d = d.to(device)
            out_net = model(d)
            out_criterion = criterion(out_net, d)
            aux_loss.update(model.aux_loss())
            bpp_loss.update(out_criterion["bpp_loss"])
            loss.update(out_criterion["loss"])
            mse_loss.update(out_criterion["mse_loss"])

    print(
        f"Test epoch {epoch}: Average losses:"
        f"\tLoss: {loss.avg:.3f} |"
        f"\tMSE loss: {mse_loss.avg * 255 ** 2 / 3:.3f} |"
        f"\tBpp loss: {bpp_loss.avg:.2f} |"
        f"\tAux loss: {aux_loss.avg:.2f}\n"
    )
    return loss.avg


def save_checkpoint(state, is_best, filename):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename[:-8]+"_best"+filename[-8:])

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-m",
        "--model",
        default="cnn",
        choices=models.keys(),
        help="Model architecture (default: %(default)s)",
    )

    parser.add_argument(
            "-wavelet_model",
            "--wavelet_model",
            dest="wavelet_model_paths",
            type=str,
            nargs="*",
            required=True,
            help="wavelet_model checkpoint path",
        )
    parser.add_argument(
            "-LH_model",
            "--LH_model",
            dest="LH_model_paths",
            type=str,
            nargs="*",
            required=True,
            help="LH_model checkpoint path",
        )
    parser.add_argument(
            "-HL_model",
            "--HL_model",
            dest="HL_model_paths",
            type=str,
            nargs="*",
            required=True,
            help="HL_model checkpoint path",
        )
    parser.add_argument(
            "-HH_model",
            "--HH_model",
            dest="HH_model_paths",
            type=str,
            nargs="*",
            required=True,
            help="HH_model checkpoint path",
        )
    
    parser.add_argument(
            "--local_rank", 
            default=-1, 
            type=int,
            help="node rank for distributed training"
        )
    
    parser.add_argument(
            "-LH_diffusion_model",
            "--LH_diffusion_model",
            dest="LH_diffusion_model_paths",
            type=str,
            nargs="*",
            required=True,
            help="LH_diffusion_model checkpoint path",
        )
    parser.add_argument(
            "-HL_diffusion_model",
            "--HL_diffusion_model",
            dest="HL_diffusion_model_paths",
            type=str,
            nargs="*",
            required=True,
            help="HL_diffusion_model checkpoint path",
        )
    parser.add_argument(
            "-HH_diffusion_model",
            "--HH_diffusion_model",
            dest="HH_diffusion_model_paths",
            type=str,
            nargs="*",
            required=True,
            help="HH_diffusion_model checkpoint path",
        )

    parser.add_argument(
        "-d", "--dataset", type=str, required=True, help="Training dataset"
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=100,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=1e-4,
        type=float,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=30,
        help="Dataloaders threads (default: %(default)s)",
    )
    parser.add_argument(
        "--lambda",
        dest="lmbda",
        type=float,
        default=1e-2,
        help="Bit-rate distortion parameter (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=8, help="Batch size (default: %(default)s)"
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=16, 
        help="Test batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--aux-learning-rate",
        default=1e-3,
        type=float,
        help="Auxiliary loss learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=2,
        default=(256, 256),
        help="Size of the patches to be cropped (default: %(default)s)",
    )
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument(
        "--save", action="store_true", default=True, help="Save model to disk"
    )
    parser.add_argument(
        "--save_path", type=str, default="ckpt/model.pth.tar", help="Where to Save model"
    )
    parser.add_argument(
        "--seed", type=int, default=3407,help="Set random seed for reproducibility"
    )
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")
    args = parser.parse_args(argv)
    return args

def main(argv):
    args = parse_args(argv)
    print(args)
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    train_transforms = transforms.Compose(
        [transforms.RandomCrop(args.patch_size), transforms.ToTensor()]
    )

    test_transforms = transforms.Compose(
        [transforms.CenterCrop(args.patch_size), transforms.ToTensor()]
    )

    wavelet_model = load_checkpoint("cnn", (args.wavelet_model_paths)[0])
    wavelet_model.update(force=True)
    wavelet_model = wavelet_model.to(device)

    LH_model = Unet()
    HL_model = Unet()
    HH_model = Unet()
    LH_model.load_state_dict(torch.load((args.LH_model_paths)[0]))
    HL_model.load_state_dict(torch.load((args.HL_model_paths)[0]))
    HH_model.load_state_dict(torch.load((args.HH_model_paths)[0]))

    timesteps=10
    diffusion_model = GaussianDiffusion(timesteps)
    # diffusion_model=torch.nn.parallel.DistributedDataParallel(diffusion_model)

    diffusion_model_HL = UNetModel(
        in_channels=3,
        model_channels=96,
        out_channels=3,
        channel_mult=(1, 2, 2),
        attention_resolutions=[]
    )
    diffusion_model_HL.load_state_dict(torch.load((args.LH_diffusion_model_paths)[0]),strict=False)
    diffusion_model_HL = diffusion_model_HL.to(device)
    #diffusion_model_HL = CustomDataParallel(diffusion_model_HL)
    diffusion_model_HL=torch.nn.parallel.DistributedDataParallel(diffusion_model_HL)

    diffusion_model_LH = UNetModel(
        in_channels=3,
        model_channels=96,
        out_channels=3,
        channel_mult=(1, 2, 2),
        attention_resolutions=[]
    )
    diffusion_model_LH.load_state_dict(torch.load((args.LH_diffusion_model_paths)[0]),strict=False)
    diffusion_model_LH = diffusion_model_LH.to(device)
    #diffusion_model_LH = CustomDataParallel(diffusion_model_LH)
    diffusion_model_LH=torch.nn.parallel.DistributedDataParallel(diffusion_model_LH)

    diffusion_model_HH = UNetModel(
        in_channels=3,
        model_channels=96,
        out_channels=3,
        channel_mult=(1, 2, 2),
        attention_resolutions=[]
    )
    diffusion_model_HH.load_state_dict(torch.load((args.HH_diffusion_model_paths)[0]),strict=False)
    diffusion_model_HH = diffusion_model_HH.to(device)
    #diffusion_model_HH = CustomDataParallel(diffusion_model_HH)
    diffusion_model_HH=torch.nn.parallel.DistributedDataParallel(diffusion_model_HH)

    if torch.cuda.is_available():
        wavelet_model = wavelet_model.to(device)
        LH_model = LH_model.to(device)
        HL_model = HL_model.to(device)
        HH_model = HH_model.to(device)
        #diffusion_model_HL = diffusion_model_HL.to(device)
        #diffusion_model_LH = diffusion_model_LH.to(device)
        #diffusion_model_HH = diffusion_model_HH.to(device)
    train_dataset = ImageFolder(args.dataset, split="train", transform=train_transforms,
                                wavelet_model=wavelet_model, 
                                LH_model=LH_model, HL_model=HL_model, HH_model=HH_model, 
                                diffusion_model=diffusion_model,
                                diffusion_model_LH=diffusion_model_LH, diffusion_model_HL=diffusion_model_HL, diffusion_model_HH=diffusion_model_HH,          
                                )
    test_dataset = ImageFolder(args.dataset, split="test", transform=test_transforms,
                               wavelet_model=wavelet_model, 
                                LH_model=LH_model, HL_model=HL_model, HH_model=HH_model, 
                                diffusion_model=diffusion_model,
                                diffusion_model_LH=diffusion_model_LH, diffusion_model_HL=diffusion_model_HL, diffusion_model_HH=diffusion_model_HH,          
                                )

    # device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    sampler = DistributedSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler = sampler,
        #num_workers=args.num_workers,
        #shuffle=False,
        #pin_memory=(device == device),
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device == device),
    )

    net = models[args.model]()
    net = net.to(device)
    net = torch.nn.parallel.DistributedDataParallel(net).module
    #net = torch.nn.parallel.DistributedDataParallel(net)

    #if args.cuda and torch.cuda.device_count() > 1:
        #net = CustomDataParallel(net)
    optimizer, aux_optimizer = configure_optimizers(net, args)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.3, patience=4)
    test_criterion = RateDistortionLoss(lmbda=args.lmbda)
    train_criterion = URDLoss(lmbda=args.lmbda)
    last_epoch = 0
    if args.checkpoint:  # load from previous checkpoint
        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        last_epoch = checkpoint["epoch"] + 1
        net.load_state_dict(checkpoint["state_dict"])

        optimizer.load_state_dict(checkpoint["optimizer"])
        aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    best_loss = float("inf")
    for epoch in range(last_epoch, args.epochs):
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        train_one_epoch(
            net,
            train_criterion,
            train_dataloader,
            optimizer,
            aux_optimizer,
            epoch,
            args.clip_max_norm,
            diffusion_model,
            diffusion_model_LH,
            diffusion_model_HL,
            diffusion_model_HH
        )
        loss = test_epoch(epoch, test_dataloader, net, test_criterion)
        lr_scheduler.step(loss)

        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        if args.save:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": net.state_dict(),
                    "loss": loss,
                    "optimizer": optimizer.state_dict(),
                    "aux_optimizer": aux_optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                },
                is_best,
                args.save_path,
            )

if __name__ == "__main__":
    main(sys.argv[1:])
