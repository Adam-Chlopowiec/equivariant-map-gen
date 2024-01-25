from typing import Any, Optional, Tuple, Dict, Callable, Union, Sequence
import os
import copy
import warnings

import tqdm
import torch
import torchvision
import hydra
import numpy as np
import pytorch_lightning as pl
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torchvision.utils import make_grid
from omegaconf import DictConfig


class GANLoss(nn.Module):
    def __init__(self, gan_mode: str, target_real_label: float = 1.0, target_fake_label: float = 1.0) -> None:
        super().__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        else:
            raise NotImplementedError(f'Gan Mode {gan_mode} not implemented')
        
    def get_target_tensor(self, prediction: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)
    
    def __call__(self, prediction: torch.Tensor, target_is_real: torch.Tensor) -> float:
        target_tensor = self.get_target_tensor(prediction, target_is_real)
        loss = self.loss(prediction, target_tensor)
        return loss
        


class MapGan(pl.LightningModule):
    def __init__(
        self,
        *args,
        **kwargs
    ) -> None:
        self.save_hyperparameters()
        
        self.G = hydra.utils.instantiate(self.hparams.generator, _recursive_=False)
        self.D = hydra.utils.instantiate(self.hparams.discriminator, _recursive_=False)
        
        self.criterionGAN = GANLoss(self.hparams.loss_gan_mode).to(self.device)
        self.criterionL1 = nn.L1Loss()
        self.lambda_l1 = self.hparams.lambda_l1
        
    def configure_optimizers(self) -> Tuple[nn.Module, nn.Module]:
        g_opt = hydra.utils.instantiate(self.hparams.g_optimizer, params=list(self.G.parameters()), _recursive_=False)
        d_opt = hydra.utils.instantiate(self.hparams.d_optimizer, params=list(self.D.parameters()), _recursive_=False)
        return g_opt, d_opt
        
    def set_input(self, input: Dict[str, torch.Tensor]) -> None:
        AtoB = self.hparams.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.images = input['A_images' if AtoB else 'B_images']
        
    def forward(self) -> None:
        self.fake_B = self.G(self.real_A)
        
    def backward_G(self) -> None:
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.D(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.D(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()
        
    def backward_G(self) -> None:
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.D(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.lambda_l1
        
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()
        
    def _set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
        
    def optimize_parameters(self):
        g_opt, d_opt = self.optimizers()
        self.forward()
        
        self._set_requires_grad(self.D, True)
        d_opt.zero_grad()
        self.backward_G()
        d_opt.step()
        
        self._set_requires_grad(self.D, False)
        g_opt.zero_grad()
        self.backward_G()
        g_opt.step()
