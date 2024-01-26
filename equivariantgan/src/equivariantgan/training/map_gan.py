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
from PIL import Image
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torchvision.utils import make_grid
import torchvision.transforms as T
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from omegaconf import DictConfig

from equivariantgan.gan.pix2pix.discriminator import UNetDiscriminator
from equivariantgan.gan.pix2pix.generator import UNetGenerator


class GANLoss(nn.Module):
    def __init__(self, gan_mode: str, target_real_label: float = 1.0, target_fake_label: float = 0.0) -> None:
        super().__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.gan_mode = gan_mode
        self.denormalize = T.Normalize(
            [-1, -1, -1],
            [2, 2, 2]
        )
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
        lambda_l1: float,
        AtoB: str,
        *args,
        **kwargs
    ) -> None:
        self.save_hyperparameters()
        
        self.G = UNetGenerator()
        self.D = UNetDiscriminator()
        
        self.criterionGAN = GANLoss(self.hparams.loss_gan_mode).to(self.device)
        self.criterionL1 = nn.L1Loss()
        self.lambda_l1 = lambda_l1
        self.AtoB = AtoB
        
    def configure_optimizers(self) -> Tuple[nn.Module, nn.Module]:
        g_opt = torch.optim.Adam(self.G.parameters(), lr=2e-4)
        d_opt = torch.optim.Adam(self.D.parameters(), lr=2e-4)
        return g_opt, d_opt
        
    def set_input(self, input: Dict[str, torch.Tensor]) -> None:
        self.real_A = input['A' if self.AtoB else 'B'].to(self.device)
        self.real_B = input['B' if self.AtoB else 'A'].to(self.device)
        self.images = input['A_images' if self.AtoB else 'B_images']
        
    def forward(self, x) -> None:
        # self.fake_B = self.G(self.real_A)
        return self.G(x)
        
    def training_step(self, batch: torch.Tensor, batch_idx: int):
        g_opt, d_opt = self.optimizers()
        
        real_A = batch['terrain' if self.AtoB else 'roadmap'].to(self.device)
        real_B = batch['roadmap' if self.AtoB else 'terrain'].to(self.device)
        fake_B = self.forward(real_A)
        
        # Update D
        d_opt.zero_grad()
        log_real_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        log_fake_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        
        fake_AB = torch.cat((real_A, fake_B), 1)
        pred_fake = self.D(fake_AB.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.D(real_AB)
        loss_D_real = self.criterionGAN(pred_real, True)
        
        loss_D = (loss_D_fake + loss_D_real) * 0.5
        loss_D.backward()
        
        log_real_loss += loss_D_real.detach()
        log_fake_loss += loss_D_fake.detach()
        d_opt.step()
        
        self.log(
            "D_real",
            log_real_loss,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "D_fake",
            log_fake_loss,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "D",
            (log_real_loss + log_fake_loss) * 0.5,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        
        # Update G
        log_gan_G_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        log_l1_G_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        
        g_opt.zero_grad()
        self._set_requires_grad(self.D, False)
        
        fake_AB = torch.cat((real_A, fake_B), 1)
        pred_fake = self.D(fake_AB)
        loss_G_GAN = self.criterionGAN(pred_fake, True)
        
        loss_G_L1 = self.criterionL1(fake_B, real_B) * self.lambda_l1
        loss_G = loss_G_GAN + loss_G_L1
        
        loss_G.backward()
        
        g_opt.step()
        self._set_requires_grad(self.D, True)
        
        log_gan_G_loss += loss_G_GAN.detach()
        log_l1_G_loss += loss_G_L1.detach()
        
        self.log(
            "G_GAN",
            log_gan_G_loss,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "G_L1",
            log_l1_G_loss,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "G",
            log_gan_G_loss + log_l1_G_loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        
    def _set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
        
    def validation_step(self, *args, **kwargs):
        pass
    
    def test_step(self, batch: torch.Tensor, batch_idx: int):
        real_A = batch['terrain' if self.AtoB else 'roadmap'].to(self.device)
        real_B = batch['roadmap' if self.AtoB else 'terrain'].to(self.device)
        fake_B = self.forward(real_A)
        fake = self.denormalize(fake_B)
        
        grid = make_grid(fake, nrow=4).permute(1, 2, 0).cpu().numpy()
        grid = (grid * 255.).astype(np.uint8)
        grid = Image.fromarray(grid)
        self.logger.log_image(key='test/fake', images=[grid])  
        
    def on_validation_epoch_end(self, *args, **kwargs) -> None:
        batch = next(iter(self.trainer.datamodule.val_dataloader()))
        real_A = batch['terrain' if self.AtoB == 'AtoB' else 'roadmap']
        fake_B = self.G(real_A)
        fake = self.denormalize(fake_B)
        
        grid = make_grid(fake, nrow=4).permute(1, 2, 0).cpu().numpy()
        grid = (grid * 255.).astype(np.uint8)
        grid = Image.fromarray(grid)
        self.logger.log_image(key='val/fake', images=[grid])
