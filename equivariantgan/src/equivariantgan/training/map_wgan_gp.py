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
import torch.autograd as autograd
from PIL import Image
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.autograd import Variable
from torchvision.utils import make_grid
import torchvision.transforms as T
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from omegaconf import DictConfig

from equivariantgan.gan.pix2pix.discriminator import UNetDiscriminator
from equivariantgan.gan.pix2pix.generator import UNetGenerator
from equivariantgan.gan.e2_pix2pix.discriminator import E2UNetDiscriminator
from equivariantgan.gan.e2_pix2pix.generator import E2UNetGenerator


class GANLoss(nn.Module):
    def __init__(self, gan_mode: str, target_real_label: float = 1.0, target_fake_label: float = 0.0) -> None:
        super().__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
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
        

class MapWGanGP(pl.LightningModule):
    def __init__(
        self,
        lambda_gp: float,
        lambda_l1: float,
        AtoB: bool,
        gan_type: str,
        *args,
        **kwargs
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        
        if gan_type == "pix2pix":
            self.G = UNetGenerator(**kwargs["generator"])
            self.D = UNetDiscriminator(**kwargs["discriminator"])
        else:
            self.G = E2UNetGenerator(**kwargs["generator"])
            self.D = E2UNetDiscriminator(**kwargs["discriminator"])
        
        # self.criterionGAN = GANLoss(self.hparams.loss_gan_mode).to(self.device)
        self.criterionL1 = nn.L1Loss()
        self.lambda_l1 = lambda_l1
        self.lambda_gp = lambda_gp
        self.AtoB = AtoB
        self.denormalize = T.Normalize(
            [-1, -1, -1],
            [2, 2, 2]
        )
        self.automatic_optimization = False
        
    def configure_optimizers(self) -> Tuple[nn.Module, nn.Module]:
        g_opt = torch.optim.Adam(self.G.parameters(), lr=self.hparams.lr_g)
        d_opt = torch.optim.Adam(self.D.parameters(), lr=self.hparams.lr_d)
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
        
        d_opt.zero_grad(set_to_none=True)
        log_real_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        log_fake_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        log_D_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        
        fake_AB = torch.cat((real_A, fake_B), 1)
        pred_fake = self.D(fake_AB.detach())
        
        real_AB = torch.cat((real_A, real_B), 1)
        pred_real = self.D(real_AB)
        
        # Gradient penalty
        gradient_penalty = self.compute_gradient_penalty(real_A, real_B, fake_B)
        # Adversarial loss
        loss_D_fake = torch.mean(pred_fake)
        loss_D_real = -torch.mean(pred_real)
        loss_D = loss_D_real + loss_D_fake + self.lambda_gp * gradient_penalty
        
        self.manual_backward(loss_D)
        
        log_real_loss += loss_D_real.detach()
        log_fake_loss += loss_D_fake.detach()
        log_D_loss += loss_D.detach()
        d_opt.step()
        
        self.log(
            "D_real",
            log_real_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "D_fake",
            log_fake_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "D",
            log_D_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        
        if batch_idx % self.hparams.g_n_steps == 0:
            # Update G
            log_G_wasserstein_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
            log_G_L1_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
            log_G_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
            
            fake_B = self.forward(real_A)
            
            g_opt.zero_grad(set_to_none=True)
            self._set_requires_grad(self.D, False)
            
            fake_AB = torch.cat((real_A, fake_B), 1)
            pred_fake = self.D(fake_AB)
            
            loss_G_L1 = self.criterionL1(fake_B, real_B) * self.lambda_l1
            loss_G_wasserstein = -torch.mean(pred_fake)
            loss_G = loss_G_wasserstein + loss_G_L1
            
            self.manual_backward(loss_G, retain_graph=True)
            
            log_G_wasserstein_loss += loss_G_wasserstein.detach()
            log_G_L1_loss += loss_G_L1.detach()
            log_G_loss += loss_G.detach()
            g_opt.step()
            self._set_requires_grad(self.D, True)
            
            self.log(
                "G_wasserstein",
                log_G_wasserstein_loss,
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                sync_dist=True,
            )
            self.log(
                "G_L1",
                log_G_L1_loss,
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                sync_dist=True,
            )
            self.log(
                "G",
                log_G_loss,
                on_step=True,
                on_epoch=True,
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
        real_B = self.denormalize(real_B)
        real_A = self.denormalize(real_A)
        
        grid = make_grid(fake, nrow=4).permute(1, 2, 0).cpu().numpy()
        grid = (grid * 255.).astype(np.uint8)
        grid = Image.fromarray(grid)
        self.logger.log_image(key='test/fake', images=[grid])  
        
        grid = make_grid(real_B, nrow=4).permute(1, 2, 0).cpu().numpy()
        grid = (grid * 255.).astype(np.uint8)
        grid = Image.fromarray(grid)
        self.logger.log_image(key='test/real_B', images=[grid])
        
        grid = make_grid(real_A, nrow=4).permute(1, 2, 0).cpu().numpy()
        grid = (grid * 255.).astype(np.uint8)
        grid = Image.fromarray(grid)
        self.logger.log_image(key='test/real_A', images=[grid])
        
        cat = torch.cat((fake, real_B), 0)
        grid = make_grid(cat, nrow=8).permute(1, 2, 0).cpu().numpy()
        grid = (grid * 255.).astype(np.uint8)
        grid = Image.fromarray(grid)
        self.logger.log_image(key='test/fake-real', images=[grid])
        
        # for fake, real in zip(fake, real_B):
        #     grid = make_grid(torch.cat(fake.unsqueeze(0), real.unsqueeze(0), dim=0), nrow=1).permute(1, 2, 0).cpu().numpy()
        #     grid = (grid * 255.).astype(np.uint8)
        #     grid = Image.fromarray(grid)
        #     self.logger.log_image(key='test/fake-real', images=[grid])
        
        
        
    def on_validation_epoch_end(self, *args, **kwargs) -> None:
        batch = next(iter(self.trainer.datamodule.val_dataloader()))
        
        real_A = batch['terrain' if self.AtoB else 'roadmap'].to(self.device)
        real_B = batch['roadmap' if self.AtoB else 'terrain'].to(self.device)
        fake_B = self.forward(real_A)
        fake = self.denormalize(fake_B)
        real_B = self.denormalize(real_B)
        real_A = self.denormalize(real_A)
        
        grid = make_grid(fake, nrow=4).permute(1, 2, 0).cpu().numpy()
        grid = (grid * 255.).astype(np.uint8)
        grid = Image.fromarray(grid)
        self.logger.log_image(key='val/fake', images=[grid])  
        
        grid = make_grid(real_B, nrow=4).permute(1, 2, 0).cpu().numpy()
        grid = (grid * 255.).astype(np.uint8)
        grid = Image.fromarray(grid)
        self.logger.log_image(key='val/real_B', images=[grid])
        
        grid = make_grid(real_A, nrow=4).permute(1, 2, 0).cpu().numpy()
        grid = (grid * 255.).astype(np.uint8)
        grid = Image.fromarray(grid)
        self.logger.log_image(key='val/real_A', images=[grid])
        
        cat = torch.cat((fake, real_B), 0)
        grid = make_grid(cat, nrow=8).permute(1, 2, 0).cpu().numpy()
        grid = (grid * 255.).astype(np.uint8)
        grid = Image.fromarray(grid)
        self.logger.log_image(key='val/fake-real', images=[grid])
        
    def compute_gradient_penalty(self, real_A, real_B, fake_B):
        """Calculates the gradient penalty loss for WGAN GP"""
        # Random weight term for interpolation between real and fake samples
        alpha = torch.tensor(np.random.random((real_B.size(0), 1, 1, 1)), dtype=torch.float).to(self.device)
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_B + ((1 - alpha) * fake_B)).requires_grad_(True)
        interp_cat = torch.cat((real_A, interpolates), 1)
        d_interpolates = self.D(interp_cat)
        # fake = torch.ones((real_B.shape[0], 1, d_interpolates.shape[2], d_interpolates.shape[3]), dtype=torch.float, requires_grad=False).to(self.device)
        # print(d_interpolates.shape)
        # print(interp_cat.shape)
        # print(fake.shape)
        # Get gradient w.r.t. interpolates
        gradients = autograd.grad(
            outputs=d_interpolates,
            inputs=interp_cat,
            grad_outputs=d_interpolates.new_ones(d_interpolates.shape),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty
