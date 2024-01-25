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


class MapGan(pl.LightningModule):
    def __init__(
        self,
        *args,
        **kwargs
    ) -> None:
        self.save_hyperparameters()
        
        self.G = hydra.utils.instantiate(self.hparams.generator, _recursive_=False)
        self.D = hydra.utils.instantiate(self.hparams.discriminator, _recursive_=False)
        
        