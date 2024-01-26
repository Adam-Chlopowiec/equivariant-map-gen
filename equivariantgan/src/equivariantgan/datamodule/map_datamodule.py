from typing import Optional, Tuple, Callable, Dict
import os
import pickle
import json
from logging import getLogger
from functools import partial
from pathlib import Path

import tqdm 
import torch
import hydra
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from omegaconf import DictConfig
from torchvision.transforms import Compose
from torch.utils.data import DataLoader, Dataset
import transformers
from transformers import AutoImageProcessor

from equivariantgan.dataset.map_dataset import MapDataset


logger = getLogger(__name__)


class MapDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_root: str,
        target_size: int,
        num_workers: int,
        batch_size: int,
    ) -> None:
        super().__init__()
        self.data_root = Path(data_root)
        self.target_size = target_size
        self.num_workers = num_workers
        self.batch_size = batch_size
        
    def prepare_data(self) -> None:
        pass
    
    def setup(self, stage: Optional[str] = None) -> None:
        pass
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            MapDataset(
                self.data_root / 'train',
                self.target_size
            ),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
            drop_last=True
        )
        
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            MapDataset(
                self.data_root / 'test',
                self.target_size
            ),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=True
        )
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            MapDataset(
                self.data_root / 'test',
                self.target_size
            ),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=True
        )
    
        
    