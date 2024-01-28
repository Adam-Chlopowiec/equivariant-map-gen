from enum import Enum
from pathlib import Path
from typing import Literal, Dict

import lightning as L
import numpy as np
import pandas as pd
import torch
import hydra
import glob
import imageio
from PIL import Image
from pydantic import BaseModel
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T


class MapDataset(Dataset):
    def __init__(
        self,
        data_root: Path,
        target_size: int,
    ) -> None:
        super().__init__()
        self.data_root = data_root
        self.terrain_map = self.data_root + '/terrain/*'
        self.roadmap_map = self.data_root + '/roadmap/*'
        self.target_size = target_size
        
        self.transforms = T.Compose([
            T.Resize(self.target_size),
            T.ToTensor(),
            # T.PILToTensor(),
            # T.ConvertImageDtype(torch.float),
            T.Normalize(
                [0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5]
                # [0.5],
                # [0.5]
            )
        ])
        
        self.terrain_paths = glob.glob(self.terrain_map)
        self.roadmap_paths = glob.glob(self.roadmap_map)
        
    def __len__(self) -> int:
        return len(self.terrain_paths)
    
    def _getimg(self, path: str) -> torch.Tensor:
        # img = Image.open(path)
        img = imageio.imread(path)
        img = np.asarray(img, dtype=np.uint8)
        img = Image.fromarray(img)
        return self.transforms(img)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        terrain_img = self._getimg(self.terrain_paths[idx])
        roadmap_img = self._getimg(self.roadmap_paths[idx])
        return {
            'terrain': terrain_img,
            'roadmap': roadmap_img,
        }
