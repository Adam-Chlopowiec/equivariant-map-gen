from enum import Enum
from pathlib import Path
from typing import Literal, Dict, Union

import lightning as L
import numpy as np
import pandas as pd
import torch
import hydra
import glob
import imageio
import random
from PIL import Image
from pydantic import BaseModel
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T


class MapDataset(Dataset):
    def __init__(
        self,
        data_root: Path,
        target_size: int,
        rotate: bool = False,
    ) -> None:
        super().__init__()
        self.data_root = data_root
        self.terrain_map = self.data_root + '/terrain/*'
        self.roadmap_map = self.data_root + '/roadmap/*'
        self.target_size = target_size
        self.rotate = rotate
        
        self.transforms = T.Compose([
            T.Resize(self.target_size),
            T.ToTensor(),
            T.Normalize(
                [0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5]
            )
        ])
        
        self.terrain_paths = glob.glob(self.terrain_map)
        self.roadmap_paths = glob.glob(self.roadmap_map)
        
    def __len__(self) -> int:
        return len(self.terrain_paths)
    
    def _getimg(self, path: str) -> torch.Tensor:
        img = imageio.imread(path)
        img = np.asarray(img, dtype=np.uint8)
        img = Image.fromarray(img)
        return self.transforms(img)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        terrain_img = self._getimg(self.terrain_paths[idx])
        roadmap_img = self._getimg(self.roadmap_paths[idx])
        if self.rotate:
            terrain_img, roadmap_img = self.__rotate_d4(terrain_img, roadmap_img)
        
        return {
            'terrain': terrain_img,
            'roadmap': roadmap_img,
        }
        
    def __rotate_d4(self, image_A: Union[torch.Tensor, np.ndarray], image_B: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        k = random.randint(0, 3)
        return torch.rot90(image_A, k, dims=[-2, -1]), torch.rot90(image_B, k, dims=[-2, -1])
