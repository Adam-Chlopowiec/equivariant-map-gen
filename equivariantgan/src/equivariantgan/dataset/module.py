from enum import Enum
from pathlib import Path
from typing import Literal

import lightning as L
import numpy as np
import pandas as pd
import torch
from PIL import Image
from pydantic import BaseModel
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T


class KvasirLabels(Enum):
    NORMAL = "normal"
    EROSION = "erosion"
    ANGIECTASIA = "angiectasia"
    UNCLEAR_VIEW = "unclear_view"
    ULCER = "ulcer"
    BLOOD_FRESH = "blood_fresh"
    FOREIGN_BODY = "foreign_body"
    LYMPHANGIECTASIA = "lymphangiectasia"
    ERYTHEMA = "erythema"


class KvasirSample(BaseModel):
    filename: str
    label: KvasirLabels
    split: str
    full_filename: Path
    patient: str
    frame_number: int
    x1: float
    y1: float
    x2: float
    y2: float
    x3: float
    y3: float
    x4: float
    y4: float


def load_image_with_bbox(kvasir_sample: KvasirSample) -> tuple[Image, Image]:
    image = Image.open(kvasir_sample.full_filename)

    if kvasir_sample.label is KvasirLabels.NORMAL:
        return image, None, Image.fromarray(np.zeros_like(np.array(image)))

    bbox_coordinates = (
        kvasir_sample.x1,
        kvasir_sample.y1,
        kvasir_sample.x2,
        kvasir_sample.y2,
        kvasir_sample.x3,
        kvasir_sample.y3,
        kvasir_sample.x4,
        kvasir_sample.y4,
    )

    min_x = int(min(bbox_coordinates[0::2]))
    min_y = int(min(bbox_coordinates[1::2]))
    max_x = int(max(bbox_coordinates[0::2]))
    max_y = int(max(bbox_coordinates[1::2]))

    cropped_image = image.crop((min_x, min_y, max_x, max_y))

    mask = np.zeros_like(np.array(image))
    mask[min_y:max_y, min_x:max_x, :] = 255
    mask = Image.fromarray(mask)

    return image, cropped_image, mask


class KvasirDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        split: Literal["train", "validation", "test"],
        target_size: int,
    ) -> None:
        super().__init__()
        self.target_size = target_size
        self.data_root = Path(data_root)
        self.split_data = self.data_root / "kvasir_with_bboxes.csv"

        self.split_df = pd.read_csv(self.split_data)
        self.split_df = self.split_df[self.split_df["split"] == split]
        self.split_df["full_filename"] = self.split_df["full_filename"].map(
            lambda path: self.data_root / path
        )

        self.transforms = T.Compose(
            [
                T.Resize(self.target_size),
                T.PILToTensor(),
                T.ConvertImageDtype(torch.float),
            ]
        )

    def __len__(self) -> int:
        return len(self.split_df)

    def __getitem__(self, index: int) -> torch.Tensor:
        sample = KvasirSample(**dict(self.split_df.iloc[index]))

        image, cropped, mask = load_image_with_bbox(sample)

        return image, cropped, mask, sample.label


class KvasirDataModule(L.LightningDataModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: str):
        if stage == "fit":
            self.train_dataset = KvasirDataset(
                data_root=self.hparams.data_root,
                split="train",
                target_size=self.hparams.target_size,
            )

            self.validation_dataset = KvasirDataset(
                data_root=self.hparams.data_root,
                split="validation",
                target_size=self.hparams.target_size,
            )

        elif stage == "test":
            self.test_dataset = KvasirDataset(
                data_root=self.hparams.data_root,
                split="test",
                target_size=self.hparams.target_size,
            )
        else:
            raise ValueError(f"No such stage implemented: {stage}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.self.hparams.batch_size,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
            num_workers=self.self.hparams.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.validation_dataset,
            batch_size=self.self.hparams.batch_size,
            drop_last=True,
            num_workers=self.self.hparams.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.self.hparams.batch_size,
            drop_last=True,
            num_workers=self.self.hparams.num_workers,
        )
