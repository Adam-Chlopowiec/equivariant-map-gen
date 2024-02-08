from typing import Optional, Sequence

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import escnn
import tqdm
import yaml
import torchvision.transforms as T
from torchvision.transforms import functional as F
from escnn import gspaces
from PIL import Image
from omegaconf import DictConfig

from equivariantgan.datamodule.map_datamodule import MapDataModule
from equivariantgan.dataset.map_dataset import MapDataset
from equivariantgan.training.map_gan import MapGan

import torch
from torchvision.transforms import Resize, ToTensor
np.set_printoptions(linewidth=10000)


parser = argparse.ArgumentParser()
parser.add_argument('--config', required=True, help='Path to config')


def test_model_single_image(config, x: torch.Tensor, N: int = 4, k: int = 5):
    x = Image.fromarray(x.cpu().numpy().transpose(1, 2, 0), mode='RGB')

    # to reduce interpolation artifacts (e.g. when testing the model on rotated images),
    # we upsample an image by a factor of 3, rotate it and finally downsample it again
    resize = Resize(224) # to upsample
    totensor = ToTensor()
    x = resize(x)

    # evaluate the model on N rotated versions of the input image x
    print()
    print('##########################################################################################')
    header = 'angle  |  ' + '  '.join(["{:5d}".format(d) for d in range(10)])
    print(header)
    results = np.zeros(shape=(k * N, 10))
    with torch.no_grad():
        for i in range(1, k+1):
            model = hydra.utils.instantiate(config.model) # Hydra
            model.eval()
            for r in range(N):
                x_transformed = totensor(x.rotate(r*360./N, Image.BILINEAR)).reshape(3, 224, 224).unsqueeze(0)

                y = model(x_transformed)
                y = y.numpy().squeeze()
                results[i * r, :] = y
                
                angle = r * 360. / N
                if i == 1:
                    print("{:6.1f} : {}".format(angle, y))
    print(f"mean: {results.mean(axis=0)}")
    print(f"std: {results.std(axis=0)}")
    print('##########################################################################################')
    print()


def denormalize(image):
    denorm = T.Normalize(
        [-1, -1, -1],
        [2, 2, 2]
    )
    return denorm(image)


def compare_two_models_relative_errors(x_A, x_B, model, eq_model, W=225, AtoB='AtoB', zoom='Z16'):
    N = 4

    # We evaluate the equivariance error on N=4 rotations
    error_equivariant = []
    error_conventional = []
    
    generations_equivariant = []
    generations_conventional = []
    
    y_equivariant_can = eq_model(x_A)
    y_conventional_can = model(x_A)

    # for each of the N rotations
    for i in tqdm.tqdm(range(N)):
        # rotate the input
        x_A_transformed = torch.rot90(x_A, i, dims=[-2, -1]) 
        
        # compute the output of both models
        y_equivariant = eq_model(x_A_transformed)
        y_conventional = model(x_A_transformed)
        
        y_equivariant_canonical = torch.rot90(y_equivariant, -i, dims=[-2, -1])
        y_conventional_canonical = torch.rot90(y_conventional, -i, dims=[-2, -1])

        # compute the relative error of both models
        rel_error_equivariant = torch.norm(y_equivariant_can - y_equivariant_canonical).item()
        rel_error_conventional = torch.norm(y_conventional_can - y_conventional_canonical).item()

        error_equivariant.append(rel_error_equivariant)
        error_conventional.append(rel_error_conventional)
        generations_equivariant.append(y_equivariant_canonical)
        generations_conventional.append(y_conventional_canonical)

    # plot the error of both models as a function of the rotation angle theta
    fig, ax = plt.subplots(figsize=(10, 6))

    xs = [i / N * 2*np.pi for i in range(N)]
    plt.plot(xs, error_equivariant, label='Rotations Augmented CNN')
    plt.plot(xs, error_conventional, label='Conventional CNN')
    plt.title('Rotations Augmented vs Conventional CNNs', fontsize=20)
    plt.xlabel(r'$g = r_\theta$', fontsize=20)
    plt.ylabel('Equivariance Error', fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=15)
    plt.legend(fontsize=20)
    i = 0
    path = f"/home/adrian/pdp/equivariant-map-gen/equivariantgan/storage/figures/{AtoB}_{zoom}/"
    os.makedirs(path, exist_ok=False)
    name = "equivariance_measure_plot.png"
    plt.savefig(path + name)
    
    # Wykres 2 rows, 5 cols, oryginalny obraz, wygenerowane co 90st przekształcone do orientacji kanonicznej, pierwszy wiersz nieekwi, drugi ekwi
    for i in range(x_B.shape[0]):
        fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(12, 7))
        ax[0, 0].imshow(denormalize(x_B[i]).numpy().transpose(1, 2, 0))
        ax[0, 0].set_title('Original Target Image')
        
        ax[1, 0].imshow(denormalize(x_B[i]).numpy().transpose(1, 2, 0))
        ax[1, 0].set_title('Original Target Image')
        
        for j in range(4):
            # print(denormalize(generations_conventional[j][0]).numpy().shape)
            ax[0, 1 + j].imshow(denormalize(generations_conventional[j][i]).numpy().transpose(1, 2, 0))
            ax[0, 1 + j].set_title(f'Conventional Rot{90*j}')
            
            ax[1, 1 + j].imshow(denormalize(generations_equivariant[j][i]).numpy().transpose(1, 2, 0))
            ax[1, 1 + j].set_title(f'RotAug Rot{90*j}')
            
        plt.savefig(path + f'plot_comparison_{i:02}.png')


def test():
    """
    Quick check, whether the model works
    """
    opt = parser.parse_args()
    with open(opt.config, 'r') as f:
        config = yaml.safe_load(f)
    
    datamodule = MapDataModule(
        **config["datamodule"]
    )
    test_loader = datamodule.test_dataloader()
    
    it = iter(test_loader)
    for i in range(2):
        x = next(it)
    x_terrain = x["terrain"].squeeze()
    x_roadmap = x['roadmap'].squeeze()
    
    print(x_terrain.shape)
    print(x_roadmap.shape)
    
    # Load model from checkpoint
    print('Loading 1st ckpt')
    lightning_model_rot = MapGan.load_from_checkpoint('/home/adrian/pdp/equivariant-map-gen/equivariantgan/storage/data/map_gan/kzv4g1t4/checkpoints/epoch=499-step=293000.ckpt')
    eq_model = lightning_model_rot.G.cpu()
    eq_model.eval()
    
    # Load model from checkpoint
    print('Loading 2nd ckpt')
    lightning_model_no_rot = MapGan.load_from_checkpoint('/home/adrian/pdp/equivariant-map-gen/equivariantgan/storage/data/map_gan/p0tm216s/checkpoints/epoch=499-step=293000.ckpt') # Z16 AtoB
    model = lightning_model_no_rot.G.cpu()
    model.eval() 
    
    with torch.no_grad():
        print('Starting comparison')
        compare_two_models_relative_errors(x_terrain, x_roadmap, model, eq_model, AtoB='AtoB', zoom='Z16')
    

if __name__ == "__main__":
    test()
