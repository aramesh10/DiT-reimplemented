from tqdm import tqdm
import matplotlib.pyplot as plt
import torch

from model.model import DiT

import utils

BATCH_SIZE = 128

if __name__ == "__main__":
    
    vae_param = utils.get_VAE_params()
    
    train_dataloader, val_dataloader, test_dataloader = utils.get_dataloaders(batch_size=BATCH_SIZE)
    arr = torch.rand((128, 3, 256, 256))
    dit = DiT()
    dit.patchify(arr)