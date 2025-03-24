from tqdm import tqdm
import matplotlib.pyplot as plt
import torch

from data.imagenet_subset import ImageNetSubset, get_dataloaders
from model.model import DiT

BATCH_SIZE = 128

if __name__ == "__main__":
    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(batch_size=BATCH_SIZE)
    arr = torch.rand((128, 3, 256, 256))
    dit = DiT()
    dit.patchify(arr)