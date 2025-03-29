
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.io import read_image

class ImageNetSubset(Dataset):
    def __init__(self, csv_file_path, transform=None, target_transform=None):
        self.data = pd.read_csv(csv_file_path, skiprows=[0])
        self.transform = transform
        self.target_transform = target_transform
        self.num_of_classes = 10

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx, 0]        
        image = read_image(img_path)
        if self.is_grayscale(image):
            image = torch.stack((image,image,image)).squeeze()
        label = self.get_onehot_label(self.data.iloc[idx, 1])

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label
    
    def is_grayscale(self, img):
        return img.shape[0] == 1

    def get_onehot_label(self, label):
        labels = ["brass_instrument", "castle", "chainsaw", "dog", "fish", "garbage_truck", "gas_station", "golf_ball", "parachute", "radio"]
        return torch.nn.functional.one_hot(torch.tensor([labels.index(label)]), num_classes=10)
    
