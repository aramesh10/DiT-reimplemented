import os
import csv
import yaml
from pprint import pprint

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from data.imagenet_subset import ImageNetSubset

def generate_csv():
    data = [["path", "label", "isVal"]]
    labels = ["brass_instrument", "castle", "chainsaw", "dog", "fish", "garbage_truck", "gas_station", "golf_ball", "parachute", "radio"]
    img_dir = "./data/"
    datasets_dir = ["train/", "val/"]
    for dataset in datasets_dir:
        isVal = dataset == "val/"
        for label in labels:
            file_path = os.path.join(img_dir, dataset, label)
            for f in os.listdir(file_path): 
                img_path = os.path.join(file_path, f)
                if os.path.isfile(img_path):
                    data.append([img_path, label, isVal])

    with open('imagenet_subset.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(data[0])
        writer.writerows(data[1:])

    print("CSV file 'imagenet_subset.csv' created successfully.")

def get_dataloaders(batch_size):
    csv_file_path = "./data/imagenet_subset.csv"
    if not os.path.exists(csv_file_path):
        generate_csv()
        
    transform = transforms.Compose([transforms.Resize((128, 128))])
    target_transform = None
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset=ImageNetSubset(csv_file_path=csv_file_path,
                               transform=transform,
                               target_transform=target_transform), 
        lengths=[0.7, 0.15, 0.15], 
        generator=torch.Generator().manual_seed(42))
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    return train_dataloader, val_dataloader, test_dataloader

def get_VAE_params():
    with open("./model/vq-f8-n256/config.yaml") as stream:
        try:
            vae_param = yaml.safe_load(stream)['model']['params']
        except yaml.YAMLError as exc:
            raise ValueError("YAML Error raised\n", yaml.YAMLError)
    return vae_param