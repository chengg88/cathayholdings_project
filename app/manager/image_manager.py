from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
import torch
import pandas as pd
import warnings
import numpy as np
import os
from PIL import Image
warnings.filterwarnings("ignore")
"""
影像讀取，篩選/前處理等在這
"""
class image_manager:
    def __init__(self, img_path, df_path, logger):
        self.img_path = img_path
        self.df_path = df_path
        self.logger = logger

    def get_image_info(self):
        self.df = pd.read_csv(self.df_path)
        self.df["Image"] = self.df["Image"].apply(lambda x: x+".jpg")
        self.logger.info("Image info loaded")
        return print("Complete image info processes")

    def creat_dataset(self, height, width, transform):
        self.dataset = CustomImageDataset(annotations_file=self.df,
                                    img_dir=self.img_path, height = height, width=width,
                                    transform = transform)

    def train_test_split(self, val_size, test_size, shuffle_dataset, random_seed, batch_size):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Training on {device}")
        dataset_size = len(self.dataset)
        indices = list(range(dataset_size))
        val_spilt = int(np.floor(val_size * dataset_size))
        test_spilt = val_spilt + int(np.floor(test_size * dataset_size))

        if shuffle_dataset:
            np.random.seed(random_seed)
            np.random.shuffle(indices)
            
        train_indices, val_indices, test_indices = indices[test_spilt:], indices[:val_spilt], indices[val_spilt:test_spilt]
        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)
        test_sampler = SubsetRandomSampler(test_indices)
        train_loader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, sampler=train_sampler)
        val_loader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, sampler=val_sampler)
        test_loader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, sampler=test_sampler)
        return train_loader, val_loader, test_loader


class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, height, width, transform=None, target_transform=None):
        self.img_labels = annotations_file[["Class"]]
        self.img_name =  annotations_file[["Image"]]
        self.img_name = self.img_name
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.height = height
        self.width = width

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_name.iloc[idx, 0])
        image = Image.open(img_path)
        label = self.img_labels.iloc[idx, 0]  # Debug: 要修改成[idx, 0]才是取數值，不會把欄位名稱一起誤抓        
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label