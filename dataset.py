import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import torch

class TongueDataset(Dataset):
    def __init__(self, csv_file, image_root, label_cols):
        self.data = pd.read_csv(csv_file)
        self.image_root = image_root
        self.label_cols = label_cols
        self.transform = T.Compose([
            T.Resize((384, 384)),
            T.ToTensor(),
            T.Normalize([0.5]*3, [0.5]*3)
        ])
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.image_root, row['image_path'])
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        labels = torch.tensor(row[self.label_cols].values.astype(float), dtype=torch.float32)
        return image, labels
