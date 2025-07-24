import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import torch
import numpy as np
import cv2

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
        img_pil = Image.open(img_path).convert("RGB")

        # 原圖（不遮）
        img_whole = self.transform(img_pil)

        # 建立 mask（舌頭為非黑）
        img_np = np.array(img_pil)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        mask = (gray > 10).astype(np.uint8) * 255  # 二值舌頭 mask

        # 舌身：腐蝕取內部
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
        body_mask = cv2.erode(mask, kernel, iterations=1)

        # 舌邊 = 原始 mask - 舌身
        edge_mask = cv2.subtract(mask, body_mask)

        # 遮罩應用在原圖
        img_body_np = cv2.bitwise_and(img_np, img_np, mask=body_mask)
        img_edge_np = cv2.bitwise_and(img_np, img_np, mask=edge_mask)

        # 轉回 PIL
        img_body = self.transform(Image.fromarray(img_body_np))
        img_edge = self.transform(Image.fromarray(img_edge_np))

        labels = torch.tensor(row[self.label_cols].values.astype(float), dtype=torch.float32)

        return (img_whole, img_body, img_edge), labels
