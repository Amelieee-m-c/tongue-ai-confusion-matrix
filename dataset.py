import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import torch
import numpy as np
import cv2

class TongueDataset(Dataset):
    def __init__(self, csv_file, image_root, label_cols, mask_root=None):
        self.data = pd.read_csv(csv_file)
        self.image_root = image_root
        self.label_cols = label_cols
        self.mask_root = mask_root  # 加入 mask 路徑
        self.transform = T.Compose([
            T.Resize((384, 384)),
            T.ToTensor(),
            T.Normalize([0.5]*3, [0.5]*3)
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_name = row['image_path']
        img_path = os.path.join(self.image_root, img_name)
        image = Image.open(img_path).convert("RGB")

        if self.mask_root:
            mask_path = os.path.join(self.mask_root, img_name)
            if os.path.exists(mask_path):
                image = self.align_tongue(image, mask_path)

        image = self.transform(image)
        labels = torch.tensor(row[self.label_cols].values.astype(float), dtype=torch.float32)
        return image, labels

    def align_tongue(self, image_pil, mask_path):
        # Convert PIL image to OpenCV format
        image_np = np.array(image_pil)
        mask = np.array(Image.open(mask_path).convert("L"))
        mask_bin = (mask > 127).astype(np.uint8)

        # Find contours
        contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return Image.fromarray(image_np)

        cnt = max(contours, key=cv2.contourArea)

        # Fit ellipse to estimate angle
        if len(cnt) < 5:
            return Image.fromarray(image_np)  # Not enough points
        ellipse = cv2.fitEllipse(cnt)
        angle = ellipse[2]

        # Compute center of tongue
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            return Image.fromarray(image_np)
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        # Rotate around center
        h, w = image_np.shape[:2]
        rot_mat = cv2.getRotationMatrix2D((cx, cy), angle-90, 1.0)
        aligned = cv2.warpAffine(image_np, rot_mat, (w, h), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))

        return Image.fromarray(aligned)
