import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from dataset import TongueDataset
from model import SwinClassifier
from sklearn.metrics import f1_score, classification_report
import numpy as np
import pandas as pd
from tqdm import tqdm

# ✅ 論文風格的 class weight: alpha_j = median / Fj
def compute_alpha_weights(csv_path, label_cols):
    df = pd.read_csv(csv_path)
    pos_counts = np.array([df[col].sum() + 1e-5 for col in label_cols])
    median_freq = np.median(pos_counts)
    alpha_weights = median_freq / pos_counts
    return torch.tensor(alpha_weights, dtype=torch.float32)

def train_model(train_csv, val_csv, image_root, label_cols,
                num_epochs=10, batch_size=8, lr=1e-4, model_path="swin_best.pth"):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 準備資料集
    train_set = TongueDataset(train_csv, "images", label_cols, mask_root="masks")
    val_set = TongueDataset(val_csv, "images", label_cols,mask_root="masks"  )
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)

    # 初始化模型
    model = SwinClassifier(num_classes=len(label_cols))
    model.to(device)

    # 計算 alpha_j 權重
    alpha_weights = compute_alpha_weights(train_csv, label_cols)
    criterion = nn.BCEWithLogitsLoss(pos_weight=alpha_weights.to(device))
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    best_f1 = 0
    for epoch in range(num_epochs):
        model.train()
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} Training", leave=False)
        for imgs, labels in loop:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            loop.set_postfix(loss=loss.item())

        # 驗證
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                outputs = model(imgs)
                preds = torch.sigmoid(outputs).cpu().numpy()
                preds = (preds > 0.5).astype(int)
                all_preds.append(preds)
                all_labels.append(labels.cpu().numpy())

        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)

        f1 = f1_score(all_labels, all_preds, average='micro')
        print(f"Epoch {epoch+1}, Val Micro F1: {f1:.4f}")

        # 顯示每個 label 的 F1 分數（建議保留）
        print(classification_report(all_labels, all_preds, target_names=label_cols, digits=4))

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), model_path)
            print(f"✅ Saved best model to {model_path}")

if __name__ == "__main__":
    label_cols = ['TonguePale', 'TipSideRed', 'Spot', 'Ecchymosis',
                  'Crack', 'Toothmark', 'FurThick', 'FurYellow']
    
    for i in range(1, 6):
        train_csv = f"train_fold{i}.csv"
        val_csv = f"val_fold{i}.csv"
        model_path = f"swin_best_fold{i}.pth"
        print(f"====== Training Fold {i} ======")
        train_model(train_csv, val_csv, "images", label_cols, model_path=model_path)
