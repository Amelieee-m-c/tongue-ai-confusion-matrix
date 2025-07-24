import torch
from torch.utils.data import DataLoader
from dataset import TongueDataset
from model import SwinClassifier
from sklearn.metrics import classification_report
import numpy as np

def test_model(test_csv, image_root, model_path="swin_best_fold1.pth"):
    label_cols = ['TonguePale', 'TipSideRed', 'RedSpot', 'Ecchymosis',
              'Crack', 'Toothmark', 'FurThick', 'FurYellow']
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = TongueDataset(test_csv, image_root, label_cols)
    loader = DataLoader(dataset, batch_size=8)

    model = SwinClassifier(num_classes=len(label_cols))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            preds = torch.sigmoid(outputs).cpu().numpy()
            preds = (preds > 0.5).astype(int)
            all_preds.append(preds)
            all_labels.append(labels.numpy())

    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    print(classification_report(all_labels, all_preds, target_names=label_cols))

if __name__ == "__main__":
    test_model("test.csv", "images")
