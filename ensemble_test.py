import torch
from torch.utils.data import DataLoader
from dataset import TongueDataset
from model import SwinClassifier
from sklearn.metrics import classification_report
import numpy as np

def ensemble_test(test_csv, image_root, model_paths):
    label_cols = ['TonguePale', 'TipSideRed', 'RedSpot', 'Ecchymosis',
              'Crack', 'Toothmark', 'FurThick', 'FurYellow']
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = TongueDataset(test_csv, image_root, label_cols)
    loader = DataLoader(dataset, batch_size=8)

    # 載入所有模型
    models = []
    for path in model_paths:
        model = SwinClassifier(num_classes=len(label_cols))
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device)
        model.eval()
        models.append(model)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            
            # 各模型 sigmoid 機率加總
            sum_probs = torch.zeros((imgs.size(0), len(label_cols)), device=device)
            for model in models:
                outputs = model(imgs)
                probs = torch.sigmoid(outputs)
                sum_probs += probs

            avg_probs = sum_probs / len(models)
            preds = (avg_probs > 0.5).int().cpu().numpy()

            all_preds.append(preds)
            all_labels.append(labels.numpy())

    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    # 🔍 顯示每個類別的 support
    print("==== Support per class (Ground Truth Count) ====")
    for idx, name in enumerate(label_cols):
        print(f"{name:12s} support: {np.sum(all_labels[:, idx])}")

    print("\n==== Classification Report ====")
    print(classification_report(
        all_labels,
        all_preds,
        target_names=label_cols,
        zero_division=0
    ))

if __name__ == "__main__":
    model_paths = [f"swin_best_fold{i}.pth" for i in range(1, 6)]
    ensemble_test("test.csv", "images", model_paths)
