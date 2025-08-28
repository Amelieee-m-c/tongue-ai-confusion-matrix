import torch
from torch.utils.data import DataLoader
from dataset import TongueDataset
from model import SwinClassifierWithBranches
from sklearn.metrics import classification_report, multilabel_confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def test_model(test_csv, image_root, model_path="swin_best_fold1.pth"):
    label_cols = ['TonguePale', 'TipSideRed', 'Spot', 'Ecchymosis',
                  'Crack', 'Toothmark', 'FurThick', 'FurYellow']

    thresholds = {
        'TonguePale': 0.3,
        'TipSideRed': 0.5,
        'Spot': 0.5,
        'Ecchymosis': 0.3,
        'Crack': 0.5,
        'Toothmark': 0.5,
        'FurThick': 0.5,
        'FurYellow': 0.3
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = TongueDataset(test_csv, image_root, label_cols)
    loader = DataLoader(dataset, batch_size=8)

    model = SwinClassifierWithBranches(num_classes=len(label_cols))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    all_preds, all_labels = [], []

    with torch.no_grad():
        for (x_whole, x_body, x_edge), labels in loader:
            x_whole = x_whole.to(device)
            x_body = x_body.to(device)
            x_edge = x_edge.to(device)
            outputs = torch.sigmoid(model(x_whole, x_body, x_edge)).cpu().numpy()

            preds = np.zeros_like(outputs, dtype=int)
            for i, col in enumerate(label_cols):
                preds[:, i] = (outputs[:, i] > thresholds[col]).astype(int)

            all_preds.append(preds)
            all_labels.append(labels.numpy())

    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    print("\n每個類別的樣本數量:")
    for i, col in enumerate(label_cols):
        positives = int(all_labels[:, i].sum())
        negatives = int(len(all_labels) - positives)
        print(f"{col:10s} → positives={positives}, negatives={negatives}, total={len(all_labels)}")


    # --- Classification report ---
    print("Classification Report:\n")
    print(classification_report(all_labels, all_preds, target_names=label_cols, zero_division=0))

    # --- Multilabel confusion matrix ---
    mcm = multilabel_confusion_matrix(all_labels, all_preds)
    
    # 畫整合圖
    n_cols = 4  # 每列 4 個子圖
    n_rows = int(np.ceil(len(label_cols) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
    axes = axes.flatten()

    for i, col in enumerate(label_cols):
        sns.heatmap(mcm[i], annot=True, fmt='d', cmap='Blues', ax=axes[i])
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Actual')
        axes[i].set_title(col)

    # 若子圖比類別多，隱藏多餘子圖
    for j in range(len(label_cols), len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    test_model("test.csv", "images")




