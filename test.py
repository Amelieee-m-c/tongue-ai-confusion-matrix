import torch
from torch.utils.data import DataLoader
from dataset import TongueDataset
from model import SwinClassifier
from sklearn.metrics import classification_report

def test_model(test_csv, image_root, model_path="swin_best.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = TongueDataset(test_csv, image_root)
    loader = DataLoader(dataset, batch_size=8)
    
    model = SwinClassifier()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            preds = torch.argmax(outputs, dim=1).cpu()
            all_preds.extend(preds)
            all_labels.extend(labels)

    print(classification_report(all_labels, all_preds))

if __name__ == "__main__":
    test_model("test.csv", "images")
