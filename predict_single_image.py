import torch
from torchvision import transforms as T
from PIL import Image
import numpy as np
from model import SwinClassifier

def predict_image(img_path, model_path):
    label_cols = ['TonguePale', 'TipSideRed', 'Spot', 'Ecchymosis', 'Crack', 'Toothmark',
                  'FurThick', 'FurYellow', 'Heart', 'Lung', 'Spleen', 'Liver', 'Kidney']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 👇 你要和 TongueDataset 用的一樣（調整這段保持一致）
    transform = T.Compose([
        T.Resize((384, 384)),
        T.ToTensor(),
        T.Normalize([0.5]*3, [0.5]*3)
    ])

    # 載入圖片
    img = Image.open(img_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)

    # 載入模型
    model = SwinClassifier(num_classes=len(label_cols))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # 預測
    with torch.no_grad():
        output = model(img)
        probs = torch.sigmoid(output).cpu().numpy()[0]
        preds = (probs > 0.5).astype(int)

    print(f"\n🧪 Predicting: {img_path}")
    for label, prob, pred in zip(label_cols, probs, preds):
        print(f"{label:12} | score: {prob:.2f} | {'✅ yes' if pred else '❌ no'}")

if __name__ == "__main__":
    predict_image("tongue_cut.jpg", "swin_best_fold1.pth")  # <- 你改成自己的圖和模型
