from sklearn.metrics import f1_score
import torch

def calculate_metrics(preds, targets, threshold=0.5):
    preds = (preds > threshold).float()
    f1 = f1_score(targets.cpu(), preds.cpu(), average="macro")
    return f1
